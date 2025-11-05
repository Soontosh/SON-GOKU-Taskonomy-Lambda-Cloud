from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch
from torch import nn, optim

from son_goku import TaskSpec  # same TaskSpec used elsewhere

from .base import MultiTaskMethod, register_method


@register_method("fairgrad")
class FairGradMethod(MultiTaskMethod):
    """
    FairGrad (Ban & Ji, ICML 2024) style method for MTL.

    FairGrad solves, at each step, an α-fair utility maximization problem in the
    space of task gradients:

        maximize_d  ∑_i (g_i^T d)^{1-α} / (1-α)
        subject to  g_i^T d ≥ 0,  d ∈ B_ε

    Under the ansatz d = Σ_i w_i g_i, the optimal weights satisfy the nonlinear
    system:

        (G^T G) w = w^{-1/α},                                  (Eq. 4 in paper)

    where G is the matrix of task gradients (columns g_i), and w^{-1/α} is
    applied elementwise. We approximate the solution w by gradient descent on
    the residual ||G^T G w - w^{-1/α}||^2 (with positivity constraints).

    Implementation notes here:

    * We use gradients w.r.t. shared parameters (via shared_param_filter) to
      construct G and solve for w.
    * Then we use those weights in a *weighted loss*:

          L_total = Σ_i w_i * L_i,

      and backpropagate L_total to update all parameters.
    * This is fully compatible with your SON-GOKU / GradNorm / MGDA harness.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter: Callable[[nn.Parameter], bool],
        optimizer: optim.Optimizer | None = None,
        base_optimizer: optim.Optimizer | None = None,
        device: torch.device | None = None,
        # FairGrad-specific hyperparameters
        alpha: float = 0.5,           # α in α-fairness (α=0 -> LS, α≈1 prop. fairness, α=2 MPD, large -> max-min-ish)
        solver_lr: float = 0.1,       # learning rate in w-space for solving Eq. (4)
        solver_steps: int = 20,       # number of iterations to refine w each batch
        w_min: float = 1e-4,          # minimum weight to avoid w<=0
        reg_eps: float = 1e-8,        # small diagonal regularizer on G^T G
    ) -> None:
        super().__init__()

        opt = base_optimizer or optimizer
        if opt is None:
            raise ValueError("FairGradMethod expects `optimizer` or `base_optimizer`.")

        self.model = model
        self.tasks: List[TaskSpec] = list(tasks)
        self.optimizer: optim.Optimizer = opt

        self.num_tasks = len(self.tasks)
        self.task_names = [t.name for t in self.tasks]

        # Device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Shared parameters for gradient geometry
        shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not shared_params:
            # Fallback: use all trainable parameters if filter is empty
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.shared_params: List[nn.Parameter] = shared_params

        # FairGrad hyperparameters
        self.alpha = float(alpha)
        self.solver_lr = float(solver_lr)
        self.solver_steps = int(solver_steps)
        self.w_min = float(w_min)
        self.reg_eps = float(reg_eps)

        # Keep a running estimate of w for warm-starting the solver
        if self.num_tasks > 0:
            init_w = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        else:
            init_w = torch.tensor([], device=self.device)
        self._w_state = init_w

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shared_grads(
        self, batch: Mapping[str, Any]
    ) -> tuple[List[torch.Tensor], List[float]]:
        """
        For each task i, compute:

          L_i = task loss (scalar)
          g_i = gradient of L_i w.r.t. shared parameters

        Returns:
          grads_flat: list of length T, each a 1-D tensor of concatenated shared grads
          losses    : list of length T, python floats with the detached loss values
        """
        grads_flat: List[torch.Tensor] = []
        losses: List[float] = []

        # We do separate forward+grad for each task.
        for spec in self.tasks:
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            losses.append(float(Li.detach().item()))

            # Gradient w.r.t. shared params only, no optimizer step here
            gi = torch.autograd.grad(
                Li,
                self.shared_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            flat = torch.cat([g.detach().reshape(-1) for g in gi])
            grads_flat.append(flat)

        return grads_flat, losses

    def _solve_weights(self, GtG: torch.Tensor) -> torch.Tensor:
        """
        Approximately solve (G^T G) w = w^{-1/α} for w>0 via gradient descent
        on 0.5 * ||G^T G w - w^{-1/α}||^2.

        Args:
            GtG: [T, T] matrix = G^T G where G has task gradients as columns.

        Returns:
            w: [T] tensor of positive weights on the same device as GtG.
        """
        T = self.num_tasks
        if T == 0:
            return torch.tensor([], device=GtG.device)

        # α=0 → LS: equal weights
        if abs(self.alpha) < 1e-8:
            return torch.ones(T, device=GtG.device) / T

        alpha = self.alpha
        A = GtG + self.reg_eps * torch.eye(T, device=GtG.device)

        # Warm-start from previous state (moved to correct device)
        w = self._w_state.to(GtG.device)

        # Make sure w is valid length
        if w.numel() != T:
            w = torch.ones(T, device=GtG.device) / T

        for _ in range(self.solver_steps):
            # Enforce positivity
            w = w.clamp(min=self.w_min)

            # Residual: r = A w - w^{-1/α}
            w_pow = w.pow(-1.0 / alpha)
            r = A @ w - w_pow  # [T]

            # Gradient of 0.5 * ||r||^2 w.r.t. w:
            #   ∇ = J^T r,  where J = ∂r/∂w = A + diag((1/α) * w^{-1/α - 1})
            diag_term = (1.0 / alpha) * w.pow(-1.0 / alpha - 1.0)  # [T]
            # A is symmetric, so A^T = A. diag_term is diagonal entries.
            grad = A @ r + diag_term * r  # [T]

            # Gradient descent step in w-space
            w = w - self.solver_lr * grad

        # Final clamp to ensure positivity
        w = w.clamp(min=self.w_min)

        # Cache for warm-start next time (detach from graph)
        self._w_state = w.detach()

        return w

    # ------------------------------------------------------------------
    # Public API: one training step
    # ------------------------------------------------------------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        One FairGrad training step.

        Assumes:
          * `batch` is already on the correct device (handled in the runner)
          * `self.model` is on that same device
        """
        self.model.train()
        logs: Dict[str, float] = {}

        # --------------------------------------------------------------
        # 1) Probe gradients on shared parameters for each task
        # --------------------------------------------------------------
        grads_flat, loss_vals = self._compute_shared_grads(batch)

        # Stack to form G, then compute G^T G
        if self.num_tasks > 0:
            # grads_flat: list of [P] → G: [P, T]
            G = torch.stack(grads_flat, dim=1)  # [P, T]
            GtG = G.t() @ G                     # [T, T]
            w = self._solve_weights(GtG)       # [T]
        else:
            w = torch.tensor([], device=self.device)

        # Log per-task probe losses and weights
        for name, lv in zip(self.task_names, loss_vals):
            logs[f"loss/{name}"] = float(lv)
        if self.num_tasks > 0:
            for i, name in enumerate(self.task_names):
                logs[f"weight/{name}"] = float(w[i].item())
            logs["weight/sum"] = float(w.sum().item())

        # --------------------------------------------------------------
        # 2) Actual update: weighted sum of task losses
        # --------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for idx, spec in enumerate(self.tasks):
            Li = spec.loss_fn(self.model, batch)  # fresh forward
            if self.num_tasks > 0:
                total_loss = total_loss + w[idx] * Li
            else:
                total_loss = total_loss + Li  # degenerate single-task case

        total_loss.backward()
        self.optimizer.step()

        logs["loss/total"] = float(total_loss.detach().item())

        return logs

    # ------------------------------------------------------------------
    # Checkpoint helpers (optional)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "solver_lr": self.solver_lr,
            "solver_steps": self.solver_steps,
            "w_min": self.w_min,
            "reg_eps": self.reg_eps,
            "w_state": self._w_state.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "alpha" in state:
            self.alpha = float(state["alpha"])
        if "solver_lr" in state:
            self.solver_lr = float(state["solver_lr"])
        if "solver_steps" in state:
            self.solver_steps = int(state["solver_steps"])
        if "w_min" in state:
            self.w_min = float(state["w_min"])
        if "reg_eps" in state:
            self.reg_eps = float(state["reg_eps"])
        if "w_state" in state:
            self._w_state = state["w_state"].to(self.device)
