from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn, optim

# We reuse the same TaskSpec you already use for SON-GOKU / GradNorm
from son_goku import TaskSpec

from .base import MultiTaskMethod, register_method


@register_method("mgda")
class MGDAMethod(MultiTaskMethod):
    """
    MGDA-UB-style multi-task learning (Sener & Koltun, NeurIPS 2018).

    At each step:
      1. Compute per-task losses L_i.
      2. Compute gradients of each L_i w.r.t. shared parameters.
      3. Solve a small convex QP on the simplex to find weights w_i that minimize
         the norm of the weighted sum of gradients: ||Σ w_i ∇_W L_i||_2.
      4. Take an update step using the scalarized loss Σ w_i L_i.

    Notes:
      * We only use "shared" parameters (given by shared_param_filter) when
        building the gradient Gram matrix, just like SON-GOKU and GradNorm.
      * The final backward pass is done on ALL model parameters, so both
        shared trunk and task heads are updated.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter: Callable[[nn.Parameter], bool],
        base_optimizer: optim.Optimizer,
        grad_normalization: str = "none",  # 'none' | 'l2' | 'loss' (optional)
        max_qp_iter: int = 250,
        qp_tol: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.base_optimizer = base_optimizer
        self.grad_normalization = grad_normalization
        self.max_qp_iter = max_qp_iter
        self.qp_tol = qp_tol
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Shared parameters used to build gradient vectors for MGDA
        shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not shared_params:
            # Fallback: use all params if the filter selects none
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.shared_params: List[nn.Parameter] = shared_params

    # ------------------------------------------------------------------
    # Utilities for gradient handling and QP on the simplex
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_grad_list(grad_list: List[torch.Tensor]) -> torch.Tensor:
        """Flatten a list of parameter gradients into a single 1D vector."""
        return torch.cat([g.contiguous().view(-1) for g in grad_list])

    @staticmethod
    def _project_to_simplex(y: np.ndarray) -> np.ndarray:
        """
        Project a vector y onto the probability simplex:
            { z : z_i >= 0, sum_i z_i = 1 }.

        Implemented via a simple bisection on the Lagrange multiplier.
        """
        y = np.asarray(y, dtype=np.float64)
        n = y.size
        if n == 1:
            return np.array([1.0], dtype=np.float64)

        # Bisection on lambda such that sum(max(y - lambda, 0)) = 1
        lower = y.min() - 1.0
        upper = y.max()
        for _ in range(50):
            mid = 0.5 * (lower + upper)
            z = y - mid
            z[z < 0.0] = 0.0
            s = z.sum()
            if s > 1.0:
                lower = mid
            else:
                upper = mid

        z = y - upper
        z[z < 0.0] = 0.0
        s = z.sum()
        if s > 0:
            z /= s
        else:
            # Degenerate case: all entries zero; fall back to uniform
            z[:] = 1.0 / n
        return z

    @staticmethod
    def _solve_two_task_qp(v11: float, v12: float, v22: float) -> np.ndarray:
        """
        Closed-form solution for the 2-task case.

        We minimize || c g1 + (1-c) g2 ||_2^2 w.r.t c in [0,1], where
        v11 = g1^T g1, v22 = g2^T g2, v12 = g1^T g2.
        """
        denom = v11 + v22 - 2.0 * v12
        if denom <= 0.0:
            # Degenerate or colinear gradients; fall back to 0.5 / 0.5
            return np.array([0.5, 0.5], dtype=np.float64)

        c_star = (v22 - v12) / denom
        c_star = float(np.clip(c_star, 0.0, 1.0))
        return np.array([c_star, 1.0 - c_star], dtype=np.float64)

    def _solve_min_norm_weights(self, gram: np.ndarray) -> np.ndarray:
        """
        Given Gram matrix G_ij = <g_i, g_j>, find weights w on the simplex
        that minimize ||Σ w_i g_i||^2 = w^T G w.

        Uses:
          * exact closed-form for 2 tasks
          * projected gradient descent on the simplex for >2 tasks
        """
        n = gram.shape[0]
        if n == 1:
            return np.array([1.0], dtype=np.float64)

        if n == 2:
            return self._solve_two_task_qp(
                float(gram[0, 0]),
                float(gram[0, 1]),
                float(gram[1, 1]),
            )

        # n >= 3: projected gradient descent on simplex
        w = np.full(n, 1.0 / n, dtype=np.float64)

        # Rough Lipschitz estimate: 2 * max diag(G)
        L = float(np.max(np.diag(gram)) * 2.0)
        if not np.isfinite(L) or L <= 0.0:
            # Degenerate case: all gradients zero
            return w

        step_size = 1.0 / L

        for _ in range(self.max_qp_iter):
            grad = 2.0 * gram.dot(w)  # gradient of w^T G w
            w_next = self._project_to_simplex(w - step_size * grad)
            if np.sum(np.abs(w_next - w)) < self.qp_tol:
                w = w_next
                break
            w = w_next

        return w

    def _normalize_grads(
        self,
        grad_lists: List[List[torch.Tensor]],
        loss_values: List[float],
    ) -> List[List[torch.Tensor]]:
        """
        Optionally normalize per-task gradients to match MGDA-UB's variants.

        grad_normalization:
          * 'none'  : leave gradients as-is
          * 'l2'    : divide each task's grads by its L2 norm
          * 'loss'  : divide each task's grads by its loss value
        """
        if self.grad_normalization == "none":
            return grad_lists

        norms: List[float] = []
        if self.grad_normalization == "l2":
            for grads in grad_lists:
                sq_sum = 0.0
                for g in grads:
                    sq_sum += float(g.pow(2).sum().item())
                norms.append(np.sqrt(sq_sum) + 1e-8)
        elif self.grad_normalization == "loss":
            for lv in loss_values:
                norms.append(abs(lv) + 1e-8)
        else:
            raise ValueError(f"Unknown grad_normalization: {self.grad_normalization}")

        normed: List[List[torch.Tensor]] = []
        for grads, n in zip(grad_lists, norms):
            inv = 1.0 / n
            normed.append([g * inv for g in grads])
        return normed

    def _compute_mgda_weights(
        self,
        grad_lists: List[List[torch.Tensor]],
        loss_values: List[float],
    ) -> np.ndarray:
        """
        Build Gram matrix from per-task gradients and solve for MGDA weights.
        """
        # Optional gradient normalization per task
        grad_lists = self._normalize_grads(grad_lists, loss_values)

        # Flatten and stack
        flat_vecs: List[torch.Tensor] = [
            self._flatten_grad_list(gs) for gs in grad_lists
        ]
        # Shape: [num_tasks, num_params]
        G = torch.stack(flat_vecs)  # type: ignore[arg-type]

        # Gram matrix on CPU / numpy for the small QP
        gram = (G @ G.t()).detach().cpu().numpy().astype(np.float64)

        # Solve for weights on simplex
        w = self._solve_min_norm_weights(gram)  # numpy array, shape [T]
        # Safety: renormalize to sum to 1
        s = w.sum()
        if s <= 0.0:
            w[:] = 1.0 / len(w)
        else:
            w /= s
        return w

    # ------------------------------------------------------------------
    # Main public API: one training step
    # ------------------------------------------------------------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        Perform one MGDA training step on a (multi-task) batch.

        Assumes:
          * `batch` has already been moved to the correct device by the caller.
          * `self.model` is on the same device.
        """
        self.model.train()
        self.base_optimizer.zero_grad(set_to_none=True)

        raw_losses: List[torch.Tensor] = []
        loss_values: List[float] = []
        shared_grads: List[List[torch.Tensor]] = []

        # 1) Per-task losses and gradients w.r.t. shared parameters
        for spec in self.tasks:
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            raw_losses.append(Li)
            loss_values.append(float(Li.detach().item()))

            grads = torch.autograd.grad(
                Li,
                self.shared_params,
                retain_graph=True,
                create_graph=False,
            )
            # Detach since we only need numerical values for MGDA QP
            shared_grads.append([g.detach() for g in grads])

        # 2) Compute MGDA weights (on CPU/numpy)
        weights = self._compute_mgda_weights(shared_grads, loss_values)  # numpy array

        # 3) Scalarized loss with MGDA weights, backprop through ALL params
        total_loss = 0.0
        for w, Li in zip(weights, raw_losses):
            total_loss = total_loss + float(w) * Li

        total_loss.backward()
        self.base_optimizer.step()

        # 4) Logging
        logs: Dict[str, float] = {}
        logs["loss/total"] = float(total_loss.detach().item())
        for w, lv, spec in zip(weights, loss_values, self.tasks):
            logs[f"loss/{spec.name}"] = float(lv)
            logs[f"weight/{spec.name}"] = float(w)

        return logs

    # ------------------------------------------------------------------
    # (Optional) checkpointing hooks
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "grad_normalization": self.grad_normalization,
            "max_qp_iter": self.max_qp_iter,
            "qp_tol": self.qp_tol,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.grad_normalization = state.get("grad_normalization", self.grad_normalization)
        self.max_qp_iter = state.get("max_qp_iter", self.max_qp_iter)
        self.qp_tol = state.get("qp_tol", self.qp_tol)
