# taskonomy_eval/methods/cagrad_method.py

from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import nn

from .base import MultiTaskMethod, register_method


@register_method("cagrad")
class CAGradMethod(MultiTaskMethod):
    """
    Conflict-Averse Gradient Descent (CAGrad) for multi-task learning.

    This implementation follows the dual formulation described in:
      Bo Liu et al., "Conflict-averse gradient descent for multi-task learning", NeurIPS 2021,
    and the re-derivation in FAMO (Liu et al., 2023) where the CAGrad dual objective is:

        F(w) = g_w^T g_0 + c ||g_w|| ||g_0||,   w in simplex

    with g_0 the gradient of the average loss, and g_w = sum_i w_i g_i.

    In practice:
      1. We compute per-task gradients g_i over all trainable parameters.
      2. Build G \in R^{K x D}, K = num tasks, D = total params.
      3. Optimize w on the simplex (projected gradient descent) to approximately minimize F(w).
      4. Form the final update direction as in the CAGrad toy code:
            g_0 = mean_i g_i
            g_w = sum_i w_i g_i
            λ = c * ||g_0|| / ||g_w||
            g = (g_0 + λ g_w) / (1 + c)
      5. Set model gradients to g and call optimizer.step().
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: List[Any],               # TaskSpec-like objects with .name and .loss_fn(model, batch)
        base_optimizer: torch.optim.Optimizer,
        device: torch.device,
        c: float = 0.5,                 # CAGrad trade-off parameter
        inner_lr: float = 0.1,          # LR for weight optimization in simplex
        inner_steps: int = 20,          # # of PGD steps on w per outer step
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.optimizer = base_optimizer
        self.device = device

        self.c = float(c)
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.eps = float(eps)

        # Treat *all* trainable params as "shared" for CAGrad
        self.params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        self._numels: List[int] = [p.numel() for p in self.params]
        self._total_numel: int = int(sum(self._numels))

    # -----------------------------
    # Helpers: flattening / unflattening
    # -----------------------------
    def _flatten_grads(self) -> torch.Tensor:
        """
        Flatten gradients of all trainable params into a single 1D tensor.
        Missing grads are treated as zeros.
        """
        flats = []
        for p in self.params:
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=self.device))
            else:
                flats.append(p.grad.view(-1).to(self.device))
        return torch.cat(flats, dim=0)

    def _set_flat_grads(self, flat_grad: torch.Tensor) -> None:
        """
        Given a flattened gradient vector (same ordering as self.params),
        reshape and assign into each parameter's .grad.
        """
        assert flat_grad.numel() == self._total_numel

        offset = 0
        for p, n in zip(self.params, self._numels):
            g_slice = flat_grad[offset : offset + n].view_as(p)
            offset += n

            # Make sure grad tensor exists and is on the right device
            if p.grad is None:
                p.grad = g_slice.detach().clone()
            else:
                p.grad.detach().copy_(g_slice)

    # -----------------------------
    # Simplex projection (Duchi et al., 2008)
    # -----------------------------
    @staticmethod
    def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        """
        Project v onto the probability simplex {w | w >= 0, sum w = 1}.

        Args:
            v: 1D tensor of shape (K,)

        Returns:
            w: 1D tensor, same shape, projected onto simplex.
        """
        assert v.dim() == 1
        K = v.numel()
        # Sort in descending order
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, K + 1, device=v.device, dtype=v.dtype)
        cond = u > cssv / ind

        if not torch.any(cond):
            # Fallback: uniform weights
            w = torch.full_like(v, 1.0 / K)
            return w

        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho + 1.0)
        w = v - theta
        w = torch.clamp(w, min=0.0)
        w_sum = w.sum()
        if w_sum <= 0:
            w = torch.full_like(v, 1.0 / K)
        else:
            w = w / w_sum
        return w

    # -----------------------------
    # CAGrad weight solver in the simplex
    # -----------------------------
    def _solve_cagrad_weights(self, G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve approximately for w in the simplex that minimizes:

            F(w) = g_w^T g_0 + c ||g_w|| ||g_0||

        where G \in R^{K x D} has task gradients as rows,
              g_0 = mean_i g_i,
              g_w = sum_i w_i g_i.

        We do a small number of projected gradient steps on w.

        Args:
            G: [K, D] tensor of flattened per-task grads

        Returns:
            g:   [D] aggregated gradient direction (CAGrad)
            w:   [K] task weights on simplex
        """
        device = G.device
        K, D = G.shape

        # g_0: gradient of average loss
        g0 = G.mean(dim=0)  # [D]
        norm0 = g0.norm() + self.eps

        # Initialize w uniformly on simplex
        w = torch.full((K,), 1.0 / K, device=device)

        for _ in range(self.inner_steps):
            gw = torch.mv(G.t(), w)  # g_w = G^T w, shape [D]
            normw = gw.norm() + self.eps

            # Gradient of F(w) wrt w:
            #   ∂F/∂w = G g0 + c * ||g0|| / ||gw|| * G gw
            grad_w = torch.mv(G, g0) + self.c * norm0 / normw * torch.mv(G, gw)

            # Gradient step then project onto simplex
            w = w - self.inner_lr * grad_w
            w = self._project_simplex(w)

        # Final direction: use the same closed-form as in the original CAGrad toy code
        gw = torch.mv(G.t(), w)
        normw = gw.norm() + self.eps

        lam = self.c * norm0 / normw
        g = g0 + lam * gw
        g = g / (1.0 + self.c)

        return g, w

    # -----------------------------
    # Main training step
    # -----------------------------
    def step(self, batch: Dict[str, Any], global_step: Optional[int] = None) -> Dict[str, float]:
        """
        One optimization step using CAGrad.

        Args:
            batch: dictionary from the DataLoader (rgb + per-task targets)
            global_step: unused, kept for API compatibility

        Returns:
            logs: dict of scalars (loss per task, mean loss, task weights, etc.)
        """
        self.model.train()

        # Move batch to device
        batch_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_device[k] = v.to(self.device, non_blocking=True)
            else:
                batch_device[k] = v

        num_tasks = len(self.tasks)
        grads = []
        losses = []

        # Compute per-task gradients (over *all* params) separately
        for spec in self.tasks:
            # Each spec.loss_fn(model, batch_device) should return a scalar
            self.optimizer.zero_grad(set_to_none=True)
            loss_t = spec.loss_fn(self.model, batch_device)
            losses.append(loss_t.detach())
            loss_t.backward()
            flat_grad = self._flatten_grads()
            grads.append(flat_grad)

        G = torch.stack(grads, dim=0)  # [K, D]

        # Compute CAGrad combined direction
        g_cagrad, w = self._solve_cagrad_weights(G)

        # Apply the aggregated gradient to the model and step
        self.optimizer.zero_grad(set_to_none=True)
        self._set_flat_grads(g_cagrad)
        self.optimizer.step()

        # Logging
        logs: Dict[str, float] = {}

        # Per-task and mean loss
        for spec, loss_t in zip(self.tasks, losses):
            logs[f"loss/{spec.name}"] = float(loss_t.item())
        logs["loss/mean"] = float(torch.stack(losses).mean().item())

        # CAGrad weights per task
        for spec, wi in zip(self.tasks, w):
            logs[f"cagrad/w_{spec.name}"] = float(wi.item())

        # Some norm diagnostics (optional)
        logs["cagrad/norm_g0"] = float(G.mean(dim=0).norm().item())
        logs["cagrad/norm_g"] = float(g_cagrad.norm().item())

        return logs
