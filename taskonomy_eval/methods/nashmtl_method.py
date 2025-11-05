from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn, optim

from son_goku import TaskSpec  # same TaskSpec used for SON-GOKU
from .base import MultiTaskMethod, register_method


@register_method("nashmtl")
class NashMTLMethod(MultiTaskMethod):
    """
    Nash-MTL (Navon et al., ICML 2022) style gradient aggregation.

    Idea (Algorithm 1 in the paper):
      * For each task i, compute the gradient g_i of its loss w.r.t shared params.
      * Form the Gram matrix G_ij = <g_i, g_j>.
      * Find positive coefficients alpha in R^K such that G^T G alpha ≈ 1 / alpha
        (elementwise), which corresponds to a Nash bargaining solution in gradient
        space.
      * Use the combined update direction sum_i alpha_i * g_i.

    Here we approximate the solution by:
      * Building the Gram matrix from flattened gradients.
      * Optimizing alpha (constrained to be positive and re-normalized to sum 1)
        to drive the residuals (log alpha_i + log beta_i(α))^2 toward zero, where
        beta = G alpha.

    We only use shared parameters to build the Gram matrix, but apply the final
    weighted loss to all parameters (shared + task-specific) via a scalarized loss.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter,
        base_optimizer: optim.Optimizer,
        inner_lr: float = 0.1,
        max_inner_iter: int = 20,
        eps: float = 1e-8,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.n_tasks = len(self.tasks)
        self.base_optimizer = base_optimizer

        if device is not None:
            self.device = device
        else:
            self.device = next(self.model.parameters()).device

        self.inner_lr = inner_lr
        self.max_inner_iter = max_inner_iter
        self.eps = eps

        # Shared parameters are the ones used to build the bargaining game.
        shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not shared_params:
            # Fallback: use all trainable params if filter selects none
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.shared_params: List[nn.Parameter] = shared_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_grad_list(grad_list: List[torch.Tensor]) -> torch.Tensor:
        """Flatten a list of parameter gradients into a 1D vector."""
        return torch.cat([g.contiguous().view(-1) for g in grad_list])

    def _compute_gram(self, shared_grads: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Build Gram matrix G where G_ij = <g_i, g_j> from per-task gradient lists.
        """
        flat_vecs = [self._flatten_grad_list(gs) for gs in shared_grads]  # [K] of [D]
        G = torch.stack(flat_vecs, dim=0)  # [K, D]
        gram = G @ G.t()  # [K, K]
        return gram

    def _solve_nash_weights(self, gram: torch.Tensor) -> torch.Tensor:
        """
        Approximate solution α to G^T G α = 1 / α in the sense of
        driving φ_i(α) = log α_i + log β_i(α) toward zero, where β = G α.

        We minimize R(α) = Σ_i φ_i(α)^2 via gradient descent on α,
        keeping α positive and re-normalizing to sum 1 for stability.

        Returns:
            alpha: tensor of shape [K] on self.device, dtype float32.
        """
        K = gram.shape[0]
        if K == 1:
            return torch.ones(1, device=self.device, dtype=torch.float32)

        # Work on CPU (K is small) with float64 for stability
        gram = gram.detach().cpu().to(dtype=torch.float64)
        eps = float(self.eps)

        alpha = np.full(K, 1.0 / K, dtype=np.float64)

        for _ in range(self.max_inner_iter):
            # Make alpha a torch tensor with gradient
            alpha_t = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
            beta = gram @ alpha_t  # [K]

            # Ensure positivity for logs
            alpha_clamped = torch.clamp(alpha_t, min=eps)
            beta_clamped = torch.clamp(beta, min=eps)

            phi = torch.log(alpha_clamped) + torch.log(beta_clamped)
            R = torch.sum(phi ** 2)

            R.backward()
            with torch.no_grad():
                grad = alpha_t.grad
                if grad is None:
                    break
                alpha_new = alpha_t - self.inner_lr * grad
                # keep α positive
                alpha_new = torch.clamp(alpha_new, min=eps)
                # normalize to sum 1 for a stable scale of the update direction
                alpha_new = alpha_new / alpha_new.sum()
                alpha = alpha_new.detach().numpy()

        # Back to model device / dtype
        alpha_final = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        # Safety renormalization
        s = float(alpha_final.sum().item())
        if s <= 0:
            alpha_final = torch.full_like(alpha_final, 1.0 / K)
        else:
            alpha_final = alpha_final / s
        return alpha_final

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        One training step of Nash-MTL.

        Assumes:
          * `batch` is already on the correct device (runner handles this).
          * `self.model` is on the same device.
        """
        self.model.train()
        self.base_optimizer.zero_grad(set_to_none=True)

        raw_losses: List[torch.Tensor] = []

        # 1) Compute per-task losses
        for spec in self.tasks:
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            raw_losses.append(Li)

        # 2) Compute gradients w.r.t. shared parameters for each task
        shared_grads: List[List[torch.Tensor]] = []
        for Li in raw_losses:
            grads_i = torch.autograd.grad(
                Li,
                self.shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grads_i_filled: List[torch.Tensor] = []
            for g, p in zip(grads_i, self.shared_params):
                if g is None:
                    grads_i_filled.append(torch.zeros_like(p))
                else:
                    grads_i_filled.append(g)
            shared_grads.append(grads_i_filled)

        # 3) Build Gram matrix and solve for Nash weights α
        gram = self._compute_gram(shared_grads)
        alpha = self._solve_nash_weights(gram)  # [K], on model device

        # 4) Scalarized loss with Nash weights; backprop through ALL params
        total_loss = torch.zeros((), device=self.device)
        for w, Li in zip(alpha, raw_losses):
            total_loss = total_loss + w * Li

        total_loss.backward()
        self.base_optimizer.step()

        # 5) Logging
        logs: Dict[str, float] = {}
        logs["loss/total"] = float(total_loss.detach().item())
        for w, Li, spec in zip(alpha.tolist(), raw_losses, self.tasks):
            logs[f"loss/{spec.name}"] = float(Li.detach().item())
            logs[f"weight/{spec.name}"] = float(w)

        return logs

    # Optional: simple state hooks
    def state_dict(self) -> Dict[str, Any]:
        return {
            "inner_lr": self.inner_lr,
            "max_inner_iter": self.max_inner_iter,
            "eps": self.eps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.inner_lr = state.get("inner_lr", self.inner_lr)
        self.max_inner_iter = state.get("max_inner_iter", self.max_inner_iter)
        self.eps = state.get("eps", self.eps)
