from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch
from torch import nn, optim

# Only used for typing / consistency with SON-GOKU code – adjust if your TaskSpec lives elsewhere.
from son_goku import TaskSpec

from .base import MultiTaskMethod, register_method


@register_method("gradnorm")
class GradNormMethod(MultiTaskMethod):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    Chen et al., ICML 2018.

    This expects:
      - a shared backbone with task-specific heads
      - TaskSpec objects with .loss_fn(model, batch) defined exactly as for SON-GOKU
      - a shared_param_filter that marks which params belong to the shared trunk
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter: Callable[[nn.Parameter], bool],
        base_optimizer: optim.Optimizer,
        alpha: float = 1.5,     # asymmetry hyperparam in the paper
        weight_lr: float = 0.025,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Shared parameters W used in gradient norms
        self.shared_params: List[nn.Parameter] = [
            p for p in self.model.parameters() if shared_param_filter(p)
        ]

        num_tasks = len(self.tasks)

        # Learnable log-weights; we exponentiate + renormalize to keep w_i > 0, sum w_i = num_tasks
        self.log_w = nn.Parameter(torch.zeros(num_tasks, device=self.device), requires_grad=True)
        self.weight_optimizer = optim.Adam([self.log_w], lr=weight_lr)

        self.initial_losses: torch.Tensor | None = None  # L_i(0), set on first step

    # Convenience: positive weights that sum to num_tasks
    @property
    def w(self) -> torch.Tensor:
        w = torch.exp(self.log_w)
        return (len(self.tasks) * w) / (w.sum() + 1e-8)

    def _raw_losses(self, batch: Mapping[str, Any]) -> torch.Tensor:
        losses = []
        for t in self.tasks:
            li = t.loss_fn(self.model, batch)
            losses.append(li)
        return torch.stack(losses)  # [T]

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        # Move tensors to device once
        batch = {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        # -----------------------------
        # 1) Compute raw per-task losses
        # -----------------------------
        self.model.zero_grad(set_to_none=True)
        self.base_optimizer.zero_grad(set_to_none=True)
        self.weight_optimizer.zero_grad(set_to_none=True)

        raw_losses = self._raw_losses(batch)  # shape [T]

        if self.initial_losses is None:
            # First-iteration losses L_i(0) (detached so we don't keep a giant graph)
            self.initial_losses = raw_losses.detach()

        # -------------------------------------------------------
        # 2) GradNorm update for loss weights (log_w parameters)
        # -------------------------------------------------------
        weights = self.w  # uses current log_w (requires_grad=True)

        # Gradient norms g_i = ||∂ (w_i * L_i)/∂W_shared ||_2
        grad_norms: List[torch.Tensor] = []
        for i, Li in enumerate(raw_losses):
            Li_scaled = weights[i] * Li
            grads = torch.autograd.grad(
                Li_scaled,
                self.shared_params,
                retain_graph=True,
                create_graph=True,   # needed so g_i depends on log_w
            )
            # Flatten over layers, L2 norm
            g_norm = torch.norm(torch.stack([g.norm(2) for g in grads]))
            grad_norms.append(g_norm)
        G = torch.stack(grad_norms)           # [T]
        G_avg = G.mean().detach()             # treat target mean as constant

        # Relative inverse training rates r_i(t)
        with torch.no_grad():
            loss_ratio = raw_losses / (self.initial_losses + 1e-8)  # L_i(t)/L_i(0)
            inv_train_rate = loss_ratio / (loss_ratio.mean() + 1e-8)  # r_i(t)

        # Target gradient norms: G_i* = G_avg * r_i(t)^alpha
        target_G = (G_avg * (inv_train_rate ** self.alpha)).detach()

        gradnorm_loss = torch.sum(torch.abs(G - target_G))
        gradnorm_loss.backward()
        self.weight_optimizer.step()

        # -------------------------------------------
        # 3) Update model weights with *new* w_i's
        #    (weights are frozen during this step)
        # -------------------------------------------
        self.model.zero_grad(set_to_none=True)
        self.base_optimizer.zero_grad(set_to_none=True)

        weights_detached = self.w.detach()
        total_loss = torch.sum(weights_detached * raw_losses)  # gradients only w.r.t model

        total_loss.backward()
        self.base_optimizer.step()

        # Logging
        out: Dict[str, float] = {
            "loss/total": float(total_loss.item()),
            "loss/gradnorm": float(gradnorm_loss.item()),
        }
        for i, t in enumerate(self.tasks):
            out[f"loss/{t.name}"] = float(raw_losses[i].item())
            out[f"weight/{t.name}"] = float(weights_detached[i].item())
            out[f"g_norm/{t.name}"] = float(G[i].detach().item())
        return out
