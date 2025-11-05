from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import random
import torch
from torch import nn, optim

from son_goku import TaskSpec  # same TaskSpec you already use

from .base import MultiTaskMethod, register_method


@register_method("pcgrad")
class PCGradMethod(MultiTaskMethod):
    """
    PCGrad: Gradient Projection for Multi-Task Learning
    (Yu et al., NeurIPS 2020).

    At each training step:

      1. Compute per-task losses L_i.
      2. For each task i, compute gradients g_i = ∇_θ L_i for *all* trainable params.
      3. For each task i, iterate over tasks j in random order; if <g_i, g_j> < 0,
         project g_i onto the normal plane of g_j:
             g_i <- g_i - ( <g_i, g_j> / ||g_j||^2 ) * g_j
      4. Average the resulting projected gradients over tasks:
             g_pcgrad = (1/T) Σ_i g_i
      5. Apply g_pcgrad as the model gradient and take an optimizer step.

    Notes:
      * We operate on gradients for all trainable parameters, not just shared ones.
      * The `shared_param_filter` is accepted for API symmetry but not used directly.
      * `batch` is assumed to already be on the correct device (runner handles this).
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter,   # kept for interface symmetry; not used directly
        base_optimizer: optim.Optimizer,
        device: torch.device | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.base_optimizer = base_optimizer
        self.eps = eps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # We'll apply PCGrad to *all* trainable parameters
        self.params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]

    # ---------------- gradient utilities ----------------

    @staticmethod
    def _dot(gi: List[torch.Tensor], gj: List[torch.Tensor]) -> torch.Tensor:
        """Inner product <g_i, g_j> over lists of parameter gradients."""
        s = torch.zeros((), device=gi[0].device)
        for a, b in zip(gi, gj):
            s = s + (a * b).sum()
        return s

    @staticmethod
    def _norm_sq(g: List[torch.Tensor]) -> torch.Tensor:
        """Squared L2 norm ||g||^2."""
        s = torch.zeros((), device=g[0].device)
        for t in g:
            s = s + (t * t).sum()
        return s

    # ---------------- main API ----------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        Perform one PCGrad training step on a multi-task batch.

        Assumes:
          * batch tensors are already on the same device as the model.
        """
        self.model.train()
        self.base_optimizer.zero_grad(set_to_none=True)

        num_tasks = len(self.tasks)

        # 1) Compute per-task losses and gradients
        raw_losses: List[torch.Tensor] = []
        losses_float: List[float] = []
        grads_per_task: List[List[torch.Tensor]] = []

        for spec in self.tasks:
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            raw_losses.append(Li)
            losses_float.append(float(Li.detach().item()))

            grads = torch.autograd.grad(
                Li,
                self.params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            # Replace None with zeros_like(param) so we can safely do vector ops
            grads_clean: List[torch.Tensor] = []
            for p, g in zip(self.params, grads):
                if g is None:
                    grads_clean.append(torch.zeros_like(p, device=self.device))
                else:
                    grads_clean.append(g.detach())
            grads_per_task.append(grads_clean)

        # 2) If only one task, PCGrad degenerates to vanilla SGD/Adam on that loss
        if num_tasks == 1:
            for p, g in zip(self.params, grads_per_task[0]):
                p.grad = g
            self.base_optimizer.step()

            logs: Dict[str, float] = {
                "loss/total": float(raw_losses[0].detach().item()),
                f"loss/{self.tasks[0].name}": float(losses_float[0]),
            }
            return logs

        # 3) PCGrad projection for each task's gradient
        #    We follow the "pairwise projection in random order" scheme.
        proj_grads: List[List[torch.Tensor]] = [
            [g.clone() for g in g_list] for g_list in grads_per_task
        ]

        task_indices = list(range(num_tasks))

        for i in range(num_tasks):
            # Shuffle the order of comparison tasks j for each i
            order = task_indices.copy()
            random.shuffle(order)
            for j in order:
                if j == i:
                    continue
                gi = proj_grads[i]           # current projected grad for task i
                gj = grads_per_task[j]       # *original* grad for task j

                gij = self._dot(gi, gj)      # <gi, gj>
                if gij.item() < 0.0:
                    gj_norm_sq = self._norm_sq(gj)
                    if gj_norm_sq.item() > self.eps:
                        coeff = gij / (gj_norm_sq + self.eps)
                        # gi <- gi - coeff * gj
                        for k in range(len(gi)):
                            gi[k] = gi[k] - coeff * gj[k]
                    # if gj_norm_sq is ~0, skip projection (undefined direction)

        # 4) Average the projected gradients over tasks to get final gradient
        final_grads: List[torch.Tensor] = []
        for param_idx in range(len(self.params)):
            accum = torch.zeros_like(self.params[param_idx], device=self.device)
            for t_idx in range(num_tasks):
                accum = accum + proj_grads[t_idx][param_idx]
            final_grads.append(accum / float(num_tasks))

        # 5) Assign final gradients and take an optimizer step
        for p, g in zip(self.params, final_grads):
            p.grad = g
        self.base_optimizer.step()

        # 6) Logging
        total_loss = sum(raw_losses) / float(num_tasks)
        logs: Dict[str, float] = {}
        logs["loss/total"] = float(total_loss.detach().item())
        for spec, lv in zip(self.tasks, losses_float):
            logs[f"loss/{spec.name}"] = float(lv)

        return logs

    # (optional) checkpointing hooks
    def state_dict(self) -> Dict[str, Any]:
        return {"eps": self.eps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.eps = state.get("eps", self.eps)
