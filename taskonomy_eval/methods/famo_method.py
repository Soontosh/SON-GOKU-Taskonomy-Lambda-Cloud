# taskonomy_eval/methods/famo_method.py

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from son_goku import TaskSpec

from .base import MultiTaskMethod, register_method


class _FAMOController:
    """
    Implements the FAMO loss-weighting update from:
      "FAMO: Fast Adaptive Multitask Optimization" (NeurIPS 2023).

    This class keeps a low-dimensional set of logits (one per task) and
    updates them so that the log-losses of different tasks decrease
    in a more balanced way.
    """

    def __init__(
        self,
        num_tasks: int,
        min_losses: torch.Tensor,
        alpha: float = 2.5e-2,
        weight_decay: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cpu")

        self.num_tasks = num_tasks
        self.device = torch.device(device)

        # Lower bounds on each task's loss (e.g. 0 for non-negative losses).
        self.min_losses = min_losses.to(self.device)

        # Raw parameters that define a simplex of weights via softmax.
        # (called "xi" or "w" in the paper; we use "logits" here)
        self.logits = torch.zeros(num_tasks, device=self.device, requires_grad=True)

        # Separate optimizer just for the weighting parameters.
        self.opt = torch.optim.Adam(
            [self.logits],
            lr=alpha,
            weight_decay=weight_decay,
        )

    def combine_losses(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Given per-task losses (shape: [K]), return a single scalar loss
        using the FAMO weighting rule.

        This should be called *before* the main model backward/step.
        """
        assert losses.shape[0] == self.num_tasks, "losses must be 1D with num_tasks entries"
        losses = losses.to(self.device)

        # Normalized weights on the simplex.
        weights = F.softmax(self.logits, dim=-1)

        # Shift by task-specific lower bounds.
        diff = losses - self.min_losses + 1e-8

        # Normalizing constant; we detach so we don't backprop through it.
        # This matches the form in the paper up to algebraic rearrangement.
        normalizer = 1.0 / (weights / diff).sum().detach()

        # Final scalar objective.
        return (normalizer * diff.log() * weights).sum()

    def update_logits(
        self,
        prev_losses: torch.Tensor | None,
        curr_losses: torch.Tensor,
    ) -> None:
        """
        Update the logits based on how each task's (log) loss changed since
        the previous step. This should be called *after* the model params
        have been updated for the current minibatch.

        prev_losses, curr_losses: shape [K]
        """
        # Skip on the very first step (no previous losses).
        if prev_losses is None:
            return

        prev_losses = prev_losses.to(self.device)
        curr_losses = curr_losses.to(self.device)

        # Log-loss differences, shifted by lower bounds.
        prev_log = (prev_losses - self.min_losses + 1e-8).log()
        curr_log = (curr_losses - self.min_losses + 1e-8).log()
        delta = prev_log - curr_log  # "improvement" per task

        # Compute gradient of weights = softmax(logits) in direction of delta.
        with torch.enable_grad():
            weights = F.softmax(self.logits, dim=-1)
            grad_logits, = torch.autograd.grad(
                outputs=weights,
                inputs=self.logits,
                grad_outputs=delta.detach(),
            )

        # Apply the update to logits.
        self.opt.zero_grad()
        self.logits.grad = grad_logits
        self.opt.step()


@register_method("famo")
class FAMOMethod(MultiTaskMethod):
    """
    FAMO as a MultiTaskMethod that plugs into the SON-GOKU Taskonomy harness.

    Usage from the runner CLI (after wiring):
        --methods famo
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: List[TaskSpec],
        optimizer: torch.optim.Optimizer,
        shared_param_filter=None,
        alpha: float = 2.5e-2,
        weight_decay: float = 1e-3,
        min_losses: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(model=model, tasks=tasks, optimizer=optimizer,
                         shared_param_filter=shared_param_filter)

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)

        self.num_tasks = len(tasks)

        # Default lower bounds: 0 for each task (works for non-negative losses).
        if min_losses is None:
            min_losses = torch.zeros(self.num_tasks, dtype=torch.float32)
        else:
            min_losses = torch.as_tensor(min_losses, dtype=torch.float32)

        self.controller = _FAMOController(
            num_tasks=self.num_tasks,
            min_losses=min_losses,
            alpha=alpha,
            weight_decay=weight_decay,
            device=self.device,
        )

        # We keep the previous-step losses to estimate log-loss improvement.
        self._prev_losses: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------
    def step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        """
        One optimization step on a single minibatch.

        1. Move batch to device
        2. Compute per-task losses via the TaskSpec.loss_fn callbacks
        3. Build a FAMO-weighted scalar loss and backprop through it
        4. Step the model optimizer
        5. Update FAMO's weighting parameters using prev vs. current losses
        """
        batch = self._move_batch_to_device(batch)

        per_task_losses: List[torch.Tensor] = []
        logs: Dict[str, float] = {}

        # Compute each task's loss.
        for spec in self.tasks:
            loss = spec.loss_fn(self.model, batch)
            per_task_losses.append(loss)
            logs[f"loss/{spec.name}"] = float(loss.detach().cpu().item())

        loss_vec = torch.stack(per_task_losses)  # shape: [K]

        # Combine losses using the FAMO objective.
        scalar_loss = self.controller.combine_losses(loss_vec)
        logs["loss/weighted"] = float(scalar_loss.detach().cpu().item())

        # Standard model update.
        self.optimizer.zero_grad()
        scalar_loss.backward()
        self.optimizer.step()

        # Update FAMO weighting based on how losses changed.
        with torch.no_grad():
            curr_losses = loss_vec.detach()
            self.controller.update_logits(self._prev_losses, curr_losses)
            self._prev_losses = curr_losses

            # Log the current task weights (softmax over logits).
            weights = torch.softmax(self.controller.logits, dim=-1)
            for i, spec in enumerate(self.tasks):
                logs[f"weight/{spec.name}"] = float(weights[i].item())

        return logs
