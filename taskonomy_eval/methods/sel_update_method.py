from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn, optim

from son_goku import TaskSpec  # same TaskSpec you already use for SON-GOKU

from .base import MultiTaskMethod, register_method


@register_method("sel_update")
class SelectiveUpdateMethod(MultiTaskMethod):
    """
    Selective Task Group Updates for Multi-Task Optimization (ICLR 2025-inspired).

    High-level behavior (per training step):
      1. For each task i:
           - Compute its loss L_i (probe).
           - Compute gradients of L_i w.r.t. shared parameters.
      2. Use these gradients to estimate an inter-task affinity matrix
         (here: cosine similarity of shared-parameter gradients).
      3. Maintain an EMA of this affinity to get a "proximal" view over time.
      4. Build task groups based on the current affinity (tasks with higher
         mutual affinity get grouped together, others split).
      5. Sequentially iterate over groups; for each group G:
           - Compute group loss sum_{i in G} L_i on the current model.
           - Backprop and take an optimizer step.
         -> both shared and task-specific parameters for tasks in G are updated,
            others are untouched in that group.

    This is designed to be comparable to SON-GOKU / GradNorm / MGDA in your harness.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter: Callable[[nn.Parameter], bool],
        # Allow either `base_optimizer` (like GradNorm/MGDA) or `optimizer`
        base_optimizer: optim.Optimizer | None = None,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | None = None,
        # Affinity tracking hyperparams
        affinity_momentum: float = 0.9,    # EMA factor for proximal affinity
        affinity_threshold: float = 0.0,   # threshold for grouping tasks together
        max_group_size: int | None = None, # optional hard cap on group size
    ) -> None:
        super().__init__()  # in case your base class does something

        self.model = model
        self.tasks: List[TaskSpec] = list(tasks)

        opt = base_optimizer or optimizer
        if opt is None:
            raise ValueError("SelectiveUpdateMethod expects `base_optimizer` or `optimizer`.")
        self.optimizer: optim.Optimizer = opt

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_tasks = len(self.tasks)
        self.task_names = [t.name for t in self.tasks]

        # Select shared params (used to measure inter-task affinity)
        shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not shared_params:
            # Fallback: use all trainable params if filter selects none
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.shared_params: List[nn.Parameter] = shared_params

        # Proximal affinity tracking (EMA over cosine similarity matrix)
        self.affinity_momentum = float(affinity_momentum)
        self.affinity_threshold = float(affinity_threshold)
        self.max_group_size = max_group_size

        # Stored on CPU as float64 for numerical stability
        self.affinity_matrix = torch.zeros(
            self.num_tasks, self.num_tasks, dtype=torch.float64
        )

        # Small epsilon to avoid divide-by-zero in norms
        self._eps = 1e-8

    # ------------------------------------------------------------------
    # Helpers for affinity and grouping
    # ------------------------------------------------------------------

    def _compute_task_gradients(
        self, batch: Mapping[str, Any]
    ) -> tuple[List[torch.Tensor], List[float]]:
        """
        For each task, compute:
          * probe loss L_i (no optimizer step yet)
          * gradient of L_i w.r.t. shared parameters
        Returns:
          flat_gradients: [T] each is 1D torch tensor (concatenated shared grads)
          loss_values   : [T] python floats (L_i)
        """
        flat_grads: List[torch.Tensor] = []
        loss_values: List[float] = []

        for spec in self.tasks:
            # Forward + loss for this task only
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            loss_values.append(float(Li.detach().item()))

            # Gradients w.r.t. shared parameters only (no optimizer step here)
            grads = torch.autograd.grad(
                Li,
                self.shared_params,
                retain_graph=False,
                create_graph=False,
            )
            flat = torch.cat([g.detach().reshape(-1) for g in grads])
            flat_grads.append(flat)

        return flat_grads, loss_values

    def _update_affinity_matrix(self, flat_grads: List[torch.Tensor]) -> None:
        """
        Update EMA of inter-task affinity using cosine similarity of shared grads.
        """
        if self.num_tasks <= 1:
            return

        # Stack to [T, P]
        G = torch.stack(flat_grads, dim=0)  # (T, P)
        norms = G.norm(dim=1, keepdim=True) + self._eps
        G_normed = G / norms

        # Cosine similarity matrix in torch (float32), then CPU float64
        cos_sim = (G_normed @ G_normed.t()).cpu().to(torch.float64)  # (T, T)

        m = self.affinity_momentum
        self.affinity_matrix = (
            m * self.affinity_matrix + (1.0 - m) * cos_sim
        )

    def _build_groups(self) -> List[List[int]]:
        """
        Build task groups based on the current affinity matrix.

        Simple greedy strategy:
          * If affinity is still almost zero (early training), group all tasks together.
          * Otherwise:
              - maintain a set of unassigned tasks
              - repeatedly:
                  - pick the task with highest mean affinity to other unassigned tasks
                  - form a group starting from this "seed", then
                  - greedily add other tasks whose affinity to the seed exceeds
                    `affinity_threshold`, until `max_group_size` (if given) is reached.
        """
        n = self.num_tasks
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        A = self.affinity_matrix.clone()

        # If we haven't learned anything yet (all near-zero), just group all tasks
        if torch.allclose(A, torch.zeros_like(A), atol=1e-4):
            return [list(range(n))]

        remaining = set(range(n))
        groups: List[List[int]] = []
        thr = float(self.affinity_threshold)
        max_sz = self.max_group_size or n

        while remaining:
            # Seed: task with highest mean affinity to others in 'remaining'
            def mean_aff(idx: int) -> float:
                others = list(remaining)
                if len(others) == 1:
                    return 0.0
                v = A[idx, others].mean().item()
                return float(v)

            seed = max(remaining, key=mean_aff)
            group = [seed]
            remaining.remove(seed)

            # Add other tasks sorted by affinity to the seed
            sorted_others = sorted(
                list(remaining),
                key=lambda j: float(A[seed, j].item()),
                reverse=True,
            )
            for j in sorted_others:
                if len(group) >= max_sz:
                    break
                if float(A[seed, j].item()) >= thr:
                    group.append(j)
                    remaining.remove(j)

            groups.append(group)

        return groups

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        One training step with selective task group updates.

        Assumes:
          * `batch` has already been moved to self.device by the caller.
          * `self.model` is on self.device.
        """
        self.model.train()
        logs: Dict[str, float] = {}

        # 1) Probe: per-task losses + per-task grads (for affinity only)
        flat_grads, probe_losses = self._compute_task_gradients(batch)

        # Update proximal affinity estimates
        self._update_affinity_matrix(flat_grads)

        # 2) Build groups from current affinity
        groups = self._build_groups()

        # Log probe losses and some affinity summary
        for name, lv in zip(self.task_names, probe_losses):
            logs[f"loss_probe/{name}"] = float(lv)

        if self.num_tasks >= 2:
            with torch.no_grad():
                # average off-diagonal affinity
                A = self.affinity_matrix
                mask = ~torch.eye(self.num_tasks, dtype=torch.bool)
                if mask.any():
                    mean_aff = A[mask].mean().item()
                    logs["affinity/mean_offdiag"] = float(mean_aff)

        # 3) Sequential group updates: new forward/backward for each group
        for g_idx, group in enumerate(groups):
            self.optimizer.zero_grad(set_to_none=True)

            group_loss = 0.0
            group_loss_value = 0.0

            # Compute summed loss over tasks in this group
            for t_idx in group:
                spec = self.tasks[t_idx]
                Li = spec.loss_fn(self.model, batch)
                group_loss = group_loss + Li
                group_loss_value += float(Li.detach().item())

            # Backprop through all params for this group
            group_loss.backward()
            self.optimizer.step()

            logs[f"loss/group{g_idx}"] = float(group_loss_value)
            logs[f"group_size/{g_idx}"] = float(len(group))

        # Also log total loss over all groups (just for monitoring; not the exact
        # optimization objective because we did multiple sequential updates)
        logs["loss/total"] = sum(
            logs[k] for k in logs if k.startswith("loss/group")
        )

        return logs

    # ------------------------------------------------------------------
    # (Optional) checkpoint methods
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "affinity_momentum": self.affinity_momentum,
            "affinity_threshold": self.affinity_threshold,
            "max_group_size": self.max_group_size,
            "affinity_matrix": self.affinity_matrix.clone().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "affinity_momentum" in state:
            self.affinity_momentum = float(state["affinity_momentum"])
        if "affinity_threshold" in state:
            self.affinity_threshold = float(state["affinity_threshold"])
        if "max_group_size" in state:
            self.max_group_size = state["max_group_size"]
        if "affinity_matrix" in state:
            self.affinity_matrix = state["affinity_matrix"].to(torch.float64)
