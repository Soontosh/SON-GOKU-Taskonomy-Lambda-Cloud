from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import torch
from torch import nn, Tensor
from .interfaces import TaskSpec, GradientTransform
from .utils import (
    all_params, clear_grads, flatten_grads, params_by_filter, set_param_grads, zero_like_params, add_in_place, vector_from_params
)
from .graph import cosine_interference_matrix, build_conflict_graph
from .coloring import welsh_powell_coloring, duplicate_min_coverage

@dataclass
class TauSchedule:
    kind: str = "log"  # 'log' | 'linear' | 'cosine' | 'constant'
    tau_initial: float = 1.0
    tau_target: float = 0.25
    warmup_steps: int = 0
    anneal_duration: int = 0  # number of steps after warmup over which to anneal; 0 -> jump to target

    def value(self, t: int) -> float:
        if t < self.warmup_steps:
            return self.tau_initial
        if self.anneal_duration <= 0:
            return self.tau_target
        progress = min(1.0, (t - self.warmup_steps) / max(1, self.anneal_duration))
        if self.kind == "linear":
            return self.tau_initial + (self.tau_target - self.tau_initial) * progress
        if self.kind == "cosine":
            # Cosine from initial -> target
            w = 0.5 * (1 - math.cos(math.pi * progress))
            return self.tau_initial + (self.tau_target - self.tau_initial) * w
        # default 'log' == exponential interpolation in log-space
        if self.tau_initial <= 0:
            return self.tau_target
        return self.tau_initial * ((self.tau_target / self.tau_initial) ** progress)

class SonGokuScheduler:
    """
    SON-GOKU: Interference-aware task scheduling via graph-coloring for MTL.

    Implements Algorithm 1 from the paper:
      - Maintain EMA of per-task shared-parameter gradients
      - Every R steps, compute interference matrix rho_ij = -cos(EMA_i, EMA_j)
      - Build conflict graph with edges where rho_ij > tau
      - Color with Welsh–Powell; use classes as a periodic schedule (one class per step)
      - Optionally duplicate tasks to satisfy min updates per cycle
      - Activate exactly one color/class per step; update shared and task-specific params for tasks in class
    """
    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: torch.optim.Optimizer,
        # Which parameters define "shared" for interference computation; default = all parameters.
        shared_param_filter: Optional[Callable[[nn.Parameter], bool]] = None,
        refresh_period: int = 32,  # R
        tau_schedule: Optional[TauSchedule] = None,
        ema_beta: float = 0.9,  # beta in [0,1)
        min_updates_per_cycle: int = 1,  # f_min: at least this many slots per cycle for each task
        eps_cosine: float = 1e-12,
        gradient_transform: Optional[GradientTransform] = None,  # e.g., PCGrad hook
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tasks = list(tasks)
        self.name_to_index = {t.name: i for i, t in enumerate(self.tasks)}
        self.K = len(self.tasks)
        self.optimizer = optimizer
        self.refresh_period = max(1, int(refresh_period))
        self.ema_beta = ema_beta
        self.eps_cosine = eps_cosine
        self.min_updates_per_cycle = max(1, int(min_updates_per_cycle))
        self.grad_transform = gradient_transform
        self.tau_schedule = tau_schedule or TauSchedule(kind="log", tau_initial=1.0, tau_target=0.25, warmup_steps=0, anneal_duration=0)
        self.device = device
        # Determine shared and head parameter sets
        if shared_param_filter is None:
            self.shared_params = all_params(self.model)
        else:
            self.shared_params = params_by_filter(self.model, shared_param_filter)
        # Build per-task head param lists if filters provided
        self.head_params: List[List[nn.Parameter]] = []
        for t in self.tasks:
            if t.head_param_filter is not None:
                self.head_params.append(params_by_filter(self.model, t.head_param_filter))
            else:
                # Fallback: assume the task loss_fn only touches its head; we cannot isolate reliably,
                # so default to empty head-params list. Users can provide filter for clarity.
                self.head_params.append([])
        # EMA buffers per task for shared-parameter gradient vector; None until first observation
        ref_vec = vector_from_params(self.shared_params)
        D = ref_vec.numel()
        self._emas: List[Optional[Tensor]] = [None for _ in range(self.K)]
        # Current schedule (list of classes, each a list of task indices), and pointer
        self._classes: List[List[int]] = [list(range(self.K))]  # warm-start: all tasks active
        self._schedule_len: int = 1
        self._step_in_cycle: int = 0
        # Step counter
        self._t: int = 0

    def current_tau(self) -> float:
        return float(self.tau_schedule.value(self._t))

    def _active_indices(self) -> List[int]:
        # One group per step (periodic)
        return list(self._classes[self._step_in_cycle % self._schedule_len])

    @torch.no_grad()
    def _set_param_grads_zero(self, params: Sequence[nn.Parameter]) -> None:
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()

    def step(self, batches_by_task: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform ONE training step on the currently active color group.
        Args:
            batches_by_task: dict {task_name: batch} for ANY tasks to be used this step.
                              Only batches for active tasks are consumed.
        Returns:
            dict of per-task scalar losses for the active tasks.
        """
        active = self._active_indices()
        # For accumulating shared and head grads
        shared_sum = [torch.zeros_like(p) for p in self.shared_params]
        head_sums: List[List[Tensor]] = [zero_like_params(hps) for hps in self.head_params]
        loss_log: Dict[str, float] = {}

        # Collect per-task flattened grads (shared) for optional transforms and EMA updates
        per_task_flat: Dict[str, Tensor] = {}
        # Compute grads task-by-task
        for i in active:
            spec = self.tasks[i]
            batch = batches_by_task.get(spec.name, None)
            if batch is None:
                raise ValueError(f"Missing batch for active task '{spec.name}'. Provide batches_by_task[\"{spec.name}\"].")
            # Clear all grads before single-task backward
            self.optimizer.zero_grad(set_to_none=True)
            loss = spec.loss_fn(self.model, batch)
            if not torch.is_tensor(loss):
                raise RuntimeError("loss_fn must return a scalar Tensor.")
            loss.backward()
            # Accumulate per-parameter grads
            # Shared
            for acc, p in zip(shared_sum, self.shared_params):
                if p.grad is not None:
                    acc.add_(p.grad)
            # Heads (only this task's head list is known)
            if len(self.head_params[i]) > 0:
                for acc, p in zip(head_sums[i], self.head_params[i]):
                    if p.grad is not None:
                        acc.add_(p.grad)
            loss_log[spec.name] = float(loss.detach().item())
            # Build flattened shared grad for EMA and optional transform
            flat = torch.cat([ (torch.zeros_like(p).view(-1) if p.grad is None else p.grad.view(-1)) for p in self.shared_params ])
            per_task_flat[spec.name] = flat.detach().clone()
            # EMA update for active tasks
            with torch.no_grad():
                prev = self._emas[i]
                if prev is None:
                    self._emas[i] = (1.0 - self.ema_beta) * flat
                else:
                    self._emas[i] = self.ema_beta * prev + (1.0 - self.ema_beta) * flat

        # Optional transform/surgery on per-task shared grads (within the active group)
        if self.grad_transform is not None:
            transformed = self.grad_transform(per_task_flat)
            # rebuild shared_sum from transformed flats to ensure consistency
            # First, zero shared_sum
            for t in shared_sum:
                t.zero_()
            # Sum transformed flats back into per-parameter buffers
            # Split by parameter shapes
            offsets = []
            off = 0
            shapes = [p.shape for p in self.shared_params]
            sizes = [p.numel() for p in self.shared_params]
            for sz in sizes:
                offsets.append((off, off + sz))
                off += sz
            for name, flat in transformed.items():
                # Safety: if any task not present (e.g., plugin dropped it), skip
                if name not in self.name_to_index:
                    continue
                for acc, (lo, hi), p in zip(shared_sum, offsets, self.shared_params):
                    acc.add_(flat[lo:hi].view_as(p))

        # Now apply the combined grads: set .grad tensors and step the optimizer once
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.shared_params, shared_sum):
            p.grad = g.clone()
        # Heads: only for active tasks
        for i in active:
            for p, g in zip(self.head_params[i], head_sums[i]):
                p.grad = g.clone()
        self.optimizer.step()

        # Refresh / recolor if needed
        self._t += 1
        if (self._t % self.refresh_period) == 0:
            self._refresh_and_recolor()
        # Advance schedule pointer
        self._step_in_cycle = (self._step_in_cycle + 1) % max(1, self._schedule_len)
        return loss_log

    @torch.no_grad()
    def _refresh_and_recolor(self) -> None:
        # At refresh, optionally probe all tasks to update EMA (small batches via provider)
        for i, spec in enumerate(self.tasks):
            if spec.refresh_batch_provider is None:
                continue
            batch = spec.refresh_batch_provider()
            # Probe grad for shared params only (no optimizer step)
            # Zero grads
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = spec.loss_fn(self.model, batch)
            loss.backward()
            flat = torch.cat([ (torch.zeros_like(p).view(-1) if p.grad is None else p.grad.view(-1)) for p in self.shared_params ])
            prev = self._emas[i]
            if prev is None:
                self._emas[i] = (1.0 - self.ema_beta) * flat
            else:
                self._emas[i] = self.ema_beta * prev + (1.0 - self.ema_beta) * flat

        tau = self.current_tau()
        rho = cosine_interference_matrix(self._emas, eps=self.eps_cosine)
        adj = build_conflict_graph(rho, tau=float(tau))
        classes = welsh_powell_coloring(adj)  # vertices are 0..K-1
        if self.min_updates_per_cycle > 1:
            classes = duplicate_min_coverage(classes, adj, self.min_updates_per_cycle)
        self._classes = classes
        self._schedule_len = max(1, len(self._classes))
        self._step_in_cycle = 0  # restart new cycle after recoloring

    def schedule_snapshot(self) -> List[List[str]]:
        return [[ self.tasks[i].name for i in cls ] for cls in self._classes]

    def debug_state(self) -> Dict[str, Any]:
        return {
            "t": self._t,
            "tau": self.current_tau(),
            "classes": self.schedule_snapshot(),
            "schedule_len": self._schedule_len,
        }
