# taskonomy_eval/methods/son_goku_static_method.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence, List, Optional
import time
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler


class _StaticOneShotScheduler(SonGokuInstrumentedScheduler):
    """
    Freeze the grouping after the FIRST refresh (one-shot coloring).
    Subsequent refreshes only update EMA (to keep cost similar) but DO NOT recolor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._did_first_refresh = False
        self._frozen_groups: Optional[List[List[int]]] = None

    def _refresh(self, batches: Mapping[str, Any]):
        # First refresh: normal behavior; freeze groups
        if not self._did_first_refresh:
            super()._refresh(batches)
            self._frozen_groups = [g[:] for g in self._groups]
            self._did_first_refresh = True
            # Tag the log as "static:init"
            if self._refresh_logs:
                self._refresh_logs[-1]["tau_mode"] = "static:init"
            return

        # Subsequent "refresh": update EMA only; keep groups frozen; do light logging
        if self.measure_refresh_memory and self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        # recompute task grads to keep EMA current (same cost order as normal refresh)
        grads = []
        for spec in self.tasks:
            g = self._compute_task_grad(spec, batches[spec.name])
            grads.append(g)
        G = torch.stack(grads, dim=0)
        self._update_ema(G)

        # keep prior frozen groups
        self._groups = [g[:] for g in self._frozen_groups] if self._frozen_groups else []
        self._group_idx = 0

        # lightweight log
        log = {
            "tau_mode": "static:frozen",
            "tau": float(self.tau_schedule.value(self._step)),
            "colors": len(self._groups),
            "refresh_ms": (time.time() - t0) * 1000.0,
            "random_groups": False,
        }
        if self.measure_refresh_memory and self.device.type == "cuda":
            torch.cuda.synchronize()
            log["mem/refresh_peak_mb"] = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        log["mem/ema_mb"] = self._ema.numel() * self._ema.element_size() / (1024.0 ** 2)
        self._refresh_logs.append(log)
        self._last_refresh_log = log


class SonGokuStaticOneShotMethod:
    """
    Paper ablation: run greedy coloring once at start, freeze groups thereafter.
    (All other hyperparameters match baseline SON-GOKU.)
    """
    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter,
        refresh_period: int = 32,
        tau_initial: float = 1.0,
        tau_target: float = 0.25,
        tau_kind: str = "log",
        tau_warmup: int = 0,
        tau_anneal: int = 0,
        ema_beta: float = 0.9,
        min_updates_per_cycle: int = 1,
        device: Optional[torch.device] = None,
        graph_density_target: float | None = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer

        tau = TauSchedule(
            kind=tau_kind,
            tau_initial=tau_initial,
            tau_target=tau_target,
            warmup_steps=tau_warmup,
            anneal_duration=tau_anneal,
        )

        self.sched = _StaticOneShotScheduler(
            model=self.model,
            tasks=list(tasks),
            optimizer=self.optimizer,
            shared_param_filter=shared_param_filter,
            tau_schedule=tau,
            refresh_period=refresh_period,
            ema_beta=ema_beta,
            min_updates_per_cycle=min_updates_per_cycle,
            device=self.device,
            random_groups_control=False,
            compute_exact_shadow=True,
            measure_refresh_memory=True,
        )
        if graph_density_target is not None:
            self.sched.graph_density_target = float(graph_density_target)

    def step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        # Move tensors to device here to keep runner simple/consistent.
        batch = {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        return self.sched.step(batch)
