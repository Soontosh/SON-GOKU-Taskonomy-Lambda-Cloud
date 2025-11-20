# taskonomy_eval/methods/son_goku_graph_ablate.py
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler

class SonGokuGraphAblateMethod:
    """
    SON-GOKU with configurable graph-building rule (threshold/knn/signed/quantile),
    optionally with density-matched calibration per refresh.
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
        # graph options
        graph_mode: str = "threshold",
        graph_knn_k: int = 3,
        graph_quantile_p: float = 0.3,
        graph_density_target: Optional[float] = None,
    ):
        self.device = device or next(model.parameters()).device
        tau = TauSchedule(
            kind=tau_kind, tau_initial=tau_initial, tau_target=tau_target,
            warmup_steps=tau_warmup, anneal_duration=tau_anneal
        )
        self.sched = SonGokuInstrumentedScheduler(
            model=model, tasks=list(tasks), optimizer=optimizer,
            shared_param_filter=shared_param_filter,
            tau_schedule=tau,
            refresh_period=refresh_period, ema_beta=ema_beta,
            min_updates_per_cycle=min_updates_per_cycle,
            device=self.device,
            compute_exact_shadow=True,
            random_groups_control=False,
            measure_refresh_memory=True,
        )
        # attach graph config (read inside _refresh)
        self.sched.graph_mode = graph_mode
        self.sched.graph_knn_k = int(graph_knn_k)
        self.sched.graph_quantile_p = float(graph_quantile_p)
        self.sched.graph_density_target = graph_density_target if graph_density_target is None else float(graph_density_target)

    def step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        return self.sched.step(batch)