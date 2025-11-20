# taskonomy_eval/methods/son_goku_single_step_method.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence, Optional
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler


class SonGokuSingleStepMethod:
    """
    Paper ablation: use only the latest mini-batch for conflict estimation
    (history length H=1) by setting EMA beta to 0.0.
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
        ema_beta: float = 0.0,                 # <- H=1
        min_updates_per_cycle: int = 1,
        device: Optional[torch.device] = None,
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

        self.sched = SonGokuInstrumentedScheduler(
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

    def step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        batch = {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        return self.sched.step(batch)