from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
from torch import nn, optim

# Adjust this import to wherever your SON-GOKU implementation lives.
# This assumes you've installed the earlier `son_goku` package I gave you.
from son_goku import SonGokuScheduler, TauSchedule, TaskSpec

from .base import MultiTaskMethod, register_method


@register_method("son_goku")
class SonGokuMethod(MultiTaskMethod):
    """
    Thin wrapper around SonGokuScheduler so it fits the MultiTaskMethod API.

    This does *not* replace your existing SON-GOKU training script – it's an
    additional entry point so SON-GOKU can participate in the unified runner.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter: Callable[[nn.Parameter], bool],
        tau_kind: str = "log",
        tau_initial: float = 1.0,
        tau_target: float = 0.25,
        tau_warmup: int = 0,
        tau_anneal: int = 0,
        refresh_period: int = 32,
        ema_beta: float = 0.9,
        min_updates_per_cycle: int = 1,
        base_method: str = "vanilla", # "vanilla" | "adatask" | "pcgrad"
        log_dir: str | None = None,
        log_interval: int = 50,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tasks = list(tasks)
        self.task_names = [t.name for t in tasks]
        log_path = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "son_goku_groups.json")

        tau = TauSchedule(
            kind=tau_kind,
            tau_initial=tau_initial,
            tau_target=tau_target,
            warmup_steps=tau_warmup,
            anneal_duration=tau_anneal,
        )
        self.scheduler = SonGokuScheduler(
            model=self.model,
            tasks=self.tasks,
            optimizer=self.optimizer,
            shared_param_filter=shared_param_filter,
            refresh_period=refresh_period,
            tau_schedule=tau,
            ema_beta=ema_beta,
            min_updates_per_cycle=min_updates_per_cycle,
            log_interval=log_interval,
            log_path=log_path,
            base_method=base_method,
        )

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        # Use *exactly* the same semantics as your existing SON-GOKU training:
        # same batch for every task in the schedule.
        task_batches = {name: batch for name in self.task_names}
        losses = self.scheduler.step(task_batches)
        return {f"loss/{k}": float(v) for k, v in losses.items()}

    def state_dict(self) -> Dict[str, Any]:
        # If SonGokuScheduler exposes state_dict() later you can forward to it.
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass
