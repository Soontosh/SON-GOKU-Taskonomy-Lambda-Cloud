# taskonomy_eval/methods/son_goku_pcgrad_method.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence
import torch
from torch import nn, optim

from son_goku import TaskSpec
from .base import MultiTaskMethod, register_method
from .son_goku_method import SonGokuMethod


@register_method("son_goku_pcgrad")
class SonGokuPCGradMethod(MultiTaskMethod):
    """
    SON-GOKU + PCGrad:
      - SON-GOKU schedules a low-conflict group each step.
      - Within that group, apply PCGrad surgery on shared grads (and keep heads unchanged).
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter,
        refresh_period: int,
        tau_initial: float,
        tau_target: float,
        tau_kind: str = "log",
        tau_warmup: int = 0,
        tau_anneal: int = 0,
        ema_beta: float = 0.9,
        min_updates_per_cycle: int = 1,
    ) -> None:
        self.inner = SonGokuMethod(
            model=model,
            tasks=tasks,
            optimizer=optimizer,
            shared_param_filter=shared_param_filter,
            refresh_period=refresh_period,
            tau_initial=tau_initial,
            tau_target=tau_target,
            tau_kind=tau_kind,
            tau_warmup=tau_warmup,
            tau_anneal=tau_anneal,
            ema_beta=ema_beta,
            min_updates_per_cycle=min_updates_per_cycle,
            base_method="pcgrad",
        )

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        return self.inner.step(batch, global_step)

    def state_dict(self) -> Dict[str, Any]:
        return getattr(self.inner, "state_dict", lambda: {})()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state)