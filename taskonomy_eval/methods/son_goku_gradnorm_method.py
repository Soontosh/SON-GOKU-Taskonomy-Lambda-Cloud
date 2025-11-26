from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Optional

import torch
from torch import nn, optim

from son_goku import TaskSpec  # same TaskSpec used elsewhere

from .base import MultiTaskMethod, register_method
from .son_goku_method import SonGokuMethod
from .gradnorm_method import GradNormMethod


@register_method("son_goku_gradnorm")
class SonGokuGradNormWarmStartMethod(MultiTaskMethod):
    """
    SON-GOKU + GradNorm Warm Start (Sec. 6.3 of the paper).

    Behavior:
      * Use pure GradNorm for the first `warmup_steps`.
      * After that, switch to standard SON-GOKU (same hyperparams
        as your existing SonGokuMethod).
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter,
        # SON-GOKU hyperparams
        refresh_period: int,
        tau_initial: float,
        tau_target: float,
        tau_kind: str = "log",
        tau_warmup: int = 0,
        tau_anneal: int = 0,
        ema_beta: float = 0.9,
        min_updates_per_cycle: int = 1,
        graph_density_target: float | None = None,
        # GradNorm hyperparams
        gradnorm_alpha: float = 1.5,
        gradnorm_weight_lr: float = 0.025,
        # warm start
        warmup_steps: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.optimizer = optimizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.warmup_steps = warmup_steps

        # Underlying methods
        self._gradnorm = GradNormMethod(
            model=self.model,
            tasks=self.tasks,
            shared_param_filter=shared_param_filter,
            base_optimizer=self.optimizer,
            alpha=gradnorm_alpha,
            weight_lr=gradnorm_weight_lr,
            device=self.device,
        )

        self._son_goku = SonGokuMethod(
            model=self.model,
            tasks=self.tasks,
            optimizer=self.optimizer,
            shared_param_filter=shared_param_filter,
            refresh_period=refresh_period,
            tau_initial=tau_initial,
            tau_target=tau_target,
            tau_kind=tau_kind,
            tau_warmup=tau_warmup,
            tau_anneal=tau_anneal,
            ema_beta=ema_beta,
            min_updates_per_cycle=min_updates_per_cycle,
            graph_density_target=graph_density_target,
        )

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        Delegate to GradNorm while global_step <= warmup_steps,
        then to SON-GOKU afterwards.
        """
        if global_step <= self.warmup_steps:
            logs = self._gradnorm.step(batch, global_step)
            # tag logs so you can tell which phase produced them
            logs["phase"] = 0.0  # 0 = GradNorm warm-up
        else:
            logs = self._son_goku.step(batch, global_step)
            logs["phase"] = 1.0  # 1 = SON-GOKU
        return logs
