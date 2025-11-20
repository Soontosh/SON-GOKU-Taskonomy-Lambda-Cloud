# taskonomy_eval/methods/joint_method.py
from __future__ import annotations
from typing import Dict, Any, Sequence, Optional
import torch
from torch import nn, optim

from son_goku import TaskSpec  # just reusing the TaskSpec container for consistency

class JointMethod:
    """
    Plain joint multi-task training: sum per-task losses every step.
    No refresh, no grouping, no extra overhead.
    """
    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter=None,  # ignored; kept for signature compatibility
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.tasks = list(tasks)
        self.device = device or next(model.parameters()).device

    def step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, float]:
        # Move tensors to device
        batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        self.model.train()
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        logs = {}
        loss_total = 0.0
        for spec in self.tasks:
            li = spec.loss_fn(self.model, batch)
            logs[f"loss/{spec.name}"] = float(li.item())
            loss_total = loss_total + li

        loss_total.backward()
        self.optimizer.step()
        logs["loss/total"] = float(loss_total.item())
        return logs
