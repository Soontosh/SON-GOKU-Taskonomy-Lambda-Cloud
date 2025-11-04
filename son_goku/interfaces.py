from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union, Any
import torch
from torch import nn, Tensor

class GradientTransform(Protocol):
    """
    Optional hook to modify per-task gradients inside a group before they're aggregated.
    Given a dict of task_name -> flattened grad tensor (shared params only),
    return a new dict with the same keys and tensors.
    This enables plug-ins like PCGrad/other surgeries without tying this library to them.
    """
    def __call__(self, grads: Dict[str, Tensor]) -> Dict[str, Tensor]: ...

@dataclass
class TaskSpec:
    name: str
    # Computes a scalar loss for this task on a provided batch.
    # The callable must only involve the model's parameters relevant to this task (i.e., its head) 
    # plus the shared backbone, so that task-specific grads are isolated correctly.
    loss_fn: Callable[[nn.Module, Any], Tensor]
    # Optional callable to provide a "small" batch for refresh-time gradient samples.
    # It must return a batch compatible with loss_fn(model, batch).
    refresh_batch_provider: Optional[Callable[[], Any]] = None
    # Optional: identify which parameters belong to this task-specific head (to update h_k only when active).
    # If None, we assume the task's loss_fn only touches its own head params.
    head_param_filter: Optional[Callable[[nn.Parameter], bool]] = None
