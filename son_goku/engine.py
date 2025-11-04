from typing import Dict, Any, Iterable
from torch import nn
from .scheduler import SonGokuScheduler

def train_loop(
    model: nn.Module,
    scheduler: SonGokuScheduler,
    batch_iterable: Iterable[Dict[str, Any]],  # yields dict {task_name: batch} per step
    steps: int
):
    """Minimal loop showing how to drive the scheduler. Not wired to any dataset by design."""
    it = iter(batch_iterable)
    for _ in range(steps):
        batches = next(it)
        losses = scheduler.step(batches)
        yield losses, scheduler.debug_state()
