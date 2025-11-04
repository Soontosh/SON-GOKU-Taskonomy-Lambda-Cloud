# SON-GOKU (PyTorch) — Interference-Aware Task Scheduler

This package implements the SON-GOKU scheduler for multi-task learning:
it measures cross-task gradient interference, builds a conflict graph,
greedily colors the graph to form compatible task groups, and activates
exactly **one color group per step**. Groups are recomputed every `refresh_period`
steps to track evolving task relations.

## Install (editable)
```bash
pip install -e .
```

## Quick Start
```python
import torch
from torch import nn, optim
from son_goku import SonGokuScheduler, TauSchedule, TaskSpec

# Define your model with a shared backbone and task-specific heads.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        self.head_a = nn.Linear(16, 3)
        self.head_b = nn.Linear(16, 2)
    def forward(self, x):
        h = self.backbone(x)
        return h, self.head_a(h), self.head_b(h)

model = Model()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Tasks: provide loss functions (and optional refresh-time batch providers)
def loss_task_a(model, batch):
    h, za, zb = model(batch["x"])  # batch is a dict with 'x', 'ya', 'yb'
    return torch.nn.functional.cross_entropy(za, batch["ya"])
def loss_task_b(model, batch):
    h, za, zb = model(batch["x"]) 
    return torch.nn.functional.cross_entropy(zb, batch["yb"])

# Optional: filters to mark task-specific head parameters (so only active heads update)
head_a_filter = lambda p: hasattr(model, "head_a") and (p is model.head_a.weight or p is model.head_a.bias)
head_b_filter = lambda p: hasattr(model, "head_b") and (p is model.head_b.weight or p is model.head_b.bias)

tasks = [
    TaskSpec(name="task_a", loss_fn=loss_task_a, refresh_batch_provider=None, head_param_filter=head_a_filter),
    TaskSpec(name="task_b", loss_fn=loss_task_b, refresh_batch_provider=None, head_param_filter=head_b_filter),
]

# Shared-parameter filter: treat everything except heads as shared
def shared_filter(p):
    return (p is model.head_a.weight or p is model.head_a.bias or p is model.head_b.weight or p is model.head_b.bias) == False

tau = TauSchedule(kind="log", tau_initial=1.0, tau_target=0.25, warmup_steps=100, anneal_duration=400)
sched = SonGokuScheduler(
    model, tasks, opt,
    shared_param_filter=shared_filter,
    refresh_period=32,
    tau_schedule=tau,
    ema_beta=0.9,
    min_updates_per_cycle=1,
)

# Drive the scheduler from your training data pipeline
batches = [
    {"task_a": {"x": torch.randn(8,16), "ya": torch.randint(0,3,(8,))},
     "task_b": {"x": torch.randn(8,16), "yb": torch.randint(0,2,(8,))}}
    for _ in range(1000)
]
for step in range(100):
    losses = sched.step(batches[step])
    if (step+1) % 32 == 0:
        print("Schedule:", sched.schedule_snapshot())
```

## Key Hyperparameters (all exposed)
- `refresh_period (R)`: steps between conflict-graph rebuilds.
- `TauSchedule`: controls `tau` (warmup + anneal to target). Kinds: `log` (default), `linear`, `cosine`, `constant`.
- `ema_beta (β)`: EMA smoothing for per-task shared gradients.
- `min_updates_per_cycle (f_min)`: ensure each task appears at least this many times per schedule period (via compatible duplication).
- `shared_param_filter`: predicate for which parameters define the shared vector used to measure interference.
- `gradient_transform`: optional hook to modify per-task grads within a group (e.g., PCGrad-like surgery).
- `eps_cosine`: numerical stabilizer for cosine computations.

## Design Notes → Paper Mapping
- Interference coefficient: `rho_ij = -cos(EMA_i, EMA_j)` (Sec. 3.1.1).
- Conflict graph: edges where `rho_ij > tau` (Sec. 3.1.2 / Eq. 10).
- Coloring: Welsh–Powell largest-first heuristic (Sec. 4.3).
- One color group per step; periodic schedule; refresh every `R` (Sec. 4.4).
- Warm-up + annealing of `tau` (Sec. 4.4.2).
- Minimum coverage via compatible-slot duplication (Sec. 4.4.1).

## Extensions
- Plug-in your gradient surgery (`gradient_transform`) to combine SON-GOKU with PCGrad/CAGrad, etc.
- Combine with loss reweighting (e.g., AdaTask/FAMO) **outside** the scheduler; the scheduler only selects/aggregates tasks.

## License
MIT
