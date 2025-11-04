import torch
from torch import nn, optim
from son_goku import SonGokuScheduler, TauSchedule, TaskSpec

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        self.head_a = nn.Linear(16, 3)
        self.head_b = nn.Linear(16, 2)
    def forward(self, x):
        h = self.backbone(x)
        return h, self.head_a(h), self.head_b(h)

def loss_task_a(model, batch):
    h, za, zb = model(batch["x"])
    return nn.functional.cross_entropy(za, batch["ya"])
def loss_task_b(model, batch):
    h, za, zb = model(batch["x"])
    return nn.functional.cross_entropy(zb, batch["yb"])

model = Model()
opt = optim.Adam(model.parameters(), lr=1e-3)

head_a_filter = lambda p: (p is model.head_a.weight) or (p is model.head_a.bias)
head_b_filter = lambda p: (p is model.head_b.weight) or (p is model.head_b.bias)

tasks = [
    TaskSpec(name="task_a", loss_fn=loss_task_a, head_param_filter=head_a_filter),
    TaskSpec(name="task_b", loss_fn=loss_task_b, head_param_filter=head_b_filter),
]

def shared_filter(p):
    return not ((p is model.head_a.weight) or (p is model.head_a.bias) or (p is model.head_b.weight) or (p is model.head_b.bias))

tau = TauSchedule(kind="log", tau_initial=1.0, tau_target=0.25, warmup_steps=10, anneal_duration=50)
sched = SonGokuScheduler(model, tasks, opt, shared_param_filter=shared_filter, refresh_period=8, tau_schedule=tau, ema_beta=0.9)

for step in range(40):
    batches = {
        "task_a": {"x": torch.randn(8,16), "ya": torch.randint(0,3,(8,))},
        "task_b": {"x": torch.randn(8,16), "yb": torch.randint(0,2,(8,))},
    }
    losses = sched.step(batches)
    if (step+1) % 8 == 0:
        print("Step", step+1, "schedule:", sched.schedule_snapshot(), "state:", sched.debug_state())
