# son_goku/affinity/scheduler_affinity.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.coloring import build_adjacency, welsh_powell_coloring
from .affinity import AffinityComputer, GradCosineAffinity, cosine_vs_tag_stats


def _ingroup_conflict_rate(A: torch.Tensor, groups: List[List[int]]) -> float:
    num_e, num_pairs = 0, 0
    for g in groups:
        for i, u in enumerate(g):
            for v in g[i+1:]:
                num_pairs += 1
                if bool(A[u, v].item()):
                    num_e += 1
    return float(num_e / max(1, num_pairs))


class AffinityScheduler:
    """
    Minimal SON-GOKU-style trainer that uses a pluggable affinity backend
    to construct the conflict graph and color groups.

    NOTE: For cosine affinity, we set adjacency A_ij = (C_ij > tau).
          For TAG affinities (exact/linearized), we set adjacency A_ij = (S_ij > gamma),
          where S_ij is the symmetrized lookahead increase. gamma >= 0 (default 0).
    """
    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter,
        tau_schedule: TauSchedule,           # used for cosine mode
        affinity: AffinityComputer,          # cosine or TAG variant
        mode: str,                           # "cosine" | "tag"
        threshold: float,                    # tau (cosine) or gamma (tag)
        refresh_period: int = 32,
        device: Optional[torch.device] = None,
        tag_eta_v: float = 1e-3,
        pair_fraction: float = 1.0,          # subsample pairs for TAG exact to limit cost
        log_compare_cosine: bool = True,     # compute cosine also (for diagnostics)
    ):
        self.model = model
        self.tasks = list(tasks)
        self.optimizer = optimizer
        self.shared_param_filter = shared_param_filter
        self.tau_schedule = tau_schedule
        self.affinity = affinity
        self.mode = mode
        self.threshold = threshold
        self.refresh_period = refresh_period
        self.device = device or next(model.parameters()).device
        self.tag_eta_v = tag_eta_v
        self.pair_fraction = pair_fraction
        self.log_compare_cosine = log_compare_cosine

        self._groups: List[List[int]] = []
        self._group_idx = 0
        self._step = 0
        self._refresh_logs: List[Dict[str, float]] = []

    def _refresh(self, batch: Mapping[str, Any]):
        t0 = time.time()
        # 1) Affinity matrix
        A_stats: Dict[str, float] = {}
        if self.mode == "cosine":
            res = self.affinity.compute(self.model, self.tasks, batch, self.device)
            S = res.S  # cosine in [-1,1]
            tau = float(self.tau_schedule.value(self._step))
            A = (S > tau)
            A.fill_diagonal_(False)
            A = A | A.t()
            A_stats["tau"] = tau
        else:
            res = self.affinity.compute(
                self.model, self.tasks, batch, self.device,
                pair_fraction=self.pair_fraction, eta_v=self.tag_eta_v
            )
            S = res.S  # symmetrized lookahead increase
            gamma = self.threshold
            A = (S > gamma)
            A.fill_diagonal_(False)
            A = A | A.t()
            A_stats["gamma"] = gamma
            A_stats["tag_eta_v"] = self.tag_eta_v

        # 2) Coloring
        colors, ctime = welsh_powell_coloring(A, None)
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(colors):
            groups.setdefault(c, []).append(i)
        color_groups = [groups[c] for c in sorted(groups.keys())]
        self._groups = color_groups
        self._group_idx = 0

        # 3) Optional compare vs cosine (diagnostics)
        if self.log_compare_cosine and self.mode != "cosine":
            cos_res = GradCosineAffinity(self.shared_param_filter).compute(self.model, self.tasks, batch, self.device)
            C = cos_res.S
            from .affinity import cosine_vs_tag_stats
            # Compare cos C vs negative lookahead (-S)
            compare = cosine_vs_tag_stats(C, -S)
            A_stats.update({f"cmp_{k}": v for k, v in compare.items()})

        # 4) Log
        total_ms = (time.time() - t0) * 1e3
        ing = _ingroup_conflict_rate(A, self._groups)
        avg_deg = float(A.float().sum().item() / max(1, A.shape[0]*(A.shape[0]-1)))
        self._refresh_logs.append({
            "refresh_ms": total_ms,
            "colors": len(self._groups),
            "avg_degree": avg_deg,
            "ingroup_conflict": ing,
            **A_stats
        })

    def step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self._step += 1
        if (self._step == 1) or (self._step % self.refresh_period == 0) or (not self._groups):
            self._refresh(batch)

        group = self._groups[self._group_idx]
        self._group_idx = (self._group_idx + 1) % len(self._groups)

        self.model.train()
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        losses = []
        for idx in group:
            spec = self.tasks[idx]
            losses.append(spec.loss_fn(self.model, batch))
        total = torch.stack(losses).sum()
        total.backward()
        self.optimizer.step()

        out = {f"loss/{self.tasks[idx].name}": float(l.item()) for idx, l in zip(group, losses)}
        out["loss/total"] = float(total.item())
        out["group_size"] = float(len(group))
        out["colors_now"] = float(len(self._groups))
        return out

    # Accessors
    def refresh_logs(self) -> List[Dict[str, float]]:
        return self._refresh_logs