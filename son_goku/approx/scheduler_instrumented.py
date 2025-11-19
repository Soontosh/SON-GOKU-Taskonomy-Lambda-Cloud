# son_goku/approx/scheduler_instrumented.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import math
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from .oracles import BaseCosineOracle, ExactOracle, CosineBuildResult
from .coloring import build_adjacency, welsh_powell_coloring


def _flatten_shared(params: List[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in params])


def _pairwise_f1(A_hat: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
    # both boolean KxK symmetric with zeros on diag
    iu, ju = torch.triu_indices(A.size(0), A.size(0), offset=1)
    y = A[iu, ju]
    yhat = A_hat[iu, ju]
    tp = torch.logical_and(yhat, y).sum().item()
    fp = torch.logical_and(yhat, ~y).sum().item()
    fn = torch.logical_and(~yhat, y).sum().item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    return {"precision": prec, "recall": rec, "f1": f1}


def _partition_pair_jaccard(colors_hat: List[int], colors: List[int]) -> float:
    # Compare partitions via Jaccard on "same-color" pairs
    import itertools
    def pairs(colors):
        K = len(colors)
        S = set()
        for i, j in itertools.combinations(range(K), 2):
            if colors[i] == colors[j]:
                S.add((i, j))
        return S
    S1, S2 = pairs(colors_hat), pairs(colors)
    inter = len(S1 & S2)
    union = len(S1 | S2)
    return inter / max(1, union)


class SonGokuInstrumentedScheduler:
    """
    SON-GOKU training loop with pluggable cosine oracle and warm-start coloring,
    plus on-refresh fidelity metrics vs. Exact.
    """
    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        optimizer: optim.Optimizer,
        shared_param_filter,
        tau_schedule: TauSchedule,
        refresh_period: int = 32,
        ema_beta: float = 0.9,
        min_updates_per_cycle: int = 1,
        cosine_oracle: Optional[BaseCosineOracle] = None,
        use_warmstart_coloring: bool = False,
        device: Optional[torch.device] = None,
        compute_exact_shadow: bool = True,
    ):
        self.model = model
        self.tasks = list(tasks)
        self.optimizer = optimizer
        self.shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not self.shared_params:
            self.shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.tau_schedule = tau_schedule
        self.refresh_period = refresh_period
        self.ema_beta = ema_beta
        self.min_updates_per_cycle = min_updates_per_cycle
        self.oracle = cosine_oracle or ExactOracle()
        self.use_warmstart = use_warmstart_coloring
        self.device = device or next(self.model.parameters()).device
        self.compute_exact_shadow = compute_exact_shadow

        self.K = len(self.tasks)
        self._ema = torch.zeros(self.K, sum(p.numel() for p in self.shared_params), device=self.device)
        self._have_ema = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        self._step = 0

        self._last_colors: Optional[List[int]] = None
        self._groups: List[List[int]] = []
        self._group_idx = 0

        # metrics accumulators
        self._refresh_logs: List[Dict[str, float]] = []

    def _compute_task_grad(self, spec: TaskSpec, batch: Mapping[str, Any]) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = spec.loss_fn(self.model, batch)
        loss.backward()
        flat = _flatten_shared(self.shared_params).detach()
        return flat

    @torch.no_grad()
    def _update_ema(self, gradients: torch.Tensor):
        if not self._have_ema.any():
            self._ema = gradients.clone()
            self._have_ema[:] = True
        else:
            self._ema = self.ema_beta * self._ema + (1.0 - self.ema_beta) * gradients

    def _refresh(self, batches: Dict[str, Mapping[str, Any]]):
        t0 = time.time()

        # 1) Compute per-task gradient snapshot on refresh batches
        t_grad0 = time.time()
        grads = []
        for i, spec in enumerate(self.tasks):
            g = self._compute_task_grad(spec, batches[spec.name])
            grads.append(g)
        G = torch.stack(grads, dim=0)  # [K, d_s]
        self._update_ema(G)
        t_grad = (time.time() - t_grad0) * 1000.0

        # 2) Build cosines using oracle
        res = self.oracle.build(self._ema)  # CosineBuildResult
        C_hat = res.cos
        t_oracle = res.timings
        tau = float(self.tau_schedule.value(self._step))

        # 3) Graph + coloring (approx path)
        t_graph0 = time.time()
        A_hat = build_adjacency(C_hat, tau)
        colors_hat, ctime = welsh_powell_coloring(A_hat, self._last_colors if self.use_warmstart else None)
        t_graph = (time.time() - t_graph0) * 1000.0

        # 4) (Optional) Exact shadow for fidelity
        fidel = {}
        if self.compute_exact_shadow:
            exact_res = ExactOracle().build(self._ema)
            C = exact_res.cos
            A = build_adjacency(C, tau)
            colors, _ = welsh_powell_coloring(A, None)
            # cosine error
            with torch.no_grad():
                iu, ju = torch.triu_indices(self.K, self.K, offset=1)
                err = (C_hat[iu, ju] - C[iu, ju]).abs()
                fidel["cos_mae"] = float(err.mean().item())
                fidel["cos_max"] = float(err.max().item())
            # graph f1
            prf = _pairwise_f1(A_hat, A)
            fidel.update({f"graph_{k}": v for k, v in prf.items()})
            # coloring similarity
            fidel["color_jaccard"] = _partition_pair_jaccard(colors_hat, colors)

        # 5) Build groups from colors (color id -> list of tasks)
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(colors_hat):
            groups.setdefault(c, []).append(i)
        # reorder groups by color id; within group keep as-is
        self._groups = [groups[c] for c in sorted(groups.keys())]
        self._group_idx = 0
        self._last_colors = colors_hat

        total_ms = (time.time() - t0) * 1000.0

        log = {
            "tau": tau,
            "refresh_ms": total_ms,
            "grad_ms": t_grad,
            "build_ms": t_oracle.get("total", 0.0),
            "embed_ms": t_oracle.get("embed", 0.0),
            "pairs_ms": t_oracle.get("pairs", 0.0),
            "graph_ms": t_graph,
            "colors": len(groups),
            "avg_degree": float(A_hat.float().sum().item() / max(1, self.K*(self.K-1))),
        }
        log.update(fidel)
        self._refresh_logs.append(log)

    def step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        """
        Perform one training step:
          - refresh schedule when needed
          - update the next color group (sum losses; one optimizer step)
        """
        self._step += 1

        # Build per-task "current" batches dict (use same batch for all)
        batches = {spec.name: batch for spec in self.tasks}

        # Refresh schedule if needed or groups exhausted
        if (self._step == 1) or (self._step % self.refresh_period == 0) or (not self._groups):
            self._refresh(batches)

        # Pick current group
        group = self._groups[self._group_idx]
        self._group_idx = (self._group_idx + 1) % len(self._groups)

        # Sum losses over tasks in group and take one optimizer step
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        loss_total = 0.0
        per_task = {}
        for idx in group:
            spec = self.tasks[idx]
            loss_i = spec.loss_fn(self.model, batch)
            per_task[spec.name] = float(loss_i.item())
            loss_total = loss_total + loss_i
        loss_total.backward()
        self.optimizer.step()

        out = {f"loss/{k}": v for k, v in per_task.items()}
        out["loss/total"] = float(loss_total.item())
        out["group_size"] = float(len(group))
        out["colors_now"] = float(len(self._groups))
        return out

    def refresh_logs(self) -> List[Dict[str, float]]:
        return self._refresh_logs
