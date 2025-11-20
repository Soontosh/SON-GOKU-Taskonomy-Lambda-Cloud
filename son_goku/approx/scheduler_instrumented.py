# son_goku/approx/scheduler_instrumented.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import itertools
import torch
from torch import nn, optim

from son_goku import TaskSpec, TauSchedule
from .oracles import BaseCosineOracle, ExactOracle
from .coloring import build_adjacency, welsh_powell_coloring


def _flatten_shared(params: List[nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
        for p in params
    ])


def _pairwise_f1(A_hat: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
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


def _ingroup_conflict_rate(A: torch.Tensor, groups: List[List[int]]) -> float:
    num_e, num_pairs = 0, 0
    for g in groups:
        if len(g) <= 1:
            continue
        for i, u in enumerate(g):
            for v in g[i+1:]:
                num_pairs += 1
                if bool(A[u, v].item()):
                    num_e += 1
    return float(num_e / max(1, num_pairs))


def _mb(numel: int, dtype: torch.dtype) -> float:
    bpe = {
        torch.float32: 4, torch.float: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.uint8: 1, torch.int8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8,
        torch.bool: 1
    }.get(dtype, 4)
    return (numel * bpe) / (1024.0 ** 2)


class SonGokuInstrumentedScheduler:
    """
    SON-GOKU training loop with timing/fidelity/memory logs.

    New:
      * random_groups_control (matched group sizes, shuffled membership)
      * mem logs at refresh
      * adaptive τ via percentile of cosine matrix (upper-triangular)
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
        random_groups_control: bool = False,
        measure_refresh_memory: bool = True,
        # ------------------ NEW: adaptive τ controls ------------------
        adaptive_tau_percentile: Optional[float] = None,  # e.g., 0.7 or 70 for 70th percentile
        adaptive_tau_clip: Tuple[float, float] = (-1.0, 1.0),
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
        self.random_groups_control = random_groups_control
        self.measure_refresh_memory = measure_refresh_memory

        # adaptive τ
        if adaptive_tau_percentile is not None and adaptive_tau_percentile > 1.0:
            adaptive_tau_percentile = adaptive_tau_percentile / 100.0
        self.adaptive_tau_percentile = adaptive_tau_percentile
        self.adaptive_tau_clip = adaptive_tau_clip

        self.K = len(self.tasks)
        self._ema = torch.zeros(self.K, sum(p.numel() for p in self.shared_params), device=self.device)
        self._have_ema = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        self._step = 0

        self._last_colors: Optional[List[int]] = None
        self._groups: List[List[int]] = []
        self._group_idx = 0

        self._refresh_logs: List[Dict[str, float]] = []
        self._last_refresh_log: Optional[Dict[str, float]] = None

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
        if self.measure_refresh_memory and self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()

        # 1) gradient snapshot
        t_grad0 = time.time()
        grads = []
        for i, spec in enumerate(self.tasks):
            g = self._compute_task_grad(spec, batches[spec.name])
            grads.append(g)
        G = torch.stack(grads, dim=0)
        self._update_ema(G)
        t_grad = (time.time() - t_grad0) * 1000.0

        # 2) cosine via oracle
        res = self.oracle.build(self._ema)
        C_hat = res.cos  # K x K in [-1,1]

        # ------------- NEW: choose τ -------------
        tau_mode = "schedule"
        tau = float(self.tau_schedule.value(self._step))
        tau_p = None
        if self.adaptive_tau_percentile is not None:
            tau_mode = "adaptive"
            iu, ju = torch.triu_indices(self.K, self.K, offset=1)
            vals = C_hat[iu, ju].detach()
            q = torch.tensor(self.adaptive_tau_percentile, device=vals.device, dtype=vals.dtype)
            q = torch.clamp(q, 0.0, 1.0)
            tau_val = torch.quantile(vals, q).item()
            lo, hi = self.adaptive_tau_clip
            tau = float(max(lo, min(hi, tau_val)))
            tau_p = float(self.adaptive_tau_percentile)

        # 3) adjacency + coloring
        A_hat = build_adjacency(C_hat, tau)
        colors_hat, ctime = welsh_powell_coloring(A_hat, self._last_colors if self.use_warmstart else None)

        # 4) groups (or random control)
        groups_coloring: Dict[int, List[int]] = {}
        for i, c in enumerate(colors_hat):
            groups_coloring.setdefault(c, []).append(i)
        color_groups = [groups_coloring[c] for c in sorted(groups_coloring.keys())]
        color_sizes = [len(g) for g in color_groups]
        if self.random_groups_control:
            perm = torch.randperm(self.K).tolist()
            active_groups: List[List[int]] = []
            offset = 0
            for sz in color_sizes:
                active_groups.append(perm[offset:offset+sz])
                offset += sz
        else:
            active_groups = color_groups

        ing_color = _ingroup_conflict_rate(A_hat, color_groups)
        ing_active = _ingroup_conflict_rate(A_hat, active_groups)

        # 5) (optional) fidelity vs exact
        fidel = {}
        if self.compute_exact_shadow:
            exact_res = ExactOracle().build(self._ema)
            C = exact_res.cos
            A = build_adjacency(C, tau)
            colors, _ = welsh_powell_coloring(A, None)
            with torch.no_grad():
                iu2, ju2 = torch.triu_indices(self.K, self.K, offset=1)
                err = (C_hat[iu2, ju2] - C[iu2, ju2]).abs()
                fidel["cos_mae"] = float(err.mean().item())
                fidel["cos_max"] = float(err.max().item())
            prf = _pairwise_f1(A_hat, A)
            fidel.update({f"graph_{k}": v for k, v in prf.items()})
            fidel["color_jaccard"] = _partition_pair_jaccard(colors_hat, colors)

        # finalize groups
        self._groups = active_groups
        self._group_idx = 0
        self._last_colors = colors_hat

        total_ms = (time.time() - t0) * 1000.0
        avg_deg = float(A_hat.float().sum().item() / max(1, self.K * (self.K - 1)))

        # memory
        mem_log = {}
        if self.measure_refresh_memory and self.device.type == "cuda":
            torch.cuda.synchronize()
            mem_log["mem/refresh_peak_mb"] = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        mem_log["mem/ema_mb"] = _mb(self._ema.numel(), self._ema.dtype)
        mem_log["mem/adj_mb"] = _mb(A_hat.numel(), torch.bool)

        log = {
            "tau": tau,
            "tau_mode": tau_mode,                 # NEW
            "tau_adaptive_p": tau_p,              # NEW (None if schedule)
            "refresh_ms": total_ms,
            "grad_ms": t_grad,
            "build_ms": res.timings.get("total", 0.0),
            "embed_ms": res.timings.get("embed", 0.0),
            "pairs_ms": res.timings.get("pairs", 0.0),
            "colors": len(color_groups),
            "avg_degree": avg_deg,
            "random_groups": bool(self.random_groups_control),
            "ing_conf_color": ing_color,
            "ing_conf_active": ing_active,
            **fidel,
            **mem_log,
        }
        self._refresh_logs.append(log)
        self._last_refresh_log = log

    def step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self._step += 1
        batches = {spec.name: batch for spec in self.tasks}
        if (self._step == 1) or (self._step % self.refresh_period == 0) or (not self._groups):
            self._refresh(batches)

        group = self._groups[self._group_idx]
        self._group_idx = (self._group_idx + 1) % len(self._groups)

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

    def current_groups(self) -> List[List[int]]:
        return [g[:] for g in self._groups]

    def last_refresh_log(self) -> Optional[Dict[str, float]]:
        return None if self._last_refresh_log is None else dict(self._last_refresh_log)

    def refresh_logs(self) -> List[Dict[str, float]]:
        return self._refresh_logs
