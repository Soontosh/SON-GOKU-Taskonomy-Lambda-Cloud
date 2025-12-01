# taskonomy_eval/overhead_curve_cli.py
from __future__ import annotations
import argparse, json, os, time, csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, METHOD_REGISTRY, resolve_requested_tasks
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig
from taskonomy_eval.methods.son_goku_method import SonGokuMethod  # we need SON-GOKU specifically


@dataclass
class Cfg:
    data_root: str
    split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int
    batch_size: int
    num_workers: int
    device: str
    # measurement
    warmup_steps: int
    measure_steps: int
    refresh_periods: Tuple[int, ...]
    tau_kind: str
    tau_initial: float
    tau_target: float
    ema_beta: float
    min_updates_per_cycle: int
    seed: int
    out_dir: str


def _loaders(cfg: Cfg):
    ds = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    tr = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True)
    return tr


def _make_specs(model: nn.Module, tasks: Tuple[str, ...], seg_classes: int):
    from son_goku import TaskSpec
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    return [TaskSpec(name=t, loss_fn=make_loss_fn(t),
                     refresh_batch_provider=None, head_param_filter=head_filters[t])
            for t in tasks]


@torch.no_grad()
def _count_batch(b: Dict[str, Any]) -> int:
    if isinstance(b.get("rgb", None), torch.Tensor):
        return int(b["rgb"].shape[0])
    # fallback
    for v in b.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    return 0


def _measure_baseline_step_time(cfg: Cfg, loader: DataLoader, device: torch.device) -> float:
    """
    Baseline = per-step time without refresh (we avoid refresh during measured window).
    Trick: set refresh_period huge, do 1 warmup to pay the initial refresh at step=1,
    then time the next 'measure_steps' where no refresh triggers.
    """
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)
    method = SonGokuMethod(
        model=model, tasks=specs, optimizer=opt, shared_param_filter=make_shared_filter(model),
        refresh_period=10**9,  # effectively no refresh after step=1
        tau_initial=cfg.tau_initial, tau_target=cfg.tau_target, tau_kind=cfg.tau_kind,
        ema_beta=cfg.ema_beta, min_updates_per_cycle=cfg.min_updates_per_cycle,
        device=device
    )

    it = iter(loader)

    # warmup 1 step to consume the mandatory step==1 refresh
    b = next(it); b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}; _ = method.step(b, 1)

    # measure
    nsteps = 0
    t0 = time.time()
    while nsteps < cfg.measure_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        _ = method.step(b, 1)
        nsteps += 1
    dt = time.time() - t0
    return float((dt * 1000.0) / max(1, nsteps))  # ms/step


def _measure_for_R(cfg: Cfg, loader: DataLoader, device: torch.device, R: int) -> Dict[str, float]:
    """
    Measure avg step time for refresh_period=R. Also gather average refresh_ms from the scheduler logs.
    """
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)
    method = SonGokuMethod(
        model=model, tasks=specs, optimizer=opt, shared_param_filter=make_shared_filter(model),
        refresh_period=R, tau_initial=cfg.tau_initial, tau_target=cfg.tau_target, tau_kind=cfg.tau_kind,
        ema_beta=cfg.ema_beta, min_updates_per_cycle=cfg.min_updates_per_cycle,
        device=device
    )

    it = iter(loader)

    # warmup: do cfg.warmup_steps to stabilize timing & trigger at least one refresh
    for _ in range(cfg.warmup_steps):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        _ = method.step(b, 1)

    # measure
    nsteps = 0
    t0 = time.time()
    while nsteps < cfg.measure_steps:
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        _ = method.step(b, 1)
        nsteps += 1
    dt = time.time() - t0
    ms_per_step = float((dt * 1000.0) / max(1, nsteps))

    # avg refresh_ms from scheduler logs during the whole run
    refresh_logs = getattr(method, "sched", None).refresh_logs() if hasattr(method, "sched") else []
    avg_refresh_ms = float(np.mean([x.get("refresh_ms", 0.0) for x in refresh_logs] or [0.0]))
    return {"R": R, "ms_per_step": ms_per_step, "avg_refresh_ms": avg_refresh_ms}


def parse_args():
    ap = argparse.ArgumentParser("Experiment B: Overhead amortization curve (SON-GOKU only)")
    # Data/model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--tasks", type=str, nargs="+",
        default=["depth_euclidean", "normal", "reshading"],
        help="Space-separated task list or 'all' to automatically use every available target."
    )
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=32)
    # Loader/timing
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--measure_steps", type=int, default=80)
    ap.add_argument("--refresh_periods", type=int, nargs="+", default=[8,16,32,64])
    # SON-GOKU τ/EMA
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="experiments/overhead_curve")
    return ap.parse_args()


def main():
    args = parse_args()
    resolved_tasks = resolve_requested_tasks(args.tasks, args.data_root, args.split, args.buildings_list)
    cfg = Cfg(
        data_root=args.data_root, split=args.split,
        tasks=resolved_tasks, resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
        warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
        refresh_periods=tuple(args.refresh_periods),
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        ema_beta=args.ema_beta, min_updates_per_cycle=args.min_updates_per_cycle,
        seed=args.seed, out_dir=args.out_dir,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    loader = _loaders(cfg)

    # 1) baseline step time (no refresh in measured window)
    baseline_ms = _measure_baseline_step_time(cfg, loader, device)

    # 2) sweep R, get measured ms/step and avg refresh_ms
    rows: List[Dict[str, Any]] = []
    for R in cfg.refresh_periods:
        res = _measure_for_R(cfg, loader, device, R)
        rows.append(res)

    # 3) compute predictions: baseline + avg_refresh_ms / R
    for r in rows:
        r["pred_ms_per_step"] = float(baseline_ms + (r["avg_refresh_ms"] / max(1, r["R"])))
        r["baseline_ms"] = float(baseline_ms)

    # Save JSON + CSV
    with open(os.path.join(cfg.out_dir, "overhead_curve.json"), "w") as f:
        json.dump({"rows": rows, "baseline_ms": baseline_ms}, f, indent=2)

    csv_path = os.path.join(cfg.out_dir, "overhead_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["R","ms_per_step","pred_ms_per_step","avg_refresh_ms","baseline_ms"])
        w.writeheader()
        for r in rows: w.writerow(r)

    # Plot
    Rs = np.array([r["R"] for r in rows], dtype=float)
    y_meas = np.array([r["ms_per_step"] for r in rows], dtype=float)
    y_pred = np.array([r["pred_ms_per_step"] for r in rows], dtype=float)

    plt.figure()
    plt.plot(Rs, y_meas, "o-", label="measured")
    plt.plot(Rs, y_pred, "s--", label="predicted = baseline + refresh_ms / R")
    plt.xlabel("Refresh period R")
    plt.ylabel("Avg step time (ms)")
    plt.title("Overhead amortization (SON-GOKU)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    png = os.path.join(cfg.out_dir, "overhead_curve.png")
    plt.savefig(png, dpi=160, bbox_inches="tight")
    print(f"[overhead_curve] wrote {csv_path} and {png}")


if __name__ == "__main__":
    main()
