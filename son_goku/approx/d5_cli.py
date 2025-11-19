# taskonomy_eval/d5_cli.py
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Reuse helpers from your runner (same file names you used earlier)
from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate,
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.oracles import (
    BaseCosineOracle, ExactOracle, RandomProjectionOracle,
    FrequentDirectionsOracle, EdgeSamplingOracle, IncrementalGramOracle,
)
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler


@dataclass
class D5Config:
    # data/model
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int
    # training
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    device: str
    # SON-GOKU core
    refresh_period: int
    tau_kind: str
    tau_initial: float
    tau_target: float
    tau_warmup: int
    tau_anneal: int
    ema_beta: float
    min_updates_per_cycle: int
    # techniques
    techniques: Tuple[str, ...]  # e.g. ("exact","jl","fd","edge","incr","warm")
    jl_dim: int
    fd_rank: int
    edge_p: float
    incr_eps: float
    warmstart: bool
    # trials
    seeds: Tuple[int, ...]
    out_dir: str


def _make_task_specs(model: nn.Module, tasks: Sequence[str], seg_classes: int):
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    return [
        TaskSpec(
            name=t,
            loss_fn=make_loss_fn(t),
            refresh_batch_provider=None,
            head_param_filter=head_filters[t],
        )
        for t in tasks
    ]


def _build_oracle(name: str, ema_dim: int, device: torch.device, cfg: D5Config) -> BaseCosineOracle:
    name = name.lower()
    if name == "exact":
        return ExactOracle()
    if name == "jl":
        return RandomProjectionOracle(d=ema_dim, r=cfg.jl_dim, device=device)
    if name == "fd":
        return FrequentDirectionsOracle(d=ema_dim, ell=cfg.fd_rank, device=device)
    if name == "edge":
        return EdgeSamplingOracle(ExactOracle(), p=cfg.edge_p, gen=torch.Generator(device="cpu"))
    if name == "incr":
        return IncrementalGramOracle(ExactOracle(), epsilon=cfg.incr_eps)
    if name == "warm":
        # warm-only uses exact cosines but warm-start coloring
        return ExactOracle()
    raise ValueError(f"Unknown technique: {name}")


def run_once(tech: str, seed: int, cfg: D5Config) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    run_dir = os.path.join(cfg.out_dir, f"{tech}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Data
    train_ds = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    val_ds = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.val_split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # Task specs
    task_specs = _make_task_specs(model, cfg.tasks, cfg.seg_classes)
    shared_filter = make_shared_filter(model)

    # Tau schedule
    tau = TauSchedule(
        kind=cfg.tau_kind,
        tau_initial=cfg.tau_initial,
        tau_target=cfg.tau_target,
        warmup_steps=cfg.tau_warmup,
        anneal_duration=cfg.tau_anneal,
    )

    # EMA vector dimension (sum of shared param counts)
    ema_dim = sum(p.numel() for p in model.parameters() if shared_filter(p))
    if ema_dim == 0:
        ema_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Oracle + scheduler
    oracle = _build_oracle(tech, ema_dim, device, cfg)
    use_warm = (tech == "warm") or cfg.warmstart
    sched = SonGokuInstrumentedScheduler(
        model=model,
        tasks=task_specs,
        optimizer=opt,
        shared_param_filter=shared_filter,
        tau_schedule=tau,
        refresh_period=cfg.refresh_period,
        ema_beta=cfg.ema_beta,
        min_updates_per_cycle=cfg.min_updates_per_cycle,
        cosine_oracle=oracle,
        use_warmstart_coloring=use_warm,
        device=device,
        compute_exact_shadow=True,  # so we can report fidelity
    )

    # Train
    t0 = time.time()
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            global_step += 1
            # move to device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            _ = sched.step(batch)
    wall_clock_min = (time.time() - t0) / 60.0

    # Eval
    with torch.no_grad():
        metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)

    # Save refresh logs & metrics
    refresh_logs = sched.refresh_logs()
    with open(os.path.join(run_dir, "refresh_logs.json"), "w") as f:
        json.dump(refresh_logs, f, indent=2)
    with open(os.path.join(run_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Aggregate useful stats
    # Overhead: mean refresh time, amortized per-step overhead, etc.
    if refresh_logs:
        mean_refresh_ms = float(np.mean([x["refresh_ms"] for x in refresh_logs]))
        mean_build_ms = float(np.mean([x["build_ms"] for x in refresh_logs]))
        mean_embed_ms = float(np.mean([x["embed_ms"] for x in refresh_logs]))
        mean_pairs_ms = float(np.mean([x["pairs_ms"] for x in refresh_logs]))
        mean_colors = float(np.mean([x["colors"] for x in refresh_logs]))
        mean_degree = float(np.mean([x["avg_degree"] for x in refresh_logs]))
        # Fidelity (available only when exact shadow computed)
        cos_mae = float(np.mean([x.get("cos_mae", 0.0) for x in refresh_logs]))
        cos_max = float(np.mean([x.get("cos_max", 0.0) for x in refresh_logs]))
        graph_f1 = float(np.mean([x.get("graph_f1", 0.0) for x in refresh_logs]))
        color_j = float(np.mean([x.get("color_jaccard", 0.0) for x in refresh_logs]))
    else:
        mean_refresh_ms = mean_build_ms = mean_embed_ms = mean_pairs_ms = mean_colors = mean_degree = 0.0
        cos_mae = cos_max = graph_f1 = color_j = 0.0

    result = {
        "technique": tech,
        "seed": seed,
        "wall_clock_min": wall_clock_min,
        "mean_refresh_ms": mean_refresh_ms,
        "mean_build_ms": mean_build_ms,
        "mean_embed_ms": mean_embed_ms,
        "mean_pairs_ms": mean_pairs_ms,
        "mean_colors": mean_colors,
        "mean_degree": mean_degree,
        "cos_mae": cos_mae,
        "cos_max": cos_max,
        "graph_f1": graph_f1,
        "color_jaccard": color_j,
        "val_metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def summarize(all_results: List[Dict[str, Any]], out_dir: str):
    import csv
    # group by technique
    by = {}
    for r in all_results:
        by.setdefault(r["technique"], []).append(r)

    rows = []
    for tech, rs in by.items():
        def avg_std(key, default=0.0):
            xs = [r.get(key, default) for r in rs]
            mu = float(np.mean(xs))
            sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
            return mu, sd

        mu_time, sd_time = avg_std("wall_clock_min")
        mu_ref, sd_ref = avg_std("mean_refresh_ms")
        mu_f1, sd_f1 = avg_std("graph_f1")
        mu_cmae, sd_cmae = avg_std("cos_mae")
        mu_cj, sd_cj = avg_std("color_jaccard")
        mu_colors, sd_colors = avg_std("mean_colors")

        # For validation, report averaged task metrics as a single number if possible
        # You can extend this to per-task columns if desired.
        # Example: take mean of all metric scalars across tasks.
        val_vals = []
        for r in rs:
            for t, d in r["val_metrics"].items():
                val_vals.extend(list(d.values()))
        mu_val = float(np.mean(val_vals)) if val_vals else 0.0
        sd_val = float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0

        rows.append({
            "technique": tech,
            "seeds": len(rs),
            "wall_clock_min_mean": mu_time, "wall_clock_min_std": sd_time,
            "refresh_ms_mean": mu_ref, "refresh_ms_std": sd_ref,
            "graph_f1_mean": mu_f1, "graph_f1_std": sd_f1,
            "cos_mae_mean": mu_cmae, "cos_mae_std": sd_cmae,
            "color_jaccard_mean": mu_cj, "color_jaccard_std": sd_cj,
            "colors_mean": mu_colors, "colors_std": sd_colors,
            "val_metric_mean": mu_val, "val_metric_std": sd_val,
        })

    # Write CSV + JSON
    csv_path = os.path.join(out_dir, "d5_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(os.path.join(out_dir, "d5_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[D5] Wrote summary to {csv_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Appendix D.5 isolation experiments")
    # Data / model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=64)
    # Train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    # SON-GOKU core
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log", "linear", "cosine", "constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # Techniques: choose any subset; "exact" always recommended as reference
    ap.add_argument("--techniques", type=str, nargs="+", default=["exact", "jl", "fd", "edge", "incr", "warm"],
                    help="Which D.5 techniques to run: exact jl fd edge incr warm")
    ap.add_argument("--jl_dim", type=int, default=128)
    ap.add_argument("--fd_rank", type=int, default=128)
    ap.add_argument("--edge_p", type=float, default=0.6)
    ap.add_argument("--incr_eps", type=float, default=5e-4)
    ap.add_argument("--warmstart", action="store_true", help="Use warm-start coloring (in addition to exact)")
    # Trials
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out_dir", type=str, default="experiments/d5")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = D5Config(
        data_root=args.data_root,
        split=args.split,
        val_split=args.val_split,
        tasks=tuple(args.tasks),
        resize=tuple(args.resize),
        buildings_list=args.buildings_list,
        seg_classes=args.seg_classes,
        base_channels=args.base_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        refresh_period=args.refresh_period,
        tau_kind=args.tau_kind,
        tau_initial=args.tau_initial,
        tau_target=args.tau_target,
        tau_warmup=args.tau_warmup,
        tau_anneal=args.tau_anneal,
        ema_beta=args.ema_beta,
        min_updates_per_cycle=args.min_updates_per_cycle,
        techniques=tuple(args.techniques),
        jl_dim=args.jl_dim,
        fd_rank=args.fd_rank,
        edge_p=args.edge_p,
        incr_eps=args.incr_eps,
        warmstart=bool(args.warmstart),
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_results: List[Dict[str, Any]] = []
    for tech in cfg.techniques:
        for seed in cfg.seeds:
            print(f"\n[D5] Running technique={tech}, seed={seed}")
            res = run_once(tech, seed, cfg)
            all_results.append(res)

    summarize(all_results, cfg.out_dir)


if __name__ == "__main__":
    main()
