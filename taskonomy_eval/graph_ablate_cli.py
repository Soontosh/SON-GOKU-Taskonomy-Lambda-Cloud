# taskonomy_eval/graph_ablate_cli.py
from __future__ import annotations
import argparse, json, os, csv, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate, METHOD_REGISTRY, maybe_set_graph_dump_dir
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

@dataclass
class Cfg:
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    device: str
    refresh_period: int
    ema_beta: float
    tau_kind: str
    tau_initial: float
    tau_target: float
    tau_warmup: int
    tau_anneal: int
    min_updates_per_cycle: int
    # graph
    graph_modes: Tuple[str, ...]
    graph_knn_k: Tuple[int, ...]
    graph_quantiles: Tuple[float, ...]
    densities: Tuple[float, ...]
    density_match: bool
    seeds: Tuple[int, ...]
    out_dir: str

def _loaders(cfg: Cfg):
    train = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    val = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.val_split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    tr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True)
    va = DataLoader(val, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    return tr, va

def _make_specs(model: nn.Module, tasks: Tuple[str, ...], seg_classes: int):
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    from son_goku import TaskSpec
    return [TaskSpec(name=t, loss_fn=make_loss_fn(t),
                     refresh_batch_provider=None, head_param_filter=head_filters[t])
            for t in tasks]

def _run_one(seed: int, cfg: Cfg, graph_mode: str,
             knn_k: int | None, q_p: float | None, dens: float | None) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    tr, va = _loaders(cfg)
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)

    Method = METHOD_REGISTRY["son_goku_graph_ablate"]
    method = Method(
        model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
        refresh_period=cfg.refresh_period, ema_beta=cfg.ema_beta,
        tau_kind=cfg.tau_kind, tau_initial=cfg.tau_initial, tau_target=cfg.tau_target,
        tau_warmup=cfg.tau_warmup, tau_anneal=cfg.tau_anneal,
        min_updates_per_cycle=cfg.min_updates_per_cycle,
        device=device,
        graph_mode=graph_mode,
        graph_knn_k=(knn_k if knn_k is not None else 3),
        graph_quantile_p=(q_p if q_p is not None else 0.3),
        graph_density_target=dens,
    )
    maybe_set_graph_dump_dir(method, os.path.join(cfg.out_dir, "graphs"))

    t0 = time.time()
    for _ in range(cfg.epochs):
        for batch in tr:
            batch = {k:(v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            _ = method.step(batch, 0)
    wall = (time.time() - t0) / 60.0
    metrics = evaluate(model, va, cfg.tasks, cfg.seg_classes, device)

    # Extract mean graph stats from scheduler logs (if present)
    logs = getattr(method, "sched", None).refresh_logs() if hasattr(method, "sched") else []
    colors = float(np.mean([x.get("colors", 0.0) for x in logs] or [0.0]))
    density = float(np.mean([x.get("graph_density", 0.0) for x in logs] or [0.0]))
    return {
        "seed": seed, "graph_mode": graph_mode,
        "graph_knn_k": knn_k, "graph_quantile_p": q_p, "graph_density_target": dens,
        "colors_mean": colors, "density_mean": density,
        "wall_min": wall, "val_metrics": metrics,
    }

def _scalarize(m):
    vals = []
    for t, d in m.items():
        vals.extend(list(d.values()))
    return float(np.mean(vals)) if vals else 0.0

def parse_args():
    ap = argparse.ArgumentParser("Graph-building ablation CLI")
    # data/model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=64)
    # train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    # SON-GOKU
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # graph ablation
    ap.add_argument("--graph_modes", type=str, nargs="+",
                    default=["threshold","knn","signed","quantile"],
                    help="Any subset of: threshold knn signed quantile")
    ap.add_argument("--graph_knn_k", type=int, nargs="+", default=[3],
                    help="Values for k in kNN (ignored for non-knn modes)")
    ap.add_argument("--graph_quantiles", type=float, nargs="+", default=[0.3],
                    help="Quantiles (0..1 or 0..100) for 'quantile' mode")
    ap.add_argument("--densities", type=float, nargs="+", default=[],
                    help="If provided with --density_match, we run density-matched calibration")
    ap.add_argument("--density_match", action="store_true",
                    help="If set, per-refresh density calibration is used for modes that support it.")
    # misc
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--out_dir", type=str, default="experiments/graph_ablate")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Cfg(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=tuple(args.tasks), resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, device=args.device,
        refresh_period=args.refresh_period, ema_beta=args.ema_beta,
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        tau_warmup=args.tau_warmup, tau_anneal=args.tau_anneal,
        min_updates_per_cycle=args.min_updates_per_cycle,
        graph_modes=tuple(args.graph_modes),
        graph_knn_k=tuple(args.graph_knn_k),
        graph_quantiles=tuple(args.graph_quantiles),
        densities=tuple(args.densities),
        density_match=bool(args.density_match),
        seeds=tuple(args.seeds), out_dir=args.out_dir,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    rows: List[Dict[str, Any]] = []
    for s in cfg.seeds:
        for mode in cfg.graph_modes:
            if cfg.density_match and cfg.densities:
                for dens in cfg.densities:
                    res = _run_one(s, cfg, mode, None, None, dens)
                    rows.append(res)
                    with open(os.path.join(cfg.out_dir, f"{mode}_dens{dens}_seed{s}.json"), "w") as f:
                        json.dump(res, f, indent=2)
            else:
                # parameterized sweeps per mode
                if mode == "knn":
                    for k in cfg.graph_knn_k:
                        res = _run_one(s, cfg, mode, k, None, None)
                        rows.append(res)
                        with open(os.path.join(cfg.out_dir, f"{mode}_k{k}_seed{s}.json"), "w") as f:
                            json.dump(res, f, indent=2)
                elif mode == "quantile":
                    for p in cfg.graph_quantiles:
                        res = _run_one(s, cfg, mode, None, p, None)
                        rows.append(res)
                        with open(os.path.join(cfg.out_dir, f"{mode}_p{p}_seed{s}.json"), "w") as f:
                            json.dump(res, f, indent=2)
                else:
                    res = _run_one(s, cfg, mode, None, None, None)
                    rows.append(res)
                    with open(os.path.join(cfg.out_dir, f"{mode}_seed{s}.json"), "w") as f:
                        json.dump(res, f, indent=2)

    # summarize
    def summarize(group_key: str):
        groups = {}
        for r in rows:
            key = r[group_key]
            groups.setdefault(key, []).append(r)
        table = []
        for key, arr in groups.items():
            vals = np.array([_scalarize(x["val_metrics"]) for x in arr], dtype=float)
            dens = np.array([x["density_mean"] for x in arr], dtype=float)
            cols = np.array([x["colors_mean"] for x in arr], dtype=float)
            wall = np.array([x["wall_min"] for x in arr], dtype=float)
            table.append({
                group_key: key,
                "n": len(arr),
                "val_mean": float(vals.mean()), "val_std": float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                "density_mean": float(dens.mean()), "colors_mean": float(cols.mean()),
                "wall_mean": float(wall.mean()),
            })
        return table

    # choose grouping key for CSV
    if cfg.density_match and cfg.densities:
        key = "graph_mode"
    else:
        key = "graph_mode"

    summary = summarize(key)
    csv_path = os.path.join(cfg.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary: w.writerow(row)
    print(f"[graph_ablate] wrote {csv_path}")

if __name__ == "__main__":
    main()
