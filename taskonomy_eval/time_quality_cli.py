# taskonomy_eval/time_quality_cli.py
from __future__ import annotations
import argparse, json, os, time, csv, inspect
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate, METHOD_REGISTRY, maybe_set_graph_dump_dir,
    resolve_requested_tasks,
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
    # compare methods fairly
    methods: Tuple[str, ...]
    seeds: Tuple[int, ...]
    out_dir: str
    refresh_period: int
    tau_kind: str
    tau_initial: float
    tau_target: float
    tau_warmup: int
    tau_anneal: int
    ema_beta: float
    min_updates_per_cycle: int


def _resolved_tasks(cfg: Cfg) -> Tuple[str, ...]:
    return resolve_requested_tasks(cfg.tasks, cfg.data_root, cfg.split, cfg.buildings_list)


def _loaders(cfg: Cfg, tasks: Tuple[str, ...]):
    train = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=tasks, resize=cfg.resize
    ))
    val = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.val_split, buildings_list=cfg.buildings_list,
        tasks=tasks, resize=cfg.resize
    ))
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def _make_specs(model: nn.Module, tasks: Tuple[str, ...], seg_classes: int):
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    from son_goku import TaskSpec
    return [
        TaskSpec(name=t, loss_fn=make_loss_fn(t),
                 refresh_batch_provider=None, head_param_filter=head_filters[t])
        for t in tasks
    ]

def _instantiate_method(method_key: str,
                        cfg: Cfg,
                        model: nn.Module,
                        specs: Tuple[Any, ...],
                        optimizer: optim.Optimizer,
                        shared_filter,
                        device: torch.device):
    Method = METHOD_REGISTRY[method_key]
    candidate_kwargs = {
        "model": model,
        "tasks": specs,
        "optimizer": optimizer,
        "base_optimizer": optimizer,
        "shared_param_filter": shared_filter,
        "device": device,
        "refresh_period": cfg.refresh_period,
        "tau_kind": cfg.tau_kind,
        "tau_initial": cfg.tau_initial,
        "tau_target": cfg.tau_target,
        "tau_warmup": cfg.tau_warmup,
        "tau_anneal": cfg.tau_anneal,
        "ema_beta": cfg.ema_beta,
        "min_updates_per_cycle": cfg.min_updates_per_cycle,
        "lr": cfg.lr,
    }
    sig = inspect.signature(Method)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in candidate_kwargs:
            kwargs[name] = candidate_kwargs[name]
    return Method(**kwargs)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _scalarize(m):
    vals = []
    for t, d in m.items():
        vals.extend(list(d.values()))
    return float(np.mean(vals)) if vals else 0.0


def _run_one(cfg: Cfg, method_key: str, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)

    tasks = _resolved_tasks(cfg)
    train_loader, val_loader = _loaders(cfg, tasks)
    model, _ = build_model(tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, tasks, cfg.seg_classes)
    method = _instantiate_method(method_key, cfg, model, tuple(specs), opt, shared_filter, device)
    maybe_set_graph_dump_dir(method, os.path.join(cfg.out_dir, "graphs"))

    # Training loop — measure wall-clock and throughput
    total_imgs = 0
    t0 = time.time()
    global_step = 0
    for _ in range(cfg.epochs):
        for batch in train_loader:
            global_step += 1
            batch = _move_batch_to_device(batch, device)
            # count images for throughput (batch of RGB always present)
            if isinstance(batch.get("rgb", None), torch.Tensor):
                total_imgs += int(batch["rgb"].shape[0])
            _ = method.step(batch, global_step)
    wall_s = time.time() - t0
    wall_min = wall_s / 60.0
    imgs_per_sec = float(total_imgs / max(1.0, wall_s))

    metrics = evaluate(model, val_loader, tasks, cfg.seg_classes, device)
    return {
        "seed": seed,
        "method": method_key,
        "wall_min": wall_min,
        "imgs_per_sec": imgs_per_sec,
        "val_metrics": metrics,
        "val_scalar": _scalarize(metrics),
    }


def parse_args():
    ap = argparse.ArgumentParser("Experiment A: end-to-end training time vs quality (per method)")
    # Data / model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=32)
    # Train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # Compare
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["joint","son_goku","gradnorm"],
                    help="Any keys from METHOD_REGISTRY; 'joint' is provided here as a true no-refresh baseline.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--out_dir", type=str, default="experiments/time_quality")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Cfg(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=tuple(args.tasks), resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, device=args.device,
        methods=tuple(args.methods), seeds=tuple(args.seeds),
        out_dir=args.out_dir,
        refresh_period=args.refresh_period,
        tau_kind=args.tau_kind,
        tau_initial=args.tau_initial,
        tau_target=args.tau_target,
        tau_warmup=args.tau_warmup,
        tau_anneal=args.tau_anneal,
        ema_beta=args.ema_beta,
        min_updates_per_cycle=args.min_updates_per_cycle,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    rows: List[Dict[str, Any]] = []
    for s in cfg.seeds:
        for m in cfg.methods:
            print(f"[time_quality] method={m} seed={s}")
            res = _run_one(cfg, m, s)
            rows.append(res)
            with open(os.path.join(cfg.out_dir, f"{m}_seed{s}.json"), "w") as f:
                json.dump(res, f, indent=2)

    # Aggregate by method
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)

    summary = []
    for m, arr in groups.items():
        val = np.array([a["val_scalar"] for a in arr], dtype=float)
        wall = np.array([a["wall_min"] for a in arr], dtype=float)
        ips  = np.array([a["imgs_per_sec"] for a in arr], dtype=float)
        summary.append({
            "method": m,
            "n": len(arr),
            "val_mean": float(val.mean()), "val_std": float(val.std(ddof=1)) if len(arr)>1 else 0.0,
            "wall_mean": float(wall.mean()), "wall_std": float(wall.std(ddof=1)) if len(arr)>1 else 0.0,
            "ips_mean": float(ips.mean()), "ips_std": float(ips.std(ddof=1)) if len(arr)>1 else 0.0,
        })

    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    # CSV
    csv_path = os.path.join(cfg.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary: w.writerow(row)
    print(f"[time_quality] wrote {csv_path}")


if __name__ == "__main__":
    main()
