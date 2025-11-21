# taskonomy_eval/ablate_cli.py
from __future__ import annotations
import argparse, json, os, time, csv
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
class AblateCfg:
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
    seeds: Tuple[int, ...]
    out_dir: str


def _dataset(cfg: AblateCfg):
    train = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    val = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.val_split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def _make_task_specs(model: nn.Module, tasks, seg_classes: int):
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    from son_goku import TaskSpec
    return [
        TaskSpec(name=t, loss_fn=make_loss_fn(t), refresh_batch_provider=None, head_param_filter=head_filters[t])
        for t in tasks
    ]


def _run_one(seed: int, cfg: AblateCfg, method_key: str):
    device = torch.device(cfg.device)
    set_seed(seed)
    train_loader, val_loader = _dataset(cfg)

    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_task_specs(model, cfg.tasks, cfg.seg_classes)

    MethodCls = METHOD_REGISTRY[method_key]
    method = MethodCls(
        model=model,
        tasks=specs,
        optimizer=opt,
        shared_param_filter=shared_filter,
        refresh_period=cfg.refresh_period,
        tau_initial=cfg.tau_initial,
        tau_target=cfg.tau_target,
        tau_kind=cfg.tau_kind,
        tau_warmup=cfg.tau_warmup,
        tau_anneal=cfg.tau_anneal,
        ema_beta=cfg.ema_beta if "single_step" not in method_key else 0.0,  # explicit H=1
        min_updates_per_cycle=cfg.min_updates_per_cycle,
        device=device,
    )
    maybe_set_graph_dump_dir(method, os.path.join(cfg.out_dir, "graphs"))

    t0 = time.time()
    for _ in range(cfg.epochs):
        for batch in train_loader:
            # Move happens inside method.step(), which mirrors the SON-GOKU method behavior.
            _ = method.step(batch, 0)
    wall_min = (time.time() - t0) / 60.0

    metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)
    return {"seed": seed, "method": method_key, "metrics": metrics, "wall_min": wall_min}


def _aggregate(rows: List[Dict[str, Any]]):
    # mean±std over the scalarized average of task metrics
    def scalarize(m):
        vals = []
        for task, d in m.items():
            vals.extend(list(d.values()))
        return float(np.mean(vals)) if vals else 0.0

    out = {}
    for mk in set(r["method"] for r in rows):
        subset = [r for r in rows if r["method"] == mk]
        vals = [scalarize(r["metrics"]) for r in subset]
        wall = [r["wall_min"] for r in subset]
        out[mk] = {
            "n": len(subset),
            "val_mean": float(np.mean(vals)),
            "val_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "wall_mean": float(np.mean(wall)),
            "wall_std": float(np.std(wall, ddof=1)) if len(wall) > 1 else 0.0,
        }
    return out


def parse_args():
    ap = argparse.ArgumentParser("Paper ablations: static one-shot & single-step conflict")
    # Data/model
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
    # SON-GOKU params (match defaults unless you want to change)
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log", "linear", "cosine", "constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # Seeds / out
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out_dir", type=str, default="experiments/ablations")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = AblateCfg(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=tuple(args.tasks), resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, device=args.device,
        refresh_period=args.refresh_period, ema_beta=args.ema_beta,
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        tau_warmup=args.tau_warmup, tau_anneal=args.tau_anneal,
        min_updates_per_cycle=args.min_updates_per_cycle,
        seeds=tuple(args.seeds), out_dir=args.out_dir
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    methods = ["son_goku", "son_goku_static", "son_goku_single_step"]
    rows: List[Dict[str, Any]] = []
    for seed in cfg.seeds:
        for mk in methods:
            print(f"[ablate] method={mk} seed={seed}")
            res = _run_one(seed, cfg, mk)
            rows.append(res)
            with open(os.path.join(cfg.out_dir, f"{mk}_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)

    agg = _aggregate(rows)
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump({"aggregate": agg, "rows": rows}, f, indent=2)

    # quick CSV
    csv_path = os.path.join(cfg.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "val_mean", "val_std", "wall_mean", "wall_std"])
        w.writeheader()
        for k, v in agg.items():
            w.writerow({"method": k, **v})
    print(f"[ablate] wrote {csv_path}")


if __name__ == "__main__":
    main()
