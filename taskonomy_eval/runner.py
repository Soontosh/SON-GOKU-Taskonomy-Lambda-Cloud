from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# These imports assume you are using the Taskonomy scaffold from earlier.
# Adjust module paths to match your repo layout.
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig
from taskonomy_eval.models.mtl_unet import TaskonomyMTL
from taskonomy_eval.utils.metrics import depth_metrics, normal_metrics, bce_f1, miou

from son_goku import TaskSpec
from taskonomy_eval.methods.base import METHOD_REGISTRY
# Import to trigger registration side effects
from taskonomy_eval.methods import son_goku_method as _son_goku_method  # noqa: F401
from taskonomy_eval.methods import gradnorm_method as _gradnorm_method  # noqa: F401


# ---------- shared helpers ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(tasks: Sequence[str], seg_classes: int, base: int = 64) -> Tuple[nn.Module, Dict[str, int]]:
    out_ch: Dict[str, int] = {}
    for t in tasks:
        if t in ("depth_euclidean", "depth_zbuffer", "reshading", "edge_occlusion", "edge_texture", "keypoints2d"):
            out_ch[t] = 1
        elif t == "normal":
            out_ch[t] = 3
        elif t == "segment_semantic":
            out_ch[t] = seg_classes
        elif t == "principal_curvature":
            out_ch[t] = 2
        else:
            raise ValueError(f"Unsupported task: {t}")
    model = TaskonomyMTL(out_ch, base=base)
    return model, out_ch


def make_head_filter(model: nn.Module, task: str):
    head = model.heads[task]
    head_params = set(list(head.parameters()))

    def pred(p: nn.Parameter) -> bool:
        return p in head_params

    return pred


def make_shared_filter(model: nn.Module):
    head_params = set()
    for h in model.heads.values():
        head_params.update(list(h.parameters()))

    def pred(p: nn.Parameter) -> bool:
        return (p not in head_params) and p.requires_grad

    return pred


def build_task_loss(task: str, seg_classes: int):
    if task in ("depth_euclidean", "depth_zbuffer"):
        return lambda model, batch: torch.nn.functional.l1_loss(model(batch["rgb"], task), batch[task])
    if task == "normal":
        def loss_fn(model, batch):
            pred = model(batch["rgb"], task)
            p = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
            t = torch.nn.functional.normalize(batch[task], dim=1, eps=1e-6)
            cos = (p * t).sum(dim=1, keepdim=True).clamp(-1, 1)
            return (1 - cos).mean()
        return loss_fn
    if task in ("edge_occlusion", "edge_texture"):
        return lambda model, batch: torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch["rgb"], task), batch[task]
        )
    if task == "reshading":
        return lambda model, batch: torch.nn.functional.l1_loss(model(batch["rgb"], task), batch[task])
    if task == "segment_semantic":
        return lambda model, batch: torch.nn.functional.cross_entropy(
            model(batch["rgb"], task), batch[task]
        )
    if task == "keypoints2d":
        return lambda model, batch: torch.nn.functional.mse_loss(
            model(batch["rgb"], task), batch[task]
        )
    if task == "principal_curvature":
        return lambda model, batch: torch.nn.functional.l1_loss(
            model(batch["rgb"], task), batch[task]
        )
    raise ValueError(f"Unsupported task: {task}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, tasks: Sequence[str], seg_classes: int, device: torch.device):
    model.eval()
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        rgb = batch["rgb"]
        for t in tasks:
            y = model(rgb, t)
            if t in ("depth_euclidean", "depth_zbuffer"):
                m = depth_metrics(y, batch[t])
            elif t == "normal":
                m = normal_metrics(y, batch[t])
            elif t in ("edge_occlusion", "edge_texture"):
                m = bce_f1(y, batch[t])
            elif t == "reshading":
                m = {"mae": torch.nn.functional.l1_loss(y, batch[t]).item()}
            elif t == "segment_semantic":
                m = miou(y, batch[t], seg_classes)
            elif t == "keypoints2d":
                m = {"mse": torch.nn.functional.mse_loss(y, batch[t]).item()}
            elif t == "principal_curvature":
                m = {"l1": torch.nn.functional.l1_loss(y, batch[t]).item()}
            else:
                continue
            sums.setdefault(t, {})
            counts[t] = counts.get(t, 0) + 1
            for k, v in m.items():
                sums[t][k] = sums[t].get(k, 0.0) + float(v)
    return {t: {k: v / max(1, counts[t]) for k, v in sums[t].items()} for t in sums}


# ---------- main training entry ----------

@dataclass
class ExperimentConfig:
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    epochs: int
    batch_size: int
    lr: float
    base_channels: int
    num_workers: int
    device: str
    method: str
    seed: int
    out_dir: str

    # SON-GOKU / GradNorm method-specific hyperparams:
    refresh_period: int = 32
    tau_initial: float = 1.0
    tau_target: float = 0.25
    tau_kind: str = "log"
    tau_warmup: int = 0
    tau_anneal: int = 0
    ema_beta: float = 0.9
    min_updates_per_cycle: int = 1
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025

def train_and_eval_once(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device(cfg.device)

    # Datasets – same splits for all methods
    train_cfg = TaskonomyConfig(
        root=cfg.data_root,
        split=cfg.split,
        buildings_list=cfg.buildings_list,
        tasks=cfg.tasks,
        resize=cfg.resize,
    )
    val_cfg = TaskonomyConfig(
        root=cfg.data_root,
        split=cfg.val_split,
        buildings_list=cfg.buildings_list,
        tasks=cfg.tasks,
        resize=cfg.resize,
    )
    train_ds = TaskonomyDataset(train_cfg)
    val_ds = TaskonomyDataset(val_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model + optimizer
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # Task specs (shared across all methods)
    shared_filter = make_shared_filter(model)
    head_filters = {t: make_head_filter(model, t) for t in cfg.tasks}

    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, cfg.seg_classes)

        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)

        return loss_fn

    task_specs = [
        TaskSpec(
            name=t,
            loss_fn=make_loss_fn(t),
            refresh_batch_provider=None,
            head_param_filter=head_filters[t],
        )
        for t in cfg.tasks
    ]

    # Instantiate method
    MethodCls = METHOD_REGISTRY[cfg.method]

    if cfg.method == "son_goku":
        from taskonomy_eval.methods.son_goku_method import SonGokuMethod

        method = SonGokuMethod(
            model=model,
            tasks=task_specs,
            optimizer=opt,
            shared_param_filter=shared_filter,
            refresh_period=cfg.refresh_period,
            tau_initial=cfg.tau_initial,
            tau_target=cfg.tau_target,
            tau_kind=cfg.tau_kind,
            tau_warmup=cfg.tau_warmup,
            tau_anneal=cfg.tau_anneal,
            ema_beta=cfg.ema_beta,
            min_updates_per_cycle=cfg.min_updates_per_cycle,
        )
    elif cfg.method == "gradnorm":
        from taskonomy_eval.methods.gradnorm_method import GradNormMethod

        method = GradNormMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            alpha=cfg.gradnorm_alpha,
            weight_lr=cfg.gradnorm_lr,
            device=device,
        )
    else:
        # Future methods can be wired here as needed
        method = MethodCls(model=model, tasks=task_specs, optimizer=opt, shared_param_filter=shared_filter)

    # Training loop
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            global_step += 1

            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            logs = method.step(batch, global_step)
            if step % 50 == 0:
                msg = " | ".join(f"{k}:{v:.3f}" for k, v in sorted(logs.items()) if k.startswith("loss/"))
                print(f"[{cfg.method}] epoch {epoch+1} step {step}: {msg}")
        dt = time.time() - t0
        print(f"[{cfg.method}] epoch {epoch+1} done in {dt/60:.1f} min")

        # Eval
        metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)
        print(f"[{cfg.method}] [VAL] epoch {epoch+1}: {metrics}")

        # Save metrics per epoch
        with open(os.path.join(cfg.out_dir, f"val_metrics_epoch{epoch+1}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Save final model
    ckpt_path = os.path.join(cfg.out_dir, f"{cfg.method}_seed{cfg.seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved {cfg.method} checkpoint to {ckpt_path}")
    return {"checkpoint": ckpt_path}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--methods", type=str, nargs="+", required=True,
                    help="E.g. son_goku gradnorm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out_dir", type=str, default="experiments")

    # SON-GOKU hyperparams
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log", "linear", "cosine", "constant"])
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)

    # GradNorm hyperparams
    ap.add_argument("--gradnorm_alpha", type=float, default=1.5)
    ap.add_argument("--gradnorm_lr", type=float, default=0.025)

    return ap.parse_args()


def main():
    args = parse_args()
    methods = args.methods
    seeds = args.seeds

    for m in methods:
        if m not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{m}'. Known: {list(METHOD_REGISTRY.keys())}")

    for method in methods:
        for seed in seeds:
            run_dir = os.path.join(args.out_dir, f"{method}_seed{seed}")
            cfg = ExperimentConfig(
                data_root=args.data_root,
                split=args.split,
                val_split=args.val_split,
                tasks=tuple(args.tasks),
                resize=tuple(args.resize),
                buildings_list=args.buildings_list,
                seg_classes=args.seg_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                base_channels=args.base_channels,
                num_workers=args.num_workers,
                device=args.device,
                method=method,
                seed=seed,
                out_dir=run_dir,
                refresh_period=args.refresh_period,
                tau_initial=args.tau_initial,
                tau_target=args.tau_target,
                tau_kind=args.tau_kind,
                tau_warmup=args.tau_warmup,
                tau_anneal=args.tau_anneal,
                ema_beta=args.ema_beta,
                min_updates_per_cycle=args.min_updates_per_cycle,
                gradnorm_alpha=args.gradnorm_alpha,
                gradnorm_lr=args.gradnorm_lr,
            )
            print(f"\n=== Running method={method}, seed={seed}, out_dir={run_dir} ===")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            train_and_eval_once(cfg)


if __name__ == "__main__":
    main()
