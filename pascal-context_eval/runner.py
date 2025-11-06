from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
import traceback

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from pascal_context_eval.datasets import PascalContextConfig, PascalContextDataset
from pascal_context_eval.methods import METHOD_REGISTRY
from taskonomy_eval.models.mtl_unet import TaskonomyMTL
from taskonomy_eval.utils.metrics import depth_metrics, normal_metrics, bce_f1, miou

from son_goku import TaskSpec

# Trigger method registration side effects.
from taskonomy_eval.methods import son_goku_method as _son_goku_method  # noqa: F401
from taskonomy_eval.methods import gradnorm_method as _gradnorm_method  # noqa: F401
from taskonomy_eval.methods import mgda_method as _mgda_method          # noqa: F401
from taskonomy_eval.methods import pcgrad_method as _pcgrad_method      # noqa: F401
from taskonomy_eval.methods import adatask_method as _adatask_method    # noqa: F401
from taskonomy_eval.methods import cagrad_method as _cagrad_method      # noqa: F401
from taskonomy_eval.methods import sel_update_method as _sel_update_method  # noqa: F401
from taskonomy_eval.methods import nashmtl_method as _nashmtl_method    # noqa: F401
from taskonomy_eval.methods import fairgrad_method as _fairgrad_method  # noqa: F401
from taskonomy_eval.methods import famo_method as _famo_method          # noqa: F401


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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


def build_task_loss(task: str, seg_classes: int, ignore_index: int):
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
            model(batch["rgb"], task), batch[task], ignore_index=ignore_index
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
def evaluate(model: nn.Module, loader: DataLoader, tasks: Sequence[str], seg_classes: int, ignore_index: int, device: torch.device):
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
                m = miou(y, batch[t], seg_classes, ignore_index=ignore_index)
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


@dataclass
class ExperimentConfig:
    data_root: str
    split: str
    val_split: str
    test_split: str | None
    tasks: Tuple[str, ...]
    resize: Tuple[int, int] | None
    label_set: str
    mask_root: str | None
    class_map: Mapping[int, int] | None
    ignore_index: int
    epochs: int
    batch_size: int
    lr: float
    base_channels: int
    num_workers: int
    device: str
    method: str
    seed: int
    out_dir: str

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
    cagrad_c: float = 0.5
    cagrad_inner_lr: float = 0.1
    cagrad_inner_steps: int = 20


def _make_loader(dataset: PascalContextDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _log_config(cfg: ExperimentConfig, seg_classes: int, log_path: Path) -> None:
    log_path.write_text(json.dumps({"config": asdict(cfg), "seg_classes": seg_classes}, indent=2) + "\n")


def _load_class_map(path: str | None) -> Dict[int, int] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        items = data.items()
    elif isinstance(data, Sequence):
        items = ((entry["raw"], entry["mapped"]) for entry in data)
    else:
        raise ValueError("Unsupported class map format. Use a dict or a list of {\"raw\", \"mapped\"} objects.")
    class_map: Dict[int, int] = {}
    for raw, mapped in items:
        class_map[int(raw)] = int(mapped)
    return class_map


def build_datasets(cfg: ExperimentConfig) -> Tuple[PascalContextDataset, PascalContextDataset, PascalContextDataset | None]:
    train_cfg = PascalContextConfig(
        root=cfg.data_root,
        split=cfg.split,
        tasks=cfg.tasks,
        resize=cfg.resize,
        label_set=cfg.label_set,
        mask_root=cfg.mask_root,
        class_map=dict(cfg.class_map) if cfg.class_map is not None else None,
        ignore_index=cfg.ignore_index,
    )
    val_cfg = PascalContextConfig(
        root=cfg.data_root,
        split=cfg.val_split,
        tasks=cfg.tasks,
        resize=cfg.resize,
        label_set=cfg.label_set,
        mask_root=cfg.mask_root,
        class_map=dict(cfg.class_map) if cfg.class_map is not None else None,
        ignore_index=cfg.ignore_index,
    )
    train_dataset = PascalContextDataset(train_cfg)
    val_dataset = PascalContextDataset(val_cfg)
    test_dataset = None
    if cfg.test_split:
        test_cfg = PascalContextConfig(
            root=cfg.data_root,
            split=cfg.test_split,
            tasks=cfg.tasks,
            resize=cfg.resize,
            label_set=cfg.label_set,
            mask_root=cfg.mask_root,
            class_map=dict(cfg.class_map) if cfg.class_map is not None else None,
            ignore_index=cfg.ignore_index,
        )
        test_dataset = PascalContextDataset(test_cfg)
    return train_dataset, val_dataset, test_dataset


def train(cfg: ExperimentConfig) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "config.json"

    log_file = open(out_dir / "stdout.txt", "w", buffering=1)
    tee = _Tee(sys.stdout, log_file)
    sys.stdout = tee  # type: ignore
    sys.stderr = tee  # type: ignore

    print("[INFO] Experiment configuration:")
    print(json.dumps(asdict(cfg), indent=2))

    set_seed(cfg.seed)

    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    seg_classes = train_dataset.num_classes
    if seg_classes is None:
        raise RuntimeError("PascalContextDataset.num_classes is None. Provide a class_map or known label_set.")
    ignore_index = train_dataset.ignore_index

    _log_config(cfg, seg_classes, log_path)

    train_loader = _make_loader(train_dataset, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = _make_loader(val_dataset, cfg.batch_size, cfg.num_workers, shuffle=False)
    test_loader = _make_loader(test_dataset, cfg.batch_size, cfg.num_workers, shuffle=False) if test_dataset else None

    device = torch.device(cfg.device)
    model, _ = build_model(cfg.tasks, seg_classes=seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    shared_filter = make_shared_filter(model)
    head_filters = {t: make_head_filter(model, t) for t in cfg.tasks}

    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes, ignore_index)

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

    method_name = cfg.method
    if method_name == "son_goku":
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
            log_dir=str(out_dir),
            log_interval=50,
        )
    elif method_name == "gradnorm":
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
    elif method_name == "mgda":
        from taskonomy_eval.methods.mgda_method import MGDAMethod

        method = MGDAMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            grad_normalization="none",
            max_qp_iter=250,
            qp_tol=1e-5,
            device=device,
        )
    elif method_name == "pcgrad":
        from taskonomy_eval.methods.pcgrad_method import PCGradMethod

        method = PCGradMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            device=device,
        )
    elif method_name == "cagrad":
        from taskonomy_eval.methods.cagrad_method import CAGradMethod

        method = CAGradMethod(
            model=model,
            tasks=task_specs,
            base_optimizer=opt,
            device=device,
            c=cfg.cagrad_c,
            inner_lr=cfg.cagrad_inner_lr,
            inner_steps=cfg.cagrad_inner_steps,
        )
    elif method_name == "sel_update":
        from taskonomy_eval.methods.sel_update_method import SelectiveUpdateMethod

        method = SelectiveUpdateMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            device=device,
            affinity_momentum=0.9,
            affinity_threshold=0.0,
            max_group_size=None,
        )
    elif method_name == "adatask":
        from taskonomy_eval.methods.adatask_method import AdaTaskMethod

        method = AdaTaskMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            lr=cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            device=device,
        )
    elif method_name == "nashmtl":
        from taskonomy_eval.methods.nashmtl_method import NashMTLMethod

        method = NashMTLMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            inner_lr=0.1,
            max_inner_iter=20,
            eps=1e-8,
            device=device,
        )
    elif method_name == "fairgrad":
        from taskonomy_eval.methods.fairgrad_method import FairGradMethod

        method = FairGradMethod(
            model=model,
            tasks=task_specs,
            shared_param_filter=shared_filter,
            base_optimizer=opt,
            device=device,
        )
    elif method_name == "famo":
        from taskonomy_eval.methods.famo_method import FAMOMethod

        method = FAMOMethod(
            model=model,
            tasks=task_specs,
            optimizer=opt,
            shared_param_filter=shared_filter,
            alpha=2.5e-2,
            weight_decay=1e-3,
            device=device,
        )
    else:
        method_cls = METHOD_REGISTRY.get(method_name)
        if method_cls is None:
            raise KeyError(f"Unknown method '{method_name}'. Available: {sorted(METHOD_REGISTRY.keys())}")
        method = method_cls(model=model, tasks=task_specs, optimizer=opt, shared_param_filter=shared_filter)

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
            if step % 10 == 0:
                msg = " | ".join(f"{k}:{v:.3f}" for k, v in sorted(logs.items()) if k.startswith("loss/"))
                print(f"[{method_name}] epoch {epoch+1} step {step}: {msg}")
        dt = time.time() - t0
        print(f"[{method_name}] epoch {epoch+1} done in {dt:.1f}s")

        metrics = evaluate(model, val_loader, cfg.tasks, seg_classes, ignore_index, device)
        print(f"[{method_name}] [VAL] epoch {epoch+1}: {metrics}")
        with open(out_dir / f"val_metrics_epoch{epoch+1}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    ckpt_path = out_dir / f"{method_name}_seed{cfg.seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    final_val = evaluate(model, val_loader, cfg.tasks, seg_classes, ignore_index, device)
    print(f"[{method_name}] [FINAL VAL]: {final_val}")

    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, cfg.tasks, seg_classes, ignore_index, device)
        print(f"[{method_name}] [TEST]: {test_metrics}")
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    sys.stdout = sys.__stdout__  # type: ignore[attr-defined]
    sys.stderr = sys.__stderr__  # type: ignore[attr-defined]
    log_file.close()
    return {"checkpoint": str(ckpt_path), "val_metrics": final_val, "test_metrics": test_metrics}


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Pascal-Context multi-task runner")
    parser.add_argument("--data-root", required=True, help="Path to Pascal-Context root (contains VOCdevkit/VOC2010 and context)")
    parser.add_argument("--split", default="train", help="Training split name")
    parser.add_argument("--val-split", default="val", help="Validation split name")
    parser.add_argument("--test-split", default=None, help="Optional test split name")
    parser.add_argument("--tasks", nargs="+", default=["segment_semantic"], help="Tasks to optimize")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--label-set", default="59", help="Label taxonomy to use (59 or custom)")
    parser.add_argument("--mask-root", default=None, help="Optional override for the context mask directory")
    parser.add_argument("--class-map", default=None, help="JSON file mapping raw IDs to contiguous class indices")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignore index for unlabeled pixels")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", default="son_goku", help="Optimization method key from METHOD_REGISTRY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=lambda: f"runs/pcontext_{datetime.now().strftime('%Y%m%d_%H%M%S')}" )

    parser.add_argument("--refresh-period", type=int, default=32)
    parser.add_argument("--tau-initial", type=float, default=1.0)
    parser.add_argument("--tau-target", type=float, default=0.25)
    parser.add_argument("--tau-kind", type=str, default="log")
    parser.add_argument("--tau-warmup", type=int, default=0)
    parser.add_argument("--tau-anneal", type=int, default=0)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--min-updates-per-cycle", type=int, default=1)
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument("--cagrad-c", type=float, default=0.5)
    parser.add_argument("--cagrad-inner-lr", type=float, default=0.1)
    parser.add_argument("--cagrad-inner-steps", type=int, default=20)

    args = parser.parse_args(argv)

    out_dir = args.out_dir() if callable(args.out_dir) else args.out_dir

    class_map = _load_class_map(args.class_map)

    cfg = ExperimentConfig(
        data_root=args.data_root,
        split=args.split,
        val_split=args.val_split,
        test_split=args.test_split,
        tasks=tuple(args.tasks),
        resize=tuple(args.resize) if args.resize is not None else None,
        label_set=args.label_set,
        mask_root=args.mask_root,
        class_map=class_map,
        ignore_index=args.ignore_index,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        device=args.device,
        method=args.method,
        seed=args.seed,
        out_dir=out_dir,
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
        cagrad_c=args.cagrad_c,
        cagrad_inner_lr=args.cagrad_inner_lr,
        cagrad_inner_steps=args.cagrad_inner_steps,
    )
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    try:
        train(cfg)
    except Exception as exc:  # pragma: no cover - debugging helper
        print("[ERROR] Training failed:", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
