
import os, argparse, time
from typing import Tuple, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig
from taskonomy_eval.models.mtl_unet import TaskonomyMTL
from taskonomy_eval.utils.metrics import depth_metrics, normal_metrics, bce_f1, miou

# Import SON-GOKU from your installed package (pip install -e /path/to/son_goku)
from son_goku import SonGokuScheduler, TauSchedule, TaskSpec

def build_model(tasks: Tuple[str,...], seg_classes: int, base: int=64) -> Tuple[nn.Module, Dict[str,int]]:
    out_ch = {}
    for t in tasks:
        if t in ("depth_euclidean","depth_zbuffer","reshading","edge_occlusion","edge_texture","keypoints2d"):
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
    # Mark parameters belonging to a given head
    head = model.heads[task]
    head_params = set([p for p in head.parameters()])
    def pred(p):
        return p in head_params
    return pred

def make_shared_filter(model: nn.Module):
    head_params = set()
    for h in model.heads.values():
        for p in h.parameters():
            head_params.add(p)
    def pred(p):
        return (p not in head_params) and p.requires_grad
    return pred

def build_task_loss(task: str, seg_classes: int):
    if task in ("depth_euclidean","depth_zbuffer"):
        return lambda model, batch: torch.nn.functional.l1_loss(model(batch["rgb"], task), batch[task])
    if task == "normal":
        def loss_fn(model, batch):
            pred = model(batch["rgb"], task)
            # cosine loss: 1 - cos
            pred_n = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
            tgt_n  = torch.nn.functional.normalize(batch[task], dim=1, eps=1e-6)
            cos = (pred_n * tgt_n).sum(dim=1, keepdim=True).clamp(-1,1)
            return (1 - cos).mean()
        return loss_fn
    if task in ("edge_occlusion","edge_texture"):
        return lambda model, batch: torch.nn.functional.binary_cross_entropy_with_logits(model(batch["rgb"], task), batch[task])
    if task == "reshading":
        return lambda model, batch: torch.nn.functional.l1_loss(model(batch["rgb"], task), batch[task])
    if task == "segment_semantic":
        return lambda model, batch: torch.nn.functional.cross_entropy(model(batch["rgb"], task), batch[task])
    if task == "keypoints2d":
        return lambda model, batch: torch.nn.functional.mse_loss(model(batch["rgb"], task), batch[task])
    if task == "principal_curvature":
        return lambda model, batch: torch.nn.functional.l1_loss(model(batch["rgb"], task), batch[task])
    raise ValueError(f"Unsupported task: {task}")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, tasks: Tuple[str,...], seg_classes: int, device: torch.device):
    model.eval()
    sums: Dict[str, Dict[str,float]] = {}
    counts: Dict[str, int] = {}
    for batch in loader:
        rgb = batch["rgb"].to(device)
        for t in tasks:
            y = model(rgb, t)
            if t in ("depth_euclidean","depth_zbuffer"):
                m = depth_metrics(y, batch[t].to(device))
            elif t == "normal":
                m = normal_metrics(y, batch[t].to(device))
            elif t in ("edge_occlusion","edge_texture"):
                m = bce_f1(y, batch[t].to(device))
            elif t == "reshading":
                mae = torch.nn.functional.l1_loss(y, batch[t].to(device)).item()
                m = {"mae": mae}
            elif t == "segment_semantic":
                m = miou(y, batch[t].to(device), seg_classes)
            elif t == "keypoints2d":
                m = {"mse": torch.nn.functional.mse_loss(y, batch[t].to(device)).item()}
            elif t == "principal_curvature":
                m = {"l1": torch.nn.functional.l1_loss(y, batch[t].to(device)).item()}
            else:
                continue
            sums.setdefault(t, {})
            counts[t] = counts.get(t, 0) + 1
            for k,v in m.items():
                sums[t][k] = sums[t].get(k, 0.0) + float(v)
    # average
    out = {}
    for t in sums:
        out[t] = {k: v / max(1, counts[t]) for k,v in sums[t].items()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to Taskonomy root with train/val/test subfolders.")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean","normal","reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40, help="Num classes for segment_semantic (if used).")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    # SON-GOKU hyperparams
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # Datasets
    train_cfg = TaskonomyConfig(root=args.data_root, split=args.split, buildings_list=args.buildings_list, tasks=tuple(args.tasks), resize=tuple(args.resize))
    val_cfg   = TaskonomyConfig(root=args.data_root, split=args.val_split, buildings_list=args.buildings_list, tasks=tuple(args.tasks), resize=tuple(args.resize))
    train_ds = TaskonomyDataset(train_cfg)
    val_ds   = TaskonomyDataset(val_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model, tasks_out = build_model(tuple(args.tasks), seg_classes=args.seg_classes, base=args.base)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Shared and head filters
    shared_filter = make_shared_filter(model)
    head_filters = {t: make_head_filter(model, t) for t in args.tasks}

    # Build SON-GOKU tasks with loss functions
    def make_loss_fn(task):
        base_fn = build_task_loss(task, args.seg_classes)
        def loss_fn(model, batch):
            # batch already has tensors on CPU; move inside to device
            local = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            return base_fn(model, local)
        return loss_fn

    task_specs = []
    for t in args.tasks:
        task_specs.append(
            TaskSpec(
                name=t,
                loss_fn=make_loss_fn(t),
                refresh_batch_provider=None,
                head_param_filter=head_filters[t],
            )
        )
    tau = TauSchedule(kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target, warmup_steps=args.tau_warmup, anneal_duration=args.tau_anneal)
    sched = SonGokuScheduler(
        model=model,
        tasks=task_specs,
        optimizer=opt,
        shared_param_filter=shared_filter,
        refresh_period=args.refresh_period,
        tau_schedule=tau,
        ema_beta=args.ema_beta,
        min_updates_per_cycle=args.min_updates_per_cycle,
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            # SON-GOKU expects dict of batches for active tasks; we pass same anchors for all tasks
            batches = {t: {k: v for k,v in batch.items()} for t in args.tasks}
            losses = sched.step(batches)
            if step % 50 == 0:
                log_line = " | ".join([f"{k}:{v:.3f}" for k,v in sorted(losses.items())])
                print(f"Epoch {epoch+1} Step {step}: {log_line} | schedule={sched.schedule_snapshot()}")
        dt = time.time() - t0
        print(f"Epoch {epoch+1} finished in {dt/60:.1f} min. Recolor schedule: {sched.schedule_snapshot()}")
        # Eval
        metrics = evaluate(model, val_loader, tuple(args.tasks), args.seg_classes, device)
        print(f"[VAL] epoch {epoch+1}: {metrics}")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "taskonomy_mtl_son_goku.pt"))
    print("Saved model to checkpoints/taskonomy_mtl_son_goku.pt")

if __name__ == "__main__":
    main()
