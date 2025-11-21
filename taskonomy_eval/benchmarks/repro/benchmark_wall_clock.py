from __future__ import annotations
import argparse, json, os, time, csv, math, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Reuse your runner utilities & datasets (keeps model/loss identical to training)
from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, METHOD_REGISTRY
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

# ------------------------
# Config dataclass
# ------------------------
@dataclass
class BenchCfg:
    # data & model
    data_root: str
    split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: Optional[str]
    seg_classes: int
    base_channels: int
    # loader
    batch_size: int
    num_workers: int
    device: str
    # timing window
    warmup_steps: int
    measure_steps: int
    # training bits (kept minimal/constant)
    lr: float
    # SON-GOKU params (used only for methods that have a scheduler)
    refresh_period: int
    tau_kind: str
    tau_initial: float
    tau_target: float
    ema_beta: float
    min_updates_per_cycle: int
    # which methods + “techniques” to sweep
    methods: Tuple[str, ...]
    techniques: Tuple[str, ...]          # applies to SON-GOKU-like schedulers only
    seeds: Tuple[int, ...]
    # logging
    out_dir: str

# ------------------------
# Helpers
# ------------------------
def _build_dataloader(cfg: BenchCfg) -> DataLoader:
    ds = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=cfg.tasks, resize=cfg.resize
    ))
    # IMPORTANT: fixed seed worker_init to keep batch order stable across methods
    g = torch.Generator()
    g.manual_seed(0)

    def _seed_worker(_):
        np.random.seed(0)
        random.seed(0)

    return DataLoader(ds,
                      batch_size=cfg.batch_size,
                      shuffle=True,
                      num_workers=cfg.num_workers,
                      pin_memory=True,
                      worker_init_fn=_seed_worker,
                      generator=g)

def _make_specs(model: nn.Module, tasks: Tuple[str, ...], seg_classes: int):
    """TaskSpec list compatible with your methods."""
    from son_goku import TaskSpec
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    return [TaskSpec(name=t,
                     loss_fn=make_loss_fn(t),
                     refresh_batch_provider=None,
                     head_param_filter=head_filters[t])
            for t in tasks]

def _set_scheduler_technique_if_any(method_obj: Any, technique: str) -> None:
    """
    Non-intrusive best-effort: if the method exposes a SON-GOKU-style scheduler
    at `method_obj.sched`, try to set technique flags seen in your repo.
    If a flag/attr doesn't exist, ignore it (keeps compatibility).
    """
    if not hasattr(method_obj, "sched"):
        return
    sched = method_obj.sched

    # Common technique flags we’ve used in earlier CLIs; set if they exist.
    # exact / jl / fd / incr / gram / edge
    # You may have slightly different names; this is best-effort and safe.
    setattr_if_exists = lambda obj, name, val: setattr(obj, name, val) if hasattr(obj, name) else None

    # A canonical "technique" switch (if you added one)
    setattr_if_exists(sched, "technique", technique)

    # Specific toggles some repos expose
    if technique == "exact":
        setattr_if_exists(sched, "use_jl", False)
        setattr_if_exists(sched, "use_fd", False)
        setattr_if_exists(sched, "use_incremental", False)
        setattr_if_exists(sched, "use_gram_trick", False)
        setattr_if_exists(sched, "edge_subsample_p", None)
    elif technique == "jl":
        setattr_if_exists(sched, "use_jl", True)
        setattr_if_exists(sched, "use_fd", False)
        setattr_if_exists(sched, "use_incremental", False)
        setattr_if_exists(sched, "use_gram_trick", False)
    elif technique in ("fd", "incr"):
        setattr_if_exists(sched, "use_jl", False)
        setattr_if_exists(sched, "use_fd", True)
        setattr_if_exists(sched, "use_incremental", True)
        setattr_if_exists(sched, "use_gram_trick", True)
    elif technique == "gram":
        setattr_if_exists(sched, "use_jl", False)
        setattr_if_exists(sched, "use_fd", False)
        setattr_if_exists(sched, "use_incremental", False)
        setattr_if_exists(sched, "use_gram_trick", True)
    elif technique == "edge":
        setattr_if_exists(sched, "edge_subsample_p", 0.25)  # default p; override via env/your CLI elsewhere

def _scalar_sink(logs: Dict[str, float]) -> float:
    # ensure compute is realized; sum loss/* logs if present
    s = 0.0
    for k, v in logs.items():
        if k.startswith("loss/"):
            s += float(v)
    return s

def _measure_once(cfg: BenchCfg, method_key: str, technique: Optional[str], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)

    # Data (fixed order across methods via generator seeding)
    loader = _build_dataloader(cfg)

    # Model + method
    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)

    Method = METHOD_REGISTRY[method_key]
    # Try to pass SON-GOKU knobs; for non-SON-GOKU methods these kwargs are ignored by signature
    try:
        method = Method(
            model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
            device=device,
            refresh_period=cfg.refresh_period,  # SON-GOKU-only
            tau_kind=cfg.tau_kind, tau_initial=cfg.tau_initial, tau_target=cfg.tau_target,
            ema_beta=cfg.ema_beta, min_updates_per_cycle=cfg.min_updates_per_cycle,
        )
    except TypeError:
        # Signature doesn’t accept SG-only fields (e.g., GradNorm) → pass minimal set
        method = Method(
            model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter, device=device
        )

    # If this method has a SON-GOKU-like scheduler, set the technique
    if technique is not None:
        _set_scheduler_technique_if_any(method, technique)

    # Warmup window (stabilize CUDA, pay first refresh)
    it = iter(loader)
    for _ in range(cfg.warmup_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        _ = method.step(b, 0)

    # Timed window
    total_imgs = 0
    sink = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(cfg.measure_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        if isinstance(b.get("rgb", None), torch.Tensor):
            total_imgs += int(b["rgb"].shape[0])

        logs = method.step(b, 0)
        sink += _scalar_sink(logs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ips = float(total_imgs / max(elapsed, 1e-8))

    # Try to fetch refresh timing info if available
    refresh_ms = None
    if hasattr(method, "sched") and hasattr(method.sched, "refresh_logs"):
        logs = method.sched.refresh_logs()
        if logs:
            refresh_ms = float(np.mean([x.get("refresh_ms", 0.0) for x in logs if isinstance(x, dict)] or [0.0]))

    # device mem stats (optional)
    mem = {}
    if device.type == "cuda":
        mem = {
            "max_alloc_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
        }

    return {
        "method": method_key,
        "technique": technique,
        "seed": seed,
        "elapsed_sec": float(elapsed),
        "imgs_per_sec": ips,
        "refresh_ms": refresh_ms,
        "sink": sink,
        **mem,
    }

def parse_args():
    ap = argparse.ArgumentParser("Unified wall-clock benchmark (new repo, old methodology).")
    # Data/model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean","normal","reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=64)
    # Loader / device
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    # Timing window
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--measure_steps", type=int, default=120)
    # Train bits
    ap.add_argument("--lr", type=float, default=1e-3)
    # SON-GOKU defaults (ignored by methods that don’t accept them)
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # Methods & techniques
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["son_goku","gradnorm","pcgrad","mgda","famo","adatask","sel_update"])
    ap.add_argument("--techniques", type=str, nargs="+",
                    default=["exact","jl","fd","incr","gram","edge"],
                    help="Scheduler internals for SON-GOKU-like methods. Ignored by others.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    # Output
    ap.add_argument("--out_dir", type=str, default="experiments/bench_wall")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = BenchCfg(
        data_root=args.data_root, split=args.split,
        tasks=tuple(args.tasks), resize=tuple(args.resize),
        buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
        warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
        lr=args.lr,
        refresh_period=args.refresh_period, tau_kind=args.tau_kind,
        tau_initial=args.tau_initial, tau_target=args.tau_target,
        ema_beta=args.ema_beta, min_updates_per_cycle=args.min_updates_per_cycle,
        methods=tuple(args.methods), techniques=tuple(args.techniques),
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    rows: List[Dict[str, Any]] = []

    # For methods *without* a SON-GOKU scheduler, we still benchmark once (technique=None)
    def technique_iter(method_key: str):
        if method_key.lower().startswith("son_goku") or method_key.lower() in {"son_goku","son_goku_graph_ablate"}:
            return cfg.techniques
        else:
            return [None]

    for m in cfg.methods:
        for s in cfg.seeds:
            for tech in technique_iter(m):
                print(f"[bench] method={m} technique={tech} seed={s}")
                res = _measure_once(cfg, m, tech, s)
                rows.append(res)
                tag = tech if tech is not None else "na"
                with open(os.path.join(cfg.out_dir, f"{m}_tech_{tag}_seed{s}.json"), "w") as f:
                    json.dump(res, f, indent=2)

    # Aggregate mean ± std by (method, technique)
    agg: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["method"], str(r["technique"]))
        agg.setdefault(key, []).append(r)

    summary: List[Dict[str, Any]] = []
    for (m, t), arr in agg.items():
        el = np.array([a["elapsed_sec"] for a in arr], dtype=float)
        ips = np.array([a["imgs_per_sec"] for a in arr], dtype=float)
        ref = np.array([a["refresh_ms"] for a in arr if a["refresh_ms"] is not None], dtype=float)
        entry = {
            "method": m,
            "technique": t,
            "n": len(arr),
            "elapsed_mean_sec": float(el.mean()), "elapsed_std_sec": float(el.std(ddof=1)) if len(arr)>1 else 0.0,
            "ips_mean": float(ips.mean()), "ips_std": float(ips.std(ddof=1)) if len(arr)>1 else 0.0,
        }
        if ref.size > 0:
            entry["refresh_ms_mean"] = float(ref.mean())
            entry["refresh_ms_std"]  = float(ref.std(ddof=1)) if ref.size>1 else 0.0
        # mem if CUDA present
        if "max_alloc_mb" in arr[0]:
            mem_alloc = np.array([a.get("max_alloc_mb", np.nan) for a in arr], dtype=float)
            mem_res   = np.array([a.get("max_reserved_mb", np.nan) for a in arr], dtype=float)
            entry["max_alloc_mb_mean"] = float(np.nanmean(mem_alloc))
            entry["max_alloc_mb_std"]  = float(np.nanstd(mem_alloc, ddof=1)) if len(arr)>1 else 0.0
            entry["max_reserved_mb_mean"] = float(np.nanmean(mem_res))
            entry["max_reserved_mb_std"]  = float(np.nanstd(mem_res, ddof=1)) if len(arr)>1 else 0.0
        summary.append(entry)

    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    csv_path = os.path.join(cfg.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(summary[0].keys()) if summary else ["method","technique","n","elapsed_mean_sec","ips_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary: w.writerow(row)

    print(f"[bench] wrote {csv_path}")

if __name__ == "__main__":
    main()
