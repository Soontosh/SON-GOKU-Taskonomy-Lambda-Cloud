# taskonomy_eval/memory_cli.py
from __future__ import annotations
import argparse, os, json, time, csv, inspect
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Reuse your helpers
from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate, maybe_set_graph_dump_dir,
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

# SON-GOKU instrumented scheduler + D.5 oracles
from son_goku import TaskSpec, TauSchedule
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler
from son_goku.approx.oracles import (
    ExactOracle, RandomProjectionOracle, FrequentDirectionsOracle,
    EdgeSamplingOracle, IncrementalGramOracle
)

# If you also have the stand-alone methods in METHOD_REGISTRY:
try:
    from taskonomy_eval.methods import METHOD_REGISTRY
except Exception:
    METHOD_REGISTRY = {}


@dataclass
class MemCfg:
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int

    epochs: int
    warmup_steps: int
    measure_steps: int
    batch_size: int
    lr: float
    num_workers: int
    device: str

    # SON-GOKU core
    refresh_period: int
    ema_beta: float
    tau_kind: str
    tau_initial: float
    tau_target: float
    tau_warmup: int
    tau_anneal: int
    min_updates_per_cycle: int

    # Experiments
    exp: str  # m1_peak | m2_refresh | m3_k_scaling
    # M1: which methods
    methods: Tuple[str, ...]      # e.g., son_goku, gradnorm, pcgrad
    # M2: which SON-GOKU techniques
    techniques: Tuple[str, ...]   # exact jl fd edge incr warm
    jl_dim: int
    fd_rank: int
    edge_p: float
    incr_eps: float
    # M3: list of K values (will slice first K tasks)
    K_list: Tuple[int, ...]
    seeds: Tuple[int, ...]
    out_dir: str


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loaders(cfg: MemCfg):
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


def _make_specs(model: nn.Module, tasks: Tuple[str, ...], seg_classes: int):
    head_filters = {t: make_head_filter(model, t) for t in tasks}
    def make_loss_fn(task_name: str):
        base_fn = build_task_loss(task_name, seg_classes)
        def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
            return base_fn(model, batch)
        return loss_fn
    return [
        TaskSpec(name=t, loss_fn=make_loss_fn(t),
                 refresh_batch_provider=None, head_param_filter=head_filters[t])
        for t in tasks
    ]


def _instantiate_method(method_key: str,
                        cfg: MemCfg,
                        model: nn.Module,
                        specs: Tuple[TaskSpec, ...],
                        optimizer: optim.Optimizer,
                        shared_filter,
                        device: torch.device):
    MethodCls = METHOD_REGISTRY[method_key]
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
    sig = inspect.signature(MethodCls)
    init_kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in candidate_kwargs:
            init_kwargs[name] = candidate_kwargs[name]
    return MethodCls(**init_kwargs)


def _shared_ema_dim(model: nn.Module, shared_filter) -> int:
    total = 0
    for p in model.parameters():
        if shared_filter(p):
            total += p.numel()
    return total


def _build_sg_scheduler(model, specs, opt, shared_filter, cfg: MemCfg, device, technique: str):
    # tau schedule
    tau = TauSchedule(
        kind=cfg.tau_kind, tau_initial=cfg.tau_initial, tau_target=cfg.tau_target,
        warmup_steps=cfg.tau_warmup, anneal_duration=cfg.tau_anneal
    )
    ema_dim = _shared_ema_dim(model, shared_filter)
    # choose oracle per technique
    tech = technique.lower()
    if tech == "exact":
        oracle = ExactOracle()
    elif tech == "jl":
        oracle = RandomProjectionOracle(d=ema_dim, r=cfg.jl_dim, device=device)
    elif tech == "fd":
        oracle = FrequentDirectionsOracle(d=ema_dim, ell=cfg.fd_rank, device=device)
    elif tech == "edge":
        oracle = EdgeSamplingOracle(base=ExactOracle(), p=cfg.edge_p)
    elif tech == "incr":
        oracle = IncrementalGramOracle(base=ExactOracle(), epsilon=cfg.incr_eps)
    elif tech == "gram":
        oracle = ExactOracle()
    elif tech == "warm":
        oracle = ExactOracle()  # warm-start affects coloring only
    else:
        raise ValueError(f"Unknown SON-GOKU technique: {tech}")

    sched = SonGokuInstrumentedScheduler(
        model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
        tau_schedule=tau,
        refresh_period=cfg.refresh_period, ema_beta=cfg.ema_beta,
        min_updates_per_cycle=cfg.min_updates_per_cycle,
        cosine_oracle=oracle,
        use_warmstart_coloring=(tech == "warm"),
        device=device,
        compute_exact_shadow=False,           # faster for memory runs
        random_groups_control=False,
        measure_refresh_memory=True,
    )
    return sched


def _measure_step_memory(cfg: MemCfg, device: torch.device,
                         step_fn, train_loader: DataLoader) -> Dict[str, float]:
    """
    step_fn: callable(batch) -> None (per-step update)
    Returns mean peak (allocated/reserved) over measured steps and throughput.
    """
    if device.type != "cuda":
        print("[memory] WARNING: device is not CUDA; memory stats will be 0.")
    # Warmup
    it = iter(train_loader)
    for _ in range(cfg.warmup_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = _to_device(batch, device)
        step_fn(batch)

    # Measure
    peaks_alloc, peaks_reserved = [], []
    n_imgs = 0
    t0 = time.time()
    it = iter(train_loader)
    for _ in range(cfg.measure_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        step_fn(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peaks_alloc.append(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
            peaks_reserved.append(torch.cuda.max_memory_reserved() / (1024.0 ** 2))
        if "rgb" in batch and torch.is_tensor(batch["rgb"]):
            n = batch["rgb"].shape[0]
        else:
            tensor_vals = [v for v in batch.values() if torch.is_tensor(v)]
            n = tensor_vals[0].shape[0] if tensor_vals else 0
        n_imgs += int(n)
    secs = time.time() - t0
    throughput = float(n_imgs / max(1e-6, secs))
    return {
        "step_peak_mb_mean": float(np.mean(peaks_alloc) if peaks_alloc else 0.0),
        "step_reserved_mb_mean": float(np.mean(peaks_reserved) if peaks_reserved else 0.0),
        "throughput_ips": throughput,
    }


# -----------------------
# M1: Peak step memory
# -----------------------
def run_m1_for_method(seed: int, cfg: MemCfg, method: str) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    train_loader, val_loader = _loaders(cfg)

    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)

    if method.lower() == "son_goku":
        sched = _build_sg_scheduler(model, specs, opt, shared_filter, cfg, device, technique="exact")
        def step_fn(batch): sched.step(batch)
    else:
        # Use your METHOD_REGISTRY if available
        if method not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Add it to METHOD_REGISTRY or choose 'son_goku'.")
        method_obj = _instantiate_method(method, cfg, model, tuple(specs), opt, shared_filter, device)
        maybe_set_graph_dump_dir(method_obj, os.path.join(cfg.out_dir, "graphs"))
        global_step = {"i": 0}
        def step_fn(batch):
            global_step["i"] += 1
            method_obj.step(batch, global_step["i"])

    stats = _measure_step_memory(cfg, device, step_fn, train_loader)
    metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)

    return {"seed": seed, "method": method, **stats, "val_metrics": metrics}


def summarize_m1(rows: List[Dict[str, Any]], out_csv: str):
    def val_scalar(m):
        vals = []
        for t, d in m.items():
            vals.extend(list(d.values()))
        return float(np.mean(vals)) if vals else 0.0

    grouped = {}
    for r in rows:
        grouped.setdefault(r["method"], []).append(r)

    table = []
    for m, arr in grouped.items():
        mu_peak = float(np.mean([a["step_peak_mb_mean"] for a in arr]))
        sd_peak = float(np.std([a["step_peak_mb_mean"] for a in arr], ddof=1)) if len(arr) > 1 else 0.0
        mu_res = float(np.mean([a["step_reserved_mb_mean"] for a in arr]))
        mu_thr = float(np.mean([a["throughput_ips"] for a in arr]))
        mu_val = float(np.mean([val_scalar(a["val_metrics"]) for a in arr]))
        table.append({
            "method": m, "seeds": len(arr),
            "step_peak_mb_mean": mu_peak, "step_peak_mb_std": sd_peak,
            "step_reserved_mb_mean": mu_res, "throughput_ips_mean": mu_thr,
            "val_mean": mu_val,
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for row in table: w.writerow(row)
    print(f"[memory] wrote {out_csv}")


# -----------------------
# M2: Refresh vs step (SON-GOKU techniques)
# -----------------------
def run_m2_for_tech(seed: int, cfg: MemCfg, technique: str) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    train_loader, val_loader = _loaders(cfg)

    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, cfg.tasks, cfg.seg_classes)
    sched = _build_sg_scheduler(model, specs, opt, shared_filter, cfg, device, technique)

    # Measure step peaks during a short pass; refresh peaks are logged by the scheduler
    stats = _measure_step_memory(cfg, device, lambda b: sched.step(b), train_loader)

    # Aggregate refresh logs
    rlogs = sched.refresh_logs()
    refresh_peak = float(np.mean([x.get("mem/refresh_peak_mb", 0.0) for x in rlogs] or [0.0]))
    ema_mb = float(np.mean([x.get("mem/ema_mb", 0.0) for x in rlogs] or [0.0]))
    adj_mb = float(np.mean([x.get("mem/adj_mb", 0.0) for x in rlogs] or [0.0]))
    colors = float(np.mean([x.get("colors", 0.0) for x in rlogs] or [0.0]))

    metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)
    return {
        "seed": seed, "technique": technique,
        **stats,
        "refresh_peak_mb_mean": refresh_peak,
        "ema_mb_mean": ema_mb,
        "adj_mb_mean": adj_mb,
        "colors_mean": colors,
        "val_metrics": metrics,
    }


def summarize_m2(rows: List[Dict[str, Any]], out_csv: str):
    def val_scalar(m):
        vals = []
        for t, d in m.items():
            vals.extend(list(d.values()))
        return float(np.mean(vals)) if vals else 0.0

    grouped = {}
    for r in rows:
        grouped.setdefault(r["technique"], []).append(r)

    table = []
    for tech, arr in grouped.items():
        def mu(key):
            return float(np.mean([a[key] for a in arr]))
        def sd(key):
            xs = [a[key] for a in arr]
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        table.append({
            "technique": tech, "seeds": len(arr),
            "step_peak_mb_mean": mu("step_peak_mb_mean"),
            "step_peak_mb_std": sd("step_peak_mb_mean"),
            "refresh_peak_mb_mean": mu("refresh_peak_mb_mean"),
            "ema_mb_mean": mu("ema_mb_mean"),
            "adj_mb_mean": mu("adj_mb_mean"),
            "throughput_ips_mean": mu("throughput_ips"),
            "val_mean": float(np.mean([val_scalar(a["val_metrics"]) for a in arr])),
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for row in table: w.writerow(row)
    print(f"[memory] wrote {out_csv}")

    # Compare techniques vs exact SON-GOKU baseline
    baseline = next((row for row in table if row["technique"] == "exact"), None)
    if baseline:
        comps = []
        base_peak = baseline["step_peak_mb_mean"]
        base_ips = baseline["throughput_ips_mean"]
        base_refresh = baseline["refresh_peak_mb_mean"]
        for row in table:
            if row["technique"] == "exact":
                continue
            comps.append({
                "technique": row["technique"],
                "step_peak_rel": float(row["step_peak_mb_mean"] / max(base_peak, 1e-8)),
                "refresh_peak_rel": float(row["refresh_peak_mb_mean"] / max(base_refresh, 1e-8)),
                "throughput_rel": float(row["throughput_ips_mean"] / max(base_ips, 1e-8)),
                "step_peak_mb_mean": row["step_peak_mb_mean"],
                "refresh_peak_mb_mean": row["refresh_peak_mb_mean"],
                "throughput_ips_mean": row["throughput_ips_mean"],
            })
        if comps:
            comp_path = os.path.join(os.path.dirname(out_csv), "m2_vs_exact.json")
            with open(comp_path, "w") as f:
                json.dump({
                    "baseline": {
                        "technique": "exact",
                        "step_peak_mb_mean": base_peak,
                        "refresh_peak_mb_mean": base_refresh,
                        "throughput_ips_mean": base_ips,
                    },
                    "comparisons": comps,
                }, f, indent=2)
            print("[memory] SON-GOKU technique memory ratios vs exact:")
            for c in comps:
                print(f"  tech={c['technique']:>5} | step_peak={c['step_peak_rel']*100:6.1f}% "
                      f"| refresh_peak={c['refresh_peak_rel']*100:6.1f}% "
                      f"| throughput={c['throughput_rel']*100:6.1f}%")


# -----------------------
# M3: Memory vs K scaling
# -----------------------
def run_m3_for_method(seed: int, cfg: MemCfg, method: str, K: int) -> Dict[str, Any]:
    subcfg = MemCfg(**{**asdict(cfg), "tasks": tuple(cfg.tasks[:K])})
    return run_m1_for_method(seed, subcfg, method)


def summarize_m3(rows: List[Dict[str, Any]], out_csv: str):
    # rows: {method, seed, step_peak_mb_mean, ..., val_metrics, and implicit K via length of tasks?}
    # We record K explicitly in rows below; group by method to fit slope (MB/task)
    grouped = {}
    for r in rows:
        grouped.setdefault((r["method"]), []).append(r)

    table = []
    for m, arr in grouped.items():
        # Build (K, MB) arrays
        Ks = np.array([r["K"] for r in arr], dtype=float)
        MB = np.array([r["step_peak_mb_mean"] for r in arr], dtype=float)
        # Linear fit MB = a*K + b
        if len(Ks) >= 2:
            A = np.vstack([Ks, np.ones_like(Ks)]).T
            a, b = np.linalg.lstsq(A, MB, rcond=None)[0]
        else:
            a, b = float("nan"), float("nan")
        table.append({
            "method": m, "points": len(arr),
            "slope_mb_per_task": float(a), "intercept_mb": float(b),
            "K_min": float(np.min(Ks)), "K_max": float(np.max(Ks)),
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for row in table: w.writerow(row)
    print(f"[memory] wrote {out_csv}")


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser("Memory experiments: M1 (peak step), M2 (refresh vs step), M3 (scaling vs K)")
    # data/model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+",
                    default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=32)
    # train
    ap.add_argument("--epochs", type=int, default=1, help="(kept for API symmetry; we measure steps)")
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--measure_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    # SON-GOKU schedule params
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--tau_kind", type=str, default="log",
                    choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    # experiments
    ap.add_argument("--exp", type=str, required=True,
                    choices=["m1_peak","m2_refresh","m3_k_scaling"])
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["son_goku","gradnorm","pcgrad","mgda","cagrad","famo","adatask"])
    ap.add_argument("--techniques", type=str, nargs="+",
                    default=["exact","jl","fd","edge","incr","warm"])
    ap.add_argument("--jl_dim", type=int, default=128)
    ap.add_argument("--fd_rank", type=int, default=128)
    ap.add_argument("--edge_p", type=float, default=0.25)
    ap.add_argument("--incr_eps", type=float, default=1e-3)
    ap.add_argument("--K_list", type=int, nargs="+", default=[2,3,5,8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--out_dir", type=str, default="experiments/memory")
    return ap.parse_args()


def main():
    args = parse_args()
    args.tasks = resolve_requested_tasks(args.tasks, args.data_root, args.split, args.buildings_list)
    cfg = MemCfg(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=tuple(args.tasks), resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
        batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers, device=args.device,
        refresh_period=args.refresh_period, ema_beta=args.ema_beta,
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        tau_warmup=args.tau_warmup, tau_anneal=args.tau_anneal,
        min_updates_per_cycle=args.min_updates_per_cycle,
        exp=args.exp, methods=tuple(args.methods), techniques=tuple(args.techniques),
        jl_dim=args.jl_dim, fd_rank=args.fd_rank, edge_p=args.edge_p, incr_eps=args.incr_eps,
        K_list=tuple(args.K_list), seeds=tuple(args.seeds), out_dir=args.out_dir,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device(cfg.device)

    if cfg.exp == "m1_peak":
        rows = []
        for m in cfg.methods:
            for s in cfg.seeds:
                print(f"[memory] M1 seed={s} method={m}")
                res = run_m1_for_method(s, cfg, m)
                rows.append(res)
                with open(os.path.join(cfg.out_dir, f"m1_{m}_seed{s}.json"), "w") as f:
                    json.dump(res, f, indent=2)
        summarize_m1(rows, os.path.join(cfg.out_dir, "m1_summary.csv"))
        return

    if cfg.exp == "m2_refresh":
        rows = []
        for tech in cfg.techniques:
            for s in cfg.seeds:
                print(f"[memory] M2 seed={s} technique={tech}")
                res = run_m2_for_tech(s, cfg, tech)
                rows.append(res)
                with open(os.path.join(cfg.out_dir, f"m2_{tech}_seed{s}.json"), "w") as f:
                    json.dump(res, f, indent=2)
        summarize_m2(rows, os.path.join(cfg.out_dir, "m2_summary.csv"))
        return

    if cfg.exp == "m3_k_scaling":
        rows = []
        for s in cfg.seeds:
            for m in cfg.methods:
                for K in cfg.K_list:
                    if K > len(cfg.tasks):
                        continue
                    print(f"[memory] M3 seed={s} method={m} K={K}")
                    r = run_m3_for_method(s, cfg, m, K)
                    r["K"] = K
                    rows.append(r)
                    with open(os.path.join(cfg.out_dir, f"m3_{m}_K{K}_seed{s}.json"), "w") as f:
                        json.dump(r, f, indent=2)
        summarize_m3(rows, os.path.join(cfg.out_dir, "m3_summary.csv"))
        return


if __name__ == "__main__":
    main()
