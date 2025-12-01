# taskonomy_eval/rebuttal_cli.py
from __future__ import annotations
import argparse, json, os, time, copy, csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Reuse your existing helpers
from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate, resolve_requested_tasks,
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

from son_goku import TaskSpec, TauSchedule
from son_goku.approx.oracles import ExactOracle
from son_goku.approx.scheduler_instrumented import SonGokuInstrumentedScheduler

import csv

@dataclass
class RebuttalConfig:
    # Data/model
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int
    # Train
    epochs: int
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
    exp: str  # random_groups | scheduled_vs_mixed | tau_sweep
    taus: Tuple[float, ...]  # used when exp == tau_sweep with tau_kind=constant
    micro_batches: int       # used in scheduled_vs_mixed
    seeds: Tuple[int, ...]
    out_dir: str
    adaptive_tau_p: float | None
    log_train_every: int

def _train_log_path(cfg: RebuttalConfig, label: str) -> str:
    safe = label.replace("/", "_").replace(" ", "_")
    return os.path.join(cfg.out_dir, f"train_log_{safe}.csv")


def _maybe_log_step(logs: Dict[str, float], global_step: int, epoch: int, step: int,
                    csv_path: str, log_every: int) -> None:
    if log_every <= 0 or not logs:
        return
    if (global_step % log_every) != 0:
        return
    loss_keys = sorted(k for k in logs if k.startswith("loss/"))
    if not loss_keys:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(f"global_step,epoch,step,{','.join(loss_keys)}\n")
    with open(csv_path, "a") as f:
        row = ",".join(f"{logs.get(k, np.nan)}" for k in loss_keys)
        f.write(f"{global_step},{epoch},{step},{row}\n")

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


def _build_sched(model, task_specs, opt, shared_filter, cfg: RebuttalConfig,
                 device, random_groups_control: bool, tau_const: float | None = None, adaptive_tau_p: float | None = None):
    tau = TauSchedule(
        kind=("constant" if tau_const is not None else cfg.tau_kind),
        tau_initial=(tau_const if tau_const is not None else cfg.tau_initial),
        tau_target=(tau_const if tau_const is not None else cfg.tau_target),
        warmup_steps=cfg.tau_warmup,
        anneal_duration=cfg.tau_anneal,
    )
    return SonGokuInstrumentedScheduler(
        model=model,
        tasks=task_specs,
        optimizer=opt,
        shared_param_filter=shared_filter,
        tau_schedule=tau,
        refresh_period=cfg.refresh_period,
        ema_beta=cfg.ema_beta,
        min_updates_per_cycle=cfg.min_updates_per_cycle,
        cosine_oracle=ExactOracle(),
        use_warmstart_coloring=False,
        device=device,
        compute_exact_shadow=True,           # keep if you want fidelity logs vs exact
        random_groups_control=random_groups_control,
        measure_refresh_memory=True,
        adaptive_tau_percentile=adaptive_tau_p,  # ← THIS enables adaptive-τ
    )

def run_adaptive_vs_default(seed: int, cfg: RebuttalConfig) -> Dict[str, Any]:
    """
    Trains two runs per seed:
      - 'scheduled': your configured τ schedule (log/linear/cosine/constant)
      - 'adaptive' : τ = percentile_p(C) each refresh (upper-triangular of cosine)
    """
    device = torch.device(cfg.device)
    set_seed(seed)

    # Dataloaders
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

    def make_specs(model: nn.Module):
        head_filters = {t: make_head_filter(model, t) for t in cfg.tasks}
        def make_loss_fn(task_name: str):
            base_fn = build_task_loss(task_name, cfg.seg_classes)
            def loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
                return base_fn(model, batch)
            return loss_fn
        return [
            TaskSpec(name=t, loss_fn=make_loss_fn(t),
                     refresh_batch_provider=None, head_param_filter=head_filters[t])
            for t in cfg.tasks
        ]

    out = {}
    for tag, adaptive_p in [("scheduled", None), ("adaptive", cfg.adaptive_tau_p)]:
        model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        shared_filter = make_shared_filter(model)
        specs = make_specs(model)

        sched = _build_sched(
            model, specs, opt, shared_filter, cfg, device,
            random_groups_control=False,
            tau_const=None,
            adaptive_tau_p=adaptive_p
        )
        label = f"adaptive_tau_seed{seed}_{tag}"
        train_csv = _train_log_path(cfg, label)
        global_step = 0

        t0 = time.time()
        for epoch in range(cfg.epochs):
            for step_idx, batch in enumerate(train_loader):
                global_step += 1
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                logs = sched.step(batch)
                _maybe_log_step(logs, global_step, epoch + 1, step_idx, train_csv, cfg.log_train_every)
        wall = (time.time() - t0)/60.0
        metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)

        out[tag] = {
            "seed": seed,
            "wall_clock_min": wall,
            "val_metrics": metrics,
            "refresh_logs": sched.refresh_logs(),
        }

    return out


def _total_loss(model: nn.Module, task_specs: Sequence[TaskSpec],
                batch: Dict[str, Any]) -> torch.Tensor:
    total = 0.0
    for spec in task_specs:
        total = total + spec.loss_fn(model, batch)
    return total


def _one_mixed_step(model: nn.Module, task_specs: Sequence[TaskSpec],
                    batch: Dict[str, Any], lr: float):
    opt = optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad(set_to_none=True)
    loss = _total_loss(model, task_specs, batch)
    loss.backward()
    opt.step()
    return float(loss.item())


def _one_scheduled_steps(model: nn.Module, task_specs: Sequence[TaskSpec],
                         groups: List[List[int]], batch: Dict[str, Any], lr: float):
    opt = optim.Adam(model.parameters(), lr=lr)
    before = _total_loss(model, task_specs, batch).item()
    for group in groups:
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for idx in group:
            loss = loss + task_specs[idx].loss_fn(model, batch)
        loss.backward()
        opt.step()
    return before


def _microtest_scheduled_vs_mixed(model: nn.Module, task_specs: Sequence[TaskSpec],
                                  groups: List[List[int]], probe_batches: List[Dict[str, Any]],
                                  lr: float, device: torch.device) -> Dict[str, float]:
    """
    Returns average delta loss across probe batches for Mixed vs Scheduled.
    Positive deltas mean larger improvement.
    """
    deltas_mixed, deltas_sched = [], []
    for pb in probe_batches:
        # ensure on device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in pb.items()}
        # Mixed
        m = copy.deepcopy(model).to(device)
        base = _total_loss(m, task_specs, batch).item()
        mixed_before = _one_mixed_step(m, task_specs, batch, lr=lr)
        mixed_after = _total_loss(m, task_specs, batch).item()
        deltas_mixed.append(base - mixed_after)

        # Scheduled
        s = copy.deepcopy(model).to(device)
        base2 = _one_scheduled_steps(s, task_specs, groups, batch, lr=lr)
        sched_after = _total_loss(s, task_specs, batch).item()
        deltas_sched.append(base2 - sched_after)

    dm = float(np.mean(deltas_mixed)) if deltas_mixed else 0.0
    ds = float(np.mean(deltas_sched)) if deltas_sched else 0.0
    return {
        "delta_mixed": dm,
        "delta_scheduled": ds,
        "advantage": float(ds - dm),
    }


def _prepare_data(cfg: RebuttalConfig):
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


def run_random_groups(seed: int, cfg: RebuttalConfig) -> Dict[str, Any]:
    """
    Train two runs per seed:
      - 'coloring'   : true SON-GOKU grouping
      - 'randgroups' : random groups with matched sizes
    """
    set_seed(seed)
    device = torch.device(cfg.device)
    out = {}

    # Shared data
    train_loader, val_loader = _prepare_data(cfg)

    for tag, random_ctrl in [("coloring", False), ("randgroups", True)]:
        model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        shared_filter = make_shared_filter(model)
        specs = _make_task_specs(model, cfg.tasks, cfg.seg_classes)
        sched = _build_sched(model, specs, opt, shared_filter, cfg, device,
                             random_groups_control=random_ctrl, tau_const=None)
        train_csv = _train_log_path(cfg, f"random_seed{seed}_{tag}")
        global_step = 0

        t0 = time.time()
        for epoch in range(cfg.epochs):
            for step_idx, batch in enumerate(train_loader):
                global_step += 1
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                logs = sched.step(batch)
                _maybe_log_step(logs, global_step, epoch + 1, step_idx, train_csv, cfg.log_train_every)
        wall = (time.time() - t0) / 60.0

        metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)
        logs = sched.refresh_logs()

        out[tag] = {
            "seed": seed,
            "wall_clock_min": wall,
            "val_metrics": metrics,
            "refresh_logs": logs,
        }

    return out


def run_scheduled_vs_mixed(seed: int, cfg: RebuttalConfig) -> Dict[str, Any]:
    """
    Train a single 'coloring' model but at each refresh, run micro-tests offline.
    """
    set_seed(seed)
    device = torch.device(cfg.device)

    train_loader, val_loader = _prepare_data(cfg)

    model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_task_specs(model, cfg.tasks, cfg.seg_classes)
    sched = _build_sched(model, specs, opt, shared_filter, cfg, device,
                         random_groups_control=False, tau_const=None)
    train_csv = _train_log_path(cfg, f"scheduled_vs_mixed_seed{seed}")
    global_step = 0

    # prepare a fixed small pool of probe batches from val
    probe_batches: List[Dict[str, Any]] = []
    with torch.no_grad():
        it = iter(val_loader)
        for _ in range(cfg.micro_batches):
            try:
                b = next(it)
                probe_batches.append(b)
            except StopIteration:
                break

    micro_logs: List[Dict[str, float]] = []
    last_refresh_count = 0

    t0 = time.time()
    for epoch in range(cfg.epochs):
        for step_idx, batch in enumerate(train_loader):
            global_step += 1
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            logs = sched.step(batch)
            _maybe_log_step(logs, global_step, epoch + 1, step_idx, train_csv, cfg.log_train_every)

            # detect a new refresh
            if len(sched.refresh_logs()) > last_refresh_count:
                last_refresh_count = len(sched.refresh_logs())
                groups = sched.current_groups()
                # run micro-test
                mt = _microtest_scheduled_vs_mixed(
                    model=model, task_specs=specs, groups=groups,
                    probe_batches=probe_batches, lr=cfg.lr, device=device
                )
                # include some context from refresh
                rlog = sched.last_refresh_log() or {}
                mt.update({"tau": rlog.get("tau", 0.0), "colors": rlog.get("colors", 0.0)})
                micro_logs.append(mt)

    wall = (time.time() - t0) / 60.0
    metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)

    return {
        "seed": seed,
        "wall_clock_min": wall,
        "val_metrics": metrics,
        "micro_tests": micro_logs,
        "refresh_logs": sched.refresh_logs(),
    }


def run_tau_sweep(seed: int, cfg: RebuttalConfig) -> Dict[str, Any]:
    """
    Runs constant-τ variants (given by --taus) with exact oracle & normal coloring.
    """
    results = {}
    train_loader, val_loader = _prepare_data(cfg)
    device = torch.device(cfg.device)

    for tau_val in cfg.taus:
        set_seed(seed)
        model, _ = build_model(cfg.tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        shared_filter = make_shared_filter(model)
        specs = _make_task_specs(model, cfg.tasks, cfg.seg_classes)
        sched = _build_sched(model, specs, opt, shared_filter, cfg, device,
                             random_groups_control=False, tau_const=float(tau_val))
        tau_label = str(tau_val).replace(".", "p")
        train_csv = _train_log_path(cfg, f"tau_sweep_seed{seed}_tau{tau_label}")
        global_step = 0

        t0 = time.time()
        for epoch in range(cfg.epochs):
            for step_idx, batch in enumerate(train_loader):
                global_step += 1
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                logs = sched.step(batch)
                _maybe_log_step(logs, global_step, epoch + 1, step_idx, train_csv, cfg.log_train_every)
        wall = (time.time() - t0) / 60.0
        metrics = evaluate(model, val_loader, cfg.tasks, cfg.seg_classes, device)

        results[str(tau_val)] = {
            "seed": seed,
            "wall_clock_min": wall,
            "val_metrics": metrics,
            "refresh_logs": sched.refresh_logs(),
        }

    return results


def _summarize_random_groups(all_runs: List[Dict[str, Any]], out_dir: str):
    # flatten
    rows = []
    for r in all_runs:
        seed = r["coloring"]["seed"]
        for tag in ["coloring", "randgroups"]:
            entry = r[tag]
            # aggregate val scalar
            vals = []
            for t, d in entry["val_metrics"].items():
                vals.extend(list(d.values()))
            val_mean = float(np.mean(vals)) if vals else 0.0
            rows.append({
                "seed": seed, "variant": tag,
                "val_scalar": val_mean,
                "wall_min": entry["wall_clock_min"],
                "refresh_ms_mean": float(np.mean([x["refresh_ms"] for x in entry["refresh_logs"]] or [0.0])),
                "ing_conf_active_mean": float(np.mean([x["ing_conf_active"] for x in entry["refresh_logs"]] or [0.0])),
                "colors_mean": float(np.mean([x["colors"] for x in entry["refresh_logs"]] or [0.0])),
            })
    # group by variant
    by = {}
    for row in rows:
        by.setdefault(row["variant"], []).append(row)

    def mu_sd(key, arr):
        xs = [a[key] for a in arr]
        mu = float(np.mean(xs))
        sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        return mu, sd

    summary = []
    for variant, arr in by.items():
        mu_val, sd_val = mu_sd("val_scalar", arr)
        mu_wall, sd_wall = mu_sd("wall_min", arr)
        mu_conf, sd_conf = mu_sd("ing_conf_active_mean", arr)
        mu_colors, sd_colors = mu_sd("colors_mean", arr)
        summary.append({
            "variant": variant, "seeds": len(arr),
            "val_mean": mu_val, "val_std": sd_val,
            "wall_mean": mu_wall, "wall_std": sd_wall,
            "ingroup_conflict_mean": mu_conf, "ingroup_conflict_std": sd_conf,
            "colors_mean": mu_colors, "colors_std": sd_colors,
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "random_groups_summary.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    with open(os.path.join(out_dir, "random_groups_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"[rebuttal] wrote random_groups_summary.csv")

def summarize_adaptive(all_runs: List[Dict[str, Any]], out_dir: str):
    def as_scalar(m):
        vals = []
        for t, d in m.items():
            vals.extend(list(d.values()))
        return float(np.mean(vals)) if vals else 0.0

    rows = []
    for r in all_runs:
        seed = r["scheduled"]["seed"]
        for tag in ["scheduled", "adaptive"]:
            entry = r[tag]
            rows.append({
                "seed": seed,
                "variant": tag,
                "val_scalar": as_scalar(entry["val_metrics"]),
                "wall_min": entry["wall_clock_min"],
                "colors_mean": float(np.mean([x["colors"] for x in entry["refresh_logs"]] or [0.0])),
                "avg_degree_mean": float(np.mean([x["avg_degree"] for x in entry["refresh_logs"]] or [0.0])),
                "tau_mode_frac_adaptive": float(np.mean([1.0 if x.get("tau_mode") == "adaptive" else 0.0
                                                         for x in entry["refresh_logs"]])),
            })

    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    def mu_sd(key, arr):
        xs = [a[key] for a in arr]
        mu = float(np.mean(xs))
        sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        return mu, sd

    table = []
    for variant, arr in grouped.items():
        mu_val, sd_val = mu_sd("val_scalar", arr)
        mu_wall, sd_wall = mu_sd("wall_min", arr)
        mu_colors, _ = mu_sd("colors_mean", arr)
        mu_deg, _ = mu_sd("avg_degree_mean", arr)
        table.append({
            "variant": variant, "seeds": len(arr),
            "val_mean": mu_val, "val_std": sd_val,
            "wall_mean": mu_wall, "wall_std": sd_wall,
            "colors_mean": mu_colors,
            "avg_degree_mean": mu_deg,
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "adaptive_summary.json"), "w") as f:
        json.dump({"rows": rows, "summary": table}, f, indent=2)
    with open(os.path.join(out_dir, "adaptive_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for t in table:
            w.writerow(t)
    print("[rebuttal] wrote adaptive_summary.csv")

def _summarize_scheduled_vs_mixed(all_runs: List[Dict[str, Any]], out_dir: str):
    # expand micro-tests
    flat = []
    for r in all_runs:
        seed = r["seed"]
        for mt in r["micro_tests"]:
            flat.append({"seed": seed, **mt})
    if not flat:
        print("[rebuttal] no micro-test logs found")
        return

    mu_adv = float(np.mean([x["advantage"] for x in flat]))
    sd_adv = float(np.std([x["advantage"] for x in flat], ddof=1)) if len(flat) > 1 else 0.0
    frac_win = float(np.mean([1.0 if x["advantage"] > 0 else 0.0 for x in flat]))

    summary = {
        "samples": len(flat),
        "advantage_mean": mu_adv,
        "advantage_std": sd_adv,
        "fraction_scheduled_better": frac_win,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scheduled_vs_mixed_summary.json"), "w") as f:
        json.dump({"flat": flat, "summary": summary}, f, indent=2)
    print(f"[rebuttal] wrote scheduled_vs_mixed_summary.json")


def _summarize_tau_sweep(all_runs: List[Dict[str, Any]], out_dir: str):
    # structure: [{ tau_str: {...} }, ...] per seed
    # build table tau -> [rows per seed]
    agg = {}
    for r in all_runs:
        for tau_str, entry in r.items():
            agg.setdefault(tau_str, []).append(entry)

    rows = []
    for tau_str, arr in agg.items():
        # val
        vals = []
        for e in arr:
            for t, d in e["val_metrics"].items():
                vals.extend(list(d.values()))
        mu_val = float(np.mean(vals)) if vals else 0.0
        sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        mu_colors = float(np.mean([np.mean([x["colors"] for x in e["refresh_logs"]] or [0.0]) for e in arr]))
        mu_deg = float(np.mean([np.mean([x["avg_degree"] for x in e["refresh_logs"]] or [0.0]) for e in arr]))

        rows.append({
            "tau": float(tau_str),
            "seeds": len(arr),
            "val_mean": mu_val, "val_std": sd_val,
            "colors_mean": mu_colors,
            "avg_degree_mean": mu_deg,
        })

    rows.sort(key=lambda r: r["tau"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tau_sweep_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "tau_sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[rebuttal] wrote tau_sweep_summary.csv")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("ICLR rebuttal experiments: B (random groups), C (scheduled vs mixed), A (tau-sweep)")
    # Data / model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=32)
    # Train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    # SON-GOKU core
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log", "linear", "cosine", "constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)
    ap.add_argument("--min_updates_per_cycle", type=int, default=1)
    ap.add_argument("--adaptive_tau_p", type=float, default=None,
                help="If set (e.g., 0.7 or 70), τ is the percentile of the cosine matrix each refresh.")
    ap.add_argument("--log_train_every", type=int, default=0,
                help="Log per-task losses every N steps (0 disables logging).")
    # Experiments
    ap.add_argument("--exp", type=str, required=True,
                choices=["random_groups", "scheduled_vs_mixed", "tau_sweep", "adaptive_tau"])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.4, 0.6],
                    help="Used only for tau_sweep; runs constant-tau variants.")
    ap.add_argument("--micro_batches", type=int, default=3,
                    help="Used only for scheduled_vs_mixed; number of probe batches from val.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out_dir", type=str, default="experiments/rebuttal")
    return ap.parse_args()


def main():
    args = parse_args()
    resolved_tasks = resolve_requested_tasks(args.tasks, args.data_root, args.split, args.buildings_list)
    cfg = RebuttalConfig(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=resolved_tasks, resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, device=args.device,
        refresh_period=args.refresh_period, ema_beta=args.ema_beta,
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        tau_warmup=args.tau_warmup, tau_anneal=args.tau_anneal,
        min_updates_per_cycle=args.min_updates_per_cycle,
        exp=args.exp, taus=tuple(args.taus), micro_batches=int(args.micro_batches),
        seeds=tuple(args.seeds), out_dir=args.out_dir,
        adaptive_tau_p=args.adaptive_tau_p,
        log_train_every=args.log_train_every,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_runs: List[Dict[str, Any]] = []
    if cfg.exp == "random_groups":
        for seed in cfg.seeds:
            print(f"[rebuttal] random_groups seed={seed}")
            res = run_random_groups(seed, cfg)
            # write per-seed
            with open(os.path.join(cfg.out_dir, f"random_groups_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            all_runs.append(res)
        _summarize_random_groups(all_runs, cfg.out_dir)

    elif cfg.exp == "scheduled_vs_mixed":
        for seed in cfg.seeds:
            print(f"[rebuttal] scheduled_vs_mixed seed={seed}")
            res = run_scheduled_vs_mixed(seed, cfg)
            with open(os.path.join(cfg.out_dir, f"scheduled_vs_mixed_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            all_runs.append(res)
        _summarize_scheduled_vs_mixed(all_runs, cfg.out_dir)

    elif cfg.exp == "tau_sweep":
        for seed in cfg.seeds:
            print(f"[rebuttal] tau_sweep seed={seed} taus={cfg.taus}")
            res = run_tau_sweep(seed, cfg)
            with open(os.path.join(cfg.out_dir, f"tau_sweep_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            all_runs.append(res)
        _summarize_tau_sweep(all_runs, cfg.out_dir)

    elif cfg.exp == "adaptive_tau":
        runs = []
        for seed in cfg.seeds:
            print(f"[rebuttal] adaptive_tau seed={seed} p={cfg.adaptive_tau_p}")
            res = run_adaptive_vs_default(seed, cfg)
            with open(os.path.join(cfg.out_dir, f"adaptive_tau_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            runs.append(res)
        summarize_adaptive(runs, cfg.out_dir)
        return


if __name__ == "__main__":
    main()
