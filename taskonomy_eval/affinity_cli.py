# taskonomy_eval/affinity_cli.py
from __future__ import annotations
import argparse, json, os, time, csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Reuse your helpers
from taskonomy_eval.runner import (
    set_seed, build_model, make_shared_filter, make_head_filter,
    build_task_loss, evaluate, resolve_requested_tasks,
)
from taskonomy_eval.datasets.taskonomy import TaskonomyDataset, TaskonomyConfig

from son_goku import TaskSpec, TauSchedule
from son_goku.affinity.affinity import (
    GradCosineAffinity, TAGExactAffinity, TAGLinearizedAffinity, cosine_vs_tag_stats
)
from son_goku.affinity.scheduler_affinity import AffinityScheduler


@dataclass
class AffinityCfg:
    # data/model
    data_root: str
    split: str
    val_split: str
    tasks: Tuple[str, ...]
    resize: Tuple[int, int]
    buildings_list: str | None
    seg_classes: int
    base_channels: int
    # train
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    device: str
    refresh_period: int
    ema_beta: float  # not used here, included for completeness
    # tau schedule (cosine)
    tau_kind: str
    tau_initial: float
    tau_target: float
    tau_warmup: int
    tau_anneal: int
    # E1–E4 knobs
    exp: str  # e1|e2|e3|e4
    tag_eta_v: float
    tag_gamma: float
    tag_pair_fraction: float
    taus: Tuple[float, ...]  # used in e4 (constant tau sweep for cosine baseline)
    seeds: Tuple[int, ...]
    out_dir: str


def _make_specs(model: nn.Module, tasks: Sequence[str], seg_classes: int):
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

def _resolved_tasks(cfg: AffinityCfg) -> Tuple[str, ...]:
    return resolve_requested_tasks(cfg.tasks, cfg.data_root, cfg.split, cfg.buildings_list)


def _loaders(cfg: AffinityCfg, tasks: Sequence[str]):
    train = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.split, buildings_list=cfg.buildings_list,
        tasks=tuple(tasks), resize=cfg.resize
    ))
    val = TaskonomyDataset(TaskonomyConfig(
        root=cfg.data_root, split=cfg.val_split, buildings_list=cfg.buildings_list,
        tasks=tuple(tasks), resize=cfg.resize
    ))
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


# --------------------------------------------------------------------------------------
# E1: Correlation & sign agreement between cosine and TAG lookahead on periodic probes
# --------------------------------------------------------------------------------------
def run_e1(seed: int, cfg: AffinityCfg) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    tasks = _resolved_tasks(cfg)
    train_loader, _ = _loaders(cfg, tasks)

    model, _ = build_model(tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, tasks, cfg.seg_classes)

    cos_aff = GradCosineAffinity(shared_filter)
    tag_aff = TAGExactAffinity(shared_filter, symmetrize=True)

    logs = []
    # Probe first N steps of first few epochs (lightweight)
    max_probes = 10
    step = 0
    for batch in train_loader:
        step += 1
        if step > max_probes:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        Cres = cos_aff.compute(model, specs, batch, device)
        Dres = tag_aff.compute(model, specs, batch, device,
                               eta_v=cfg.tag_eta_v, pair_fraction=cfg.tag_pair_fraction)
        stats = cosine_vs_tag_stats(Cres.S, -Dres.S)
        logs.append({
            "step": step,
            "spearman": stats["spearman"],
            "sign_agree": stats["sign_agree"],
            "tag_eta_v": cfg.tag_eta_v,
            "pair_frac": cfg.tag_pair_fraction,
        })

    return {"seed": seed, "logs": logs}


def summarize_e1(all_runs: List[Dict[str, Any]], out_dir: str):
    flat = []
    for r in all_runs:
        seed = r["seed"]
        for x in r["logs"]:
            flat.append({"seed": seed, **x})
    if not flat:
        return
    mu_spear = float(np.mean([x["spearman"] for x in flat]))
    sd_spear = float(np.std([x["spearman"] for x in flat], ddof=1)) if len(flat) > 1 else 0.0
    mu_sign = float(np.mean([x["sign_agree"] for x in flat]))
    sd_sign = float(np.std([x["sign_agree"] for x in flat], ddof=1)) if len(flat) > 1 else 0.0
    summ = {"samples": len(flat), "spearman_mean": mu_spear, "spearman_std": sd_spear,
            "sign_agree_mean": mu_sign, "sign_agree_std": sd_sign}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e1_correlation_summary.json"), "w") as f:
        json.dump({"flat": flat, "summary": summ}, f, indent=2)
    print("[affinity] wrote e1_correlation_summary.json")


# --------------------------------------------------------------------------------------
# E2/E3/E4: Train with an affinity (cosine or TAG variants) and compare schedules/metrics
# --------------------------------------------------------------------------------------
def _train_with_affinity(
    cfg: AffinityCfg,
    seed: int,
    mode: str,           # "cosine" | "tag_exact" | "tag_linearized"
    tau_const: float | None = None
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(cfg.device)
    tasks = _resolved_tasks(cfg)
    train_loader, val_loader = _loaders(cfg, tasks)

    model, _ = build_model(tasks, seg_classes=cfg.seg_classes, base=cfg.base_channels)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    shared_filter = make_shared_filter(model)
    specs = _make_specs(model, tasks, cfg.seg_classes)

    if mode == "cosine":
        affinity = GradCosineAffinity(shared_filter)
        tau = TauSchedule(
            kind="constant" if tau_const is not None else cfg.tau_kind,
            tau_initial=tau_const if tau_const is not None else cfg.tau_initial,
            tau_target=tau_const if tau_const is not None else cfg.tau_target,
            warmup_steps=cfg.tau_warmup, anneal_duration=cfg.tau_anneal
        )
        sched = AffinityScheduler(
            model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
            tau_schedule=tau, affinity=affinity, mode="cosine", threshold=tau.tau_initial,
            refresh_period=cfg.refresh_period, device=device, log_compare_cosine=False
        )
    elif mode == "tag_exact":
        affinity = TAGExactAffinity(shared_filter, symmetrize=True)
        tau = TauSchedule(kind="constant", tau_initial=1.0, tau_target=1.0, warmup_steps=0, anneal_duration=0)
        sched = AffinityScheduler(
            model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
            tau_schedule=tau, affinity=affinity, mode="tag", threshold=cfg.tag_gamma,
            refresh_period=cfg.refresh_period, device=device,
            tag_eta_v=cfg.tag_eta_v, pair_fraction=cfg.tag_pair_fraction, log_compare_cosine=True
        )
    elif mode == "tag_linearized":
        affinity = TAGLinearizedAffinity(shared_filter, cosine_normalize=False)
        tau = TauSchedule(kind="constant", tau_initial=1.0, tau_target=1.0, warmup_steps=0, anneal_duration=0)
        sched = AffinityScheduler(
            model=model, tasks=specs, optimizer=opt, shared_param_filter=shared_filter,
            tau_schedule=tau, affinity=affinity, mode="tag", threshold=cfg.tag_gamma,
            refresh_period=cfg.refresh_period, device=device,
            tag_eta_v=cfg.tag_eta_v, pair_fraction=1.0, log_compare_cosine=True
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    t0 = time.time()
    for epoch in range(cfg.epochs):
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            _ = sched.step(batch)
    wall = (time.time() - t0)/60.0
    metrics = evaluate(model, val_loader, tasks, cfg.seg_classes, device)
    return {
        "seed": seed, "mode": mode, "tau_const": tau_const,
        "wall_clock_min": wall, "val_metrics": metrics,
        "refresh_logs": sched.refresh_logs(),
    }


def summarize_train(rows: List[Dict[str, Any]], out_csv: str):
    # flatten validation to a scalar mean
    def val_scalar(m):
        vals = []
        for t, d in m.items():
            vals.extend(list(d.values()))
        return float(np.mean(vals)) if vals else 0.0

    # group by mode (and tau_const if present)
    groups = {}
    for r in rows:
        key = (r["mode"], r.get("tau_const", None))
        groups.setdefault(key, []).append(r)

    table = []
    for key, arr in groups.items():
        mode, tau_c = key
        mu_val = float(np.mean([val_scalar(a["val_metrics"]) for a in arr]))
        sd_val = float(np.std([val_scalar(a["val_metrics"]) for a in arr], ddof=1)) if len(arr) > 1 else 0.0
        mu_wall = float(np.mean([a["wall_clock_min"] for a in arr]))
        mu_colors = float(np.mean([np.mean([x["colors"] for x in a["refresh_logs"]] or [0.0]) for a in arr]))
        mu_deg = float(np.mean([np.mean([x["avg_degree"] for x in a["refresh_logs"]] or [0.0]) for a in arr]))
        # diagnostics: when comparing tag vs cosine, cmp_* may be present
        cmp_spear = []
        for a in arr:
            cmp_vals = [x.get("cmp_spearman", None) for x in a["refresh_logs"] if "cmp_spearman" in x]
            if cmp_vals:
                cmp_spear.append(float(np.mean(cmp_vals)))
        row = {
            "mode": mode, "tau_const": tau_c, "seeds": len(arr),
            "val_mean": mu_val, "val_std": sd_val,
            "wall_min_mean": mu_wall,
            "colors_mean": mu_colors, "avg_degree_mean": mu_deg,
            "cmp_spearman_mean": float(np.mean(cmp_spear)) if cmp_spear else None,
        }
        table.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for r in table:
            w.writerow(r)
    print(f"[affinity] wrote {out_csv}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Affinity experiments E1–E4 (TAG vs gradient cosine)")
    # data/model
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--tasks", type=str, nargs="+", default=["depth_euclidean", "normal", "reshading"])
    ap.add_argument("--resize", type=int, nargs=2, default=[256,256])
    ap.add_argument("--buildings_list", type=str, default=None)
    ap.add_argument("--seg_classes", type=int, default=40)
    ap.add_argument("--base_channels", type=int, default=32)
    # train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--refresh_period", type=int, default=32)
    ap.add_argument("--ema_beta", type=float, default=0.9)
    # tau schedule (cosine)
    ap.add_argument("--tau_kind", type=str, default="log", choices=["log","linear","cosine","constant"])
    ap.add_argument("--tau_initial", type=float, default=1.0)
    ap.add_argument("--tau_target", type=float, default=0.25)
    ap.add_argument("--tau_warmup", type=int, default=0)
    ap.add_argument("--tau_anneal", type=int, default=0)

    # experiments
    ap.add_argument("--exp", type=str, required=True, choices=["e1","e2","e3","e4"])
    ap.add_argument("--tag_eta_v", type=float, default=1e-3, help="lookahead step size for TAG")
    ap.add_argument("--tag_gamma", type=float, default=0.0, help="conflict margin for TAG (S_ij > gamma)")
    ap.add_argument("--tag_pair_fraction", type=float, default=0.4, help="fraction of pairs for TAG exact (e1/e2/e3)")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.4, 0.6], help="used in e4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--out_dir", type=str, default="experiments/affinity")
    return ap.parse_args()


def main():
    args = parse_args()
    resolved_tasks = resolve_requested_tasks(args.tasks, args.data_root, args.split, args.buildings_list)
    cfg = AffinityCfg(
        data_root=args.data_root, split=args.split, val_split=args.val_split,
        tasks=resolved_tasks, resize=tuple(args.resize), buildings_list=args.buildings_list,
        seg_classes=args.seg_classes, base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
        device=args.device, refresh_period=args.refresh_period, ema_beta=args.ema_beta,
        tau_kind=args.tau_kind, tau_initial=args.tau_initial, tau_target=args.tau_target,
        tau_warmup=args.tau_warmup, tau_anneal=args.tau_anneal,
        exp=args.exp, tag_eta_v=args.tag_eta_v, tag_gamma=args.tag_gamma,
        tag_pair_fraction=args.tag_pair_fraction, taus=tuple(args.taus),
        seeds=tuple(args.seeds), out_dir=args.out_dir
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # E1: correlation/sign agreement (no training loop changes)
    if cfg.exp == "e1":
        all_runs = []
        for s in cfg.seeds:
            print(f"[affinity] E1 seed={s}")
            res = run_e1(s, cfg)
            with open(os.path.join(cfg.out_dir, f"e1_seed{s}.json"), "w") as f:
                json.dump(res, f, indent=2)
            all_runs.append(res)
        summarize_e1(all_runs, cfg.out_dir)
        return

    # E2: schedule equivalence: train with cosine baseline AND TAG exact
    if cfg.exp == "e2":
        rows = []
        for s in cfg.seeds:
            print(f"[affinity] E2 cosine seed={s}")
            rows.append(_train_with_affinity(cfg, s, mode="cosine", tau_const=None))
            print(f"[affinity] E2 TAG-exact seed={s}")
            rows.append(_train_with_affinity(cfg, s, mode="tag_exact"))
        with open(os.path.join(cfg.out_dir, "e2_rows.json"), "w") as f:
            json.dump(rows, f, indent=2)
        summarize_train(rows, os.path.join(cfg.out_dir, "e2_summary.csv"))
        return

    # E3: budgeted TAG: TAG-exact vs TAG-linearized (and optional cosine if you want)
    if cfg.exp == "e3":
        rows = []
        for s in cfg.seeds:
            print(f"[affinity] E3 TAG-exact seed={s}")
            rows.append(_train_with_affinity(cfg, s, mode="tag_exact"))
            print(f"[affinity] E3 TAG-linearized seed={s}")
            rows.append(_train_with_affinity(cfg, s, mode="tag_linearized"))
        with open(os.path.join(cfg.out_dir, "e3_rows.json"), "w") as f:
            json.dump(rows, f, indent=2)
        summarize_train(rows, os.path.join(cfg.out_dir, "e3_summary.csv"))
        return

    # E4: eta sweep: for each eta_v, run TAG-exact(+compare); also run cosine with constant tau grid if desired
    if cfg.exp == "e4":
        rows = []
        for s in cfg.seeds:
            for eta in [v for v in [cfg.tag_eta_v/10, cfg.tag_eta_v, cfg.tag_eta_v*5] if v > 0]:
                print(f"[affinity] E4 TAG-exact eta={eta} seed={s}")
                cfg2 = cfg
                cfg2 = AffinityCfg(**{**asdict(cfg), "tag_eta_v": eta})
                rows.append(_train_with_affinity(cfg2, s, mode="tag_exact"))
            # Optional cosine constant tau sweep (comment out if not needed)
            for tau_c in cfg.taus:
                print(f"[affinity] E4 cosine tau={tau_c} seed={s}")
                rows.append(_train_with_affinity(cfg, s, mode="cosine", tau_const=float(tau_c)))
        with open(os.path.join(cfg.out_dir, "e4_rows.json"), "w") as f:
            json.dump(rows, f, indent=2)
        summarize_train(rows, os.path.join(cfg.out_dir, "e4_summary.csv"))
        return


if __name__ == "__main__":
    main()
