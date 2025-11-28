#!/usr/bin/env python
"""
Fast Taskonomy reshaper with auto domain detection.

This utility quickly scans a subset of buildings to determine which
domains (tasks) are actually present in a raw Omnidata/Taskonomy download.
It then reshapes the data into Taskonomy layout, creating symlinks only
for the detected domains.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Sequence

from prepare_taskonomy_data import (
    discover_layout,
    make_splits,
    src_dir,
    KNOWN_DOMAIN_NAMES,
)


def _p(msg: str) -> None:
    print(msg, flush=True)


def list_buildings(download_root: Path, component: str, max_buildings: int | None = None) -> tuple[str, Path, List[str]]:
    layout, rgb_root, buildings = discover_layout(download_root, component)
    if max_buildings:
        buildings = buildings[:max_buildings]
    return layout, rgb_root, buildings


def candidate_domains(download_root: Path, component: str, layout: str) -> List[str]:
    if layout == "domain_first":
        base_dirs = [d.name for d in download_root.iterdir() if d.is_dir()]
    else:
        comp_dir = download_root / component
        if not comp_dir.is_dir():
            return []
        base_dirs = [d.name for d in comp_dir.iterdir() if d.is_dir()]
    names = []
    for d in base_dirs:
        if d.lower() in KNOWN_DOMAIN_NAMES:
            names.append(d)
    return sorted(set(names))


def detect_domains(
    download_root: Path,
    component: str,
    sample_buildings: int,
    min_hits: int,
    require_all: bool = False,
) -> List[str]:
    layout, rgb_root, buildings = discover_layout(download_root, component)
    if not buildings:
        raise RuntimeError("No buildings discovered.")
    sample = buildings if len(buildings) <= sample_buildings else buildings[:sample_buildings]
    _p(f"[INFO] Auto-detecting domains using {len(sample)} buildings (layout={layout}, rgb root={rgb_root})")

    candidates = candidate_domains(download_root, component, layout)
    if "rgb" not in candidates:
        candidates.insert(0, "rgb")
    hits = {}
    for domain in candidates:
        if domain == "rgb":
            continue
        cnt = 0
        for b in sample:
            d_dir = src_dir(download_root, layout, component, domain, b)
            if d_dir.is_dir():
                try:
                    next(d_dir.iterdir())
                    cnt += 1
                    if cnt >= min_hits:
                        break
                except StopIteration:
                    continue
        hits[domain] = cnt

    selected = ["rgb"]
    for domain, count in sorted(hits.items(), key=lambda kv: kv[1], reverse=True):
        _p(f"[INFO] Domain '{domain}' present in {count}/{len(sample)} sampled buildings.")
        threshold = len(sample) if require_all else min_hits
        if count >= threshold:
            selected.append(domain)

    if len(selected) == 1:
        raise RuntimeError("No domains satisfied the detection threshold. Use --domains to override.")

    _p(f"[INFO] Selected domains: {', '.join(selected)}")
    return selected


def reshape(
    download_root: Path,
    reshape_root: Path,
    component: str,
    domains: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    force: bool,
    no_split: bool,
    exclude_partial: bool,
) -> None:
    if reshape_root.exists():
        if force:
            _p(f"[WARN] Removing existing reshape root: {reshape_root}")
            shutil.rmtree(reshape_root)
        else:
            raise RuntimeError(f"{reshape_root} already exists. Use --force.")

    layout, rgb_root, buildings = list_buildings(download_root, component)
    _p(f"[INFO] Reshaping {len(buildings)} buildings from {rgb_root}")
    splits = {b: "train"} if no_split else make_splits(buildings, train_frac, val_frac, test_frac, seed)
    reshape_root.mkdir(parents=True, exist_ok=True)

    # Persist domain list
    with open(reshape_root / "domains_used.txt", "w") as f:
        for d in domains:
            f.write(d + "\n")

    total_created = 0
    total_skipped = 0
    begin = time.time()
    for idx, building in enumerate(buildings, start=1):
        split = splits[building]
        rgb_dir = src_dir(download_root, layout, component, "rgb", building)
        if not rgb_dir.is_dir():
            _p(f"[WARN] Missing RGB dir for {building}, skipping building.")
            continue
        rgb_files = sorted(p for p in rgb_dir.iterdir() if p.is_file())
        if not rgb_files:
            continue

        created = 0
        skipped = 0
        partial = 0
        for rgb_path in rgb_files:
            fname = rgb_path.name
            sample_paths = {"rgb": rgb_path}
            missing = []
            for domain in domains:
                if domain == "rgb":
                    continue
                d_dir = src_dir(download_root, layout, component, domain, building)
                if not d_dir.is_dir():
                    missing.append(domain)
                    continue
                d_name = fname.replace("domain_rgb", f"domain_{domain}") if "domain_rgb" in fname else fname
                cand = d_dir / d_name
                if not cand.is_file():
                    missing.append(domain)
                    continue
                sample_paths[domain] = cand

            if missing:
                if exclude_partial:
                    skipped += 1
                    total_skipped += 1
                    continue
                partial += 1
            for domain, src in sample_paths.items():
                dest_dir = reshape_root / split / building / domain
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src.name
                if not dest.exists():
                    dest.symlink_to(src)
            created += 1
            total_created += 1

        _p(f"[INFO] [{idx}/{len(buildings)}] {building}: created {created} (partial {partial}), skipped {skipped}")

    _p(f"[INFO] Reshape complete in {(time.time() - begin)/60.0:.2f} min. Samples={total_created}, strict skips={total_skipped}")


def main() -> None:
    ap = argparse.ArgumentParser("Fast Taskonomy domain detector + reshaper.")
    ap.add_argument("--raw-root", type=Path, required=True, help="Path to the raw Taskonomy/Omnidata download.")
    ap.add_argument("--reshape-root", type=Path, required=True, help="Destination root for reshaped layout.")
    ap.add_argument("--component", type=str, default="taskonomy", help="Dataset component name (default: taskonomy).")
    ap.add_argument("--auto-domains", action="store_true", help="Automatically detect domains instead of specifying --domains.")
    ap.add_argument("--domains-everywhere", action="store_true", help="When auto-detecting, only keep domains that appear in every sampled building.")
    ap.add_argument("--domains", type=str, nargs="+", default=["rgb", "depth_euclidean", "normal", "reshading"], help="Domains to reshape when not using --auto-domains.")
    ap.add_argument("--sample-buildings", type=int, default=25, help="Number of buildings to scan when auto-detecting domains.")
    ap.add_argument("--min-domain-hits", type=int, default=3, help="Minimum sample-building hits required to keep a domain.")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite existing reshape root.")
    ap.add_argument("--exclude-partial-samples", action="store_true", help="Drop RGB views that are missing any selected domain.")
    args = ap.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    reshape_root = args.reshape_root.expanduser().resolve()
    if not raw_root.exists():
        ap.error(f"{raw_root} does not exist.")

    if args.auto_domains:
        try:
            if args.domains_everywhere:
                _p("[INFO] Restricting to domains present in every sampled building (--domains-everywhere).")
            domains = detect_domains(
                raw_root,
                args.component,
                args.sample_buildings,
                args.min_domain_hits,
                require_all=args.domains_everywhere,
            )
        except RuntimeError as exc:
            _p(f"[ERROR] {exc}")
            sys.exit(2)
    else:
        domains = list(dict.fromkeys(args.domains))
        if "rgb" not in domains:
            domains.insert(0, "rgb")
        _p(f"[INFO] Using provided domains: {', '.join(domains)}")

    reshape(
        raw_root,
        reshape_root,
        args.component,
        domains,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.force,
        args.no_split,
        args.exclude_partial_samples,
    )


if __name__ == "__main__":
    main()
