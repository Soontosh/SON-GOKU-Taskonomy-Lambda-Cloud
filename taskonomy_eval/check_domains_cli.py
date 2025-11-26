#!/usr/bin/env python
"""
Utility CLI to inspect which Taskonomy domains are actually available under a
raw download root. This helps validate `--domains all` by reporting coverage
per domain (how many buildings have the domain directory and whether sample
files exist).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from prepare_taskonomy_data import (
    discover_layout,
    discover_domains,
    src_dir,
)


def _p(msg: str) -> None:
    print(msg, flush=True)


def list_files(p: Path, limit: int | None = None) -> List[Path]:
    files = [f for f in p.iterdir() if f.is_file()]
    if limit is not None and len(files) > limit:
        random.shuffle(files)
        files = files[:limit]
    return files


def check_domains(
    raw_root: Path,
    component: str,
    max_buildings: int | None,
    samples_per_building: int,
) -> Dict[str, Dict[str, float]]:
    layout, rgb_root, buildings = discover_layout(raw_root, component)
    if max_buildings:
        buildings = buildings[:max_buildings]
    _p(f"[INFO] Inspecting {len(buildings)} buildings (layout={layout}) under {rgb_root}")

    candidates = [d for d in discover_domains(raw_root) if d != "rgb"]
    if not candidates:
        _p("[WARN] No non-RGB domains discovered.")
        return {}

    rng = random.Random(0)
    results: Dict[str, Dict[str, float]] = {d: {"building_hits": 0, "file_hits": 0, "file_checks": 0} for d in candidates}

    for b in buildings:
        rgb_dir = src_dir(raw_root, layout, component, "rgb", b)
        rgb_files = list_files(rgb_dir, samples_per_building)
        if not rgb_files:
            continue
        rng.shuffle(rgb_files)
        for domain in candidates:
            d_dir = src_dir(raw_root, layout, component, domain, b)
            if not d_dir.is_dir():
                continue
            files = list_files(d_dir, samples_per_building)
            if not files:
                continue
            results[domain]["building_hits"] += 1
            # Sample RGB filenames and see if aligned files exist
            for rgb_path in rgb_files:
                d_fname = rgb_path.name.replace("domain_rgb", f"domain_{domain}") if "domain_rgb" in rgb_path.name else rgb_path.name
                cand = d_dir / d_fname
                results[domain]["file_checks"] += 1
                if cand.is_file():
                    results[domain]["file_hits"] += 1

    total_buildings = max(1, len(buildings))
    for domain, stats in results.items():
        stats["building_frac"] = stats["building_hits"] / total_buildings
        stats["file_frac"] = 0.0 if stats["file_checks"] == 0 else stats["file_hits"] / stats["file_checks"]
    return results


def main() -> None:
    ap = argparse.ArgumentParser("Inspect Taskonomy domain coverage under a raw download root.")
    ap.add_argument("--download-root", type=Path, required=True, help="Path that contains the raw Omnidata download.")
    ap.add_argument("--component", type=str, default="taskonomy", help="Component used during download (default: taskonomy).")
    ap.add_argument("--max-buildings", type=int, default=None, help="Optional limit on number of buildings to inspect.")
    ap.add_argument("--samples-per-building", type=int, default=25, help="How many RGB views to sample per building for file checks.")
    ap.add_argument("--min-building-frac", type=float, default=0.8, help="Threshold for recommending a domain (fraction of buildings with data).")
    ap.add_argument("--min-file-frac", type=float, default=0.8, help="Threshold for recommending a domain based on sampled file availability.")
    args = ap.parse_args()

    raw_root = args.download_root.expanduser().resolve()
    if not raw_root.exists():
        ap.error(f"{raw_root} does not exist.")

    results = check_domains(raw_root, args.component, args.max_buildings, args.samples_per_building)
    if not results:
        _p("[INFO] No domains to report.")
        return

    _p("\n[REPORT] Domain coverage")
    for domain, stats in sorted(results.items(), key=lambda kv: kv[1]["building_frac"], reverse=True):
        _p(
            f"  {domain:20s} buildings: {stats['building_hits']:4.0f} "
            f"({stats['building_frac']*100:5.1f}%)  "
            f"sample hits: {stats['file_hits']:5.0f}/{stats['file_checks']:5.0f} "
            f"({stats['file_frac']*100:5.1f}%)"
        )

    recommended = [
        d for d, stats in results.items()
        if (stats["building_frac"] >= args.min_building_frac and stats["file_frac"] >= args.min_file_frac)
    ]
    _p("\n[REPORT] Recommended domains for `--domains all`:")
    if recommended:
        _p("  " + ", ".join(["rgb"] + recommended))
    else:
        _p("  (no domains met the thresholds)")


if __name__ == "__main__":
    main()
