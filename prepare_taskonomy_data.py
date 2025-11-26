#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path
import random
import shutil
import os
import time
from typing import List, Set, Optional


# ---- Domain catalog (used for discovery) ------------------------------------
KNOWN_DOMAIN_NAMES: Set[str] = {
    "rgb", "albedo", "depth_euclidean", "depth_zbuffer",
    "edge_occlusion", "edge_texture",
    "keypoints2d", "keypoints3d",
    "mask_valid", "mist",
    "normal", "reshading",
    "principal_curvature", "point_info",
    "segment_semantic", "segment_instance",
    "segment_unsup2d", "segment_unsup25d",
    "room_layout", "vanishing_point",
    "surface_orientation", "curvature", "depth", "shading",
}


# ---- Small helpers ----------------------------------------------------------
def _p(s: str) -> None:
    print(s, flush=True)


def run_cmd(cmd: List[str], log_file: Optional[Path], max_retries: int, retry_wait: int) -> None:
    """Run a CLI with retry + optional logging. Raises on final failure."""
    attempt = 0
    while True:
        attempt += 1
        _p(f"[INFO] Running (attempt {attempt}/{max_retries + 1}): {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "a") as f:
                    if proc.stdout:
                        f.write(proc.stdout)
                    if proc.stderr:
                        f.write(proc.stderr)
            if proc.stdout:
                _p(proc.stdout)
            if proc.stderr:
                _p(proc.stderr)
            return  # success
        except FileNotFoundError:
            _p("[ERROR] 'omnitools.download' not found. Did you `pip install omnidata-tools`?")
            raise
        except subprocess.CalledProcessError as exc:
            # Append logs so you can inspect failures later
            if log_file:
                with open(log_file, "a") as f:
                    if exc.stdout:
                        f.write(exc.stdout)
                    if exc.stderr:
                        f.write(exc.stderr)
            _p(f"[WARN] Command failed with code {exc.returncode}.")
            if attempt > max_retries:
                _p("[ERROR] Exhausted retries, giving up.")
                raise
            _p(f"[INFO] Sleeping {retry_wait}s then retrying …")
            time.sleep(retry_wait)


def discover_layout(download_root: Path, component: str):
    """
    Detect whether omnitools created:
        <dest>/<domain>/<component>/<building>  OR
        <dest>/<component>/<domain>/<building>
    Return (layout_type, rgb_root, building_names).
    """
    rgb_domain_first = download_root / "rgb" / component
    rgb_component_first = download_root / component / "rgb"

    if rgb_domain_first.is_dir():
        layout, rgb_root = "domain_first", rgb_domain_first
    elif rgb_component_first.is_dir():
        layout, rgb_root = "component_first", rgb_component_first
    else:
        raise RuntimeError(
            f"Could not find RGB root under {download_root} "
            f"(looked for rgb/{component} and {component}/rgb)."
        )

    buildings = sorted(d.name for d in rgb_root.iterdir() if d.is_dir())
    if not buildings:
        raise RuntimeError(f"No building directories found under {rgb_root}")

    return layout, rgb_root, buildings


def src_dir(download_root: Path, layout: str, component: str, domain: str, building: str) -> Path:
    """Return source directory for given domain + building."""
    return (download_root / domain / component / building) if layout == "domain_first" \
           else (download_root / component / domain / building)


def make_splits(buildings, train_frac, val_frac, test_frac, seed):
    """Randomly assign buildings to train/val/test."""
    if train_frac + val_frac + test_frac > 1.0 + 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must be <= 1.0")

    rng = random.Random(seed)
    buildings = list(buildings)
    rng.shuffle(buildings)

    n = len(buildings)
    n_tr = int(n * train_frac)
    n_v  = int(n * val_frac)
    n_te = int(n * test_frac)

    if (n_tr + n_v + n_te) < n:
        n_te += n - (n_tr + n_v + n_te)

    splits = {}
    for i, b in enumerate(buildings):
        if i < n_tr:
            splits[b] = "train"
        elif i < n_tr + n_v:
            splits[b] = "val"
        else:
            splits[b] = "test"
    return splits


def discover_domains(raw_root: Path) -> List[str]:
    """
    Walk the raw download tree and discover which domain directories exist.
    Handles both <dest>/<domain>/<component>/<building> and
    <dest>/<component>/<domain>/<building> layouts.
    """
    found: Set[str] = set()
    for root, dirs, files in os.walk(raw_root):
        base = os.path.basename(root)
        if base in KNOWN_DOMAIN_NAMES and (dirs or any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files)):
            found.add(base)

    domains = sorted(d for d in found if d != "rgb")
    return (["rgb"] + domains) if "rgb" in found else domains


def report_progress(raw_root: Path, component: str) -> None:
    """Print a per-domain progress report vs RGB count."""
    try:
        _, rgb_root, _ = discover_layout(raw_root, component)
    except Exception as e:
        _p(f"[INFO] Layout not ready yet: {e}")
        return

    rgb_n = sum(1 for _ in rgb_root.rglob("*") if _.is_file())
    _p(f"[REPORT] RGB files: {rgb_n}")

    domains = discover_domains(raw_root)
    _p(f"[REPORT] Domains seen: {', '.join(domains)}")

    def count_domain(d):
        cand = [raw_root / d / component, raw_root / component / d]
        droot = next((p for p in cand if p.is_dir()), None)
        return 0 if not droot else sum(1 for _ in droot.rglob("*") if _.is_file())

    for d in domains:
        if d == "rgb":
            continue
        n = count_domain(d)
        pct = 0.0 if rgb_n == 0 else 100.0 * (n / rgb_n)
        _p(f"[REPORT] {d:20s} {n:9d} files  ~ {pct:5.1f}% of RGB")


# ---- Download & reshape -----------------------------------------------------
def run_download(args) -> Path:
    """
    Call `omnitools.download` to fetch Taskonomy data (resumable + retry).
    Returns the *actual* raw destination directory used so the reshape step
    can point to the right place (especially for --domains all).
    """
    if args.skip_download:
        _p("[INFO] Skipping download step.")
        return Path(args.download_root)

    use_all = (len(args.domains) == 1 and args.domains[0].lower() == "all")
    raw_dest = (Path(args.download_root) / f"{args.component}_{args.subset}_all") if use_all else Path(args.download_root)
    raw_dest.parent.mkdir(parents=True, exist_ok=True)

    if not args.name or not args.email:
        _p("ERROR: --name and --email are required unless --skip-download.")
        sys.exit(1)

    cmd = ["omnitools.download"]
    if use_all:
        if not args.agree_all:
            _p("ERROR: Using --domains all requires --agree_all.")
            sys.exit(2)
        cmd += ["all"]
    else:
        cmd += args.domains

    cmd += [
        "--components", args.component,
        "--subset", args.subset,            # debug/tiny/medium/full/fullplus
        "--split", args.download_split,     # train/val/test/all
        "--dest", str(raw_dest),
        "--connections_total", str(args.connections_total),
        "--name", args.name,                # plain text; don't URL-encode
        "--email", args.email,
    ]
    if args.agree_all:
        cmd.append("--agree_all")

    run_cmd(cmd, args.log_file, args.max_retries, args.retry_wait)
    return raw_dest.resolve()


def reshape(args) -> None:
    """
    Build Taskonomy-style layout:
        reshape_root/split/building/domain/*.png
    using RGB filenames as anchors and enforcing that all requested domains
    exist for each sample (otherwise that view is skipped).
    """
    download_root = Path(args.download_root)
    reshape_root = Path(args.reshape_root)

    if reshape_root.exists():
        if args.force:
            _p(f"[INFO] Removing existing reshape_root: {reshape_root}")
            shutil.rmtree(reshape_root)
        else:
            raise RuntimeError(f"{reshape_root} already exists. Use --force to overwrite.")

    t0 = time.time()
    layout, rgb_root, buildings = discover_layout(download_root, args.component)
    _p(f"[INFO] Found {len(buildings)} buildings under {rgb_root}")

    splits = {b: "train"} if args.no_split else make_splits(buildings, args.train_frac, args.val_frac, args.test_frac, args.seed)
    reshape_root.mkdir(parents=True, exist_ok=True)
    if args.no_split:
        _p("[INFO] No split requested; all buildings go to 'train'.")
    else:
        counts = {"train": 0, "val": 0, "test": 0}
        for v in splits.values():
            counts[v] += 1
        _p(f"[INFO] Building split counts -> train:{counts['train']}  val:{counts['val']}  test:{counts['test']}")

    # Determine domains to reshape
    use_all = (len(args.domains) == 1 and args.domains[0].lower() == "all")
    if use_all:
        domains = discover_domains(download_root)
        if not domains:
            _p(f"[ERROR] Could not discover any domains under {download_root}")
            sys.exit(2)
        _p(f"[INFO] Discovered domains: {', '.join(domains)}")
    else:
        # ensure rgb is present and domains are unique, keep order
        domains = list(dict.fromkeys(["rgb"] + args.domains))
        _p(f"[INFO] Domains: {domains}")

    # Save final list for reproducibility
    with open(reshape_root / "domains_used.txt", "w") as f:
        for d in domains:
            f.write(d + "\n")

    total_samples = 0
    skipped_samples = 0

    for idx_b, b in enumerate(buildings, start=1):
        split = splits[b]
        _p(f"[INFO] [{idx_b}/{len(buildings)}] Processing building '{b}' (split={split})")
        rgb_dir = src_dir(download_root, layout, args.component, "rgb", b)
        if not rgb_dir.is_dir():
            _p(f"[WARN] No rgb dir for building {b} at {rgb_dir}, skipping building.")
            continue

        rgb_files = sorted(p for p in rgb_dir.iterdir() if p.is_file())
        if not rgb_files:
            _p(f"[WARN] No rgb files for building {b}, skipping building.")
            continue

        created_here = 0
        skipped_here = 0
        checkpoint_every = max(1, len(rgb_files) // 40)

        for view_idx, rgb_path in enumerate(rgb_files, start=1):
            fname = rgb_path.name

            # candidate src paths for all domains
            src_paths = {"rgb": rgb_path}
            ok = True

            for d in domains:
                if d == "rgb":
                    continue

                d_dir = src_dir(download_root, layout, args.component, d, b)
                if not d_dir.is_dir():
                    ok = False
                    break

                # Omnidata naming: ...domain_rgb.png -> ...domain_<domain>.png
                d_fname = fname.replace("domain_rgb", f"domain_{d}") if "domain_rgb" in fname else fname
                cand = d_dir / d_fname
                if not cand.is_file():
                    ok = False
                    break

                src_paths[d] = cand

            if not ok:
                skipped_samples += 1
                skipped_here += 1
                continue

            # Create symlinks for this multi-task sample
            for d, src in src_paths.items():
                dest_dir = reshape_root / split / b / d
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src.name
                if not dest.exists():
                    dest.symlink_to(src)

            total_samples += 1
            created_here += 1

            if (view_idx % checkpoint_every) == 0 or view_idx == len(rgb_files):
                pct = 100.0 * (view_idx / max(1, len(rgb_files)))
                _p(
                    f"[INFO] Building {b}: {view_idx}/{len(rgb_files)} RGB views processed "
                    f"({pct:4.1f}%), created={created_here}, skipped={skipped_here}"
                )

        _p(f"[INFO] Finished building '{b}': created {created_here}, skipped {skipped_here}")

    _p("[INFO] Finished reshaping.")
    _p(f"[INFO] Total multi-task samples created: {total_samples}")
    if skipped_samples:
        _p(f"[INFO] Skipped {skipped_samples} rgb views with missing targets.")
    _p(f"[INFO] Reshape wall-clock: {(time.time() - t0)/60.0:.2f} min")


# ---- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Download and reshape Omnidata/Taskonomy data for SON-GOKU Taskonomy training."
    )
    ap.add_argument("--download-root", type=str, required=True,
                    help="Directory where raw data will be stored. For --domains all, a subfolder "
                         "<component>_<subset>_all is created inside this directory.")
    ap.add_argument("--reshape-root", type=str, required=True,
                    help="Output directory for Taskonomy-style layout (root/split/building/domain/).")
    ap.add_argument("--subset", type=str, default="debug",
                    choices=["debug", "tiny", "medium", "full", "fullplus"],
                    help="Subset to download (omnitools --subset).")
    ap.add_argument("--download-split", type=str, default="all",
                    choices=["train", "val", "test", "all"],
                    help="Split to download from Omnidata (omnitools --split).")
    ap.add_argument("--component", type=str, default="taskonomy",
                    help="Component dataset (omnitools --components). Usually 'taskonomy'.")
    ap.add_argument("--domains", type=str, nargs="+",
                    default=["rgb", "depth_euclidean", "normal", "reshading"],
                    help="Domains to download & reshape. Use 'all' to include *all* domains available for the subset. "
                         "RGB is always included for reshaping.")
    ap.add_argument("--agree_all", action="store_true",
                    help="Automatically accept the Omnidata EULA for omnitools (required for --domains all).")
    ap.add_argument("--connections-total", type=int, default=16,
                    help="omnitools --connections_total.")
    ap.add_argument("--name", type=str, required=False,
                    help="Your name for omnitools license agreement (required unless --skip-download).")
    ap.add_argument("--email", type=str, required=False,
                    help="Your email for omnitools license agreement (required unless --skip-download).")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip download step and only reshape existing data.")
    ap.add_argument("--no-split", action="store_true",
                    help="Do not create train/val/test splits; put everything into 'train'.")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of buildings in train split.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of buildings in val split.")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="Fraction of buildings in test split.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for building splits.")
    ap.add_argument("--force", action="store_true",
                    help="Delete existing reshape-root directory before reshaping.")

    # NEW: robustness & reporting
    ap.add_argument("--max-retries", type=int, default=10,
                    help="Retries for omnitools if it fails.")
    ap.add_argument("--retry-wait", type=int, default=60,
                    help="Seconds to wait between retries.")
    ap.add_argument("--log-file", type=str, default=None,
                    help="If set, append omnitools stdout/stderr here.")
    ap.add_argument("--report-only", action="store_true",
                    help="Only print a progress report and exit.")

    args = ap.parse_args()
    args.download_root = Path(args.download_root).expanduser().resolve()
    args.reshape_root = Path(args.reshape_root).expanduser().resolve()
    args.log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None

    # Report-only mode (no download/reshape)
    if args.report_only:
        raw = args.download_root if args.skip_download else (
            (args.download_root / f"{args.component}_{args.subset}_all")
            if (len(args.domains) == 1 and args.domains[0].lower() == "all")
            else args.download_root
        )
        report_progress(raw, args.component)
        return

    # Download (resumable + retry), then reshape
    raw_root = args.download_root if args.skip_download else run_download(args)
    args.download_root = raw_root  # point reshape to actual location
    reshape(args)


if __name__ == "__main__":
    main()
