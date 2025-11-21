#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path
import random
import shutil
import os
from typing import List, Set


# A broad superset of Taskonomy/Omnidata domain names for discovery.
# (Safe to expand over time; discovery walks the tree anyway.)
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


def run(cmd: List[str]):
    print("[INFO] Running:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("ERROR: 'omnitools.download' not found in PATH. Did you install 'omnidata-tools'?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] omnitools.download exited with code {exc.returncode}.", file=sys.stderr)
        if exc.stdout:
            print("[omnitools stdout]\n" + exc.stdout, file=sys.stderr)
        if exc.stderr:
            print("[omnitools stderr]\n" + exc.stderr, file=sys.stderr)
        raise
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)


def run_download(args) -> Path:
    """
    Call `omnitools.download` to fetch Taskonomy data.

    Returns the *actual* raw destination directory used so the reshape step
    can point to the right place (especially for --domains all).
    """
    if args.skip_download:
        print("[INFO] Skipping download step.")
        # If skipping, assume the provided path already contains the data.
        return Path(args.download_root)

    use_all = (len(args.domains) == 1 and args.domains[0].lower() == "all")

    # For 'all', download into a subdir to avoid clobbering prior runs.
    if use_all:
        raw_dest = Path(args.download_root) / f"{args.component}_{args.subset}_all"
        raw_dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        raw_dest = Path(args.download_root)
        raw_dest.mkdir(parents=True, exist_ok=True)

    if not args.name or not args.email:
        print("ERROR: --name and --email are required unless you use --skip-download.", file=sys.stderr)
        sys.exit(1)

    cmd = ["omnitools.download"]
    if use_all:
        if not args.agree_all:
            print("ERROR: Using --domains all requires --agree_all to accept the license.", file=sys.stderr)
            sys.exit(2)
        cmd += ["all"]
    else:
        cmd += args.domains

    cmd.extend([
        "--components", args.component,
        "--subset", args.subset,             # debug / tiny / medium / full / fullplus
        "--split", args.download_split,      # train / val / test / all
        "--dest", str(raw_dest),
        "--connections_total", str(args.connections_total),
        "--name", args.name,
        "--email", args.email,
    ])

    if args.agree_all:
        cmd.append("--agree_all")

    run(cmd)
    return raw_dest.resolve()


def discover_layout(download_root: Path, component: str):
    """
    Figure out whether omnitools created:

        <dest>/<domain>/<component>/<building> or
        <dest>/<component>/<domain>/<building>

    and return (layout_type, rgb_root, building_names).
    """
    # domain-first layout: <dest>/<domain>/<component>/<building>
    rgb_domain_first = download_root / "rgb" / component
    # component-first layout: <dest>/<component>/<domain>/<building>
    rgb_component_first = download_root / component / "rgb"

    if rgb_domain_first.is_dir():
        layout = "domain_first"
        rgb_root = rgb_domain_first
    elif rgb_component_first.is_dir():
        layout = "component_first"
        rgb_root = rgb_component_first
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
    if layout == "domain_first":
        # <dest>/<domain>/<component>/<building>
        return download_root / domain / component / building
    else:
        # <dest>/<component>/<domain>/<building>
        return download_root / component / domain / building


def make_splits(buildings, train_frac, val_frac, test_frac, seed):
    """Randomly assign buildings to train/val/test."""
    if train_frac + val_frac + test_frac > 1.0 + 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must be <= 1.0")

    rng = random.Random(seed)
    buildings = list(buildings)
    rng.shuffle(buildings)

    n = len(buildings)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    # Ensure all buildings get used: put leftovers into test
    assigned = n_train + n_val + n_test
    if assigned < n:
        n_test += n - assigned

    splits = {}
    for i, b in enumerate(buildings):
        if i < n_train:
            splits[b] = "train"
        elif i < n_train + n_val:
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
        if base in KNOWN_DOMAIN_NAMES:
            # Keep if this looks like a real domain folder (has subdirs or image files)
            if dirs or any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                found.add(base)
    # Ensure rgb, put it first
    domains = sorted(d for d in found if d != "rgb")
    if "rgb" in found:
        return ["rgb"] + domains
    return domains


def reshape(args):
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
            print(f"[INFO] Removing existing reshape_root: {reshape_root}")
            shutil.rmtree(reshape_root)
        else:
            raise RuntimeError(f"{reshape_root} already exists. Use --force to overwrite.")

    layout, rgb_root, buildings = discover_layout(download_root, args.component)
    print(f"[INFO] Found {len(buildings)} buildings under {rgb_root}")

    if args.no_split:
        splits = {b: "train" for b in buildings}  # everything goes into 'train'
    else:
        splits = make_splits(buildings, args.train_frac, args.val_frac, args.test_frac, args.seed)

    reshape_root.mkdir(parents=True, exist_ok=True)

    # Determine domains to reshape
    use_all = (len(args.domains) == 1 and args.domains[0].lower() == "all")
    if use_all:
        discovered = discover_domains(download_root)
        if not discovered:
            print(f"[ERROR] Could not discover any domains under {download_root}", file=sys.stderr)
            sys.exit(2)
        domains = discovered if "rgb" in discovered else (["rgb"] + discovered)
        print(f"[INFO] Discovered domains: {', '.join(domains)}")
        # Save the final list for reproducibility
        with open(reshape_root / "domains_used.txt", "w") as f:
            for d in domains:
                f.write(d + "\n")
    else:
        # ensure rgb is present and domains are unique, keep order
        domains = list(dict.fromkeys(["rgb"] + args.domains))
        print(f"[INFO] Domains: {domains}")
        with open(reshape_root / "domains_used.txt", "w") as f:
            for d in domains:
                f.write(d + "\n")

    total_samples = 0
    skipped_samples = 0

    for b in buildings:
        split = splits[b]
        rgb_dir = src_dir(download_root, layout, args.component, "rgb", b)
        if not rgb_dir.is_dir():
            print(f"[WARN] No rgb dir for building {b} at {rgb_dir}, skipping building.")
            continue

        rgb_files = sorted(p for p in rgb_dir.iterdir() if p.is_file())
        if not rgb_files:
            print(f"[WARN] No rgb files for building {b}, skipping building.")
            continue

        for rgb_path in rgb_files:
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
                if "domain_rgb" in fname:
                    d_fname = fname.replace("domain_rgb", f"domain_{d}")
                else:
                    # Fallback: identical filename
                    d_fname = fname

                cand = d_dir / d_fname
                if not cand.is_file():
                    ok = False
                    break

                src_paths[d] = cand

            if not ok:
                skipped_samples += 1
                continue

            # Create symlinks for this multi-task sample
            for d, src in src_paths.items():
                dest_dir = reshape_root / split / b / d
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src.name
                if not dest.exists():
                    dest.symlink_to(src)

            total_samples += 1

    print("[INFO] Finished reshaping.")
    print(f"[INFO] Total multi-task samples created: {total_samples}")
    if skipped_samples:
        print(f"[INFO] Skipped {skipped_samples} rgb views with missing targets: {skipped_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and reshape Omnidata/Taskonomy data for SON-GOKU Taskonomy training."
    )
    parser.add_argument(
        "--download-root",
        type=str,
        required=True,
        help="Directory where raw data will be stored. For --domains all, a subfolder "
             "<component>_<subset>_all is created inside this directory.",
    )
    parser.add_argument(
        "--reshape-root",
        type=str,
        required=True,
        help="Output directory for Taskonomy-style layout (root/split/building/domain/).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="debug",
        choices=["debug", "tiny", "medium", "full", "fullplus"],
        help="Subset to download (omnitools --subset).",
    )
    parser.add_argument(
        "--download-split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Split to download from Omnidata (omnitools --split).",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="taskonomy",
        help="Component dataset (omnitools --components). Usually 'taskonomy'.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["rgb", "depth_euclidean", "normal", "reshading"],
        help="Domains to download & reshape. Use the special value 'all' to include *all* "
             "domains available for the subset. RGB is always included for reshaping.",
    )
    parser.add_argument(
        "--agree_all",
        action="store_true",
        help="Automatically accept the Omnidata EULA for omnitools (required for --domains all).",
    )
    parser.add_argument(
        "--connections-total",
        type=int,
        default=16,
        help="omnitools --connections_total.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Your name for omnitools license agreement (required unless --skip-download).",
    )
    parser.add_argument(
        "--email",
        type=str,
        required=False,
        help="Your email for omnitools license agreement (required unless --skip-download).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step and only reshape existing data.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not create train/val/test splits; put everything into 'train'.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of buildings in train split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of buildings in val split.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of buildings in test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for building splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing reshape-root directory before reshaping.",
    )

    args = parser.parse_args()
    # keep these as Paths throughout
    args.download_root = Path(args.download_root).expanduser().resolve()
    args.reshape_root = Path(args.reshape_root).expanduser().resolve()

    # Download (or skip) and capture actual raw path used
    if args.skip_download:
        raw_root = args.download_root
    else:
        raw_root = run_download(args)
    # Point reshape step at the actual raw directory
    args.download_root = raw_root

    reshape(args)


if __name__ == "__main__":
    main()