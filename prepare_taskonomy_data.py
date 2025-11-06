#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path
import random
import shutil


def run_download(args):
    """Call `omnitools.download` to fetch Taskonomy data."""
    if args.skip_download:
        print("[INFO] Skipping download step.")
        return

    if not args.domains:
        raise ValueError("Need at least one domain (e.g., rgb depth_euclidean normal reshading).")

    cmd = ["omnitools.download"]
    cmd.extend(args.domains)
    cmd.extend([
        "--components", args.component,
        "--subset", args.subset,             # debug / tiny / medium / full / fullplus
        "--split", args.download_split,      # train / val / test / all
        "--dest", str(args.download_root),
        "--connections_total", str(args.connections_total),
        "--name", args.name,
        "--email", args.email,
        "--agree_all",
    ])

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

    # ensure rgb is present and domains are unique, keep order
    domains = list(dict.fromkeys(["rgb"] + args.domains))
    print(f"[INFO] Domains: {domains}")

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
        description="Download and reshape Omnidata/Taskonomy data for Son-Goku Taskonomy training."
    )
    parser.add_argument(
        "--download-root",
        type=str,
        required=True,
        help="Directory where omnitools.download will put raw data (omnitools --dest).",
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
        help="Domains to download & reshape (RGB will be added if missing).",
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
    args.download_root = Path(args.download_root).expanduser().resolve()
    args.reshape_root = Path(args.reshape_root).expanduser().resolve()

    if not args.skip_download:
        if not args.name or not args.email:
            print("ERROR: --name and --email are required unless you use --skip-download.", file=sys.stderr)
            sys.exit(1)
        run_download(args)

    reshape(args)


if __name__ == "__main__":
    main()
