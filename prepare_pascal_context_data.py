#!/usr/bin/env python
"""Utility script to reshape Pascal-Context into the Taskonomy-style layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def discover_roots(root: Path) -> Tuple[Path, Path]:
    """Return (voc_root, context_root) under the provided directory."""
    candidates = [
        root,
        root / "VOC2010",
        root / "VOCdevkit" / "VOC2010",
    ]
    voc_root = None
    for cand in candidates:
        if (cand / "JPEGImages").is_dir():
            voc_root = cand
            break
    if voc_root is None:
        raise FileNotFoundError(
            f"Could not locate JPEGImages under {root}. "
            "Point --pascal-root to VOCdevkit/VOC2010 or a parent directory."
        )

    context_candidates = [
        root / "context",
        root / "VOC2010" / "context",
        root / "VOCdevkit" / "context",
        root.parent / "context",
    ]
    context_root = None
    for cand in context_candidates:
        if cand.is_dir():
            context_root = cand
            break
    if context_root is None:
        raise FileNotFoundError(
            f"Could not locate a 'context' directory near {root}. "
            "Verify that the Pascal-Context annotations were extracted."
        )

    return voc_root, context_root


def iter_ids(mask_dir: Path, limit: int | None = None) -> Iterable[str]:
    ids = sorted(p.stem for p in mask_dir.glob("*.png"))
    if limit is not None:
        ids = ids[:limit]
    for image_id in ids:
        yield image_id


def find_rgb(voc_root: Path, image_id: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = voc_root / "JPEGImages" / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def prepare_split(
    voc_root: Path,
    context_root: Path,
    split: str,
    output_root: Path,
    copy_mode: str,
    limit: int | None = None,
) -> Tuple[int, int]:
    mask_dir = context_root / split
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory for split '{split}': {mask_dir}")

    created = 0
    skipped = 0
    for image_id in iter_ids(mask_dir, limit=limit):
        rgb_path = find_rgb(voc_root, image_id)
        if rgb_path is None:
            skipped += 1
            continue
        mask_path = mask_dir / f"{image_id}.png"
        sample_root = output_root / split / image_id
        rgb_dest = sample_root / "rgb" / f"{image_id}.jpg"
        seg_dest = sample_root / "segment_semantic" / f"{image_id}.png"
        seg_dest.parent.mkdir(parents=True, exist_ok=True)
        rgb_dest.parent.mkdir(parents=True, exist_ok=True)

        if copy_mode == "copy":
            shutil.copy2(rgb_path, rgb_dest)
            shutil.copy2(mask_path, seg_dest)
        elif copy_mode == "symlink":
            if rgb_dest.exists():
                rgb_dest.unlink()
            if seg_dest.exists():
                seg_dest.unlink()
            rgb_dest.symlink_to(rgb_path)
            seg_dest.symlink_to(mask_path)
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        created += 1
    return created, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Pascal-Context data")
    parser.add_argument("--pascal-root", required=True, help="Directory containing VOC2010 and context folders")
    parser.add_argument("--output-root", required=True, help="Destination for the Taskonomy-style layout")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="Which splits to process")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on images per split (useful for dummy runs)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pascal_root = Path(args.pascal_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    voc_root, context_root = discover_roots(pascal_root)
    print(f"[INFO] VOC root: {voc_root}")
    print(f"[INFO] Context root: {context_root}")

    if output_root.exists():
        if args.force:
            print(f"[INFO] Removing existing directory: {output_root}")
            shutil.rmtree(output_root)
        else:
            raise RuntimeError(f"{output_root} already exists. Use --force to overwrite.")

    total_created = 0
    total_skipped = 0
    for split in args.splits:
        created, skipped = prepare_split(
            voc_root,
            context_root,
            split,
            output_root,
            args.copy_mode,
            limit=args.limit,
        )
        total_created += created
        total_skipped += skipped
        print(f"[INFO] Split {split}: created {created} samples, skipped {skipped} (missing RGB)")

    print(f"[DONE] Wrote {total_created} samples to {output_root}. Skipped {total_skipped} items.")


if __name__ == "__main__":
    main()
