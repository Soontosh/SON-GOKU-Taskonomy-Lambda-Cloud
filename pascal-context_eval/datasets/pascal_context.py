"""Pascal-Context dataset loader compatible with the Taskonomy scaffold."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

PASCAL_CONTEXT_59_CLASSES: Tuple[str, ...] = (
    "background",
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

LABEL_SETS: Dict[str, Dict[str, Any]] = {
    "59": {
        "classes": PASCAL_CONTEXT_59_CLASSES,
        "num_classes": len(PASCAL_CONTEXT_59_CLASSES),
        "ignore_index": 255,
    }
}

SUPPORTED_TASKS = {"segment_semantic"}


@dataclass
class PascalContextConfig:
    """Configuration for :class:`PascalContextDataset`.

    Attributes
    ----------
    root:
        Directory containing the extracted Pascal-Context assets. The loader
        looks for ``JPEGImages`` as well as ``context/<split>`` directories
        under this root. You can point it either to ``VOCdevkit/VOC2010`` or to
        a parent folder that contains both ``VOCdevkit/VOC2010`` and
        ``context``.
    split:
        Which split to load. Official Pascal-Context releases provide ``train``
        and ``val``. The loader will also accept custom split names as long as
        corresponding folders exist under the context masks directory.
    tasks:
        Target tasks to include alongside the RGB anchor. ``segment_semantic``
        is currently supported.
    resize:
        Optional ``(width, height)`` pair used to resize both RGB images and
        masks. ``None`` keeps native resolution.
    label_set:
        Which semantic label taxonomy to use. ``"59"`` corresponds to the
        standard 59-class setup (plus background). If you pass a value that is
        not part of :data:`LABEL_SETS`, you must provide ``class_map`` so that
        labels are mapped into a contiguous ``0..C-1`` range.
    mask_root:
        Overrides the automatic discovery of the context mask directory. This
        should point to the folder that directly contains the per-split
        subfolders (e.g. ``.../context``).
    class_map:
        Optional dictionary mapping raw pixel values found in the annotation to
        the output class indices. Values not present in the mapping are treated
        as ``ignore_index``.
    ignore_index:
        Label value reserved for unlabeled pixels. Defaults to ``255`` to match
        the official annotations.
    reduce_zero_label:
        Whether to shift all labels down by one (useful if you want to discard
        the background class but keep the native mask files untouched).
    transforms:
        Callable applied to the sample dictionary after all tensors have been
        created. You can plug data augmentation pipelines here if needed.
    """

    root: str
    split: str = "train"
    tasks: Tuple[str, ...] = ("segment_semantic",)
    resize: Optional[Tuple[int, int]] = None
    label_set: str = "59"
    mask_root: Optional[str] = None
    class_map: Optional[Dict[int, int]] = None
    ignore_index: int = 255
    reduce_zero_label: bool = False
    transforms: Optional[Any] = None


class PascalContextDataset(Dataset):
    """Dataset that mirrors the :class:`TaskonomyDataset` contract for Pascal-Context."""

    def __init__(self, cfg: PascalContextConfig):
        super().__init__()
        self.cfg = cfg
        self.root = Path(os.path.expanduser(cfg.root)).resolve()
        self.split = cfg.split
        self.tasks = tuple(cfg.tasks)

        unknown_tasks = set(self.tasks) - SUPPORTED_TASKS
        if unknown_tasks:
            raise ValueError(f"Unsupported tasks for Pascal-Context: {sorted(unknown_tasks)}")

        self.voc_root = self._discover_voc_root(self.root)
        self.mask_root = self._discover_mask_root(self.root, cfg)
        self.mask_dir = self.mask_root / self.split
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Split directory '{self.mask_dir}' not found.")

        self.samples: List[Tuple[str, Path, Path]] = []  # (image_id, rgb_path, mask_path)
        self.skipped: List[Tuple[str, str]] = []

        rgb_dir = self.voc_root / "JPEGImages"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Could not find JPEGImages directory under {self.voc_root}")

        for mask_path in sorted(self.mask_dir.glob("*.png")):
            image_id = mask_path.stem
            rgb_path = self._find_rgb(rgb_dir, image_id)
            if rgb_path is None:
                self.skipped.append((image_id, "missing_rgb"))
                continue
            if not self._is_valid_image(rgb_path):
                self.skipped.append((str(rgb_path), "corrupt_rgb"))
                continue
            if not self._is_valid_image(mask_path):
                self.skipped.append((str(mask_path), "corrupt_mask"))
                continue
            self.samples.append((image_id, rgb_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split '{self.split}'. "
                f"Checked masks under {self.mask_dir} and RGBs under {rgb_dir}."
            )

        self.class_map = cfg.class_map
        if cfg.label_set in LABEL_SETS:
            meta = LABEL_SETS[cfg.label_set]
            self.num_classes = int(meta.get("num_classes", 0))
            # Align ignore index with metadata unless explicitly overridden.
            if cfg.ignore_index == PascalContextConfig.__dataclass_fields__["ignore_index"].default:
                self.ignore_index = int(meta.get("ignore_index", cfg.ignore_index))
            else:
                self.ignore_index = cfg.ignore_index
            self.classes = tuple(meta.get("classes", ()))
        else:
            if cfg.class_map is None:
                raise ValueError(
                    "PascalContextDataset requires `class_map` when `label_set` is not one of "
                    f"{sorted(LABEL_SETS.keys())}."
                )
            self.num_classes = max(cfg.class_map.values()) + 1
            self.ignore_index = cfg.ignore_index
            self.classes: Tuple[str, ...] = ()

    @staticmethod
    def _discover_voc_root(root: Path) -> Path:
        candidates = [
            root,
            root / "VOC2010",
            root / "VOCdevkit" / "VOC2010",
        ]
        for cand in candidates:
            if (cand / "JPEGImages").is_dir():
                return cand
        raise FileNotFoundError(
            "Could not locate the VOC2010 JPEGImages directory. "
            "Tried the provided root, root/VOC2010, and root/VOCdevkit/VOC2010."
        )

    @staticmethod
    def _discover_mask_root(root: Path, cfg: PascalContextConfig) -> Path:
        if cfg.mask_root is not None:
            mask_root = Path(os.path.expanduser(cfg.mask_root)).resolve()
            if not mask_root.is_dir():
                raise FileNotFoundError(f"mask_root '{mask_root}' does not exist or is not a directory.")
            return mask_root

        candidates = [
            root / "context",
            root / "VOC2010" / "context",
            root / "VOCdevkit" / "context",
            root.parent / "context",
        ]
        for cand in candidates:
            if cand.is_dir():
                return cand
        raise FileNotFoundError(
            "Could not find a 'context' directory. Specify PascalContextConfig.mask_root explicitly."
        )

    @staticmethod
    def _find_rgb(rgb_dir: Path, image_id: str) -> Optional[Path]:
        candidates = [
            rgb_dir / f"{image_id}.jpg",
            rgb_dir / f"{image_id}.png",
            rgb_dir / f"{image_id}.jpeg",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (OSError, IOError):
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.cfg.resize is not None:
                img = img.resize(self.cfg.resize, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as mask_img:
            if self.cfg.resize is not None:
                mask_img = mask_img.resize(self.cfg.resize, Image.NEAREST)
            mask = np.array(mask_img, dtype=np.int64)

        if self.class_map is not None:
            remapped = np.full(mask.shape, self.ignore_index, dtype=np.int64)
            for src, dst in self.class_map.items():
                remapped[mask == src] = dst
            mask = remapped

        if self.cfg.reduce_zero_label:
            mask = mask - 1
            mask[mask < 0] = self.ignore_index

        return torch.from_numpy(mask).long()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, rgb_path, mask_path = self.samples[idx]
        sample: Dict[str, Any] = {
            "rgb": self._load_rgb(rgb_path),
            "filename": f"{image_id}.jpg",
            "image_id": image_id,
            "split": self.split,
        }
        for task in self.tasks:
            if task == "segment_semantic":
                sample[task] = self._load_mask(mask_path)
            else:
                raise ValueError(f"Unsupported task '{task}'")

        if self.cfg.transforms is not None:
            sample = self.cfg.transforms(sample)

        return sample

    def class_frequencies(self) -> Optional[np.ndarray]:
        """Estimate per-class pixel frequencies by scanning the dataset once.

        The method can be helpful for computing class-balanced loss weights on
        small dummy datasets where a precomputed statistics file is not
        available. The result is cached after the first call.
        """

        if not hasattr(self, "_class_freq_cache"):
            freq: Optional[np.ndarray]
            if self.num_classes is None:
                freq = None
            else:
                counts = np.zeros(self.num_classes, dtype=np.int64)
                total = 0
                for _, _, mask_path in self.samples:
                    with Image.open(mask_path) as mask_img:
                        mask = np.array(mask_img, dtype=np.int64)
                    mask = mask[(mask != self.ignore_index)]
                    if mask.size == 0:
                        continue
                    bins = np.bincount(mask, minlength=self.num_classes)
                    counts[: len(bins)] += bins[: self.num_classes]
                    total += mask.size
                if total == 0:
                    freq = None
                else:
                    freq = counts / float(total)
            self._class_freq_cache = freq
        return getattr(self, "_class_freq_cache")

    def extra_repr(self) -> str:
        details = [
            f"root={self.root}",
            f"split={self.split}",
            f"samples={len(self.samples)}",
            f"tasks={self.tasks}",
            f"resize={self.cfg.resize}",
            f"label_set={self.cfg.label_set}",
            f"num_classes={self.num_classes}",
            f"ignore_index={self.ignore_index}",
        ]
        return ", ".join(details)
