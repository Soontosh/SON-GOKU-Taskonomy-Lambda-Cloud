from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


_TASK_CHANNELS: Dict[str, int] = {
    "depth_euclidean": 1,
    "depth_zbuffer": 1,
    "normal": 3,
    "reshading": 1,
    "edge_occlusion": 1,
    "edge_texture": 1,
    "segment_semantic": 1,  # integer labels, handled separately
    "keypoints2d": 1,
    "principal_curvature": 2,
}


@dataclass
class SyntheticConfig:
    tasks: Tuple[str, ...]
    resize: Tuple[int, int] | None
    length: int
    seg_classes: int
    seed: int


class SyntheticTaskonomyDataset(Dataset):
    """
    Synthetic drop-in replacement for TaskonomyDataset that emits deterministic
    random tensors with Taskonomy-like shapes. This avoids filesystem + decode
    variance so we can isolate method-only memory footprints.
    """

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        if cfg.resize is None:
            self.height, self.width = 256, 256
        else:
            self.height, self.width = cfg.resize

    def __len__(self) -> int:
        return self.cfg.length

    def _task_tensor(self, task: str, gen: torch.Generator) -> torch.Tensor:
        H, W = self.height, self.width
        if task not in _TASK_CHANNELS:
            raise ValueError(f"SyntheticTaskonomyDataset cannot synthesize task '{task}'.")
        channels = _TASK_CHANNELS[task]
        if task == "segment_semantic":
            return torch.randint(
                low=0,
                high=max(2, self.cfg.seg_classes),
                size=(H, W),
                generator=gen,
                dtype=torch.long,
            )
        tensor = torch.randn(channels, H, W, generator=gen)
        if task == "normal":
            norm = tensor.norm(dim=0, keepdim=True).clamp_min(1e-6)
            tensor = tensor / norm
        if task in ("edge_occlusion", "edge_texture"):
            probs = torch.rand(channels, H, W, generator=gen)
            tensor = torch.bernoulli(probs)
        if task == "reshading":
            tensor = torch.rand(channels, H, W, generator=gen)
        return tensor

    def __getitem__(self, idx: int):
        # make per-index generator so samples repeat deterministically
        gen = torch.Generator()
        gen.manual_seed(self.cfg.seed + idx)
        rgb = torch.randn(3, self.height, self.width, generator=gen)
        sample = {"rgb": rgb}
        for task in self.cfg.tasks:
            sample[task] = self._task_tensor(task, gen)
        sample["filename"] = f"synthetic_{idx:06d}.png"
        sample["building"] = "synthetic"
        return sample
