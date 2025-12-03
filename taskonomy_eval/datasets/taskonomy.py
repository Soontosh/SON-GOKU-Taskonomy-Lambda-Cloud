
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

TASK_FOLDERS = {
    "rgb": "rgb",
    "depth_euclidean": "depth_euclidean",
    "depth_zbuffer": "depth_zbuffer",
    "normal": "normal",
    "reshading": "reshading",
    "edge_occlusion": "edge_occlusion",
    "edge_texture": "edge_texture",
    "segment_semantic": "segment_semantic",
    "keypoints2d": "keypoints2d",
    "principal_curvature": "principal_curvature",
}

def _load_png(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)

def _load_rgb(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    arr = _load_png(path)  # H,W,3 uint8
    if resize is not None:
        arr = np.array(Image.fromarray(arr).resize(resize, Image.BILINEAR))
    # to tensor float in [0,1], then normalize ImageNet
    t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mean) / std

def _load_depth(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    # 16-bit PNG with units of 1/512 meters per Taskonomy sample repo
    arr = _load_png(path).astype(np.float32)  # H,W or H,W,1
    if arr.ndim == 3:
        arr = arr[...,0]
    if resize is not None:
        arr = np.array(Image.fromarray(arr).resize(resize, Image.NEAREST))
    depth_m = torch.from_numpy(arr) / 512.0  # meters
    return depth_m.unsqueeze(0)  # 1,H,W

def _load_normals(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    arr = _load_png(path).astype(np.float32)  # H,W,3 uint8 127-centered
    if resize is not None:
        arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(resize, Image.BILINEAR)).astype(np.float32)
    # convert to [-1,1], 127-centered
    n = (arr - 127.0) / 127.0
    n = torch.from_numpy(n).permute(2,0,1)  # 3,H,W
    # re-normalize to unit length to be safe
    eps = 1e-6
    norm = torch.clamp(torch.linalg.norm(n, dim=0, keepdim=True), min=eps)
    n = n / norm
    return n

def _load_gray01(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    arr = _load_png(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[...,0]
    if resize is not None:
        arr = np.array(Image.fromarray(arr).resize(resize, Image.BILINEAR))
    # assume uint8 0..255 -> 0..1
    t = torch.from_numpy(arr) / 255.0
    return t.unsqueeze(0)  # 1,H,W

def _load_edges01(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    arr = _load_png(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[...,0]
    if resize is not None:
        arr = np.array(Image.fromarray(arr).resize(resize, Image.NEAREST))
    t = (torch.from_numpy(arr) > 127.5).float()  # binary mask 0/1
    return t.unsqueeze(0)

def _load_semantic(path: str, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    # Semantic segmentation label ids as uint8/uint16 (we keep as long)
    arr = _load_png(path)
    if arr.ndim == 3:
        arr = arr[...,0]
    if resize is not None:
        arr = np.array(Image.fromarray(arr).resize(resize, Image.NEAREST))
    # long tensor H,W
    return torch.from_numpy(arr).long()

def _resolve_buildings(root: str, split: str, buildings_list: Optional[str]) -> List[str]:
    bdir = os.path.join(root, split)
    if not os.path.isdir(bdir):
        raise FileNotFoundError(f"Split directory not found: {bdir}")
    if buildings_list:
        list_path = os.path.expanduser(buildings_list)
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                buildings = [ln.strip() for ln in f if ln.strip()]
            if buildings:
                return buildings
    return sorted([d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d))])


_GLOBAL_BAD_LOGGED: Set[str] = set()


def list_available_tasks(root: str, split: str, buildings_list: Optional[str] = None) -> Tuple[str, ...]:
    """
    Enumerate which supervision targets exist for the requested subset by scanning
    buildings for Taskonomy domains that contain at least one PNG file.
    """
    root = os.path.expanduser(root)
    bdir = os.path.join(root, split)
    buildings = _resolve_buildings(root, split, buildings_list)
    available: List[str] = []
    for task, folder in TASK_FOLDERS.items():
        if task == "rgb":
            continue
        for b in buildings:
            domain_dir = os.path.join(bdir, b, folder)
            if not os.path.isdir(domain_dir):
                continue
            if glob(os.path.join(domain_dir, "*.png")):
                available.append(task)
                break
    return tuple(sorted(available))


@dataclass
class TaskonomyConfig:
    root: str
    split: str = "train"  # train/val/test
    buildings_list: Optional[str] = None  # optional path to a txt listing building names to include
    tasks: Tuple[str, ...] = ("depth_euclidean","normal","reshading")
    resize: Optional[Tuple[int,int]] = (256,256)

class TaskonomyDataset(Dataset):
    """
    Expects directory layout like:
    <root>/<split>/<building>/<domain>/<filename>.png
    where <domain> is one of TASK_FOLDERS.
    We index by listing all RGB files and deriving aligned target paths
    for each selected task using the same filename.
    """
    def __init__(self, cfg: TaskonomyConfig):
        super().__init__()
        self.cfg = cfg
        self.root = os.path.expanduser(cfg.root)
        self.split = cfg.split
        self.tasks = tuple(cfg.tasks)
        # collect buildings
        bdir = os.path.join(self.root, self.split)
        buildings = _resolve_buildings(self.root, self.split, cfg.buildings_list)
        # enumerate RGB images as anchors
        self.items: List[Tuple[str,str,str]] = []  # (building, rgb_dir, filename)
        for b in buildings:
            rgb_dir = os.path.join(bdir, b, TASK_FOLDERS["rgb"])
            if not os.path.isdir(rgb_dir):
                continue
            files = sorted(glob(os.path.join(rgb_dir, "*.png")))
            for f in files:
                self.items.append((b, rgb_dir, os.path.basename(f)))
        if not self.items:
            raise RuntimeError(f"No samples found for split '{self.split}' with tasks {self.tasks}.")
        self._bad_indices: set[int] = set()
        self._bad_logged: set[str] = set()
        self._skip_count: int = 0
        self._skip_log_limit: int = 5
        self._skip_log_interval: int = 50
        # verify coverage of targets lazily during __getitem__

    def __len__(self) -> int:
        return len(self.items)

    def _path(self, building: str, domain: str, fname: str) -> str:
        # Omnidata/Taskonomy encodes domain in the filename, e.g.
        #   point_47_view_3_domain_rgb.png
        # and the corresponding depth file is
        #   point_47_view_3_domain_depth_euclidean.png
        # Our dataset uses the RGB filenames as anchors, so for non-RGB
        # tasks we need to swap the suffix appropriately.
        basename = fname
        if "domain_rgb" in basename and domain != "rgb":
            basename = basename.replace("domain_rgb", f"domain_{domain}")
        return os.path.join(
            self.root,
            self.split,
            building,
            TASK_FOLDERS[domain],
            basename,
        )

    def __getitem__(self, idx: int):
        num_items = len(self.items)
        attempt = 0
        current_idx = idx % num_items
        while attempt < num_items:
            if current_idx in self._bad_indices:
                current_idx = (current_idx + 1) % num_items
                attempt += 1
                continue
            b, rgb_dir, fname = self.items[current_idx]
            rgb_path = os.path.join(rgb_dir, fname)
            try:
                sample: Dict[str, Any] = {"rgb": _load_rgb(rgb_path, self.cfg.resize)}
                for t in self.tasks:
                    path = self._path(b, t, fname)
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Missing target for task '{t}': {path}")
                    if t in ("depth_euclidean","depth_zbuffer"):
                        sample[t] = _load_depth(path, self.cfg.resize)
                    elif t == "normal":
                        sample[t] = _load_normals(path, self.cfg.resize)
                    elif t == "reshading":
                        sample[t] = _load_gray01(path, self.cfg.resize)
                    elif t in ("edge_occlusion","edge_texture"):
                        sample[t] = _load_edges01(path, self.cfg.resize)
                    elif t == "segment_semantic":
                        sample[t] = _load_semantic(path, self.cfg.resize)
                    elif t == "keypoints2d":
                        sample[t] = _load_gray01(path, self.cfg.resize)
                    elif t == "principal_curvature":
                        arr = _load_png(path).astype(np.float32)
                        if arr.ndim == 2:
                            arr = np.expand_dims(arr, -1)
                        if self.cfg.resize is not None:
                            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(self.cfg.resize, Image.BILINEAR)).astype(np.float32)
                        tcurv = (torch.from_numpy(arr).permute(2,0,1) - 127.0) / 127.0
                        sample[t] = tcurv
                    else:
                        raise ValueError(f"Unsupported task: {t}")
                sample["filename"] = fname
                sample["building"] = b
                return sample
            except (OSError, IOError, SyntaxError) as err:
                self._bad_indices.add(current_idx)
                self._bad_logged.add(rgb_path)
                self._skip_count += 1
                log_this = (
                    self._skip_count <= self._skip_log_limit
                    or (self._skip_count % self._skip_log_interval) == 0
                )
                if log_this:
                    already_reported = rgb_path in _GLOBAL_BAD_LOGGED
                    _GLOBAL_BAD_LOGGED.add(rgb_path)
                    detail = rgb_path if already_reported else f"{rgb_path} ({err})"
                    print(
                        "[TaskonomyDataset] "
                        f"Skipped {self._skip_count} samples so far; latest: {detail}"
                    )
                else:
                    _GLOBAL_BAD_LOGGED.add(rgb_path)
                current_idx = (current_idx + 1) % num_items
                attempt += 1
        raise RuntimeError("No valid samples available after skipping corrupt entries.")
