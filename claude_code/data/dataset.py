"""BEDLAM2 frame-level dataset for Sapiens-based pose estimation.

Each sample is a single video frame paired with its depth map and the
corresponding 3D SMPL-X joint annotations in camera space.

Sample dict (before transform):
    rgb:       np.ndarray (H, W, 3)  uint8   — original video frame
    depth:     np.ndarray (H, W)     float32 — depth in metres (None if unavailable)
    joints:    np.ndarray (J, 3)     float32 — camera-space XYZ, J=127
    intrinsic: np.ndarray (3, 3)     float32 — camera intrinsic matrix
    folder_name: str
    seq_name:    str
    frame_idx:   int                 — index within the downsampled (6 fps) sequence

After applying ToTensor (or a Compose that ends with it):
    rgb:       Tensor (3, H, W)      float32 — ImageNet-normalised
    depth:     Tensor (1, H, W)      float32 — clipped & normalised to [0, 1]
    joints:    Tensor (J, 3)         float32 — unchanged
    intrinsic: Tensor (3, 3)         float32
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import FRAME_STRIDE




class BedlamFrameDataset(Dataset):
    """One sample = one frame from one BEDLAM2 sequence.

    Args:
        seq_paths:      List of relative paths like ``"folder/seq.npz"``
                        (relative to ``data_root/data/label/``).
        data_root:      Absolute path to the BEDLAM2 data root directory
                        (the one containing ``data/label``, ``data/mp4``, etc.).
        transform:      Optional callable applied to each sample dict.
        depth_required: Raise if a depth file is missing when True.
        frame_stride:   Step between sampled video frames (default 5 → 6 fps
                        from 30 fps source).
    """

    def __init__(
        self,
        seq_paths: list[str],
        data_root: str,
        transform=None,
        depth_required: bool = True,
        frame_stride: int = FRAME_STRIDE,
    ):
        self.data_root = data_root
        self.transform = transform
        self.depth_required = depth_required
        self.frame_stride = frame_stride

        # Per-worker label cache (populated lazily; each worker fills its own copy).
        # Stores only small scalar metadata + open mmap NPZ handle for joints_cam.
        self._label_cache: dict[str, dict] = {}
        # Per-worker mmap cache for NPY depth files.
        # np.load(mmap_mode='r') returns a view; the OS manages paging — no size limit needed.
        self._depth_mmap: dict[str, np.ndarray | None] = {}
        # LRU fallback cache for legacy NPZ files (bounded to avoid OOM).
        self._depth_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._depth_cache_maxsize = 3

        # Build flat index: list of (label_abs_path, frame_idx).
        # Only read n_frames here; full metadata is cached lazily per worker.
        self.index: list[tuple[str, int]] = []
        for seq_rel in seq_paths:
            label_path = os.path.join(data_root, "data", "label", seq_rel)
            try:
                meta = np.load(label_path, allow_pickle=True)
                n_frames = int(meta["n_frames"])
            except Exception as e:
                raise RuntimeError(f"Failed to read label {label_path}: {e}") from e
            for frame_idx in range(n_frames):
                self.index.append((label_path, frame_idx))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        label_path, frame_idx = self.index[idx]

        # Lazily populate label cache per worker (each worker fills its own copy)
        if label_path not in self._label_cache:
            with np.load(label_path, allow_pickle=True) as meta:
                self._label_cache[label_path] = {
                    "folder_name":      str(meta["folder_name"]),
                    "seq_name":         str(meta["seq_name"]),
                    "intrinsic_matrix": meta["intrinsic_matrix"].astype(np.float32),
                    "joints_cam":       meta["joints_cam"][0].astype(np.float32),  # (n_frames, 127, 3)
                }
        cached = self._label_cache[label_path]

        folder_name = cached["folder_name"]
        seq_name    = cached["seq_name"]
        intrinsic   = cached["intrinsic_matrix"]

        joints = cached["joints_cam"][frame_idx]  # (127, 3)

        # --- RGB ----------------------------------------------------------
        rgb = self._read_frame(folder_name, seq_name, frame_idx, label_path)

        # --- Depth --------------------------------------------------------
        # Prefer pre-converted NPY (mmappable, no decompression) over NPZ.
        npy_path = os.path.join(
            self.data_root, "data", "depth", "npy",
            folder_name, f"{seq_name}.npy",
        )
        npz_path = os.path.join(
            self.data_root, "data", "depth", "npz",
            folder_name, f"{seq_name}.npz",
        )
        depth = self._read_depth(npy_path, npz_path, frame_idx, label_path)

        # NOTE:
        # RGB is loaded from pre-extracted JPEGs in data/frames, which are
        # already stored in the upright orientation. No runtime rotation.

        sample = {
            "rgb":         rgb,
            "depth":       depth,
            "joints":      joints,
            "intrinsic":   intrinsic,
            "folder_name": folder_name,
            "seq_name":    seq_name,
            "frame_idx":   frame_idx,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_frame(
        self,
        folder_name: str,
        seq_name: str,
        frame_idx: int,
        label_path: str,
    ) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB frame.

        JPG-only mode: reads pre-extracted JPEG from ``data/frames`` and raises
        a clear error if the frame is missing.

        Pre-extracted JPEG layout:
            data/frames/<folder>/<seq_name>/<frame_idx:05d>.jpg
        """
        jpeg_path = (
            Path(self.data_root)
            / "data"
            / "frames"
            / folder_name
            / seq_name
            / f"{frame_idx:05d}.jpg"
        )

        if not jpeg_path.exists():
            raise FileNotFoundError(
                f"Missing extracted JPG frame for {label_path}: {jpeg_path}. "
                "Run extract_frames.py first."
            )

        img = cv2.imread(str(jpeg_path))
        if img is None:
            raise RuntimeError(f"Failed to decode JPG frame: {jpeg_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_depth(
        self, npy_path: str, npz_path: str, frame_idx: int, label_path: str
    ) -> np.ndarray | None:
        """Return a single (H, W) float32 depth frame.

        Fast path: memory-map the pre-converted NPY file — the OS pages in only
        the frame requested, with zero decompression cost.
        Slow fallback: load from the original compressed NPZ with an LRU cache.
        """
        # ── Fast path: NPY mmap ───────────────────────────────────────────────
        if npy_path not in self._depth_mmap:
            if os.path.exists(npy_path):
                self._depth_mmap[npy_path] = np.load(npy_path, mmap_mode="r")
            else:
                self._depth_mmap[npy_path] = None  # not available

        arr = self._depth_mmap[npy_path]
        if arr is not None:
            return arr[frame_idx].astype(np.float32)

        # ── Slow fallback: NPZ with LRU cache ────────────────────────────────
        if npz_path not in self._depth_cache:
            if not os.path.exists(npz_path):
                if self.depth_required:
                    raise FileNotFoundError(
                        f"Depth not found for {label_path}: {npz_path}"
                    )
                val = None
            else:
                with np.load(npz_path) as f:
                    val = f["depth"].astype(np.float32)
            if len(self._depth_cache) >= self._depth_cache_maxsize:
                self._depth_cache.popitem(last=False)
            self._depth_cache[npz_path] = val
        else:
            self._depth_cache.move_to_end(npz_path)
        arr = self._depth_cache[npz_path]
        return None if arr is None else arr[frame_idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors; pass metadata as lists."""
    out: dict = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals  # str / int metadata
    return out


def build_dataloader(
    seq_paths: list[str],
    data_root: str,
    transform=None,
    depth_required: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = BedlamFrameDataset(
        seq_paths=seq_paths,
        data_root=data_root,
        transform=transform,
        depth_required=depth_required,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        multiprocessing_context=("spawn" if num_workers > 0 else None),
    )
