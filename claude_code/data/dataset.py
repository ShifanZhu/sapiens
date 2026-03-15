"""BEDLAM2 frame-level dataset for Sapiens-based pose estimation.

Each sample is a single video frame for a single person, paired with its depth
map and the corresponding 3D SMPL-X joint annotations in camera space.

Supports multi-person sequences: the index is
``(label_path, body_idx, frame_idx)`` so each person in each frame is a
separate sample.  For single-person sequences ``body_idx`` is always 0.

Sample dict (before transform):
    rgb:       np.ndarray (H, W, 3)  uint8   — original video frame
    depth:     np.ndarray (H, W)     float32 — depth in metres (None if unavailable)
    joints:    np.ndarray (J, 3)     float32 — camera-space XYZ, J=NUM_JOINTS (active subset)
    intrinsic: np.ndarray (3, 3)     float32 — camera intrinsic matrix
    bbox:      np.ndarray (4,)       float32 — (x1, y1, x2, y2) person box
    body_idx:  int
    folder_name: str
    seq_name:    str
    frame_idx:   int                 — index within the downsampled (6 fps) sequence

After applying ToTensor (or a Compose that ends with it):
    rgb:       Tensor (3, H, W)      float32 — ImageNet-normalised
    depth:     Tensor (1, H, W)      float32 — clipped & normalised to [0, 1]
    joints:    Tensor (J, 3)         float32 — root-relative (after SubtractRoot)
    intrinsic: Tensor (3, 3)         float32
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import FRAME_STRIDE, ACTIVE_JOINT_INDICES

# Minimum bbox dimension (pixels) — samples below this are skipped via retry
_MIN_BBOX_PX = 32


class BedlamFrameDataset(Dataset):
    """One sample = one person in one frame from one BEDLAM2 sequence.

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
        self._label_cache: dict[str, dict] = {}
        # Per-worker mmap cache for NPY depth files.
        self._depth_mmap: dict[str, np.ndarray | None] = {}
        # LRU fallback cache for legacy NPZ files (bounded to avoid OOM).
        self._depth_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._depth_cache_maxsize = 3

        # Build flat index: list of (label_abs_path, body_idx, frame_idx).
        self.index: list[tuple[str, int, int]] = []
        for seq_rel in seq_paths:
            label_path = os.path.join(data_root, "data", "label", seq_rel)
            try:
                with np.load(label_path, allow_pickle=True) as meta:
                    n_frames = int(meta["n_frames"])
                    n_body = int(meta["joints_cam"].shape[0])
            except Exception as e:
                raise RuntimeError(f"Failed to read label {label_path}: {e}") from e
            for body_idx in range(n_body):
                for frame_idx in range(n_frames):
                    self.index.append((label_path, body_idx, frame_idx))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        # Retry loop for tiny bboxes (guarantees constant batch size)
        max_retries = 10
        for attempt in range(max_retries + 1):
            if attempt > 0:
                idx = random.randint(0, len(self.index) - 1)

            label_path, body_idx, frame_idx = self.index[idx]

            # Lazily populate label cache per worker
            if label_path not in self._label_cache:
                with np.load(label_path, allow_pickle=True) as meta:
                    cache_entry = {
                        "folder_name":      str(meta["folder_name"]),
                        "seq_name":         str(meta["seq_name"]),
                        "intrinsic_matrix": meta["intrinsic_matrix"].astype(np.float32),
                        "joints_cam":       meta["joints_cam"].astype(np.float32),
                    }
                    # Cache joints_2d if available (multi-person bbox computation)
                    if "joints_2d" in meta:
                        cache_entry["joints_2d"] = meta["joints_2d"].astype(np.float32)
                    else:
                        cache_entry["joints_2d"] = None
                    self._label_cache[label_path] = cache_entry
            cached = self._label_cache[label_path]

            folder_name = cached["folder_name"]
            seq_name    = cached["seq_name"]
            intrinsic   = cached["intrinsic_matrix"]

            joints = cached["joints_cam"][body_idx, frame_idx]  # (127, 3) raw

            # --- Bbox from joints_2d (all keypoints, visible + invisible) ---
            bbox = self._compute_bbox(cached, body_idx, frame_idx)

            # Check minimum bbox size
            if bbox is not None:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw < _MIN_BBOX_PX or bh < _MIN_BBOX_PX:
                    continue  # retry with a different sample

            # --- RGB --------------------------------------------------------
            rgb = self._read_frame(folder_name, seq_name, frame_idx, label_path)

            # --- OOB filter: skip if >70% of joints are outside the image --
            if cached["joints_2d"] is not None:
                H_raw, W_raw = rgb.shape[:2]
                kpts = cached["joints_2d"][body_idx, frame_idx]  # (127, 2)
                x_oob = (kpts[:, 0] < 0) | (kpts[:, 0] >= W_raw)
                y_oob = (kpts[:, 1] < 0) | (kpts[:, 1] >= H_raw)
                n_oob = int(np.sum(x_oob | y_oob))
                if n_oob / kpts.shape[0] > 0.70:
                    continue  # retry with a different sample

            # Reduce to active joint subset (body + eyes + hands + non-face surface)
            joints = joints[ACTIVE_JOINT_INDICES]  # (NUM_JOINTS, 3)

            # --- Depth ------------------------------------------------------
            npy_path = os.path.join(
                self.data_root, "data", "depth", "npy",
                folder_name, f"{seq_name}.npy",
            )
            npz_path = os.path.join(
                self.data_root, "data", "depth", "npz",
                folder_name, f"{seq_name}.npz",
            )
            depth = self._read_depth(npy_path, npz_path, frame_idx, label_path)

            # Clamp bbox to actual image dimensions
            H, W = rgb.shape[:2]
            if bbox is not None:
                bbox[0] = max(0.0, min(bbox[0], float(W)))
                bbox[1] = max(0.0, min(bbox[1], float(H)))
                bbox[2] = max(0.0, min(bbox[2], float(W)))
                bbox[3] = max(0.0, min(bbox[3], float(H)))

            sample = {
                "rgb":         rgb,
                "depth":       depth,
                "joints":      joints,
                "intrinsic":   intrinsic,
                "folder_name": folder_name,
                "seq_name":    seq_name,
                "frame_idx":   frame_idx,
                "body_idx":    body_idx,
            }
            if bbox is not None:
                sample["bbox"] = bbox

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

        # All retries exhausted — return last attempt anyway
        return sample  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_bbox(
        self, cached: dict, body_idx: int, frame_idx: int
    ) -> np.ndarray | None:
        """Compute person bounding box from 2D joint projections.

        Uses ALL keypoints (visible and invisible) to mimic a full-body
        detector.  Returns ``(x1, y1, x2, y2)`` float32 array, or None
        if no joints_2d data is available.
        """
        joints_2d = cached["joints_2d"]
        if joints_2d is None:
            return None

        # joints_2d shape: (n_body, n_frames, 127, 2)
        kpts = joints_2d[body_idx, frame_idx]  # (127, 2)

        # Use all keypoints (no visibility filtering)
        x_min = kpts[:, 0].min()
        y_min = kpts[:, 1].min()
        x_max = kpts[:, 0].max()
        y_max = kpts[:, 1].max()

        # Add 10% padding around the tight bbox
        w = x_max - x_min
        h = y_max - y_min
        pad_x = w * 0.1
        pad_y = h * 0.1

        return np.array(
            [x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y],
            dtype=np.float32,
        )

    def _read_frame(
        self,
        folder_name: str,
        seq_name: str,
        frame_idx: int,
        label_path: str,
    ) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB frame."""
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
        """Return a single (H, W) float32 depth frame."""
        # ── Fast path: NPY mmap ──────────────────────────────────────────
        if npy_path not in self._depth_mmap:
            if os.path.exists(npy_path):
                self._depth_mmap[npy_path] = np.load(npy_path, mmap_mode="r")
            else:
                self._depth_mmap[npy_path] = None

        arr = self._depth_mmap[npy_path]
        if arr is not None:
            return arr[frame_idx].astype(np.float32)

        # ── Slow fallback: NPZ with LRU cache ───────────────────────────
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
