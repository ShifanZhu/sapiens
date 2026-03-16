# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Transforms for BEDLAM2 RGBD pose estimation.

These transforms operate on a ``results`` dict produced by
``Bedlam2Dataset.prepare_data``.  They port the standalone transforms from
``claude_code/data/transforms.py`` to the mmpose ``BaseTransform`` interface.

Transform chain (train):
    LoadBedlamLabels → NoisyBBoxTransform → CropPersonRGBD →
    SubtractRootJoint → PackBedlamInputs

Transform chain (val):
    LoadBedlamLabels → CropPersonRGBD → SubtractRootJoint → PackBedlamInputs

The ``results`` dict keys used across the chain:

  Input (from dataset.load_data_list):
    label_path, body_idx, frame_idx, img_path,
    depth_npy_path, depth_npz_path, folder_name, seq_name

  Added by LoadBedlamLabels:
    img          (H, W, 3) uint8   RGB
    depth        (H, W)    float32 depth in metres
    joints_cam   (70, 3)   float32 camera-space XYZ in metres
    K            (3, 3)    float32 camera intrinsic matrix
    bbox         (4,)      float32 (x1, y1, x2, y2)

  Added by SubtractRootJoint:
    pelvis_abs   (3,)      float32 original pelvis XYZ
    pelvis_depth (1,)      float32 pelvis forward distance (X)
    pelvis_uv    (2,)      float32 pelvis pixel position normalised to [-1, 1]

  Produced by PackBedlamInputs:
    inputs               Tensor (4, H, W)   stacked RGBD
    data_samples         PoseDataSample with gt_instances and gt_instance_labels
"""

from __future__ import annotations

import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData

from mmpose.registry import TRANSFORMS
from mmpose.structures import PoseDataSample

# ── Constants ────────────────────────────────────────────────────────────────
# Active joint index for the pelvis (root joint)
_PELVIS_IDX = 0

# Active joint subset indices (body+eyes+hands+surface, 70 total)
# Matches claude_code/data/constants.py::ACTIVE_JOINT_INDICES
_ACTIVE_JOINT_INDICES = (
    list(range(0, 22))      # body (pelvis → right_wrist)
    + [23, 24]              # eyes
    + list(range(25, 55))   # hands
    + list(range(60, 76))   # non-face surface
)

# Flip pairs in active-joint index space (ported from constants.py::FLIP_PAIRS)
_FLIP_PAIRS = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
    (16, 17), (18, 19), (20, 21),
    (23, 24),   # eyes (note: uses original-space indices 23/24 for legacy compat)
    (25, 40), (26, 41), (27, 42),
    (28, 43), (29, 44), (30, 45),
    (31, 46), (32, 47), (33, 48),
    (34, 49), (35, 50), (36, 51),
    (37, 52), (38, 53), (39, 54),
)

_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_DEPTH_MAX_METERS = 20.0
_MIN_BBOX_PX = 32


# ── Per-worker caches (dict keyed by file path) ───────────────────────────────
# These are global module-level dicts; each worker process gets its own copy
# after fork/spawn, giving per-worker isolation without explicit worker_init_fn.
_label_cache: dict[str, dict] = {}
_depth_mmap: dict[str, Optional[np.ndarray]] = {}
_depth_npz_cache: OrderedDict[str, Optional[np.ndarray]] = OrderedDict()
_DEPTH_NPZ_MAXSIZE = 3


# ── Helper: depth loading ─────────────────────────────────────────────────────

def _read_depth(npy_path: str, npz_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Return a single (H, W) float32 depth frame using NPY mmap or NPZ LRU."""
    global _depth_mmap, _depth_npz_cache

    # Fast path: NPY memory-mapped file
    if npy_path not in _depth_mmap:
        _depth_mmap[npy_path] = (
            np.load(npy_path, mmap_mode='r') if os.path.exists(npy_path) else None
        )
    arr = _depth_mmap[npy_path]
    if arr is not None:
        return arr[frame_idx].astype(np.float32)

    # Slow path: NPZ with LRU cache
    if npz_path not in _depth_npz_cache:
        if not os.path.exists(npz_path):
            val = None
        else:
            with np.load(npz_path) as f:
                val = f['depth'].astype(np.float32)
        if len(_depth_npz_cache) >= _DEPTH_NPZ_MAXSIZE:
            _depth_npz_cache.popitem(last=False)
        _depth_npz_cache[npz_path] = val
    else:
        _depth_npz_cache.move_to_end(npz_path)

    arr = _depth_npz_cache[npz_path]
    return None if arr is None else arr[frame_idx]


# ── Transform 1: LoadBedlamLabels ─────────────────────────────────────────────

@TRANSFORMS.register_module()
class LoadBedlamLabels(BaseTransform):
    """Load RGB frame, depth map, 3D joints, intrinsic K, and bounding box.

    Reads from the BEDLAM2 data root using paths stored in the ``results``
    dict by ``Bedlam2Dataset.load_data_list``.

    Filtering behaviour (returns ``None`` to trigger MMEngine retry):
      - OOB: more than 70% of joints outside the full image bounds.
      - Tiny bbox: width or height < 32 pixels.

    Args:
        depth_required (bool): Raise if depth is missing. Default True.
        filter_invalid (bool): Return None (triggering MMEngine retry) for
            OOB or tiny-bbox samples. Set False for val/test to avoid the
            "Test time pipeline should not get None" error. Default True.
    """

    def __init__(self, depth_required: bool = True,
                 filter_invalid: bool = True):
        self.depth_required = depth_required
        self.filter_invalid = filter_invalid

    def transform(self, results: dict) -> Optional[dict]:
        global _label_cache

        label_path = results['label_path']
        body_idx = results['body_idx']
        frame_idx = results['frame_idx']

        # ── Label cache (lazy, per-worker) ────────────────────────────────
        if label_path not in _label_cache:
            with np.load(label_path, allow_pickle=True) as meta:
                entry = {
                    'folder_name': str(meta['folder_name']),
                    'seq_name': str(meta['seq_name']),
                    'intrinsic_matrix': meta['intrinsic_matrix'].astype(np.float32),
                    'joints_cam': meta['joints_cam'].astype(np.float32),
                    'joints_2d': (meta['joints_2d'].astype(np.float32)
                                  if 'joints_2d' in meta else None),
                }
            _label_cache[label_path] = entry
        cached = _label_cache[label_path]

        folder_name = cached['folder_name']
        seq_name = cached['seq_name']
        intrinsic = cached['intrinsic_matrix']     # (3, 3)
        joints_raw = cached['joints_cam'][body_idx, frame_idx]  # (127, 3)

        # ── Bounding box from joints_2d ────────────────────────────────────
        joints_2d = cached['joints_2d']
        bbox = None
        if joints_2d is not None:
            kpts = joints_2d[body_idx, frame_idx]   # (127, 2)
            x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
            x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()
            w = x_max - x_min
            h = y_max - y_min
            pad_x, pad_y = w * 0.1, h * 0.1
            bbox = np.array(
                [x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y],
                dtype=np.float32,
            )
            # Tiny bbox check
            if self.filter_invalid:
                if ((bbox[2] - bbox[0]) < _MIN_BBOX_PX
                        or (bbox[3] - bbox[1]) < _MIN_BBOX_PX):
                    return None   # trigger retry

        # ── RGB frame ─────────────────────────────────────────────────────
        img_path = results['img_path']
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f'Failed to decode frame: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── OOB filter ────────────────────────────────────────────────────
        if joints_2d is not None and self.filter_invalid:
            H_raw, W_raw = img.shape[:2]
            kpts = joints_2d[body_idx, frame_idx]
            x_oob = (kpts[:, 0] < 0) | (kpts[:, 0] >= W_raw)
            y_oob = (kpts[:, 1] < 0) | (kpts[:, 1] >= H_raw)
            if float(np.sum(x_oob | y_oob)) / kpts.shape[0] > 0.70:
                return None   # trigger retry

        # ── Reduce to active joints ────────────────────────────────────────
        joints = joints_raw[_ACTIVE_JOINT_INDICES]   # (70, 3)

        # ── Depth ─────────────────────────────────────────────────────────
        npy_path = results['depth_npy_path']
        npz_path = results['depth_npz_path']
        depth = _read_depth(npy_path, npz_path, frame_idx)
        if depth is None and self.depth_required:
            raise FileNotFoundError(
                f'Depth not found for frame {frame_idx} in {label_path}')

        # ── Clamp bbox to image bounds ─────────────────────────────────────
        if bbox is not None:
            H, W = img.shape[:2]
            bbox[0] = max(0.0, min(bbox[0], float(W)))
            bbox[1] = max(0.0, min(bbox[1], float(H)))
            bbox[2] = max(0.0, min(bbox[2], float(W)))
            bbox[3] = max(0.0, min(bbox[3], float(H)))

        results['img'] = img
        results['depth'] = depth
        results['joints_cam'] = joints
        results['K'] = intrinsic
        if bbox is not None:
            results['bbox'] = bbox
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


# ── Transform 2: NoisyBBoxTransform ──────────────────────────────────────────

@TRANSFORMS.register_module()
class NoisyBBoxTransform(BaseTransform):
    """Add random jitter to a bounding box (training augmentation only).

    Args:
        pos_jitter (float): Position jitter as fraction of bbox size.
        scale_lo (float): Lower bound of scale jitter.
        scale_hi (float): Upper bound of scale jitter.
    """

    def __init__(
        self,
        pos_jitter: float = 0.1,
        scale_lo: float = 0.15,
        scale_hi: float = 0.15,
    ):
        self.pos_jitter = pos_jitter
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def transform(self, results: dict) -> dict:
        if 'bbox' not in results:
            return results

        bbox = results['bbox'].copy()
        img_h, img_w = results['img'].shape[:2]

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        cx += random.uniform(-self.pos_jitter, self.pos_jitter) * w
        cy += random.uniform(-self.pos_jitter, self.pos_jitter) * h

        scale = random.uniform(1.0 - self.scale_lo, 1.0 + self.scale_hi)
        w *= scale
        h *= scale

        bbox[0] = max(0.0, cx - w / 2.0)
        bbox[1] = max(0.0, cy - h / 2.0)
        bbox[2] = min(float(img_w), cx + w / 2.0)
        bbox[3] = min(float(img_h), cy + h / 2.0)

        if bbox[2] - bbox[0] < 2.0 or bbox[3] - bbox[1] < 2.0:
            return results  # skip jitter, keep original bbox

        results['bbox'] = bbox
        return results


# ── Transform 3: CropPersonRGBD ──────────────────────────────────────────────

@TRANSFORMS.register_module()
class CropPersonRGBD(BaseTransform):
    """Crop RGB + depth to the person bounding box and resize.

    The bbox is expanded to match the target aspect ratio before cropping.
    Out-of-bounds regions are zero-padded.  The intrinsic K is updated.
    Falls back to a plain resize if no ``bbox`` key is present.

    Args:
        out_h (int): Output height in pixels.
        out_w (int): Output width in pixels.
    """

    def __init__(self, out_h: int = 640, out_w: int = 384):
        self.out_h = out_h
        self.out_w = out_w

    def transform(self, results: dict) -> dict:
        img: np.ndarray = results['img']    # (H, W, 3)
        depth = results.get('depth')        # (H, W) or None
        K: np.ndarray = results['K']        # (3, 3)

        if 'bbox' not in results:
            # Fallback: plain resize
            H, W = img.shape[:2]
            sx, sy = self.out_w / W, self.out_h / H
            results['img'] = cv2.resize(img, (self.out_w, self.out_h),
                                        interpolation=cv2.INTER_LINEAR)
            if depth is not None:
                results['depth'] = cv2.resize(depth, (self.out_w, self.out_h),
                                              interpolation=cv2.INTER_NEAREST)
            K = K.copy()
            K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
            results['K'] = K
            results['img_shape'] = (self.out_h, self.out_w)
            return results

        bbox = results['bbox']
        H, W = img.shape[:2]

        cx_box = (bbox[0] + bbox[2]) / 2.0
        cy_box = (bbox[1] + bbox[3]) / 2.0
        w_box = max(bbox[2] - bbox[0], 1.0)
        h_box = max(bbox[3] - bbox[1], 1.0)

        # Expand to target aspect ratio
        target_aspect = self.out_w / self.out_h   # w/h
        box_aspect = w_box / h_box
        if box_aspect < target_aspect:
            w_exp = h_box * target_aspect
            h_exp = h_box
        else:
            h_exp = w_box / target_aspect
            w_exp = w_box

        x0 = cx_box - w_exp / 2.0
        y0 = cy_box - h_exp / 2.0
        x1 = cx_box + w_exp / 2.0
        y1 = cy_box + h_exp / 2.0

        pad_left = max(0, int(math.ceil(-x0)))
        pad_top = max(0, int(math.ceil(-y0)))
        pad_right = max(0, int(math.ceil(x1 - W)))
        pad_bottom = max(0, int(math.ceil(y1 - H)))

        # Align depth resolution with RGB if pre-converted NPY differs
        if depth is not None:
            dH, dW = depth.shape[:2]
            if dH != H or dW != W:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if depth is not None:
                depth = cv2.copyMakeBorder(depth, pad_top, pad_bottom, pad_left,
                                           pad_right, cv2.BORDER_CONSTANT, value=0.0)
            x0 += pad_left; y0 += pad_top; x1 += pad_left; y1 += pad_top

        ix0, iy0 = int(round(x0)), int(round(y0))
        ix1, iy1 = int(round(x1)), int(round(y1))
        crop_w = max(ix1 - ix0, 1)
        crop_h = max(iy1 - iy0, 1)
        sx = self.out_w / crop_w
        sy = self.out_h / crop_h

        results['img'] = cv2.resize(img[iy0:iy1, ix0:ix1],
                                    (self.out_w, self.out_h),
                                    interpolation=cv2.INTER_LINEAR)
        if depth is not None:
            results['depth'] = cv2.resize(depth[iy0:iy1, ix0:ix1],
                                          (self.out_w, self.out_h),
                                          interpolation=cv2.INTER_NEAREST)

        # Update K using original (pre-padding) crop origin
        orig_x0 = ix0 - pad_left
        orig_y0 = iy0 - pad_top
        K = K.copy()
        K[0, 0] = K[0, 0] * sx
        K[1, 1] = K[1, 1] * sy
        K[0, 2] = (K[0, 2] - orig_x0) * sx
        K[1, 2] = (K[1, 2] - orig_y0) * sy
        results['K'] = K
        results['img_shape'] = (self.out_h, self.out_w)

        return results


# ── Transform 4: SubtractRootJoint ───────────────────────────────────────────

@TRANSFORMS.register_module()
class SubtractRootJoint(BaseTransform):
    """Subtract pelvis (root joint) from all joints.

    Must run **after** ``CropPersonRGBD`` so that ``K`` is the crop
    intrinsic and ``img`` has the crop dimensions.

    Stores:
      ``pelvis_abs``   (3,) original pelvis XYZ in camera space
      ``pelvis_depth`` (1,) pelvis forward distance (X coordinate) in metres
      ``pelvis_uv``    (2,) pelvis 2D position normalised to [-1, 1]
                            (0, 0) = crop centre; ±1 = crop edges
    After this transform, ``joints_cam`` are root-relative (pelvis = origin).
    """

    def transform(self, results: dict) -> dict:
        joints: np.ndarray = results['joints_cam']   # (70, 3)
        pelvis_3d = joints[_PELVIS_IDX].copy()       # (3,)

        results['pelvis_abs'] = pelvis_3d
        results['joints_cam'] = joints - pelvis_3d   # root-relative

        # Pelvis depth = forward distance (X coordinate in BEDLAM2 camera space)
        results['pelvis_depth'] = np.array([pelvis_3d[0]], dtype=np.float32)

        # Project pelvis through crop K and normalise to [-1, 1]
        # BEDLAM2 camera: u = fx*(-Y/X) + cx,  v = fy*(-Z/X) + cy
        K = results['K']
        X, Y, Z = float(pelvis_3d[0]), float(pelvis_3d[1]), float(pelvis_3d[2])
        if X > 0.01:
            u_px = K[0, 0] * (-Y / X) + K[0, 2]
            v_px = K[1, 1] * (-Z / X) + K[1, 2]
        else:
            u_px, v_px = K[0, 2], K[1, 2]   # degenerate: place at image centre

        crop_h, crop_w = results['img'].shape[:2]
        u_norm = float(u_px) / crop_w * 2.0 - 1.0
        v_norm = float(v_px) / crop_h * 2.0 - 1.0
        results['pelvis_uv'] = np.array([u_norm, v_norm], dtype=np.float32)

        return results


# ── Transform 5: PackBedlamInputs ────────────────────────────────────────────

@TRANSFORMS.register_module()
class PackBedlamInputs(BaseTransform):
    """Pack results into mmpose's ``PoseDataSample`` format.

    Applies ImageNet normalisation to RGB and clips/normalises depth to
    ``[0, 1]``, then concatenates into a 4-channel tensor ``(4, H, W)``.

    GT is stored as:
      - ``data_sample.gt_instances.lifting_target``: ``(1, 70, 3)`` joints
      - ``data_sample.gt_instance_labels.pelvis_depth``: ``(1,)``
      - ``data_sample.gt_instance_labels.pelvis_uv``: ``(2,)``

    Args:
        meta_keys: Tuple of keys from ``results`` to forward as metainfo.
    """

    _DEFAULT_META_KEYS = (
        'img_path', 'depth_npy_path', 'folder_name', 'seq_name',
        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K',
    )

    def __init__(self, meta_keys: tuple = _DEFAULT_META_KEYS):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        # ── RGB normalisation ─────────────────────────────────────────────
        rgb = results['img'].astype(np.float32) / 255.0
        rgb = (rgb - _RGB_MEAN) / _RGB_STD             # (H, W, 3)
        rgb_t = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1)))   # (3, H, W)

        # ── Depth normalisation ────────────────────────────────────────────
        depth = results.get('depth')
        if depth is not None:
            depth_n = np.clip(depth, 0.0, _DEPTH_MAX_METERS) / _DEPTH_MAX_METERS
            depth_t = torch.from_numpy(depth_n[np.newaxis].astype(np.float32))  # (1,H,W)
        else:
            H, W = results['img'].shape[:2]
            depth_t = torch.zeros(1, H, W, dtype=torch.float32)

        # ── Concatenate RGBD ──────────────────────────────────────────────
        inputs = torch.cat([rgb_t, depth_t], dim=0)    # (4, H, W)

        # ── Ground truth ──────────────────────────────────────────────────
        joints_np = results['joints_cam'].astype(np.float32)   # (70, 3)
        # lifting_target: (1, 70, 3) to match mmpose convention (N_inst, K, D)
        lifting_target = torch.from_numpy(joints_np).unsqueeze(0)

        pelvis_depth = torch.from_numpy(results['pelvis_depth'])          # (1,)
        pelvis_uv = torch.from_numpy(results['pelvis_uv']).unsqueeze(0)  # (1,2)

        # ── Build PoseDataSample ──────────────────────────────────────────
        data_sample = PoseDataSample()

        gt_instances = InstanceData()
        gt_instances.lifting_target = lifting_target
        data_sample.gt_instances = gt_instances

        gt_instance_labels = InstanceData()
        gt_instance_labels.pelvis_depth = pelvis_depth
        gt_instance_labels.pelvis_uv = pelvis_uv
        data_sample.gt_instance_labels = gt_instance_labels

        # Metainfo
        meta = {}
        for key in self.meta_keys:
            if key in results:
                meta[key] = results[key]
        data_sample.set_metainfo(meta)

        packed = {
            'inputs': inputs,
            'data_samples': data_sample,
        }

        return packed
