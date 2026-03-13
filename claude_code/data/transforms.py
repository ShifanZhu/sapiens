"""Transforms for paired RGB + depth samples.

Each transform operates on the sample dict produced by BedlamFrameDataset:
    {
        "rgb":     np.ndarray (H, W, 3) uint8
        "depth":   np.ndarray (H, W)    float32  [metres, may be None]
        "joints":  np.ndarray (J, 3)    float32  camera-space XYZ
        "intrinsic": np.ndarray (3, 3)  float32
        ...metadata fields...
    }

After ToTensor:
    "rgb":   torch.Tensor (3, H, W)  float32  ImageNet-normalised
    "depth": torch.Tensor (1, H, W)  float32  clipped & normalised to [0, 1]
    "joints": torch.Tensor (J, 3)    float32  unchanged (camera-space metres)
"""

from __future__ import annotations

import math
import random
import numpy as np
import cv2
import torch

from .constants import (
    DEPTH_MAX_METERS,
    FLIP_PAIRS,
    PELVIS_IDX,
    RGB_MEAN,
    RGB_STD,
)


class Resize:
    """Resize RGB and depth to (out_h, out_w).

    Also updates the intrinsic matrix to account for the scale change.
    """

    def __init__(self, out_h: int, out_w: int):
        self.out_h = out_h
        self.out_w = out_w

    def __call__(self, sample: dict) -> dict:
        rgb: np.ndarray = sample["rgb"]          # (H, W, 3)
        orig_h, orig_w = rgb.shape[:2]

        scale_x = self.out_w / orig_w
        scale_y = self.out_h / orig_h

        sample["rgb"] = cv2.resize(
            rgb, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR
        )

        if sample.get("depth") is not None:
            sample["depth"] = cv2.resize(
                sample["depth"],
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,  # nearest to avoid depth bleeding at edges
            )

        # Scale intrinsic matrix
        K: np.ndarray = sample["intrinsic"].copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        sample["intrinsic"] = K

        return sample


class RandomHorizontalFlip:
    """Flip RGB, depth, and joints horizontally with probability p.

    Joint x-coordinates are negated and left/right pairs are swapped.
    The intrinsic matrix cx is updated accordingly.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample

        rgb: np.ndarray = sample["rgb"]
        w = rgb.shape[1]

        sample["rgb"] = np.ascontiguousarray(rgb[:, ::-1])

        if sample.get("depth") is not None:
            sample["depth"] = np.ascontiguousarray(sample["depth"][:, ::-1])

        # Flip joints: negate Y (the left-right axis in BEDLAM2 camera space),
        # swap left-right pairs. X=forward and Z=up are unaffected.
        joints: np.ndarray = sample["joints"].copy()
        joints[:, 1] = -joints[:, 1]
        for left, right in FLIP_PAIRS:
            joints[left], joints[right] = joints[right].copy(), joints[left].copy()
        sample["joints"] = joints

        # Update cx in intrinsic matrix (cx' = W - cx)
        K: np.ndarray = sample["intrinsic"].copy()
        K[0, 2] = w - K[0, 2]
        sample["intrinsic"] = K

        return sample


class ColorJitter:
    """Random brightness / contrast / saturation jitter applied only to RGB."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: dict) -> dict:
        rgb = sample["rgb"].astype(np.float32)

        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness) * 255
            rgb = np.clip(rgb + delta, 0, 255)

        # Contrast
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = rgb.mean()
            rgb = np.clip(mean + factor * (rgb - mean), 0, 255)

        # Saturation (operate in HSV)
        if self.saturation > 0:
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        sample["rgb"] = rgb.astype(np.uint8)
        return sample


class RandomResizedCropRGBD:
    """Random scale-jitter crop applied simultaneously to RGB, depth, and intrinsic K.

    Randomly crops a sub-region (area fraction in ``scale``, aspect ratio in
    ``ratio``) then resizes back to ``(out_h, out_w)``.  The intrinsic matrix
    K is updated so that projected joint coordinates remain geometrically
    consistent.  3D joint labels (camera-space XYZ) are **not** affected.

    K update math (crop origin ``(x0, y0)``, resize scales ``sx, sy``):
        fx' = fx * sx,  fy' = fy * sy
        cx' = (cx - x0) * sx,  cy' = (cy - y0) * sy
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        scale: tuple[float, float] = (0.7, 1.0),
        ratio: tuple[float, float] = (0.55, 0.65),
    ):
        self.out_h = out_h
        self.out_w = out_w
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample: dict) -> dict:
        rgb: np.ndarray = sample["rgb"]   # (H, W, 3)
        h, w = rgb.shape[:2]

        # Sample crop box; fall back to full image after 10 failed attempts
        area = h * w
        y0, x0, crop_h, crop_w = 0, 0, h, w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)          # crop_w / crop_h
            crop_h_f = math.sqrt(target_area / aspect)
            crop_w_f = math.sqrt(target_area * aspect)
            crop_h = int(round(crop_h_f))
            crop_w = int(round(crop_w_f))
            if 0 < crop_h <= h and 0 < crop_w <= w:
                y0 = random.randint(0, h - crop_h)
                x0 = random.randint(0, w - crop_w)
                break

        # Resize scales
        sx = self.out_w / crop_w
        sy = self.out_h / crop_h

        # Crop + resize RGB
        sample["rgb"] = cv2.resize(
            rgb[y0:y0 + crop_h, x0:x0 + crop_w],
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Crop + resize depth (nearest to avoid edge bleeding)
        if sample.get("depth") is not None:
            sample["depth"] = cv2.resize(
                sample["depth"][y0:y0 + crop_h, x0:x0 + crop_w],
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Update intrinsic K
        K: np.ndarray = sample["intrinsic"].copy()
        K[0, 0] = K[0, 0] * sx           # fx' = fx * sx
        K[1, 1] = K[1, 1] * sy           # fy' = fy * sy
        K[0, 2] = (K[0, 2] - x0) * sx   # cx' = (cx - x0) * sx
        K[1, 2] = (K[1, 2] - y0) * sy   # cy' = (cy - y0) * sy
        sample["intrinsic"] = K

        return sample


class NoisyBBox:
    """Add random jitter to a bounding box (training-time only).

    Shifts the bbox center by up to ``pos_jitter`` fraction of its size
    and scales the bbox by a random factor in ``[1-scale_lo, 1+scale_hi]``.
    Result is clamped to image bounds.

    Expects ``sample["bbox"]`` as ``(x1, y1, x2, y2)`` float ndarray and
    ``sample["rgb"]`` to determine image dimensions.
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

    def __call__(self, sample: dict) -> dict:
        if "bbox" not in sample:
            return sample

        bbox = sample["bbox"].copy()  # (x1, y1, x2, y2)
        H, W = sample["rgb"].shape[:2]

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # Position jitter
        cx += random.uniform(-self.pos_jitter, self.pos_jitter) * w
        cy += random.uniform(-self.pos_jitter, self.pos_jitter) * h

        # Scale jitter
        scale = random.uniform(1.0 - self.scale_lo, 1.0 + self.scale_hi)
        w *= scale
        h *= scale

        # Rebuild and clamp
        bbox[0] = max(0.0, cx - w / 2.0)
        bbox[1] = max(0.0, cy - h / 2.0)
        bbox[2] = min(float(W), cx + w / 2.0)
        bbox[3] = min(float(H), cy + h / 2.0)

        # Ensure minimum size after clamping
        if bbox[2] - bbox[0] < 2.0 or bbox[3] - bbox[1] < 2.0:
            return sample  # skip jitter, keep original bbox

        sample["bbox"] = bbox
        return sample


class CropPerson:
    """Crop RGB + depth to the person bounding box and resize.

    The bbox is expanded to match the target aspect ratio (out_h:out_w)
    before cropping.  If the expanded box extends beyond image boundaries,
    ``cv2.copyMakeBorder`` pads the out-of-bounds region.

    Intrinsic K is updated to remain geometrically consistent.

    If no ``bbox`` key is present in the sample, falls back to a simple
    ``Resize`` (backward compatible with single-person full-image mode).
    """

    def __init__(self, out_h: int, out_w: int):
        self.out_h = out_h
        self.out_w = out_w

    def __call__(self, sample: dict) -> dict:
        if "bbox" not in sample:
            # Fall back to plain resize (backward compatible)
            return Resize(self.out_h, self.out_w)(sample)

        bbox = sample["bbox"]  # (x1, y1, x2, y2)
        rgb: np.ndarray = sample["rgb"]  # (H, W, 3)
        H, W = rgb.shape[:2]

        # Bbox center and size
        cx_box = (bbox[0] + bbox[2]) / 2.0
        cy_box = (bbox[1] + bbox[3]) / 2.0
        w_box = max(bbox[2] - bbox[0], 1.0)
        h_box = max(bbox[3] - bbox[1], 1.0)

        # Expand to target aspect ratio (H:W = out_h:out_w)
        target_aspect = self.out_w / self.out_h  # w/h
        box_aspect = w_box / h_box

        if box_aspect < target_aspect:
            # Too tall — expand width
            w_exp = h_box * target_aspect
            h_exp = h_box
        else:
            # Too wide — expand height
            h_exp = w_box / target_aspect
            w_exp = w_box

        # Expanded box coordinates (may go out of bounds)
        x0 = cx_box - w_exp / 2.0
        y0 = cy_box - h_exp / 2.0
        x1 = cx_box + w_exp / 2.0
        y1 = cy_box + h_exp / 2.0

        # Compute padding needed if box extends beyond image
        pad_left = max(0, int(math.ceil(-x0)))
        pad_top = max(0, int(math.ceil(-y0)))
        pad_right = max(0, int(math.ceil(x1 - W)))
        pad_bottom = max(0, int(math.ceil(y1 - H)))

        # Ensure depth matches RGB resolution (pre-converted NPY may differ)
        depth = sample.get("depth")
        if depth is not None:
            dH, dW = depth.shape[:2]
            if dH != H or dW != W:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            rgb = cv2.copyMakeBorder(
                rgb, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            if depth is not None:
                depth = cv2.copyMakeBorder(
                    depth, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=0.0,
                )
            # Shift coordinates to account for padding
            x0 += pad_left
            y0 += pad_top
            x1 += pad_left
            y1 += pad_top

        # Integer crop coordinates
        ix0, iy0 = int(round(x0)), int(round(y0))
        ix1, iy1 = int(round(x1)), int(round(y1))
        crop_w = max(ix1 - ix0, 1)
        crop_h = max(iy1 - iy0, 1)

        sx = self.out_w / crop_w
        sy = self.out_h / crop_h

        # Crop + resize RGB
        sample["rgb"] = cv2.resize(
            rgb[iy0:iy1, ix0:ix1],
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Crop + resize depth
        if depth is not None:
            sample["depth"] = cv2.resize(
                depth[iy0:iy1, ix0:ix1],
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Update intrinsic K — use original (pre-padding) crop origin
        orig_x0 = ix0 - pad_left
        orig_y0 = iy0 - pad_top
        K: np.ndarray = sample["intrinsic"].copy()
        K[0, 0] = K[0, 0] * sx               # fx' = fx * sx
        K[1, 1] = K[1, 1] * sy               # fy' = fy * sy
        K[0, 2] = (K[0, 2] - orig_x0) * sx   # cx' = (cx - x0) * sx
        K[1, 2] = (K[1, 2] - orig_y0) * sy   # cy' = (cy - y0) * sy
        sample["intrinsic"] = K

        return sample


class SubtractRoot:
    """Subtract pelvis (root joint) from all joints.

    Stores:
        ``pelvis_abs``   — (3,) original pelvis XYZ in camera space
        ``pelvis_depth`` — (1,) pelvis forward distance (X coordinate) in metres
        ``pelvis_uv``    — (2,) pelvis 2D position normalized to [-1, 1]
                           (0, 0) = crop center; computed from crop pixels as
                           ``u_norm = u_px / W * 2 - 1``, ``v_norm = v_px / H * 2 - 1``

    After this transform, ``joints`` are root-relative (pelvis = origin).

    Must run **after** CropPerson so that ``intrinsic`` is the crop K
    and ``rgb`` has the crop dimensions.
    """

    def __call__(self, sample: dict) -> dict:
        joints: np.ndarray = sample["joints"]  # (J, 3)
        pelvis = joints[PELVIS_IDX: PELVIS_IDX + 1].copy()  # (1, 3)
        pelvis_3d = pelvis.squeeze(0)                        # (3,)
        sample["pelvis_abs"] = pelvis_3d
        sample["joints"] = joints - pelvis                   # (J, 3) root-relative

        # Pelvis depth = forward distance (X coordinate)
        sample["pelvis_depth"] = np.array([pelvis_3d[0]], dtype=np.float32)  # (1,)

        # Pelvis UV = project pelvis through crop K, then normalize to [-1, 1]
        # BEDLAM2: u = fx*(-Y/X) + cx,  v = fy*(-Z/X) + cy
        K = sample["intrinsic"]  # (3, 3) — crop K after CropPerson
        X, Y, Z = pelvis_3d[0], pelvis_3d[1], pelvis_3d[2]
        if X > 0.01:
            u_px = K[0, 0] * (-Y / X) + K[0, 2]
            v_px = K[1, 1] * (-Z / X) + K[1, 2]
        else:
            # Degenerate — place at image center
            u_px = K[0, 2]
            v_px = K[1, 2]

        # Normalize: pixel [0, W) → [-1, 1],  pixel [0, H) → [-1, 1]
        crop_h, crop_w = sample["rgb"].shape[:2]  # after CropPerson
        u_norm = u_px / crop_w * 2.0 - 1.0
        v_norm = v_px / crop_h * 2.0 - 1.0
        sample["pelvis_uv"] = np.array([u_norm, v_norm], dtype=np.float32)  # (2,)

        return sample


class ToTensor:
    """Convert numpy arrays to torch tensors and apply normalisation.

    RGB:   (H,W,3) uint8  -> (3,H,W) float32, ImageNet mean/std normalised
    Depth: (H,W)   float32 -> (1,H,W) float32, clipped to [0, DEPTH_MAX] / DEPTH_MAX
    """

    def __init__(
        self,
        rgb_mean: tuple[float, ...] = RGB_MEAN,
        rgb_std: tuple[float, ...] = RGB_STD,
        depth_max: float = DEPTH_MAX_METERS,
    ):
        self.mean = np.array(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array(rgb_std,  dtype=np.float32).reshape(1, 1, 3)
        self.depth_max = depth_max

    def __call__(self, sample: dict) -> dict:
        rgb = sample["rgb"].astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        sample["rgb"] = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

        if sample.get("depth") is not None:
            depth = np.clip(sample["depth"], 0.0, self.depth_max) / self.depth_max
            sample["depth"] = torch.from_numpy(depth[np.newaxis])  # (1, H, W)

        sample["joints"]    = torch.from_numpy(sample["joints"])
        sample["intrinsic"] = torch.from_numpy(sample["intrinsic"])
        if "pelvis_abs" in sample:
            sample["pelvis_abs"] = torch.from_numpy(sample["pelvis_abs"])
        if "pelvis_depth" in sample:
            sample["pelvis_depth"] = torch.from_numpy(sample["pelvis_depth"])
        if "pelvis_uv" in sample:
            sample["pelvis_uv"] = torch.from_numpy(sample["pelvis_uv"])
        return sample


class Compose:
    """Chain multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Ready-to-use transform presets
# ---------------------------------------------------------------------------

def build_train_transform(out_h: int, out_w: int, scale_jitter: bool = True) -> Compose:
    return Compose([
        NoisyBBox(),
        CropPerson(out_h, out_w),
        SubtractRoot(),
        ToTensor(),
    ])


def build_val_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        CropPerson(out_h, out_w),
        SubtractRoot(),
        ToTensor(),
    ])
