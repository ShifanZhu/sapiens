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
        Resize(out_h, out_w),
        ToTensor(),
    ])


def build_val_transform(out_h: int, out_w: int) -> Compose:
    return Compose([
        Resize(out_h, out_w),
        ToTensor(),
    ])
