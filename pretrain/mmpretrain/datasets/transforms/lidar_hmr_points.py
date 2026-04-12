# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Pipeline helpers for LiDAR-HMR + ViT fusion (human point clouds per sample).

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomHumanPointsPlaceholder(BaseTransform):
    """Fills ``human_points_local`` with random points for integration testing.

    Replace this in real training with transforms that load actual LiDAR-HMR
    coordinates (same frame as upstream ``pose_meshgraphormer`` expects).

    **Added keys:** ``human_points_local`` — ``np.ndarray`` of shape ``(num_points, 3)``.

    Args:
        num_points (int): Point count (LiDAR-HMR defaults to 1024). Defaults to 1024.
        scale (float): Sample in ``[-scale, scale]``. Defaults to 1.0.
    """

    def __init__(self, num_points: int = 1024, scale: float = 1.0) -> None:
        super().__init__()
        self.num_points = num_points
        self.scale = scale

    def transform(self, results: dict) -> dict:
        rng = np.random.default_rng()
        pts = (rng.random((self.num_points, 3), dtype=np.float32) * 2.0 - 1.0)
        pts *= self.scale
        results['human_points_local'] = pts
        return results


@TRANSFORMS.register_module()
class DummySingleClsLabel(BaseTransform):
    """Set a constant ``gt_label`` for datasets without real class IDs (e.g. BEDLAM2 + ViT smoke)."""

    def __init__(self, label: int = 0) -> None:
        super().__init__()
        self.label = int(label)

    def transform(self, results: dict) -> dict:
        results['gt_label'] = self.label
        return results
