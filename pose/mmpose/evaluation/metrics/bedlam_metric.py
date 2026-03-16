# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""MPJPE evaluation metric for BEDLAM2 3D pose estimation.

Reports per-group MPJPE:
  - ``mpjpe/all``:  all 70 active joints
  - ``mpjpe/body``: body joints (active indices 0-21)
  - ``mpjpe/hand``: hand joints (active indices 24-53)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS

# Active joint group slices (body, hand)
_BODY_INDICES = list(range(0, 22))          # 22 body joints
_HAND_INDICES = list(range(24, 54))         # 30 hand joints (left+right)


@METRICS.register_module()
class BedlamMPJPEMetric(BaseMetric):
    """MPJPE evaluation metric for BEDLAM2 with per-group breakdown.

    Reads predictions from ``data_sample['pred_instances']['keypoints']``
    (shape ``(1, 70, 3)``) and ground truth from
    ``data_sample['gt_instances']['lifting_target']``
    (shape ``(1, 70, 3)``), both in metres.

    Args:
        collect_device (str): Device for gathering results across ranks.
        prefix (str, optional): Metric name prefix.
    """

    default_prefix = 'bedlam'

    def __init__(
        self,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(
        self,
        data_batch: Sequence[dict],
        data_samples: Sequence[dict],
    ) -> None:
        """Accumulate one batch of predictions and ground truth."""
        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints']
            # pred_coords: (1, 70, 3) or (70, 3)
            if pred_coords.ndim == 3:
                pred_coords = pred_coords[0]   # → (70, 3)

            gt_coords = data_sample['gt_instances']['lifting_target']
            # gt_coords: (1, 70, 3) or (70, 3)
            if hasattr(gt_coords, 'numpy'):
                gt_coords = gt_coords.detach().cpu().numpy()
            if gt_coords.ndim == 3:
                gt_coords = gt_coords[0]       # → (70, 3)

            self.results.append({
                'pred': pred_coords,
                'gt': gt_coords,
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute per-group MPJPE from accumulated results."""
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating BedlamMPJPE...')

        pred_all = np.stack([r['pred'] for r in results])   # (N, 70, 3)
        gt_all = np.stack([r['gt'] for r in results])       # (N, 70, 3)

        def mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
            """Mean per-joint position error in millimetres."""
            return float(np.linalg.norm(pred - gt, axis=-1).mean()) * 1000.0

        metrics = {
            'mpjpe/all': mpjpe_mm(pred_all, gt_all),
            'mpjpe/body': mpjpe_mm(pred_all[:, _BODY_INDICES],
                                   gt_all[:, _BODY_INDICES]),
            'mpjpe/hand': mpjpe_mm(pred_all[:, _HAND_INDICES],
                                   gt_all[:, _HAND_INDICES]),
        }

        for k, v in metrics.items():
            logger.info(f'  {k}: {v:.2f} mm')

        return metrics
