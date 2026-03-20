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


def _safe_get(obj, key, default=None):
    """Get attribute/key from dict or object (PoseDataSample)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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

    default_prefix = ''

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
        """Accumulate one batch of predictions and ground truth.

        Note: MMEngine's ``Evaluator.process`` calls ``to_dict()`` on each
        data sample before passing it here, which flattens metainfo fields
        (e.g. ``K``, ``img_shape``) to the top level of the dict.
        """
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

            # Pelvis predictions + GT for absolute MPJPE.
            # K is at the top level (to_dict() flattens metainfo).
            pred_inst = data_sample['pred_instances']
            gt_labels = _safe_get(data_sample, 'gt_instance_labels', {})

            result = {
                'pred': pred_coords,
                'gt': gt_coords,
            }

            # Collect pelvis data if available
            if ('pelvis_depth' in pred_inst and 'pelvis_uv' in pred_inst
                    and 'K' in data_sample):
                result['pred_pelvis_depth'] = np.asarray(
                    pred_inst['pelvis_depth']).ravel()[0]
                pred_uv = np.asarray(pred_inst['pelvis_uv']).ravel()
                result['pred_pelvis_uv'] = pred_uv

                def _to_np(v):
                    if hasattr(v, 'detach'):
                        v = v.detach().cpu()
                    return np.asarray(v)

                result['gt_pelvis_depth'] = _to_np(
                    gt_labels['pelvis_depth']).ravel()[0]
                gt_uv = _to_np(gt_labels['pelvis_uv']).ravel()
                result['gt_pelvis_uv'] = gt_uv

                result['K'] = np.asarray(data_sample['K'], dtype=np.float32)
                img_shape = data_sample.get('img_shape', (640, 384))
                result['crop_h'] = int(img_shape[0])
                result['crop_w'] = int(img_shape[1])

            self.results.append(result)

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
            'mpjpe/rel/val': mpjpe_mm(pred_all, gt_all),
            'mpjpe/body/val': mpjpe_mm(pred_all[:, _BODY_INDICES],
                                       gt_all[:, _BODY_INDICES]),
            'mpjpe/hand/val': mpjpe_mm(pred_all[:, _HAND_INDICES],
                                       gt_all[:, _HAND_INDICES]),
        }

        # Absolute MPJPE using predicted pelvis
        if 'K' in results[0]:
            abs_pred_list = []
            abs_gt_list = []
            for r in results:
                K = r['K']
                ch, cw = r['crop_h'], r['crop_w']
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])

                def _recover(depth, uv):
                    X = float(depth)
                    u_px = (float(uv[0]) + 1.0) / 2.0 * cw
                    v_px = (float(uv[1]) + 1.0) / 2.0 * ch
                    Y = -(u_px - cx) * X / fx
                    Z = -(v_px - cy) * X / fy
                    return np.array([X, Y, Z], dtype=np.float32)

                pred_pelvis = _recover(
                    r['pred_pelvis_depth'], r['pred_pelvis_uv'])
                gt_pelvis = _recover(
                    r['gt_pelvis_depth'], r['gt_pelvis_uv'])

                abs_pred_list.append(r['pred'] + pred_pelvis[np.newaxis, :])
                abs_gt_list.append(r['gt'] + gt_pelvis[np.newaxis, :])

            abs_pred = np.stack(abs_pred_list)
            abs_gt = np.stack(abs_gt_list)
            metrics['mpjpe/abs/val'] = mpjpe_mm(abs_pred, abs_gt)

        for k, v in metrics.items():
            logger.info(f'  {k}: {v:.2f} mm')

        return metrics
