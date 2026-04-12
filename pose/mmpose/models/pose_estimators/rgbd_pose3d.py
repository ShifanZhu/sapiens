# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""RGBD 3D pose estimator for BEDLAM2.

A minimal pose estimator that:
  1. Passes 4-channel (RGB+D) input through a backbone.
  2. Regresses 3D joint coordinates directly (no heatmap, no codec).
  3. Skips the 2D coordinate-space transform that TopdownPoseEstimator
     applies (which assumes 2D affine-warped inputs).

This class inherits from ``BasePoseEstimator`` and implements only the
``loss()`` and ``predict()`` methods needed by MMEngine's Runner.
"""

from __future__ import annotations

import inspect
from typing import Optional, Union

import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptMultiConfig,
                                  SampleList)
from .base import BasePoseEstimator


def _human_points_from_pose_samples(data_samples: Optional[SampleList]):
    """Stack ``human_points_local`` from each sample when all define it."""
    if not data_samples:
        return None
    tensors = []
    for s in data_samples:
        if s is None or 'human_points_local' not in s:
            return None
        pc = s.get('human_points_local')
        if pc is None:
            return None
        if not isinstance(pc, torch.Tensor):
            pc = torch.as_tensor(pc, dtype=torch.float32)
        if pc.dim() != 2 or pc.size(-1) != 3:
            raise ValueError(
                f'human_points_local must be (N, 3), got {tuple(pc.shape)}')
        tensors.append(pc)
    return torch.stack(tensors, dim=0)


@MODELS.register_module()
class RGBDPose3dEstimator(BasePoseEstimator):
    """Backbone + 3D regression head for RGBD pose estimation.

    Args:
        backbone (dict): Backbone config (e.g. ``SapiensBackboneRGBD``).
        head (dict): Head config (e.g. ``Pose3dRegressionHead``).
        train_cfg (dict, optional): Training config forwarded to the head.
        test_cfg (dict, optional): Test config forwarded to the head.
        data_preprocessor (dict, optional): Data preprocessor config.
        init_cfg: MMEngine init config.
        metainfo (dict, optional): Dataset metainfo override.
    """

    def __init__(
        self,
        backbone: ConfigType,
        head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        metainfo: Optional[dict] = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=None,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo,
        )

    def extract_feat(
        self,
        inputs: Tensor,
        data_samples: Optional[SampleList] = None,
    ):
        """Forward backbone; pass LiDAR point clouds when the backbone supports it."""
        point_clouds = _human_points_from_pose_samples(data_samples)
        if point_clouds is not None:
            point_clouds = point_clouds.to(
                device=inputs.device, dtype=torch.float32, non_blocking=True)
        sig = inspect.signature(self.backbone.forward)
        if point_clouds is not None and 'point_clouds' in sig.parameters:
            return self.backbone(inputs, point_clouds=point_clouds)
        return self.backbone(inputs)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Forward pass + loss computation.

        Args:
            inputs: ``(B, 4, H, W)`` RGBD tensor.
            data_samples: List of ``PoseDataSample`` with GT annotations.

        Returns:
            Tuple of ``(losses_dict, pred_dict)``.
        """
        feats = self.extract_feat(inputs, data_samples)
        losses, _ = self.head.loss(feats, data_samples,
                                   train_cfg=self.train_cfg)
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Inference: extract features, run head, store preds in data_samples.

        Args:
            inputs: ``(B, 4, H, W)`` RGBD tensor.
            data_samples: List of ``PoseDataSample``.

        Returns:
            List of ``PoseDataSample`` with ``pred_instances`` set.
        """
        feats = self.extract_feat(inputs, data_samples)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        # Store predictions directly — no 2D affine back-transform needed
        # because our joints are already in camera 3D space.
        for pred_inst, data_sample in zip(preds, data_samples):
            data_sample.pred_instances = pred_inst

        return data_samples
