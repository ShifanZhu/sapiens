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

from typing import Optional, Union

import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptMultiConfig,
                                  SampleList)
from .base import BasePoseEstimator


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

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Forward pass + loss computation.

        Args:
            inputs: ``(B, 4, H, W)`` RGBD tensor.
            data_samples: List of ``PoseDataSample`` with GT annotations.

        Returns:
            Tuple of ``(losses_dict, pred_dict)``.
        """
        feats = self.extract_feat(inputs)    # (feat,) tuple
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
        feats = self.extract_feat(inputs)
        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        # Store predictions directly — no 2D affine back-transform needed
        # because our joints are already in camera 3D space.
        for pred_inst, data_sample in zip(preds, data_samples):
            data_sample.pred_instances = pred_inst

        return data_samples
