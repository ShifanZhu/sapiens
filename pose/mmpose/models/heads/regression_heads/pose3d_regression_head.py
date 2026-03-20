# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""3D pose regression head for Sapiens-RGBD.

Takes a ViT feature map ``(B, C, H', W')`` and regresses:
  - ``(B, num_joints, 3)`` root-relative joint XYZ in metres
  - ``(B, 1)``             pelvis depth (forward distance) in metres
  - ``(B, 2)``             pelvis 2D position in crop pixels, normalised to [-1, 1]

Architecture::

    feats[-1]  (B, C, H', W')
        → AdaptiveAvgPool2d(1)          (B, C, 1, 1)
        → flatten                        (B, C)
        → Linear(C, hidden)              (B, hidden)
        → LayerNorm + GELU + Dropout
        ├→ Linear(hidden, num_joints*3)  → joints_rel  (B, num_joints, 3)
        ├→ Linear(hidden, 1)             → pelvis_depth (B, 1)
        └→ Linear(hidden, 2)             → pelvis_uv    (B, 2)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                  Predictions)
from ..base_head import BaseHead
from .pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs


@MODELS.register_module()
class Pose3dRegressionHead(BaseHead):
    """MLP regression head for 3D joint prediction and pelvis localisation.

    Args:
        in_channels (int): Embedding dimension from the backbone (e.g. 1024).
        num_joints (int): Number of output joints (70 for BEDLAM2 active set).
        hidden_dim (int): Width of the hidden MLP layer.
        dropout (float): Dropout probability applied before the output branches.
        loss_joints (ConfigType): Config for the joint coordinate loss.
            Defaults to SmoothL1Loss with beta=0.05.
        loss_depth (ConfigType): Config for the pelvis depth loss.
        loss_uv (ConfigType): Config for the pelvis 2D position loss.
        loss_weight_depth (float): Weight for the depth loss term.
        loss_weight_uv (float): Weight for the UV loss term.
        init_cfg: Standard MMEngine init config.
    """

    def __init__(
        self,
        in_channels: int,
        num_joints: int = 70,
        hidden_dim: int = 2048,
        dropout: float = 0.2,
        loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                       beta=0.05, loss_weight=1.0),
        loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                      beta=0.05, loss_weight=1.0),
        loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
        loss_weight_depth: float = 1.0,
        loss_weight_uv: float = 1.0,
        init_cfg: OptConfigType = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.loss_weight_depth = loss_weight_depth
        self.loss_weight_uv = loss_weight_uv

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific output branches
        self.joints_out = nn.Linear(hidden_dim, num_joints * 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

        self._init_head_weights()

    @property
    def default_init_cfg(self):
        # Weight initialisation is handled manually in _init_head_weights()
        return []

    def _init_head_weights(self) -> None:
        for m in [self.trunk, self.joints_out, self.depth_out, self.uv_out]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def forward(
        self, feats: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            feats: Tuple of feature tensors from the backbone.
                   ``feats[-1]`` is ``(B, C, H', W')``.

        Returns:
            Dict with keys:
                ``joints``:       ``(B, num_joints, 3)`` root-relative (metres)
                ``pelvis_depth``: ``(B, 1)`` pelvis forward distance (metres)
                ``pelvis_uv``:    ``(B, 2)`` pelvis (u, v) normalised to [-1, 1]
        """
        feat = feats[-1]                          # (B, C, H', W')
        x = self.pool(feat).flatten(1)            # (B, C)
        h = self.trunk(x)                         # (B, hidden)

        joints = self.joints_out(h).view(h.size(0), self.num_joints, 3)
        pelvis_depth = self.depth_out(h)          # (B, 1)
        pelvis_uv = self.uv_out(h)               # (B, 2)

        return {
            'joints': joints,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
        }

    def loss(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {},
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Ground truth is read from:
          - ``data_sample.gt_instances.lifting_target``  → ``(1, J, 3)``
          - ``data_sample.gt_instance_labels.pelvis_depth`` → ``(1,)``
          - ``data_sample.gt_instance_labels.pelvis_uv``    → ``(2,)``

        Returns:
            Dict of scalar losses: ``loss_joints``, ``loss_depth``,
            ``loss_uv``, ``mpjpe`` (mm, for logging).
        """
        pred = self.forward(feats)   # dict

        # ── Ground truth collection ──────────────────────────────────────
        gt_joints = torch.cat([
            d.gt_instances.lifting_target
            for d in batch_data_samples
        ], dim=0)                    # (B, 1, J, 3) or (B, J, 3)
        if gt_joints.dim() == 4:
            gt_joints = gt_joints.squeeze(1)   # (B, J, 3)
        gt_joints = gt_joints.to(pred['joints'].device)

        gt_depth = torch.stack([
            d.gt_instance_labels.pelvis_depth
            for d in batch_data_samples
        ]).to(pred['pelvis_depth'].device)    # (B, 1)
        if gt_depth.dim() == 1:
            gt_depth = gt_depth.unsqueeze(-1)

        gt_uv = torch.cat([
            d.gt_instance_labels.pelvis_uv
            for d in batch_data_samples
        ], dim=0).to(pred['pelvis_uv'].device)   # (B, 2)

        # ── Losses ───────────────────────────────────────────────────────
        losses = dict()

        losses['loss/joints/train'] = self.loss_joints_module(
            pred['joints'], gt_joints)

        losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
            pred['pelvis_depth'], gt_depth)

        losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
            pred['pelvis_uv'], gt_uv)

        # ── MPJPE (mm) — stored as attributes for TrainMPJPEAveragingHook.
        # Not included in the losses dict to avoid MMEngine auto-logging
        # noisy per-batch scalars (the hook writes epoch-averaged values).
        with torch.no_grad():
            self._train_mpjpe = (
                (pred['joints'] - gt_joints).norm(dim=-1).mean() * 1000.0)
            self._train_mpjpe_abs = _compute_mpjpe_abs(
                pred['joints'], gt_joints,
                pred['pelvis_depth'], gt_depth,
                pred['pelvis_uv'], gt_uv,
                batch_data_samples)

        # Return (losses_dict, pred_dict) — the pred_dict is used by the
        # estimator for visualization hooks.
        return losses, pred

    def predict(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {},
    ) -> Predictions:
        """Predict results from feature maps.

        Returns:
            List of ``InstanceData``, one per sample, each with:
              - ``keypoints``: ``(1, J, 3)`` predicted 3D joint coords
              - ``keypoint_scores``: ``(1, J)`` all ones
              - ``pelvis_depth``: ``(1,)``
              - ``pelvis_uv``: ``(2,)``
        """
        pred = self.forward(feats)   # dict, each (B, ...)
        B = pred['joints'].size(0)

        preds: List[InstanceData] = []
        for i in range(B):
            inst = InstanceData()
            # Shape (1, J, 3) to match mmpose convention (N_instances, K, D)
            inst.keypoints = pred['joints'][i:i+1].detach().cpu().numpy()
            inst.keypoint_scores = torch.ones(
                1, self.num_joints, dtype=torch.float32
            ).numpy()
            inst.pelvis_depth = pred['pelvis_depth'][i].detach().cpu().numpy()
            inst.pelvis_uv = pred['pelvis_uv'][i:i+1].detach().cpu().numpy()  # (1, 2)
            preds.append(inst)

        return preds
