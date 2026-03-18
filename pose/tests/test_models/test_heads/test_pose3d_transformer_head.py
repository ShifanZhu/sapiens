"""Tests for Pose3dTransformerHead."""

import torch
import numpy as np
import pytest
from mmengine.structures import InstanceData

_TEST_K = np.array([[500.0, 0, 192.0],
                    [0, 500.0, 320.0],
                    [0, 0, 1.0]], dtype=np.float32)


def _build_head(in_channels=1024, num_joints=70):
    """Build a Pose3dTransformerHead with default config."""
    from mmpose.models.heads.regression_heads.pose3d_transformer_head import (
        Pose3dTransformerHead,
    )
    return Pose3dTransformerHead(
        in_channels=in_channels,
        num_joints=num_joints,
    )


def _make_feats(batch_size=2, embed_dim=1024, h=40, w=24):
    """Create a fake backbone feature tuple."""
    return (torch.randn(batch_size, embed_dim, h, w),)


class TestForwardShapes:
    """forward() should produce correctly shaped outputs."""

    def test_output_keys(self):
        head = _build_head()
        out = head.forward(_make_feats())
        assert set(out.keys()) == {'joints', 'pelvis_depth', 'pelvis_uv'}

    def test_joints_shape(self):
        head = _build_head()
        out = head.forward(_make_feats(batch_size=3))
        assert out['joints'].shape == (3, 70, 3)

    def test_pelvis_depth_shape(self):
        head = _build_head()
        out = head.forward(_make_feats(batch_size=3))
        assert out['pelvis_depth'].shape == (3, 1)

    def test_pelvis_uv_shape(self):
        head = _build_head()
        out = head.forward(_make_feats(batch_size=3))
        assert out['pelvis_uv'].shape == (3, 2)


def _make_data_samples(batch_size=2, num_joints=70):
    """Create mock batch_data_samples for loss()."""
    samples = []
    for _ in range(batch_size):
        ds = InstanceData()
        gt_inst = InstanceData()
        gt_inst.lifting_target = torch.randn(1, num_joints, 3)
        ds.gt_instances = gt_inst

        gt_labels = InstanceData()
        gt_labels.pelvis_depth = torch.tensor([3.5])
        gt_labels.pelvis_uv = torch.randn(1, 2)
        ds.gt_instance_labels = gt_labels
        ds.set_metainfo({'K': _TEST_K, 'img_shape': (640, 384)})
        samples.append(ds)
    return samples


class TestLoss:
    """loss() should return finite scalar losses."""

    def test_loss_keys(self):
        head = _build_head()
        losses, _ = head.loss(_make_feats(), _make_data_samples())
        assert set(losses.keys()) == {'loss/joints/train', 'loss/depth/train', 'loss/uv/train', 'mpjpe', 'mpjpe_abs'}

    def test_losses_finite(self):
        head = _build_head()
        losses, _ = head.loss(_make_feats(), _make_data_samples())
        for key, val in losses.items():
            assert torch.isfinite(val), f'{key} is not finite: {val}'

    def test_losses_scalar(self):
        head = _build_head()
        losses, _ = head.loss(_make_feats(), _make_data_samples())
        for key, val in losses.items():
            assert val.dim() == 0, f'{key} is not scalar: shape {val.shape}'


class TestPredict:
    """predict() should return well-formed InstanceData list."""

    def test_returns_list_of_correct_length(self):
        head = _build_head()
        preds = head.predict(_make_feats(batch_size=3), _make_data_samples(batch_size=3))
        assert len(preds) == 3

    def test_keypoints_shape(self):
        head = _build_head()
        preds = head.predict(_make_feats(), _make_data_samples())
        assert preds[0].keypoints.shape == (1, 70, 3)

    def test_keypoint_scores_shape(self):
        head = _build_head()
        preds = head.predict(_make_feats(), _make_data_samples())
        assert preds[0].keypoint_scores.shape == (1, 70)

    def test_pelvis_depth_shape(self):
        head = _build_head()
        preds = head.predict(_make_feats(), _make_data_samples())
        assert preds[0].pelvis_depth.shape == (1,)

    def test_pelvis_uv_shape(self):
        head = _build_head()
        preds = head.predict(_make_feats(), _make_data_samples())
        assert preds[0].pelvis_uv.shape == (1, 2)


class TestEmbedDimGeneralization:
    """Head should work with different embed_dim values."""

    def test_embed_dim_1280(self):
        head = _build_head(in_channels=1280)
        out = head.forward(_make_feats(batch_size=2, embed_dim=1280))
        assert out['joints'].shape == (2, 70, 3)
        assert out['pelvis_depth'].shape == (2, 1)
        assert out['pelvis_uv'].shape == (2, 2)
