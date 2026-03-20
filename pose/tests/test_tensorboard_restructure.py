"""Tests for TensorBoard tag restructure (PRD: tensorboard_restructure)."""

import torch
import numpy as np
import pytest
from mmengine.structures import InstanceData


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_feats(batch_size=2, embed_dim=1024, h=40, w=24):
    """Create fake backbone feature tuple."""
    return (torch.randn(batch_size, embed_dim, h, w),)


def _make_data_samples(batch_size=2, num_joints=70):
    """Create mock batch_data_samples with GT + K metadata."""
    K = np.array([[500.0, 0, 192.0],
                  [0, 500.0, 320.0],
                  [0, 0, 1.0]], dtype=np.float32)
    samples = []
    for _ in range(batch_size):
        ds = InstanceData()
        gt_inst = InstanceData()
        gt_inst.lifting_target = torch.randn(1, num_joints, 3)
        ds.gt_instances = gt_inst

        gt_labels = InstanceData()
        gt_labels.pelvis_depth = torch.tensor([3.5])
        gt_labels.pelvis_uv = torch.randn(1, 2) * 0.3
        ds.gt_instance_labels = gt_labels

        ds.set_metainfo({'K': K, 'img_shape': (640, 384)})
        samples.append(ds)
    return samples


# ── Test 1: Head loss() returns restructured key names ───────────────────────

class TestHeadLossKeyNames:
    """Both heads should return loss/joints/train, loss/depth/train,
    loss/uv/train instead of loss_joints, loss_depth, loss_uv."""

    EXPECTED_KEYS = {'loss/joints/train', 'loss/depth/train', 'loss/uv/train'}

    def test_regression_head_keys(self):
        from mmpose.models.heads.regression_heads.pose3d_regression_head import (
            Pose3dRegressionHead,
        )
        head = Pose3dRegressionHead(in_channels=1024)
        losses, _ = head.loss(_make_feats(), _make_data_samples())
        assert set(losses.keys()) == self.EXPECTED_KEYS

    def test_transformer_head_keys(self):
        from mmpose.models.heads.regression_heads.pose3d_transformer_head import (
            Pose3dTransformerHead,
        )
        head = Pose3dTransformerHead(in_channels=1024)
        losses, _ = head.loss(_make_feats(), _make_data_samples())
        assert set(losses.keys()) == self.EXPECTED_KEYS


# ── Test 2: BedlamMPJPEMetric returns restructured key names ─────────────────

class TestMetricKeyNames:
    """BedlamMPJPEMetric should return mpjpe/rel/val, mpjpe/body/val,
    mpjpe/hand/val instead of bedlam/mpjpe/all, etc."""

    EXPECTED_KEYS = {'mpjpe/rel/val', 'mpjpe/body/val', 'mpjpe/hand/val',
                     'mpjpe/abs/val'}

    def test_metric_keys(self):
        from mmpose.evaluation.metrics.bedlam_metric import BedlamMPJPEMetric
        metric = BedlamMPJPEMetric()
        K = np.array([[500.0, 0, 192.0],
                      [0, 500.0, 320.0],
                      [0, 0, 1.0]], dtype=np.float32)
        num_joints = 70
        for _ in range(5):
            pred = np.random.randn(1, num_joints, 3).astype(np.float32)
            gt = pred + np.random.randn(1, num_joints, 3).astype(np.float32) * 0.01
            data_sample = {
                'pred_instances': {
                    'keypoints': pred[0],
                    'pelvis_depth': np.array([3.5]),
                    'pelvis_uv': np.array([[0.1, -0.1]]),
                },
                'gt_instances': {'lifting_target': gt[0]},
                'gt_instance_labels': {
                    'pelvis_depth': np.array([3.5]),
                    'pelvis_uv': np.array([[0.0, 0.0]]),
                },
                'metainfo': {'K': K, 'img_shape': (640, 384)},
            }
            metric.process([], [data_sample])
        results = metric.compute_metrics(metric.results)
        assert set(results.keys()) == self.EXPECTED_KEYS


# ── Test 3: Head loss() returns mpjpe_abs as finite scalar ───────────────────

class TestHeadMpjpeAbs:
    """Head loss() should return mpjpe_abs: finite, scalar, no gradient."""

    def test_regression_head_mpjpe_abs(self):
        from mmpose.models.heads.regression_heads.pose3d_regression_head import (
            Pose3dRegressionHead,
        )
        head = Pose3dRegressionHead(in_channels=1024)
        head.loss(_make_feats(), _make_data_samples())
        v = head._train_mpjpe_abs
        assert v.dim() == 0, f'not scalar: {v.shape}'
        assert torch.isfinite(v), f'not finite: {v}'
        assert not v.requires_grad

    def test_transformer_head_mpjpe_abs(self):
        from mmpose.models.heads.regression_heads.pose3d_transformer_head import (
            Pose3dTransformerHead,
        )
        head = Pose3dTransformerHead(in_channels=1024)
        head.loss(_make_feats(), _make_data_samples())
        v = head._train_mpjpe_abs
        assert v.dim() == 0
        assert torch.isfinite(v)
        assert not v.requires_grad


# ── Test 4: mpjpe_abs correctness with known values ─────────────────────────

class TestMpjpeAbsCorrectness:
    """Absolute MPJPE should match hand-computed value for known inputs."""

    def test_known_values(self):
        """With identical pred/gt relative joints but different pelvis,
        absolute MPJPE = pelvis position error (same for all joints)."""
        from mmpose.models.heads.regression_heads.pelvis_utils import (
            compute_mpjpe_abs, recover_pelvis_3d,
        )
        K = np.array([[500.0, 0, 192.0],
                      [0, 500.0, 320.0],
                      [0, 0, 1.0]], dtype=np.float32)
        crop_h, crop_w = 640, 384
        num_joints = 70

        # Both pred and gt have the same relative joints (zero)
        joints = torch.zeros(1, num_joints, 3)

        # GT pelvis: depth=5.0, uv=(0, 0) → center of crop
        gt_depth = torch.tensor([[5.0]])
        gt_uv = torch.tensor([[0.0, 0.0]])

        # Pred pelvis: depth=5.0, uv=(0.1, 0) → slightly off center
        pred_depth = torch.tensor([[5.0]])
        pred_uv = torch.tensor([[0.1, 0.0]])

        # Hand-compute expected pelvis positions
        gt_pelvis = recover_pelvis_3d(gt_depth, gt_uv, K, crop_h, crop_w)
        pred_pelvis = recover_pelvis_3d(pred_depth, pred_uv, K, crop_h, crop_w)

        # Since joints are zero, abs error = pelvis error
        expected_err = (pred_pelvis - gt_pelvis).norm().item() * 1000.0  # mm

        # Build data samples
        ds = InstanceData()
        ds.set_metainfo({'K': K, 'img_shape': (crop_h, crop_w)})

        result = compute_mpjpe_abs(
            joints, joints, pred_depth, gt_depth, pred_uv, gt_uv, [ds])

        assert abs(result.item() - expected_err) < 0.01, \
            f'Expected {expected_err:.2f}, got {result.item():.2f}'

    def test_recover_pelvis_known(self):
        """Pelvis at crop center with depth=5 → Y=0, Z=0."""
        from mmpose.models.heads.regression_heads.pelvis_utils import (
            recover_pelvis_3d,
        )
        K = np.array([[500.0, 0, 192.0],
                      [0, 500.0, 320.0],
                      [0, 0, 1.0]], dtype=np.float32)
        # uv=(0,0) → u_px = 192, v_px = 320 → exactly cx, cy → Y=0, Z=0
        pelvis = recover_pelvis_3d(
            torch.tensor([[5.0]]), torch.tensor([[0.0, 0.0]]),
            K, 640, 384)
        assert pelvis.shape == (1, 3)
        assert abs(pelvis[0, 0].item() - 5.0) < 1e-5   # X = depth
        assert abs(pelvis[0, 1].item()) < 1e-5           # Y = 0
        assert abs(pelvis[0, 2].item()) < 1e-5           # Z = 0


# ── Test 6: BedlamMPJPEMetric absolute MPJPE correctness ────────────────────

class TestMetricAbsCorrectness:
    """BedlamMPJPEMetric absolute MPJPE should match hand-computed value."""

    def test_known_pelvis_offset(self):
        """Zero relative joints, known pelvis offset → abs MPJPE = pelvis error."""
        from mmpose.evaluation.metrics.bedlam_metric import BedlamMPJPEMetric
        metric = BedlamMPJPEMetric()

        K = np.array([[500.0, 0, 192.0],
                      [0, 500.0, 320.0],
                      [0, 0, 1.0]], dtype=np.float32)
        num_joints = 70

        # Zero relative joints for both pred and GT
        joints = np.zeros((num_joints, 3), dtype=np.float32)

        # GT pelvis at crop center: depth=5, uv=(0,0) → [5, 0, 0]
        # Pred pelvis offset: depth=5, uv=(0.1, 0.05)
        gt_depth, gt_uv = 5.0, np.array([0.0, 0.0])
        pred_depth, pred_uv = 5.0, np.array([0.1, 0.05])

        # Hand-compute pelvis positions
        def _recover(depth, uv):
            X = depth
            u_px = (uv[0] + 1.0) / 2.0 * 384
            v_px = (uv[1] + 1.0) / 2.0 * 640
            Y = -(u_px - 192.0) * X / 500.0
            Z = -(v_px - 320.0) * X / 500.0
            return np.array([X, Y, Z])

        gt_pelvis = _recover(gt_depth, gt_uv)
        pred_pelvis = _recover(pred_depth, pred_uv)
        expected_err = float(np.linalg.norm(pred_pelvis - gt_pelvis)) * 1000.0

        data_sample = {
            'pred_instances': {
                'keypoints': joints,
                'pelvis_depth': np.array([pred_depth]),
                'pelvis_uv': np.array([pred_uv]),
            },
            'gt_instances': {'lifting_target': joints},
            'gt_instance_labels': {
                'pelvis_depth': np.array([gt_depth]),
                'pelvis_uv': np.array([gt_uv]),
            },
            'metainfo': {'K': K, 'img_shape': (640, 384)},
        }
        metric.process([], [data_sample])
        results = metric.compute_metrics(metric.results)

        assert abs(results['mpjpe/abs/val'] - expected_err) < 0.1, \
            f'Expected {expected_err:.2f}, got {results["mpjpe/abs/val"]:.2f}'


# ── Test 7: Epoch-averaging hook ────────────────────────────────────────────

class TestTrainMPJPEHook:
    """TrainMPJPEAveragingHook should accumulate per-batch values
    and flush correct epoch averages."""

    def test_accumulate_and_flush(self):
        from mmpose.engine.hooks.train_mpjpe_hook import TrainMPJPEAveragingHook

        hook = TrainMPJPEAveragingHook()

        # Simulate 3 batches with known mpjpe values stored on a fake head
        values = [100.0, 200.0, 300.0]
        abs_values = [150.0, 250.0, 350.0]

        class FakeHead:
            pass

        class FakeModel:
            head = FakeHead()

        class FakeRunnerWithModel:
            epoch = 5
            model = FakeModel()

        runner_with_model = FakeRunnerWithModel()
        for mpjpe_val, abs_val in zip(values, abs_values):
            runner_with_model.model.head._train_mpjpe = torch.tensor(mpjpe_val)
            runner_with_model.model.head._train_mpjpe_abs = torch.tensor(abs_val)
            hook.after_train_iter(runner_with_model, 0, None, {})

        assert len(hook._mpjpe_buffer) == 3
        assert len(hook._mpjpe_abs_buffer) == 3

        # Flush — should compute averages and write to tensorboard
        # We'll mock the runner to capture what gets logged
        logged = {}

        class FakeTB:
            def add_scalar(self, tag, value, global_step):
                logged[tag] = value

        class FakeBackend:
            _env_initialized = True
            _tensorboard = FakeTB()

        class FakeVisualizer:
            _vis_backends = {'TensorboardVisBackend': FakeBackend()}

        class FakeRunner:
            epoch = 5
            visualizer = FakeVisualizer()

        hook.after_train_epoch(FakeRunner())

        assert abs(logged['mpjpe/rel/train'] - 200.0) < 0.01  # mean of 100,200,300
        assert abs(logged['mpjpe/abs/train'] - 250.0) < 0.01  # mean of 150,250,350
        assert len(hook._mpjpe_buffer) == 0  # reset

    def test_empty_epoch(self):
        """No crash if no batches were accumulated."""
        from mmpose.engine.hooks.train_mpjpe_hook import TrainMPJPEAveragingHook

        hook = TrainMPJPEAveragingHook()

        class FakeRunner:
            epoch = 0

        # Should not crash
        hook.after_train_epoch(FakeRunner())
