# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""3D pose visualization hook with BEDLAM2 TensorBoard video support.

Supports two modes:
1. **Heatmap mode** (original): per-iteration image grid output for
   heatmap-based 3D pose tasks (Goliath 308-joint, etc.).
2. **BEDLAM2 mode** (``bedlam2_video=True``): per-epoch TensorBoard video
   clips for RGBD 3D pose regression.

BEDLAM2 video mode renders 16-frame clips for fixed + random sequences
from both val and train datasets.  Two videos per sequence:

- ``{split}/scene_N/gt_pelvis``   — predicted joints + GT pelvis on crop
- ``{split}/scene_N/pred_pelvis`` — predicted joints + pred pelvis on
  original image (multi-person scenes show all bodies)

Usage in config (BEDLAM2)::

    custom_hooks = [
        dict(type='Pose3dVisualizationHook',
             enable=True,
             bedlam2_video=True,
             vis_interval=1),
    ]
"""

import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS, VISUALIZERS
from mmpose.structures import PoseDataSample, merge_data_samples

# ── BEDLAM2 constants ────────────────────────────────────────────────────────
_BODY_LINKS: List[Tuple[int, int]] = [
    # SMPL-X body joint ordering (indices 0-21 in active joint space):
    # 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
    # 6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
    # 12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder,
    # 17=R_shoulder, 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # spine / head
    (0, 1), (1, 4), (4, 7), (7, 10),             # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),             # right leg
    (9, 13), (13, 16), (16, 18), (18, 20),       # left arm
    (9, 14), (14, 17), (17, 19), (19, 21),       # right arm
]
_PERSON_COLORS: List[Tuple[int, int, int]] = [
    (0, 200, 0), (220, 120, 0), (0, 80, 200),
    (180, 0, 180), (0, 180, 180),
]
_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@HOOKS.register_module()
class Pose3dVisualizationHook(Hook):
    """3D pose visualization hook.

    Args:
        enable (bool): Whether to enable. Default: False.
        interval (int): Per-iteration vis interval (heatmap mode).
        bedlam2_video (bool): Enable BEDLAM2 TensorBoard video mode.
        vis_interval (int): Per-epoch vis interval (BEDLAM2 mode).
        n_vis_frames (int): Frames per video clip. Default: 16.
        fps (int): TensorBoard video fps. Default: 4.
        kpt_thr (float): Keypoint threshold (heatmap mode).
        show (bool): Display in window. Default: False.
        wait_time (float): Display wait time.
        max_vis_samples (int): Max samples per batch (heatmap mode).
        scale (int): Heatmap-to-image scale factor.
        line_width (int): Skeleton line width.
        radius (int): Joint dot radius.
        out_dir (str, optional): Output directory.
        backend_args (dict, optional): Backend args.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        bedlam2_video: bool = False,
        vis_interval: int = 1,
        n_vis_frames: int = 16,
        fps: int = 4,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        max_vis_samples: int = 16,
        scale: int = 4,
        line_width: int = 4,
        radius: int = 4,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args
        self.max_vis_samples = max_vis_samples
        self.scale = scale
        self.init_visualizer = False
        self._visualizer.line_width = line_width
        self._visualizer.radius = radius

        # BEDLAM2 video mode
        self.bedlam2_video = bedlam2_video
        self.vis_interval = vis_interval
        self.n_vis_frames = n_vis_frames
        self.fps = fps
        self._fixed_indices: Dict[str, List[Optional[int]]] = {
            'val': [None, None, None],
            'train': [None, None, None],
        }
        self._orig_K_cache: Dict[str, np.ndarray] = {}

    # ══════════════════════════════════════════════════════════════════════
    # MMEngine lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def before_run(self, runner: Runner) -> None:
        """Select fixed BEDLAM2 visualisation sequences once."""
        if not self.bedlam2_video or not self.enable:
            return
        try:
            val_ds = runner.val_dataloader.dataset
            train_ds = runner.train_dataloader.dataset
            self._fixed_indices['val'] = self._bedlam2_select_fixed(
                val_ds, runner)
            self._fixed_indices['train'] = self._bedlam2_select_fixed(
                train_ds, runner)
            runner.logger.info(
                f'Pose3dVisHook BEDLAM2: val fixed={self._fixed_indices["val"]}'
                f', train fixed={self._fixed_indices["train"]}')
        except Exception as exc:
            runner.logger.warning(
                f'Pose3dVisHook BEDLAM2 before_run failed: {exc}')

    def after_val_epoch(self, runner: Runner, metrics: dict) -> None:
        """Log BEDLAM2 TensorBoard videos after each val epoch."""
        if not self.bedlam2_video or not self.enable:
            return
        epoch = runner.epoch
        if epoch % self.vis_interval != 0:
            return
        if hasattr(runner, 'rank') and runner.rank != 0:
            return

        tb_writer = self._bedlam2_get_tb_writer(runner)
        if tb_writer is None:
            return

        try:
            device = next(runner.model.parameters()).device
            model = runner.model
            for split in ('val', 'train'):
                try:
                    dataset = (runner.val_dataloader.dataset
                               if split == 'val'
                               else runner.train_dataloader.dataset)
                except AttributeError:
                    continue

                fixed = self._fixed_indices[split]
                rand_idx = self._bedlam2_random_start(dataset)
                scenes = [
                    ('scene_0', fixed[0]),
                    ('scene_1', fixed[1]),
                    ('scene_2', fixed[2]),
                    ('scene_3_random', rand_idx),
                ]
                for scene_name, start_idx in scenes:
                    if start_idx is None:
                        continue
                    try:
                        self._bedlam2_visualize_sequence(
                            runner, model, dataset, start_idx, device,
                            tb_writer, split, scene_name, epoch)
                    except Exception as exc:
                        runner.logger.warning(
                            f'Pose3dVisHook BEDLAM2: {split}/{scene_name} '
                            f'failed: {exc}')
        except Exception as exc:
            runner.logger.warning(
                f'Pose3dVisHook BEDLAM2 after_val_epoch: {exc}')

    # ══════════════════════════════════════════════════════════════════════
    # Original heatmap-mode methods (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        if self.enable is False or self.bedlam2_video:
            return

        if not runner.rank == 0:
            return

        total_curr_iter = runner.iter

        if total_curr_iter % self.interval != 0:
            return

        image = torch.cat([input.unsqueeze(dim=0)/255 for input in data_batch['inputs']], dim=0)
        output = outputs['vis_preds']['pose2d'].detach()
        output_pose3d = outputs['vis_preds']['pose3d'].detach()

        batch_size = min(self.max_vis_samples, len(image))

        if self.init_visualizer == False:
            self._visualizer.set_dataset_meta(runner.train_dataloader.dataset.metainfo)
            self.init_visualizer = True

        image = image[:batch_size]
        output = output[:batch_size]
        output_pose3d = output_pose3d[:batch_size].detach().cpu()

        target = []
        for i in range(batch_size):
            target.append(data_batch['data_samples'][i].get('gt_fields').get('heatmaps').unsqueeze(dim=0))
        target = torch.cat(target, dim=0)

        target_weight = []
        for i in range(batch_size):
            target_weight.append(torch.tensor(data_batch['data_samples'][i].get('gt_instances').get('keypoints_visible')))
        target_weight = torch.cat(target_weight, dim=0)

        gt_K = []
        for i in range(batch_size):
            gt_K.append(torch.from_numpy(data_batch['data_samples'][i].K.astype(np.float32)).unsqueeze(dim=0))
        gt_K = torch.cat(gt_K, dim=0)

        gt_pose3d = []
        for i in range(batch_size):
            gt_pose3d.append(torch.from_numpy(data_batch['data_samples'][i].gt_instances.pose3d[0].astype(np.float32)))
        gt_pose3d = torch.stack(gt_pose3d)

        pose2d_homogeneous = torch.bmm(output_pose3d, gt_K.transpose(1, 2))
        pose2d = pose2d_homogeneous[:, :, :2] / (pose2d_homogeneous[:, :, 2:3] + 1e-5)

        gt_pose2d_homogeneous = torch.bmm(gt_pose3d, gt_K.transpose(1, 2))
        gt_pose2d = gt_pose2d_homogeneous[:, :, :2] / (gt_pose2d_homogeneous[:, :, 2:3] + 1e-5)

        pose2d_vis_dir = os.path.join(runner.work_dir, 'vis_data', '2d')
        pose3d_vis_dir = os.path.join(runner.work_dir, 'vis_data', '3d')

        if not os.path.exists(pose2d_vis_dir):
            os.makedirs(pose2d_vis_dir, exist_ok=True)

        if not os.path.exists(pose3d_vis_dir):
            os.makedirs(pose3d_vis_dir, exist_ok=True)

        pose2d_prefix = os.path.join(pose2d_vis_dir, 'train')
        pose3d_prefix = os.path.join(pose3d_vis_dir, 'train')

        suffix = str(total_curr_iter).zfill(6)
        original_image = image

        self.save_batch_image_with_joints(255*original_image, target, target_weight, '{}_{}_gt.jpg'.format(pose2d_prefix, suffix), scale=self.scale, is_rgb=False)
        self.save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(pose2d_prefix, suffix), scale=self.scale, is_rgb=False)

        self.save_batch_image_with_pose3d(255*original_image, gt_pose2d, torch.ones_like(target_weight), '{}_{}_pose3d_gt.jpg'.format(pose3d_prefix, suffix), is_rgb=False)
        self.save_batch_image_with_pose3d(255*original_image, pose2d, torch.ones_like(target_weight), '{}_{}_pose3d_pred.jpg'.format(pose3d_prefix, suffix), is_rgb=False)

        return

    def save_batch_image_with_pose3d(self, batch_image, batch_joints, batch_target_weight, file_name, dataset_info=None, is_rgb=True, nrow=8, padding=2):
        B, C, H, W = batch_image.size()
        num_joints = batch_joints.size(1)

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.detach().cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, num_joints)

        grid = []

        for i in range(B):
            image = batch_image[i].permute(1, 2, 0).cpu().numpy()
            image = image.copy()
            kps = batch_joints[i]
            kps_vis = batch_target_weight[i]
            kps_score = batch_target_weight[i]

            kps = np.maximum(kps, 0.0)
            kps[:, 0] = np.minimum(kps[:, 0], image.shape[1])
            kps[:, 1] = np.minimum(kps[:, 1], image.shape[0])

            if is_rgb == False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype(np.uint8)

            instances = InstanceData(metainfo=dict(keypoints=[kps], keypoints_visible=[kps_vis], keypoint_scores=[kps_score]))
            kp_vis_image = self._visualizer._draw_instances_kpts(image, instances=instances)
            kp_vis_image = cv2.cvtColor(kp_vis_image, cv2.COLOR_RGB2BGR)

            kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
            kp_vis_image = torch.from_numpy(kp_vis_image.copy())
            grid.append(kp_vis_image)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(file_name, ndarr)
        return

    def save_batch_image_with_joints(self, batch_image, batch_heatmaps, batch_target_weight, file_name, dataset_info=None, is_rgb=True, scale=4, nrow=8, padding=2):
        B, C, H, W = batch_image.size()
        num_joints = batch_heatmaps.size(1)

        if isinstance(batch_heatmaps, np.ndarray):
            batch_joints, batch_scores = get_max_preds(batch_heatmaps)
        else:
            batch_joints, batch_scores = get_max_preds(batch_heatmaps.detach().cpu().numpy())

        batch_joints = batch_joints*scale

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, num_joints)

        grid = []

        for i in range(B):
            image = batch_image[i].permute(1, 2, 0).cpu().numpy()
            image = image.copy()
            kps = batch_joints[i]
            kps_vis = batch_target_weight[i]
            kps_score = batch_scores[i].reshape(-1)

            if is_rgb == False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            instances = InstanceData(metainfo=dict(keypoints=[kps], keypoints_visible=[kps_vis], keypoint_scores=[kps_score]))
            kp_vis_image = self._visualizer._draw_instances_kpts(image, instances=instances)
            kp_vis_image = cv2.cvtColor(kp_vis_image, cv2.COLOR_RGB2BGR)

            kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
            kp_vis_image = torch.from_numpy(kp_vis_image.copy())
            grid.append(kp_vis_image)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(file_name, ndarr)
        return

    # ══════════════════════════════════════════════════════════════════════
    # BEDLAM2 video mode — index selection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _bedlam2_get_data_list(dataset) -> List[dict]:
        """Reconstruct data_list from serialized dataset.

        MMEngine's BaseDataset clears data_list after serialization
        (for shared-memory efficiency). This helper deserializes all
        items so visualization code can iterate them.
        """
        n = len(dataset)
        if n == 0:
            return []
        return [dataset.get_data_info(i) for i in range(n)]

    def _bedlam2_select_fixed(
        self, dataset, runner: Runner,
    ) -> List[Optional[int]]:
        """Return [rotate_true_idx, rotate_false_idx, multi_person_idx]."""
        data_list = self._bedlam2_get_data_list(dataset)
        label_first: Dict[str, int] = {}
        for i, s in enumerate(data_list):
            if s['label_path'] not in label_first:
                label_first[s['label_path']] = i

        slots: List[Optional[int]] = [None, None, None]
        for lp, first_idx in label_first.items():
            try:
                with np.load(lp, allow_pickle=True) as npz:
                    rot = bool(npz['rotate_flag'])
                    nb = int(npz['joints_cam'].shape[0])
                    self._orig_K_cache[lp] = (
                        npz['intrinsic_matrix'].astype(np.float32))
            except Exception as exc:
                runner.logger.warning(
                    f'Pose3dVisHook BEDLAM2: cannot load {lp}: {exc}')
                continue
            if slots[0] is None and rot:
                slots[0] = first_idx
            if slots[1] is None and not rot:
                slots[1] = first_idx
            if slots[2] is None and nb > 1:
                slots[2] = first_idx
            if all(s is not None for s in slots):
                break
        return slots

    def _bedlam2_random_start(self, dataset) -> Optional[int]:
        """Pick a random (label_path, body_idx) and return its first index."""
        n = len(dataset)
        if n == 0:
            return None
        kf: Dict[Tuple[str, int], int] = {}
        for i in range(n):
            s = dataset.get_data_info(i)
            key = (s['label_path'], s['body_idx'])
            if key not in kf:
                kf[key] = i
        return kf[random.choice(list(kf.keys()))] if kf else None

    def _bedlam2_collect_frames(
        self, dataset, start_idx: int,
    ) -> List[int]:
        """Collect up to n_vis_frames consecutive same-sequence indices."""
        anchor = dataset.get_data_info(start_idx)
        key = (anchor['label_path'], anchor['body_idx'])
        indices = []
        for i in range(start_idx, len(dataset)):
            s = dataset.get_data_info(i)
            if (s['label_path'], s['body_idx']) != key:
                break
            indices.append(i)
            if len(indices) >= self.n_vis_frames:
                break
        return indices

    def _bedlam2_get_orig_K(self, label_path: str) -> np.ndarray:
        """Get original intrinsic matrix from NPZ (cached)."""
        if label_path not in self._orig_K_cache:
            with np.load(label_path, allow_pickle=True) as npz:
                self._orig_K_cache[label_path] = (
                    npz['intrinsic_matrix'].astype(np.float32))
        return self._orig_K_cache[label_path]

    # ══════════════════════════════════════════════════════════════════════
    # BEDLAM2 video mode — geometry helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _bedlam2_recover_pelvis(
        pelvis_depth, pelvis_uv, K, crop_h, crop_w,
    ) -> np.ndarray:
        """(depth, uv, crop K) -> absolute [X, Y, Z] metres.

        BEDLAM2: X=forward, Y=left, Z=up.
        """
        X = float(np.asarray(pelvis_depth).ravel()[0])
        uv = np.asarray(pelvis_uv).ravel()
        u_px = (float(uv[0]) + 1.0) / 2.0 * crop_w
        v_px = (float(uv[1]) + 1.0) / 2.0 * crop_h
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        Y = -(u_px - cx) * X / fx
        Z = -(v_px - cy) * X / fy
        return np.array([X, Y, Z], dtype=np.float32)

    @staticmethod
    def _bedlam2_project_2d(joints_abs, K):
        """Project absolute joints -> pixel coords (BEDLAM2 convention)."""
        J = joints_abs.shape[0]
        uv = np.full((J, 2), -1.0, dtype=np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        for j in range(J):
            X = float(joints_abs[j, 0])
            if X <= 0.01:
                continue
            Y, Z = float(joints_abs[j, 1]), float(joints_abs[j, 2])
            uv[j, 0] = fx * (-Y / X) + cx
            uv[j, 1] = fy * (-Z / X) + cy
        return uv

    # ══════════════════════════════════════════════════════════════════════
    # BEDLAM2 video mode — drawing
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _bedlam2_draw_frame(
        img_hwc: np.ndarray,
        joints_abs: np.ndarray,
        K: np.ndarray,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Draw body skeleton (joints 0-21) on img. Returns (3,H,W) uint8."""
        canvas = img_hwc.copy()
        uv = Pose3dVisualizationHook._bedlam2_project_2d(joints_abs, K)
        H, W = canvas.shape[:2]

        def _ok(pt):
            return 0 <= pt[0] < W and 0 <= pt[1] < H

        for (i, j) in _BODY_LINKS:
            if i >= len(uv) or j >= len(uv):
                continue
            pi, pj = uv[i], uv[j]
            if pi[0] < 0 or pj[0] < 0 or not _ok(pi) or not _ok(pj):
                continue
            pt1 = (int(round(float(pi[0]))), int(round(float(pi[1]))))
            pt2 = (int(round(float(pj[0]))), int(round(float(pj[1]))))
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

        for j in range(min(22, len(uv))):
            pt = uv[j]
            if pt[0] < 0 or not _ok(pt):
                continue
            center = (int(round(float(pt[0]))), int(round(float(pt[1]))))
            cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)

        return canvas.transpose(2, 0, 1)  # (3, H, W)

    @staticmethod
    def _bedlam2_build_video(frames_chw):
        """Stack (3,H,W) uint8 list -> (1,T,3,H,W) uint8 tensor."""
        return torch.from_numpy(np.stack(frames_chw, axis=0)).unsqueeze(0)

    # ══════════════════════════════════════════════════════════════════════
    # BEDLAM2 video mode — per-sequence visualisation
    # ══════════════════════════════════════════════════════════════════════

    def _bedlam2_visualize_sequence(
        self, runner, model, dataset, start_idx, device,
        tb_writer, split, scene_name, epoch,
    ) -> None:
        """Render and log gt_pelvis + pred_pelvis videos for one sequence."""
        frame_indices = self._bedlam2_collect_frames(dataset, start_idx)
        if not frame_indices:
            return

        anchor = dataset.get_data_info(start_idx)
        anchor_label = anchor['label_path']
        anchor_body = anchor['body_idx']
        K_orig = self._bedlam2_get_orig_K(anchor_label)

        # Reverse lookup: frame_idx -> all flat indices (all bodies)
        needed = {dataset.get_data_info(fi)['frame_idx']
                  for fi in frame_indices}
        frame_to_bodies: Dict[int, List[int]] = defaultdict(list)
        for fi in range(len(dataset)):
            s = dataset.get_data_info(fi)
            if s['label_path'] == anchor_label and s['frame_idx'] in needed:
                frame_to_bodies[s['frame_idx']].append(fi)

        was_training = model.training
        model.eval()

        gt_frames: List[np.ndarray] = []
        pred_frames: List[np.ndarray] = []

        with torch.no_grad():
            for flat_idx in frame_indices:
                try:
                    sample = dataset.prepare_data(flat_idx)
                except Exception as exc:
                    runner.logger.warning(
                        f'Pose3dVisHook: dataset[{flat_idx}] failed: {exc}')
                    continue
                if sample is None:
                    continue

                inputs_4ch = sample['inputs']
                ds = sample['data_samples']
                inp = inputs_4ch.unsqueeze(0).to(device)

                try:
                    preds = model(inp, [ds], mode='predict')
                except Exception as exc:
                    runner.logger.warning(
                        f'Pose3dVisHook: forward {flat_idx} failed: {exc}')
                    continue

                pred_ds = preds[0]

                # Unpack GT
                gt_rel = ds.gt_instances.lifting_target[0]
                if isinstance(gt_rel, torch.Tensor):
                    gt_rel = gt_rel.cpu().numpy()
                gt_pdepth = ds.gt_instance_labels.pelvis_depth
                gt_puv = ds.gt_instance_labels.pelvis_uv
                K_crop = np.asarray(ds.metainfo.get('K'), dtype=np.float32)

                # Unpack predictions
                pred_rel = pred_ds.pred_instances.keypoints[0]
                if isinstance(pred_rel, torch.Tensor):
                    pred_rel = pred_rel.cpu().numpy()
                pred_pdepth = pred_ds.pred_instances.pelvis_depth
                pred_puv = pred_ds.pred_instances.pelvis_uv

                # Unnormalize RGB crop
                rgb_t = inputs_4ch[:3]
                crop_h, crop_w = rgb_t.shape[1], rgb_t.shape[2]
                rgb_np = rgb_t.permute(1, 2, 0).numpy()
                rgb_np = rgb_np * _IMG_STD + _IMG_MEAN
                rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

                color = _PERSON_COLORS[anchor_body % len(_PERSON_COLORS)]

                # gt_pelvis frame (crop image, crop K)
                gt_xyz = self._bedlam2_recover_pelvis(
                    gt_pdepth, gt_puv, K_crop, crop_h, crop_w)
                gt_abs = pred_rel + gt_xyz[np.newaxis, :]
                gt_frames.append(
                    self._bedlam2_draw_frame(rgb_np, gt_abs, K_crop, color))

                # pred_pelvis frame (original image, original K)
                img_path = ds.metainfo.get('img_path', '')
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    orig_rgb, K_proj = rgb_np.copy(), K_crop
                else:
                    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    K_proj = K_orig

                pred_xyz = self._bedlam2_recover_pelvis(
                    pred_pdepth, pred_puv, K_crop, crop_h, crop_w)
                pred_abs = pred_rel + pred_xyz[np.newaxis, :]
                canvas = self._bedlam2_draw_frame(
                    orig_rgb, pred_abs, K_proj, color)
                canvas_hwc = canvas.transpose(1, 2, 0)

                # Multi-person: draw sibling bodies
                fidx = dataset.get_data_info(flat_idx)['frame_idx']
                for sib_fi in frame_to_bodies.get(fidx, []):
                    sib_body = dataset.get_data_info(sib_fi)['body_idx']
                    if sib_body == anchor_body:
                        continue
                    try:
                        sib_d = dataset.prepare_data(sib_fi)
                    except Exception:
                        continue
                    if sib_d is None:
                        continue
                    sib_inp = sib_d['inputs'].unsqueeze(0).to(device)
                    sib_ds = sib_d['data_samples']
                    try:
                        sib_preds = model(
                            sib_inp, [sib_ds], mode='predict')
                    except Exception:
                        continue
                    sib_pred = sib_preds[0]
                    sib_rel = sib_pred.pred_instances.keypoints[0]
                    if isinstance(sib_rel, torch.Tensor):
                        sib_rel = sib_rel.cpu().numpy()
                    sib_K = np.asarray(
                        sib_ds.metainfo.get('K'), dtype=np.float32)
                    sib_h, sib_w = sib_d['inputs'].shape[1:]
                    sib_xyz = self._bedlam2_recover_pelvis(
                        sib_pred.pred_instances.pelvis_depth,
                        sib_pred.pred_instances.pelvis_uv,
                        sib_K, sib_h, sib_w)
                    sib_abs = sib_rel + sib_xyz[np.newaxis, :]
                    sib_color = _PERSON_COLORS[
                        sib_body % len(_PERSON_COLORS)]
                    sib_chw = self._bedlam2_draw_frame(
                        canvas_hwc, sib_abs, K_proj, sib_color)
                    canvas_hwc = sib_chw.transpose(1, 2, 0)

                pred_frames.append(canvas_hwc.transpose(2, 0, 1))

        if was_training:
            model.train()

        if not gt_frames or not pred_frames:
            return

        gt_tag = f'video/{split}/{scene_name}/gt_pelvis'
        pred_tag = f'video/{split}/{scene_name}/pred_pelvis'
        try:
            tb_writer.add_video(
                gt_tag, self._bedlam2_build_video(gt_frames),
                global_step=epoch, fps=self.fps)
            tb_writer.add_video(
                pred_tag, self._bedlam2_build_video(pred_frames),
                global_step=epoch, fps=self.fps)
        except Exception as exc:
            runner.logger.warning(
                f'Pose3dVisHook BEDLAM2: add_video failed for '
                f'{split}/{scene_name}: {exc}')

    @staticmethod
    def _bedlam2_get_tb_writer(runner):
        """Return TensorBoard SummaryWriter or None."""
        try:
            # Access _vis_backends directly to avoid @master_only decorator
            # issues on get_backend
            tb = runner.visualizer._vis_backends.get(
                'TensorboardVisBackend')
            if tb is None:
                raise KeyError('TensorboardVisBackend not in _vis_backends')
            if not tb._env_initialized:
                tb._init_env()
            return tb._tensorboard
        except Exception as exc:
            runner.logger.warning(
                f'Pose3dVisHook: TensorboardVisBackend not found: {exc}')
            return None


# ── Module-level helpers (heatmap mode) ──────────────────────────────────────

def batch_unnormalize_image(images, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    normalize = transforms.Normalize(mean=mean, std=std)
    images[:, 0, :, :] = (images[:, 0, :, :]*normalize.std[0]) + normalize.mean[0]
    images[:, 1, :, :] = (images[:, 1, :, :]*normalize.std[1]) + normalize.mean[1]
    images[:, 2, :, :] = (images[:, 2, :, :]*normalize.std[2]) + normalize.mean[2]
    return images

def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
