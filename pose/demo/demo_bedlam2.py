# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""BEDLAM2 RGBD 3D pose inference demo.

Runs the trained RGBDPose3dEstimator on BEDLAM2 test sequences and saves
per-sample PNG files with two side-by-side panels:

  Left  — RGB crop with 2D-projected GT (green) and predicted (red) body skeletons
  Right — 3D skeleton plot (Matplotlib) with predicted (blue) and GT (orange) joints

Usage::

    cd /home/<user>/repos/sapiens/pose
    python demo/demo_bedlam2.py \\
        configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \\
        /path/to/checkpoint.pth \\
        --data-root /media/s/SF_backup/bedlam2 \\
        --seq-paths-file data/bedlam2_splits/test_seqs.txt \\
        --output-root Outputs/demo/bedlam2 \\
        --num-samples 200 \\
        --batch-size 8 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

from mmpose.utils import register_all_modules
register_all_modules()  # registers all mmpose models, datasets, transforms
from mmpose.registry import DATASETS, MODELS

# ── Constants (must match bedlam2_transforms.py) ──────────────────────────────
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Body joint indices 0–21 used for visualisation
_BODY_IDX = list(range(22))

# Skeleton links over body joints (SMPL-X ordering, index into _BODY_IDX).
# SMPL-X body joint ordering (indices 0-21 in active joint space):
# 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
# 6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
# 12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder,
# 17=R_shoulder, 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist
BODY_LINKS = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # spine / head
    (0, 1), (1, 4), (4, 7), (7, 10),             # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),             # right leg
    (9, 13), (13, 16), (16, 18), (18, 20),       # left arm
    (9, 14), (14, 17), (17, 19), (19, 21),       # right arm
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unnorm_rgb(tensor_chw: torch.Tensor) -> np.ndarray:
    """Inverse ImageNet normalisation; returns (H, W, 3) uint8 RGB."""
    rgb = tensor_chw[:3].permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
    rgb = rgb * _RGB_STD + _RGB_MEAN
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def _recover_pelvis(pelvis_depth: np.ndarray,
                    pelvis_uv: np.ndarray,
                    K: np.ndarray,
                    crop_h: int, crop_w: int) -> np.ndarray:
    """Back-project pelvis_uv + pelvis_depth to 3D camera space.

    pelvis_depth: (1,)  X = forward distance in metres
    pelvis_uv:    (2,) or (1,2)  normalised UV in [-1,1]
    K:            (3,3) crop intrinsic matrix
    Returns (3,) absolute pelvis XYZ.
    """
    uv = pelvis_uv.flatten()
    X = float(pelvis_depth.flat[0])
    u_px = (float(uv[0]) + 1.0) / 2.0 * crop_w
    v_px = (float(uv[1]) + 1.0) / 2.0 * crop_h
    Y = -(u_px - K[0, 2]) * X / K[0, 0]
    Z = -(v_px - K[1, 2]) * X / K[1, 1]
    return np.array([X, Y, Z], dtype=np.float32)


def _project_joints(joints_abs: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project absolute camera-space joints to 2D pixel coords (crop space).

    BEDLAM2 convention: X=forward, Y=left, Z=up.
    u = fx * (-Y/X) + cx,  v = fy * (-Z/X) + cy

    joints_abs: (J, 3)
    Returns (J, 2) float32 pixel coords.
    """
    X = joints_abs[:, 0]
    Y = joints_abs[:, 1]
    Z = joints_abs[:, 2]
    # Mask degenerate joints (X <= 0)
    valid = X > 0.01
    u = np.where(valid, K[0, 0] * (-Y / np.where(valid, X, 1.0)) + K[0, 2], K[0, 2])
    v = np.where(valid, K[1, 1] * (-Z / np.where(valid, X, 1.0)) + K[1, 2], K[1, 2])
    return np.stack([u, v], axis=-1).astype(np.float32)


def _draw_skeleton_2d(img: np.ndarray, kpts_uv: np.ndarray,
                      color: tuple, radius: int = 4) -> np.ndarray:
    """Draw 2D body skeleton on a copy of img (RGB uint8)."""
    out = img.copy()
    H, W = out.shape[:2]

    def _ok(pt):
        return 0 <= pt[0] < W and 0 <= pt[1] < H

    # Draw links
    for (i, j) in BODY_LINKS:
        if i >= len(kpts_uv) or j >= len(kpts_uv):
            continue
        pi = (int(round(kpts_uv[i, 0])), int(round(kpts_uv[i, 1])))
        pj = (int(round(kpts_uv[j, 0])), int(round(kpts_uv[j, 1])))
        if _ok(pi) and _ok(pj):
            cv2.line(out, pi, pj, color, 2, cv2.LINE_AA)

    # Draw joints
    for pt in kpts_uv:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(out, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return out


def _draw_skeleton_3d(ax, joints_abs: np.ndarray,
                      color: str, label: str = '') -> None:
    """Draw 3D body skeleton on a Matplotlib 3D axes."""
    # BEDLAM2: X=depth, Y=left, Z=up → plot as (Y, Z, X) = (lateral, vert, depth)
    x = joints_abs[:, 1]   # lateral (Y)
    y = joints_abs[:, 2]   # vertical (Z)
    z = joints_abs[:, 0]   # depth (X)

    # Draw links
    for (i, j) in BODY_LINKS:
        if i >= len(joints_abs) or j >= len(joints_abs):
            continue
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                c=color, linewidth=1.5, alpha=0.7)

    ax.scatter(x, y, z, c=color, s=20, depthshade=True, label=label)


def _save_vis(rgb_crop: np.ndarray,
              pred_joints_abs: np.ndarray,
              gt_joints_abs: np.ndarray,
              K: np.ndarray,
              out_path: str) -> None:
    """Create and save the side-by-side 2D + 3D visualisation PNG."""
    H, W = rgb_crop.shape[:2]

    # ── 2D projection panel ──────────────────────────────────────────────────
    body_pred = pred_joints_abs[_BODY_IDX]
    body_gt = gt_joints_abs[_BODY_IDX]

    pred_uv = _project_joints(body_pred, K)
    gt_uv = _project_joints(body_gt, K)

    img_vis = rgb_crop.copy()
    img_vis = _draw_skeleton_2d(img_vis, gt_uv, color=(60, 200, 60))     # GT: green
    img_vis = _draw_skeleton_2d(img_vis, pred_uv, color=(220, 60, 60))   # Pred: red
    img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # ── 3D skeleton panel (Matplotlib) ───────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))

    # Left: 2D panel embedded as imshow
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.imshow(img_vis)
    ax2d.axis('off')
    ax2d.set_title('2D projection  (green=GT, red=pred)')

    # Right: 3D panel
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    _draw_skeleton_3d(ax3d, body_gt, color='tab:green', label='GT')
    _draw_skeleton_3d(ax3d, body_pred, color='tab:red', label='Pred')

    lim = 1.0
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-0.2, 2 * lim)
    ax3d.set_xlabel('Y (lateral, m)')
    ax3d.set_ylabel('Z (vert, m)')
    ax3d.set_zlabel('X (depth, m)')
    ax3d.set_title('3D skeleton  (pred vs GT)')
    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        ax3d.legend(handles[:2], labels[:2], loc='upper right', fontsize=8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Build model ───────────────────────────────────────────────────────────────

def build_model(config_file: str, checkpoint: str, device: str):
    cfg = Config.fromfile(config_file)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    model.to(device)
    return model


# ── Build dataset ──────────────────────────────────────────────────────────────

def build_val_dataset(cfg, data_root: str, seq_paths_file: str):
    """Build a Bedlam2Dataset with the val (no-augmentation) pipeline."""
    ds_cfg = cfg.val_dataloader.dataset.copy()
    ds_cfg['data_root'] = data_root
    ds_cfg['seq_paths_file'] = seq_paths_file
    return DATASETS.build(ds_cfg)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BEDLAM2 inference demo')
    p.add_argument('config', help='Config file')
    p.add_argument('checkpoint', help='Checkpoint file')
    p.add_argument('--data-root', required=True,
                   help='BEDLAM2 data root (contains data/label/, data/frames/, etc.)')
    p.add_argument('--seq-paths-file', default='data/bedlam2_splits/test_seqs.txt',
                   help='Text file with one sequence path per line')
    p.add_argument('--output-root', default='Outputs/demo/bedlam2',
                   help='Directory to save output PNGs')
    p.add_argument('--num-samples', type=int, default=200,
                   help='Max number of samples to visualise (0 = all)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    print(f'Building model from {args.config} ...')
    model = build_model(args.config, args.checkpoint, args.device)

    print(f'Building dataset from {args.seq_paths_file} ...')
    dataset = build_val_dataset(cfg, args.data_root, args.seq_paths_file)
    print(f'  {len(dataset)} samples found.')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=False,
        shuffle=False,
        collate_fn=lambda x: x,   # list of dicts — we handle batching manually
    )

    n_saved = 0
    max_samples = args.num_samples if args.num_samples > 0 else len(dataset)

    for batch in loader:
        if n_saved >= max_samples:
            break

        # batch is a list of dicts produced by the val pipeline
        # Each dict has 'inputs' (4,H,W tensor) and 'data_samples' (PoseDataSample)
        inputs_list = [item['inputs'] for item in batch]
        data_samples_list = [item['data_samples'] for item in batch]

        inputs = torch.stack(inputs_list).to(args.device)    # (B, 4, H, W)

        with torch.no_grad():
            results = model(inputs, data_samples_list, mode='predict')

        for i, (data_sample, item) in enumerate(zip(results, batch)):
            if n_saved >= max_samples:
                break

            meta = data_sample.metainfo
            seq_name = meta.get('seq_name', 'seq')
            frame_idx = meta.get('frame_idx', 0)
            body_idx = meta.get('body_idx', 0)
            K = np.array(meta['K'], dtype=np.float32)   # (3,3)

            # ── Unnormalise RGB crop ────────────────────────────────────────
            rgb_crop = _unnorm_rgb(item['inputs'])     # (H, W, 3) uint8
            crop_h, crop_w = rgb_crop.shape[:2]

            # ── Predicted joints (root-relative) ───────────────────────────
            pred_inst = data_sample.pred_instances
            joints_rel = pred_inst.keypoints[0]               # (70, 3) numpy
            pelvis_depth = pred_inst.pelvis_depth              # (1,)
            pelvis_uv = pred_inst.pelvis_uv                    # (1,2) or (2,)

            pelvis_abs = _recover_pelvis(pelvis_depth, pelvis_uv, K,
                                         crop_h, crop_w)
            pred_joints_abs = joints_rel + pelvis_abs          # (70, 3)

            # ── GT joints (root-relative) ───────────────────────────────────
            gt_inst = data_sample.gt_instances
            gt_joints_rel = gt_inst.lifting_target[0].cpu().numpy()  # (70, 3)

            gt_inst_labels = data_sample.gt_instance_labels
            gt_depth = gt_inst_labels.pelvis_depth.cpu().numpy()  # (1,)
            gt_uv = gt_inst_labels.pelvis_uv.cpu().numpy()        # (1,2)

            gt_pelvis_abs = _recover_pelvis(gt_depth, gt_uv, K,
                                            crop_h, crop_w)
            gt_joints_abs = gt_joints_rel + gt_pelvis_abs

            # ── Save ────────────────────────────────────────────────────────
            out_path = os.path.join(
                args.output_root, seq_name,
                f'{frame_idx:05d}_body{body_idx:02d}.png')

            _save_vis(rgb_crop, pred_joints_abs, gt_joints_abs, K, out_path)
            n_saved += 1

            if n_saved % 20 == 0:
                print(f'  Saved {n_saved}/{max_samples}  → {out_path}')

    print(f'\nDone. {n_saved} images saved to {args.output_root}')


if __name__ == '__main__':
    main()
