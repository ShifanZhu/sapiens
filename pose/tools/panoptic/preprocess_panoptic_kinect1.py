#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess one CMU Panoptic Kinect1 sequence: align depth to RGB (toolbox math).

Pipeline per HD-synced frame (see CMU ``demo_kinoptic_gen_ptcloud.m``):

1. Pick universal time from ``synctables_*.json`` (HD timeline).
2. Map to Kinect color / depth frame indices via ``ksynctables_*.json``.
3. Load RGB frame from ``kinectVideos/kinect_50_XX.mp4``.
4. Load depth slice from ``kinect_shared_depth/KINECTNODEX/depthdata.dat``.
5. Unproject depth with ``kcalibration`` (same as MATLAB ``unprojectDepth_release``).
6. Project 3D points onto the Kinect RGB sensor (``PoseProject2D``).
7. Optionally undistort the RGB image with OpenCV (``K_color``, ``distCoeffs_color``).
8. Splat a z-buffered depth map in RGB pixel space; save overlay for QA.
9. Optional: project ``hdPose3d_stage1_coco19`` 3D joints onto the same Kinect RGB
   (``--gt-pose``) and save ``*_pose_gt_*.jpg`` for label QA. Pose JSON filenames use
   ``body3DScene_{(hd-output-frame + body3d-scene-offset):08d}.json`` when offset is
   non-zero (releases often start at 1000+).

This script does **not** train Sapiens — it only writes portable arrays/images for a
later dataset loader (Part B).

Example::

  python pose/tools/panoptic/preprocess_panoptic_kinect1.py \\
    --sequence-dir /path/to/170224_haggling_b2 \\
    --output-dir /path/out/170224_haggling_b2_kinect1 \\
    --kinect-node 1 \\
    --hd-output-frames 500 501 \\
    --max-frames 2 \\
    --gt-pose \\
    --body3d-scene-offset 546
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# ``tools.*`` resolves when ``pose/`` is on ``sys.path`` (same as ``python tools/train.py``).
_POSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _POSE_DIR not in sys.path:
    sys.path.insert(0, _POSE_DIR)

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        'preprocess_panoptic_kinect1 requires OpenCV (pip install opencv-python-headless)'
    ) from e

from tools.panoptic.depth_io import (
    depth_frame_count,
    get_depthdata_path,
    read_depth_frame_1based,
)
from tools.panoptic.kinect_project import splat_depth_to_rgb, unproject_depth_release
from tools.panoptic.panoptic_body3d import (
    body3d_scene_path,
    draw_skeleton_bgr,
    joints19_to_xyz_conf,
    load_body3d_scene,
    project_joints_world_to_uv_distorted,
    undistort_uv,
)
from tools.panoptic.sync_tables import (
    hd_psync_index_for_output_frame,
    load_ksync,
    load_psync,
    select_kinect_frames,
)


def _load_kcal(sequence_dir: str) -> dict:
    import glob
    hits = glob.glob(os.path.join(sequence_dir, 'kcalibration*.json'))
    if not hits:
        raise FileNotFoundError(f'No kcalibration*.json under {sequence_dir}')
    hits.sort()
    return json.load(open(hits[0]))


def _kinect_video_path(sequence_dir: str, kinect_node: int) -> str:
    return os.path.join(
        sequence_dir, 'kinectVideos', f'kinect_50_{kinect_node:02d}.mp4')


def _read_rgb_frame(video_path: str, frame_idx_0based: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx_0based))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(
            f'Failed to read frame {frame_idx_0based} from {video_path}')
    return bgr


def _undistort_rgb_and_new_k(
    bgr: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (bgr.shape[1], bgr.shape[0]), alpha=1.0)
    return cv2.undistort(bgr, K, dist, None, new_k), new_k


def _depth_overlay(bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """False-color overlay of depth (metres) on BGR."""
    mask = depth_m > 0
    if not np.any(mask):
        return bgr
    z = depth_m.copy()
    z[~mask] = np.nan
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    norm = (z - zmin) / (zmax - zmin + 1e-6)
    norm = np.clip(norm, 0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat[~mask] = 0
    out = bgr.copy()
    out[mask] = cv2.addWeighted(bgr, 0.45, heat, 0.55, 0)[mask]
    return out


def run_one_frame(
    sequence_dir: str,
    output_dir: str,
    kinect_node: int,
    hd_output_frame: int,
    ksync: dict,
    psync: dict,
    *,
    relax_sync: bool,
    gt_pose: bool,
    pose_subdir: str,
    pose_conf_thr: float,
    body3d_scene_offset: int,
) -> Dict[str, Any]:
    kcal = _load_kcal(sequence_dir)
    sensors: List[dict] = kcal['sensors']
    if kinect_node < 1 or kinect_node > len(sensors):
        raise ValueError(f'kinect_node {kinect_node} out of range 1..{len(sensors)}')
    cam = sensors[kinect_node - 1]

    hd_ut = psync['hd']['univ_time']
    py_idx = hd_psync_index_for_output_frame(hd_output_frame)
    if py_idx < 0 or py_idx >= len(hd_ut):
        raise IndexError(
            f'HD sync index {py_idx} out of range for {len(hd_ut)} HD times '
            f'(hd_output_frame={hd_output_frame})')
    sel_t = float(hd_ut[py_idx])

    kw = dict(
        max_color_dt_ms=300.0 if relax_sync else 30.0,
        max_depth_dt_ms=170.0 if relax_sync else 17.0,
        max_color_depth_skew_ms=65.0 if relax_sync else 6.5,
    )
    c1, d1, dbg = select_kinect_frames(ksync, sel_t, kinect_node, **kw)

    depth_path = get_depthdata_path(sequence_dir, kinect_node)
    n_depth = depth_frame_count(os.path.getsize(depth_path))
    if d1 > n_depth:
        raise IndexError(
            f'Depth index {d1} exceeds depth frames {n_depth} in {depth_path}')

    video_path = _kinect_video_path(sequence_dir, kinect_node)
    rgb_bgr = _read_rgb_frame(video_path, c1 - 1)
    depth_mm = read_depth_frame_1based(depth_path, d1)

    _p3d, uv_rgb = unproject_depth_release(depth_mm, cam, gen_color_map=True)
    assert uv_rgb is not None
    rgb_h, rgb_w = int(cam['color_height']), int(cam['color_width'])
    aligned_m, _hits = splat_depth_to_rgb(depth_mm, uv_rgb, rgb_h=rgb_h, rgb_w=rgb_w)

    Kc = np.asarray(cam['K_color'], dtype=np.float64)
    distc = np.asarray(cam['distCoeffs_color'], dtype=np.float64).reshape(-1)
    rgb_u, new_k = _undistort_rgb_and_new_k(rgb_bgr, Kc, distc)

    stem = f'hd{hd_output_frame:08d}_c{c1:06d}_d{d1:06d}'
    files: Dict[str, str] = {}
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f'{stem}_rgb_bgr.jpg'), rgb_bgr)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_rgb_undistort.jpg'), rgb_u)
    np.save(os.path.join(output_dir, f'{stem}_depth_mm_kinect.npy'), depth_mm.astype(np.float32))
    np.save(os.path.join(output_dir, f'{stem}_depth_m_aligned_rgb.npy'), aligned_m)
    overlay = _depth_overlay(rgb_bgr, aligned_m)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_overlay_bgr.jpg'), overlay)
    overlay_u = _depth_overlay(rgb_u, aligned_m)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_overlay_undistort.jpg'), overlay_u)

    gt_pose_info: Dict[str, Any] = {'enabled': gt_pose}
    if gt_pose:
        body3d_id = hd_output_frame + body3d_scene_offset
        pose_path = body3d_scene_path(sequence_dir, body3d_id, pose_subdir)
        gt_pose_info['pose_json'] = pose_path
        gt_pose_info['body3d_scene_index'] = body3d_id
        gt_pose_info['body3d_scene_offset'] = body3d_scene_offset
        if not os.path.isfile(pose_path):
            gt_pose_info['status'] = 'missing_json'
            print(f'Warning: GT pose JSON not found (skip skeleton overlay): {pose_path}')
        else:
            scene = load_body3d_scene(pose_path)
            pose_bgr = rgb_bgr.copy()
            pose_u = rgb_u.copy()
            for body in scene.get('bodies', []):
                xyz, conf = joints19_to_xyz_conf(body['joints19'])
                uv_d = project_joints_world_to_uv_distorted(xyz, cam)
                pose_bgr = draw_skeleton_bgr(
                    pose_bgr, uv_d, conf, conf_thr=pose_conf_thr)
                uv_u = undistort_uv(uv_d, Kc, distc, new_k)
                pose_u = draw_skeleton_bgr(
                    pose_u, uv_u, conf, conf_thr=pose_conf_thr)
            gt_pose_info['status'] = 'ok'
            gt_pose_info['num_bodies'] = len(scene.get('bodies', []))
            cv2.imwrite(
                os.path.join(output_dir, f'{stem}_pose_gt_bgr.jpg'), pose_bgr)
            cv2.imwrite(
                os.path.join(output_dir, f'{stem}_pose_gt_undistort.jpg'), pose_u)
            files['pose_gt_bgr'] = f'{stem}_pose_gt_bgr.jpg'
            files['pose_gt_undistort'] = f'{stem}_pose_gt_undistort.jpg'

    meta = {
        'sequence_dir': os.path.abspath(sequence_dir),
        'kinect_node': kinect_node,
        'hd_output_frame': hd_output_frame,
        'body3d_scene_offset': body3d_scene_offset,
        'hd_univ_time': sel_t,
        'color_frame_1based': c1,
        'depth_frame_1based': d1,
        'sync_debug': dbg,
        'depthdata_path': depth_path,
        'video_path': video_path,
        'gt_pose': gt_pose_info,
        'files': {
            'rgb_bgr': f'{stem}_rgb_bgr.jpg',
            'rgb_undistort': f'{stem}_rgb_undistort.jpg',
            'depth_mm_kinect': f'{stem}_depth_mm_kinect.npy',
            'depth_m_aligned_rgb': f'{stem}_depth_m_aligned_rgb.npy',
            'overlay_bgr': f'{stem}_overlay_bgr.jpg',
            'overlay_undistort': f'{stem}_overlay_undistort.jpg',
            **files,
        },
    }
    with open(os.path.join(output_dir, f'{stem}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-dir', required=True, help='One Panoptic sequence folder')
    p.add_argument('--output-dir', required=True, help='Where to write preprocessed assets')
    p.add_argument('--kinect-node', type=int, default=1, help='1..10 (default: 1 = kinect_50_01)')
    p.add_argument(
        '--hd-output-frames',
        type=int,
        nargs='+',
        required=True,
        help='HD frame labels matching toolbox output numbering (e.g. 500 501). '
        'Synced via synctables using the same +2 rule as demo_kinoptic_gen_ptcloud.',
    )
    p.add_argument(
        '--relax-sync',
        action='store_true',
        help='Use looser sync thresholds (still selects nearest color/depth frames).',
    )
    p.add_argument('--max-frames', type=int, default=0,
                   help='Process at most this many frames (0 = all listed).')
    p.add_argument(
        '--gt-pose',
        action='store_true',
        help='Overlay hdPose3d_stage1_coco19 GT skeleton (body3DScene_XXXXXXXX.json) '
        'on Kinect RGB using the same projection as depth.',
    )
    p.add_argument(
        '--pose-subdir',
        default='hdPose3d_stage1_coco19',
        help='Subfolder under sequence-dir with body3DScene_*.json (default: %(default)s).',
    )
    p.add_argument(
        '--pose-conf-thr',
        type=float,
        default=0.2,
        help='Min joint confidence to draw edges/points (default: %(default)s).',
    )
    p.add_argument(
        '--body3d-scene-offset',
        type=int,
        default=0,
        help='GT pose file index = hd-output-frame + this offset. '
        'Many releases use body3DScene indices that do not start at 0 '
        '(e.g. first file body3DScene_00001046.json → use offset 546 when '
        'pairing with hd-output-frame 500). Default: %(default)s.',
    )
    args = p.parse_args(argv)

    seq = os.path.abspath(args.sequence_dir)
    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    ksync = load_ksync(seq)
    psync = load_psync(seq)
    frames = args.hd_output_frames
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    manifest = []
    for hd_of in frames:
        subdir = os.path.join(out_root, f'frame_{hd_of:08d}')
        meta = run_one_frame(
            seq,
            subdir,
            args.kinect_node,
            hd_of,
            ksync,
            psync,
            relax_sync=args.relax_sync,
            gt_pose=args.gt_pose,
            pose_subdir=args.pose_subdir,
            pose_conf_thr=args.pose_conf_thr,
            body3d_scene_offset=args.body3d_scene_offset,
        )
        manifest.append(meta)

    with open(os.path.join(out_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {len(manifest)} frames under {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
