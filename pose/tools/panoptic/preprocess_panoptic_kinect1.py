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
8. Splat a z-buffered depth map in RGB pixel space using **undistorted** intrinsics
   (``cv2.getOptimalNewCameraMatrix`` + ``undistortPoints`` with ``P=new_K``) so
   ``depth_m_aligned_rgb.npy`` matches ``rgb_undistort``; a separate distorted
   splat is used only for ``overlay_bgr`` on raw distorted video frames.
   Supersampled downsampling can use **mean** or **min** pooling (``--splat-pool``);
   optional **mean hole-fill** (``--depth-mean-fill-ksize``) borrows neighboring depth.
9. Optional: project ``hdPose3d_stage1_coco19`` 3D joints onto the same Kinect RGB
   (``--gt-pose``). By default we pick the JSON whose ``univTime`` is closest to the
   HD time ``synctables['hd']['univ_time'][hd_output_frame + 1]`` (not a fixed index
   offset). Use ``--body3d-index-mode offset`` for legacy ``hd + offset`` naming.
   With ``--gt-pose``, HD frames with no usable JSON are skipped unless
   ``--include-hd-without-pose-json`` is set.

This script does **not** train Sapiens — it only writes portable arrays/images for a
later dataset loader (Part B).

Example::

  python pose/tools/panoptic/preprocess_panoptic_kinect1.py \\
    --sequence-dir /path/to/170224_haggling_b2 \\
    --output-dir /path/out/170224_haggling_b2_kinect1 \\
    --kinect-node 1 \\
    --all-hd-frames \\
    --gt-pose

  # Or a subset of frames only::

    --hd-output-frames 500 501 --max-frames 2
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
from tools.panoptic.kinect_project import (
    splat_depth_to_rgb,
    undistort_color_uv_to_new_k,
    unproject_depth_release,
)
from tools.panoptic.panoptic_body3d import (
    body3d_scene_id_for_hd_univ_time,
    body3d_scene_path,
    draw_skeleton_bgr,
    joints19_to_xyz_conf,
    kinect_color_world_rt_cm,
    load_body3d_scene,
    project_joints_world_to_uv_distorted,
    undistort_uv,
)
from tools.panoptic.sync_tables import (
    all_hd_output_frames,
    hd_psync_index_for_output_frame,
    load_ksync,
    load_psync,
    select_kinect_frames,
)


def _hd_frames_with_pose_json(
    frames: List[int],
    sequence_dir: str,
    pose_subdir: str,
    psync: dict,
    *,
    body3d_index_mode: str,
    body3d_scene_offset: int,
    max_univ_time_error_ms: float,
) -> List[int]:
    """Keep HD frames that have a usable GT pose JSON (see ``body3d_index_mode``)."""
    out: List[int] = []
    if body3d_index_mode == 'univ_time':
        for hd_of in frames:
            try:
                bid, _tgt, err = body3d_scene_id_for_hd_univ_time(
                    sequence_dir, pose_subdir, psync, hd_of)
            except (IndexError, FileNotFoundError, ValueError):
                continue
            if err > max_univ_time_error_ms:
                continue
            path = body3d_scene_path(sequence_dir, bid, pose_subdir)
            if os.path.isfile(path):
                out.append(hd_of)
    else:
        for hd_of in frames:
            body3d_id = hd_of + body3d_scene_offset
            path = body3d_scene_path(sequence_dir, body3d_id, pose_subdir)
            if os.path.isfile(path):
                out.append(hd_of)
    return out


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


def _compute_new_k(
    K: np.ndarray, dist: np.ndarray, rgb_w: int, rgb_h: int
) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (int(rgb_w), int(rgb_h)), alpha=1.0)
    return new_k


def _undistort_rgb_with_new_k(
    bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, new_k: np.ndarray
) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    return cv2.undistort(bgr, K, dist, None, new_k)


def _depth_mean_fill_holes(depth_m: np.ndarray, ksize: int) -> np.ndarray:
    """Fill pixels with no depth using the local mean of valid neighbors (box window).

    Uses normalized convolution: each output pixel is (sum of neighbor depths) /
    (count of valid neighbors) in a ``ksize×ksize`` window. Only **empty** pixels
    (zero or non-finite) are overwritten when the local mean is valid and positive.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError('depth mean-fill ksize must be an odd integer >= 3')
    d = depth_m.astype(np.float64)
    valid = ((depth_m > 0) & np.isfinite(depth_m)).astype(np.float32)
    num = cv2.boxFilter(
        d * valid, ddepth=-1, ksize=(ksize, ksize), normalize=False)
    den = cv2.boxFilter(valid, ddepth=-1, ksize=(ksize, ksize), normalize=False)
    filled = np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=np.float64),
        where=den > 1e-9,
    )
    hole = (depth_m <= 0) | ~np.isfinite(depth_m)
    take = hole & (filled > 0) & np.isfinite(filled)
    return np.where(take, filled, d).astype(np.float32)


def _depth_overlay(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    *,
    heat_weight: float,
) -> np.ndarray:
    """False-color overlay of depth (metres) on BGR.

    ``heat_weight`` in (0,1): blend is ``(1-w)*rgb + w*jet`` on valid-depth pixels.
    """
    w = float(heat_weight)
    w = min(0.95, max(0.05, w))
    # splat_depth_to_rgb uses inf for empty cells; exclude from normalization
    mask = (depth_m > 0) & np.isfinite(depth_m)
    if not np.any(mask):
        return bgr
    z = depth_m.copy()
    z[~mask] = np.nan
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    norm = (z - zmin) / (zmax - zmin + 1e-6)
    norm = np.clip(norm, 0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat[~mask] = 0
    out = bgr.copy()
    blended = cv2.addWeighted(bgr, 1.0 - w, heat, w, 0)
    out[mask] = blended[mask]
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
    body3d_index_mode: str,
    body3d_scene_offset: int,
    body3d_max_univ_time_error_ms: float,
    overlay_heat_weight: float,
    splat_supersample: int = 2,
    splat_pool: str = 'mean',
    depth_mean_fill_ksize: int = 0,
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

    if relax_sync:
        kw = dict(
            max_color_dt_ms=300.0,
            max_depth_dt_ms=170.0,
            max_color_depth_skew_ms=65.0,
        )
    else:
        # HD univ_time step ~33 ms; default 30 ms was tighter than one tick and
        # often failed mid-sequence. Kinect color vs depth timestamps can skew
        # tens of ms at the argmin pair (see kinoptic sync tables).
        kw = dict(
            max_color_dt_ms=40.0,
            max_depth_dt_ms=20.0,
            max_color_depth_skew_ms=50.0,
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

    rgb_h, rgb_w = int(cam['color_height']), int(cam['color_width'])
    Kc = np.asarray(cam['K_color'], dtype=np.float64)
    distc = np.asarray(cam['distCoeffs_color'], dtype=np.float64).reshape(-1)
    new_k = _compute_new_k(Kc, distc, rgb_w, rgb_h)
    rgb_u = _undistort_rgb_with_new_k(rgb_bgr, Kc, distc, new_k)

    _p3d, uv_dist = unproject_depth_release(depth_mm, cam, gen_color_map=True)
    assert uv_dist is not None
    uv_undist = undistort_color_uv_to_new_k(uv_dist, Kc, distc, new_k)
    aligned_m, _hits = splat_depth_to_rgb(
        depth_mm,
        uv_undist,
        rgb_h=rgb_h,
        rgb_w=rgb_w,
        supersample=splat_supersample,
        pool=splat_pool,
    )
    aligned_m_dist, _hits_d = splat_depth_to_rgb(
        depth_mm,
        uv_dist,
        rgb_h=rgb_h,
        rgb_w=rgb_w,
        supersample=splat_supersample,
        pool=splat_pool,
    )
    if depth_mean_fill_ksize > 0:
        aligned_m = _depth_mean_fill_holes(aligned_m, depth_mean_fill_ksize)
        aligned_m_dist = _depth_mean_fill_holes(aligned_m_dist, depth_mean_fill_ksize)

    stem = f'hd{hd_output_frame:08d}_c{c1:06d}_d{d1:06d}'
    files: Dict[str, str] = {}
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f'{stem}_rgb_bgr.jpg'), rgb_bgr)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_rgb_undistort.jpg'), rgb_u)
    np.save(os.path.join(output_dir, f'{stem}_depth_mm_kinect.npy'), depth_mm.astype(np.float32))
    np.save(os.path.join(output_dir, f'{stem}_depth_m_aligned_rgb.npy'), aligned_m)
    np.save(
        os.path.join(output_dir, f'{stem}_depth_m_aligned_rgb_distorted.npy'),
        aligned_m_dist.astype(np.float32),
    )
    overlay = _depth_overlay(
        rgb_bgr, aligned_m_dist, heat_weight=overlay_heat_weight)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_overlay_bgr.jpg'), overlay)
    overlay_u = _depth_overlay(rgb_u, aligned_m, heat_weight=overlay_heat_weight)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_overlay_undistort.jpg'), overlay_u)

    gt_pose_info: Dict[str, Any] = {'enabled': gt_pose}
    if gt_pose:
        if body3d_index_mode == 'univ_time':
            body3d_id, tgt_ut, ut_err = body3d_scene_id_for_hd_univ_time(
                sequence_dir, pose_subdir, psync, hd_output_frame)
            gt_pose_info['body3d_univ_time_error_ms'] = ut_err
            gt_pose_info['hd_univ_time_ms'] = tgt_ut
        else:
            body3d_id = hd_output_frame + body3d_scene_offset
            ut_err = None
            tgt_ut = None
        pose_path = body3d_scene_path(sequence_dir, body3d_id, pose_subdir)
        gt_pose_info['pose_json'] = pose_path
        gt_pose_info['body3d_scene_index'] = body3d_id
        gt_pose_info['body3d_index_mode'] = body3d_index_mode
        if body3d_index_mode == 'offset':
            gt_pose_info['body3d_scene_offset'] = body3d_scene_offset
        time_too_loose = (
            body3d_index_mode == 'univ_time'
            and ut_err is not None
            and ut_err > body3d_max_univ_time_error_ms
        )
        if time_too_loose:
            gt_pose_info['status'] = 'univ_time_error'
            print(
                f'Warning: GT pose |ΔunivTime|={ut_err:.1f} ms > '
                f'{body3d_max_univ_time_error_ms} (skip skeleton): {pose_path}',
                flush=True,
            )
        elif not os.path.isfile(pose_path):
            gt_pose_info['status'] = 'missing_json'
            print(f'Warning: GT pose JSON not found (skip skeleton overlay): {pose_path}')
        else:
            scene = load_body3d_scene(pose_path)
            R_w2c, t_w2c = kinect_color_world_rt_cm(sequence_dir, kinect_node)
            pose_bgr = rgb_bgr.copy()
            pose_u = rgb_u.copy()
            for body in scene.get('bodies', []):
                xyz, conf = joints19_to_xyz_conf(body['joints19'])
                uv_d = project_joints_world_to_uv_distorted(
                    xyz,
                    cam['K_color'],
                    cam['distCoeffs_color'],
                    R_w2c,
                    t_w2c,
                )
                pose_bgr = draw_skeleton_bgr(
                    pose_bgr, uv_d, conf, conf_thr=pose_conf_thr)
                uv_u = undistort_uv(uv_d, Kc, distc, new_k)  # same new_k as depth splat
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
        'overlay_heat_weight': overlay_heat_weight,
        'splat_supersample': splat_supersample,
        'splat_pool': splat_pool,
        'depth_mean_fill_ksize': depth_mean_fill_ksize,
        'body3d_index_mode': body3d_index_mode,
        'hd_univ_time': sel_t,
        'color_frame_1based': c1,
        'depth_frame_1based': d1,
        'sync_debug': dbg,
        'depthdata_path': depth_path,
        'video_path': video_path,
        'gt_pose': gt_pose_info,
        'depth_uv_space': 'undistorted_new_k',
        'files': {
            'rgb_bgr': f'{stem}_rgb_bgr.jpg',
            'rgb_undistort': f'{stem}_rgb_undistort.jpg',
            'depth_mm_kinect': f'{stem}_depth_mm_kinect.npy',
            'depth_m_aligned_rgb': f'{stem}_depth_m_aligned_rgb.npy',
            'depth_m_aligned_rgb_distorted': f'{stem}_depth_m_aligned_rgb_distorted.npy',
            'overlay_bgr': f'{stem}_overlay_bgr.jpg',
            'overlay_undistort': f'{stem}_overlay_undistort.jpg',
            **files,
        },
    }
    if body3d_index_mode == 'offset':
        meta['body3d_scene_offset'] = body3d_scene_offset
    else:
        meta['body3d_max_univ_time_error_ms'] = body3d_max_univ_time_error_ms
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
        nargs='*',
        default=None,
        metavar='N',
        help='HD frame labels (e.g. 500 501). Omit when using --all-hd-frames.',
    )
    p.add_argument(
        '--all-hd-frames',
        action='store_true',
        help='Process every valid HD label from synctables (length of HD univ_time '
        'minus one). Use --max-frames to cap for testing.',
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
        '--body3d-index-mode',
        choices=('univ_time', 'offset'),
        default='univ_time',
        help='How to choose body3DScene_XXXXXXXX.json: ``univ_time`` (default) picks '
        'the file whose univTime is nearest to this HD frame’s synctables time; '
        '``offset`` uses index = hd-output-frame + --body3d-scene-offset (legacy).',
    )
    p.add_argument(
        '--body3d-scene-offset',
        type=int,
        default=1,
        help='Only for --body3d-index-mode offset: scene index = hd-output-frame + '
        'this value (default %(default)s). Ignored for univ_time mode.',
    )
    p.add_argument(
        '--body3d-max-univ-time-error-ms',
        type=float,
        default=500.0,
        metavar='MS',
        help='univ_time mode: skip GT overlay when min |pose.univTime − HD.univTime| '
        'exceeds this (default: %(default)s). Use a large value (e.g. 20000) if '
        'early HD frames must use the nearest pose file despite a large gap.',
    )
    p.add_argument(
        '--include-hd-without-pose-json',
        action='store_true',
        help='With --gt-pose, still process every selected HD frame even when '
        'the matching body3DScene_*.json is missing (warnings only). '
        'Default is to skip those frames.',
    )
    p.add_argument(
        '--overlay-heat-weight',
        type=float,
        default=0.78,
        metavar='W',
        help='Depth QA overlay: blend weight for Jet colormap on pixels with '
        'valid aligned depth (0–1). RGB weight is 1−W. Higher = stronger depth '
        'color (default: %(default)s; was ~0.55 before).',
    )
    p.add_argument(
        '--splat-supersample',
        type=int,
        default=2,
        metavar='S',
        help='Depth→RGB splat grid multiplier (default: %(default)s): splat min-depth '
        'on an S× finer grid, then min-pool to RGB pixels. Reduces silhouette bleed '
        'without masking bodies. Use 1 for legacy single-bin rounding.',
    )
    p.add_argument(
        '--splat-pool',
        choices=('min', 'mean'),
        default='mean',
        help='How to combine subcells when downsampling the supersampled splat to RGB '
        'resolution (ignored when --splat-supersample is 1). ``mean`` averages finite '
        'subcell depths (denser, smoother; default). ``min`` keeps closest depth per '
        'macro pixel (sharper z-buffer).',
    )
    p.add_argument(
        '--depth-mean-fill-ksize',
        type=int,
        default=0,
        metavar='K',
        help='If K>0 (odd, >=3), fill empty depth pixels using the local mean of valid '
        'neighbors in a K×K box (normalized convolution). 0 disables (default).',
    )
    p.add_argument(
        '--skip-on-sync-error',
        action='store_true',
        help='If HD→Kinect sync matching fails for a frame, log and continue '
        'instead of aborting (manifest records status=sync_skipped).',
    )
    p.add_argument(
        '--append-manifest',
        action='store_true',
        help='If manifest.json already exists under --output-dir, load it and '
        'append new entries (for resuming after a partial run).',
    )
    args = p.parse_args(argv)
    if args.all_hd_frames and args.hd_output_frames:
        p.error('Use either --all-hd-frames or --hd-output-frames, not both')
    if not 0.0 < args.overlay_heat_weight < 1.0:
        p.error('--overlay-heat-weight must be strictly between 0 and 1')
    if args.splat_supersample < 1:
        p.error('--splat-supersample must be >= 1')
    if args.depth_mean_fill_ksize < 0:
        p.error('--depth-mean-fill-ksize must be >= 0')
    if args.depth_mean_fill_ksize > 0:
        if args.depth_mean_fill_ksize % 2 == 0 or args.depth_mean_fill_ksize < 3:
            p.error('--depth-mean-fill-ksize must be 0 or an odd integer >= 3')

    seq = os.path.abspath(args.sequence_dir)
    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    ksync = load_ksync(seq)
    psync = load_psync(seq)
    if args.all_hd_frames:
        frames = all_hd_output_frames(psync)
        print(
            f'--all-hd-frames: {len(frames)} HD frames from synctables '
            f'(hd_output_frame 0..{len(frames)-1 if frames else 0}).',
            flush=True,
        )
    elif args.hd_output_frames:
        frames = list(args.hd_output_frames)
    else:
        p.error('Provide --hd-output-frames N [N ...] or --all-hd-frames')
    if (
        args.gt_pose
        and not args.include_hd_without_pose_json
    ):
        n_before = len(frames)
        frames = _hd_frames_with_pose_json(
            frames,
            seq,
            args.pose_subdir,
            psync,
            body3d_index_mode=args.body3d_index_mode,
            body3d_scene_offset=args.body3d_scene_offset,
            max_univ_time_error_ms=args.body3d_max_univ_time_error_ms,
        )
        skipped = n_before - len(frames)
        if skipped:
            print(
                f'--gt-pose: skipping {skipped} HD frame(s) with no usable '
                f'body3DScene under {args.pose_subdir!r} '
                f'({len(frames)} remaining).',
                flush=True,
            )
        if not frames:
            p.error(
                'No HD frames left after requiring pose JSON files '
                '(check --pose-subdir, --body3d-index-mode, '
                '--body3d-max-univ-time-error-ms, or use '
                '--include-hd-without-pose-json).')
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    manifest_path = os.path.join(out_root, 'manifest.json')
    manifest: List[Any] = []
    if args.append_manifest and os.path.isfile(manifest_path):
        try:
            manifest = json.load(open(manifest_path))
        except json.JSONDecodeError:
            manifest = []
    n_total = len(frames)
    n_done0 = len(manifest)
    try:
        for k, hd_of in enumerate(frames, start=1):
            subdir = os.path.join(out_root, f'frame_{hd_of:08d}')
            try:
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
                    body3d_index_mode=args.body3d_index_mode,
                    body3d_scene_offset=args.body3d_scene_offset,
                    body3d_max_univ_time_error_ms=args.body3d_max_univ_time_error_ms,
                    overlay_heat_weight=args.overlay_heat_weight,
                    splat_supersample=args.splat_supersample,
                    splat_pool=args.splat_pool,
                    depth_mean_fill_ksize=args.depth_mean_fill_ksize,
                )
            except RuntimeError as err:
                if args.skip_on_sync_error and (
                    'sync match too loose' in str(err)
                    or 'color/depth skew too large' in str(err)
                ):
                    rec = {
                        'hd_output_frame': hd_of,
                        'status': 'sync_skipped',
                        'error': str(err),
                        'output_dir': subdir,
                    }
                    manifest.append(rec)
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest, f, indent=2)
                    print(
                        f'[{n_done0 + k}/{n_done0 + n_total}] '
                        f'hd_output_frame={hd_of} SKIP (sync): {err}',
                        flush=True,
                    )
                    continue
                raise
            manifest.append(meta)
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(
                f'[{n_done0 + k}/{n_done0 + n_total}] hd_output_frame={hd_of} '
                f'wrote {subdir} (manifest.json updated)',
                flush=True,
            )
    except KeyboardInterrupt:
        if manifest:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(
                f'\nStopped after {len(manifest)} frame(s); partial manifest saved to '
                f'{manifest_path}',
                flush=True,
            )
        raise
    print(f'Wrote {len(manifest)} frames under {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
