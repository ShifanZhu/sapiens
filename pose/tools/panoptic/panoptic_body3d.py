# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""CMU Panoptic ``hdPose3d_stage1_coco19`` body JSON (``body3DScene_*.json``).

3D joints are in **Panoptic world coordinates (cm)**. Project them onto the Kinect
RGB image using ``calibration_*.json`` camera ``50_XX`` (``R``, ``t``) per CMU
docs ``x = K*(R*X + t)`` — **not** ``kcalibration['M_color']``, which maps
color↔depth on the device (see panoptic-toolbox ``demo_kinoptic_gen_ptcloud.m``).
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.panoptic.kinect_project import pose_project_2d

# MATLAB 1-based edges from CMU demo, converted to 0-based indices.
BODY_EDGES_COCO19 = (
    np.array(
        [
            [1, 2],
            [1, 4],
            [4, 5],
            [5, 6],
            [1, 3],
            [3, 7],
            [7, 8],
            [8, 9],
            [3, 13],
            [13, 14],
            [14, 15],
            [1, 10],
            [10, 11],
            [11, 12],
        ],
        dtype=np.int64,
    )
    - 1
)


def body3d_scene_path(sequence_dir: str, scene_id: int, pose_subdir: str) -> str:
    return os.path.join(
        sequence_dir, pose_subdir, f'body3DScene_{scene_id:08d}.json'
    )


_BODY3D_TIME_TABLE_CACHE: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}


def body3d_sorted_univtime_table(
    sequence_dir: str,
    pose_subdir: str,
) -> List[Tuple[float, int]]:
    """All ``(univTime, body3DScene id)`` pairs sorted by time (cached per folder)."""
    seq = os.path.abspath(sequence_dir)
    key = (seq, pose_subdir)
    if key in _BODY3D_TIME_TABLE_CACHE:
        return _BODY3D_TIME_TABLE_CACHE[key]

    pose_root = os.path.join(seq, pose_subdir)
    cache_file = os.path.join(pose_root, '.body3d_univtime_index.json')
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            raw = json.load(f)
        pairs = [(float(t), int(i)) for t, i in raw]
        _BODY3D_TIME_TABLE_CACHE[key] = pairs
        return pairs

    pat = re.compile(r'body3DScene_(\d+)\.json$')
    pairs: List[Tuple[float, int]] = []
    for path in glob.glob(os.path.join(pose_root, 'body3DScene_*.json')):
        m = pat.search(path)
        if not m:
            continue
        sid = int(m.group(1))
        with open(path) as f:
            t = float(json.load(f)['univTime'])
        pairs.append((t, sid))
    pairs.sort(key=lambda x: x[0])
    try:
        with open(cache_file, 'w') as f:
            json.dump(pairs, f)
    except OSError:
        pass
    _BODY3D_TIME_TABLE_CACHE[key] = pairs
    return pairs


def body3d_scene_id_for_hd_univ_time(
    sequence_dir: str,
    pose_subdir: str,
    psync: dict,
    hd_output_frame: int,
) -> Tuple[int, float, float]:
    """Pick ``body3DScene_XXXXXXXX`` whose ``univTime`` is closest to HD time.

    Uses ``psync['hd']['univ_time'][hd_output_frame + 1]`` (same anchor as
    ``hd_psync_index_for_output_frame``).

    Returns:
        ``(scene_id, target_univ_time_ms, abs_error_ms)``.
    """
    ut = np.asarray(psync['hd']['univ_time'], dtype=np.float64)
    py_idx = hd_output_frame + 1
    if py_idx < 0 or py_idx >= len(ut):
        raise IndexError(f'HD sync index {py_idx} out of range ({len(ut)} times)')
    target = float(ut[py_idx])
    table = body3d_sorted_univtime_table(sequence_dir, pose_subdir)
    if not table:
        raise FileNotFoundError(
            f'No body3DScene_*.json under {os.path.join(sequence_dir, pose_subdir)}')
    times = [x[0] for x in table]
    ids = [x[1] for x in table]
    i = bisect.bisect_left(times, target)
    cand: List[int] = []
    if i > 0:
        cand.append(i - 1)
    if i < len(times):
        cand.append(i)
    best_k = min(cand, key=lambda k: abs(times[k] - target))
    err = abs(times[best_k] - target)
    return ids[best_k], target, err


def load_body3d_scene(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def joints19_to_xyz_conf(flat) -> Tuple[np.ndarray, np.ndarray]:
    """Return (19, 3) xyz in cm and (19,) confidence."""
    a = np.asarray(flat, dtype=np.float64).reshape(19, 4)
    return a[:, :3], a[:, 3]


_KINECT50_RT_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}


def kinect_color_world_rt_cm(sequence_dir: str, kinect_node: int) -> Tuple[np.ndarray, np.ndarray]:
    """``R`` (3x3) and ``t`` (3x1) mapping Panoptic world (cm) -> Kinect color 3D.

    Loaded from ``calibration_*.json`` entry ``name == '50_XX'`` with
    ``XX = kinect_node`` (same index as ``kinectVideos/kinect_50_XX.mp4``).
    """
    seq = os.path.abspath(sequence_dir)
    key = (seq, kinect_node)
    if key in _KINECT50_RT_CACHE:
        return _KINECT50_RT_CACHE[key]

    hits = glob.glob(os.path.join(seq, 'calibration_*.json'))
    if not hits:
        raise FileNotFoundError(
            f'No calibration_*.json under {seq} (need Panoptic dome calibration '
            'for Kinect 50_XX extrinsics)')
    hits.sort()
    with open(hits[0]) as f:
        data = json.load(f)
    name = f'50_{kinect_node:02d}'
    for cam in data.get('cameras', []):
        if cam.get('name') == name:
            R = np.asarray(cam['R'], dtype=np.float64)
            t = np.asarray(cam['t'], dtype=np.float64).reshape(3, 1)
            _KINECT50_RT_CACHE[key] = (R, t)
            return R, t
    raise KeyError(
        f'No camera {name!r} in {hits[0]} (expected kinect-color entry in calibration file)')


def project_joints_world_to_uv_distorted(
    xyz_cm: np.ndarray,
    K_color: np.ndarray,
    dist_color: np.ndarray,
    R_world_to_color: np.ndarray,
    t_world_to_color: np.ndarray,
) -> np.ndarray:
    """World (cm) -> distorted Kinect RGB pixels (Nx2)."""
    return pose_project_2d(
        xyz_cm,
        np.asarray(R_world_to_color, dtype=np.float64),
        np.asarray(t_world_to_color, dtype=np.float64).reshape(3, 1),
        np.asarray(K_color, dtype=np.float64),
        np.asarray(dist_color, dtype=np.float64).reshape(-1),
        True,
    )


def undistort_uv(
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    new_K: np.ndarray,
) -> np.ndarray:
    """Map distorted UV to pixel coords matching ``cv2.undistort`` + *new_K*."""
    import cv2

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(uv, K, dist, P=np.asarray(new_K, dtype=np.float64))
    return out.reshape(-1, 2)


# ``cv2.line`` / ``cv2.circle`` pass coords to C++ as 32-bit ints; values outside
# this range make the bindings fail with "Can't parse 'pt1'/'pt2'".
_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1


def _cv_int_pt(uv: np.ndarray, i: int) -> Optional[Tuple[int, int]]:
    """Pixel coords as plain ints, or ``None`` if not drawable in OpenCV (OOR / bad proj)."""
    x = int(round(float(uv[i, 0])))
    y = int(round(float(uv[i, 1])))
    if x < _I32_MIN or x > _I32_MAX or y < _I32_MIN or y > _I32_MAX:
        return None
    return (x, y)


def draw_skeleton_bgr(
    bgr: np.ndarray,
    uv: np.ndarray,
    conf: np.ndarray,
    *,
    conf_thr: float = 0.2,
    joint_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw COCO-19 edges and joints on a copy of ``bgr``."""
    import cv2

    out = np.ascontiguousarray(bgr.copy())
    h, w = out.shape[:2]
    lt = int(line_thickness)
    jr = int(joint_radius)

    def ok(i: int) -> bool:
        return conf[i] >= conf_thr and np.all(np.isfinite(uv[i]))

    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 128, 0),
        (255, 0, 128),
        (200, 255, 0),
        (100, 100, 255),
        (255, 100, 100),
        (180, 180, 0),
        (0, 180, 180),
        (220, 0, 220),
    ]
    for e, ij in enumerate(BODY_EDGES_COCO19):
        i, j = int(ij[0]), int(ij[1])
        if ok(i) and ok(j):
            p0 = _cv_int_pt(uv, i)
            p1 = _cv_int_pt(uv, j)
            if p0 is None or p1 is None:
                continue
            b, g, r = palette[e % len(palette)]
            col = (int(b), int(g), int(r))
            cv2.line(out, p0, p1, col, lt, lineType=cv2.LINE_AA)

    for ji in range(19):
        if not ok(ji):
            continue
        pt = _cv_int_pt(uv, ji)
        if pt is None:
            continue
        cx, cy = pt
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(out, (cx, cy), jr, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    return out
