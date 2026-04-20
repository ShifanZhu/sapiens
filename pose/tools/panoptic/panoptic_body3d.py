# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""CMU Panoptic ``hdPose3d_stage1_coco19`` body JSON (``body3DScene_*.json``).

3D joints are in **world coordinates (cm)**, same frame as ``kcalibration`` /
``unprojectDepth_release``. See panoptic-toolbox ``demo_3Dkeypoints_3dview.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

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


def body3d_scene_path(sequence_dir: str, hd_frame_index: int, pose_subdir: str) -> str:
    return os.path.join(
        sequence_dir, pose_subdir, f'body3DScene_{hd_frame_index:08d}.json'
    )


def load_body3d_scene(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def joints19_to_xyz_conf(flat) -> Tuple[np.ndarray, np.ndarray]:
    """Return (19, 3) xyz in cm and (19,) confidence."""
    a = np.asarray(flat, dtype=np.float64).reshape(19, 4)
    return a[:, :3], a[:, 3]


def project_joints_world_to_uv_distorted(
    xyz_cm: np.ndarray,
    cam_calib: dict,
) -> np.ndarray:
    """World (cm) -> distorted Kinect RGB pixels (Nx2)."""
    M = np.asarray(cam_calib['M_color'], dtype=np.float64)
    R = M[0:3, 0:3]
    t = M[0:3, 3:4]
    K = np.asarray(cam_calib['K_color'], dtype=np.float64)
    dist = np.asarray(cam_calib['distCoeffs_color'], dtype=np.float64).reshape(-1)
    return pose_project_2d(xyz_cm, R, t, K, dist, True)


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

    out = bgr.copy()
    h, w = out.shape[:2]

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
            p0 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p1 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
            col = palette[e % len(palette)]
            cv2.line(out, p0, p1, col, line_thickness, lineType=cv2.LINE_AA)

    for ji in range(19):
        if not ok(ji):
            continue
        cx, cy = int(round(uv[ji, 0])), int(round(uv[ji, 1]))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(out, (cx, cy), joint_radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    return out
