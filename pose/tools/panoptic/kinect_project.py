# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Geometry ports from CMU PanopticStudio Toolbox (MATLAB):
# ``unproject.m``, ``unprojectDepth_release.m``, ``PoseProject2D.m``.
"""Kinect depth unprojection and projection to RGB (Panoptic calibration)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_float_mat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def unproject_depth_pixels(
    p2d: np.ndarray,
    K_depth: np.ndarray,
    dist_depth: np.ndarray,
    M_depth_3x4: np.ndarray,
) -> np.ndarray:
    """MATLAB ``unproject.m`` — depth pixels + Z -> 3D (Panoptic world, cm).

    ``p2d`` is Nx3: [u_pix, v_pix, depth_mm].
    """
    p2d = np.asarray(p2d, dtype=np.float64)
    if p2d.ndim != 2 or p2d.shape[1] != 3:
        raise ValueError(f'p2d must be (N,3), got {p2d.shape}')

    cam_k = _as_float_mat(K_depth)
    cam_m = _as_float_mat(M_depth_3x4)
    dist = np.asarray(dist_depth, dtype=np.float64).reshape(-1)

    # k vector layout matches unproject.m (12 coeffs; only first 5 from file)
    k = np.zeros(12, dtype=np.float64)
    k[:5] = dist[:5]

    # Normalized points: inv(K) * [u,v,1]^T
    uv1 = np.stack([p2d[:, 0], p2d[:, 1], np.ones(len(p2d))], axis=0)
    pn2d = (np.linalg.inv(cam_k) @ uv1).T[:, :2]

    x0 = pn2d[:, 0].copy()
    y0 = pn2d[:, 1].copy()
    x = x0.copy()
    y = y0.copy()

    for _ in range(5):
        r2 = x * x + y * y
        icdist = (
            (1.0 + ((k[1 + 7] * r2 + k[1 + 6]) * r2 + k[1 + 5]) * r2)
            / (1.0 + ((k[1 + 4] * r2 + k[1 + 1]) * r2 + k[1 + 0]) * r2)
        )
        delta_x = (
            2 * k[1 + 2] * x * y
            + k[1 + 3] * (r2 + 2 * x * x)
            + k[1 + 8] * r2
            + k[1 + 9] * r2 * r2
        )
        delta_y = (
            k[1 + 2] * (r2 + 2 * y * y)
            + 2 * k[1 + 3] * x * y
            + k[1 + 10] * r2
            + k[1 + 11] * r2 * r2
        )
        x = (x0 - delta_x) * icdist
        y = (y0 - delta_y) * icdist

    pn2d = np.stack([x, y], axis=1)
    z_m = p2d[:, 2] * 0.001
    p3d = np.concatenate(
        [pn2d * z_m[:, np.newaxis], z_m[:, np.newaxis]], axis=1)

    m4 = np.vstack([cam_m, np.array([[0.0, 0.0, 0.0, 1.0]])])
    p_h = np.concatenate([p3d, np.ones((len(p3d), 1))], axis=1)
    out = (np.linalg.inv(m4) @ p_h.T).T[:, :3]
    return out


def pose_project_2d(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist_coef: np.ndarray,
    apply_distort: bool,
) -> np.ndarray:
    """MATLAB ``PoseProject2D.m`` — world points (cm) -> image pixels."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    R = _as_float_mat(R)
    t = _as_float_mat(t).reshape(3, 1)
    K = _as_float_mat(K)

    x = (R @ pts.T + t).T  # N x 3
    xp = x[:, :2] / (x[:, 2:3] + 1e-12)

    if apply_distort:
        k = np.asarray(dist_coef, dtype=np.float64).reshape(-1)
        X2 = xp[:, 0] * xp[:, 0]
        Y2 = xp[:, 1] * xp[:, 1]
        XY = xp[:, 0] * xp[:, 1]
        r2 = X2 + Y2
        r4 = r2 * r2
        r6 = r2 * r4
        radial = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6
        tangential_x = 2.0 * k[2] * XY + k[3] * (r2 + 2.0 * X2)
        tangential_y = 2.0 * k[3] * XY + k[2] * (r2 + 2.0 * Y2)
        xp = np.stack(
            [radial * xp[:, 0] + tangential_x, radial * xp[:, 1] + tangential_y],
            axis=1,
        )

    pt = xp[:, 0] * K[0, 0] + K[0, 2]
    pv = xp[:, 1] * K[1, 1] + K[1, 2]
    return np.stack([pt, pv], axis=1)


def unproject_depth_release(
    depth_mm: np.ndarray,
    cam_calib: dict,
    gen_color_map: bool,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """MATLAB ``unprojectDepth_release.m``.

    Args:
        depth_mm: (H, W) depth in millimetres (424, 512) after ``read_depth``.
        cam_calib: One entry from ``kcalibration['sensors'][i]``.
        gen_color_map: If True, compute RGB-plane projections for each pixel.

    Returns:
        p3d: (N, 3) world coordinates (cm) after unproject transform.
        p2d_color: (N, 2) RGB image pixel coords (distorted), or None.
    """
    h, w = depth_mm.shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            depth_mm.astype(np.float64).ravel(),
        ],
        axis=1,
    )

    K_depth = cam_calib['K_depth']
    M_depth = np.asarray(cam_calib['M_depth'], dtype=np.float64)
    dist_d = np.asarray(cam_calib['distCoeffs_depth'], dtype=np.float64).reshape(-1)

    p3d = unproject_depth_pixels(flat, K_depth, dist_d, M_depth[0:3, :])

    p2d_color = None
    if gen_color_map:
        M_color = np.asarray(cam_calib['M_color'], dtype=np.float64)
        K_color = np.asarray(cam_calib['K_color'], dtype=np.float64)
        dc = cam_calib['distCoeffs_color']
        dist_c = np.asarray(dc, dtype=np.float64).reshape(-1)

        p2d_color = pose_project_2d(
            p3d,
            M_color[0:3, 0:3],
            M_color[0:3, 3:4],
            K_color,
            dist_c,
            True,
        )
    return p3d, p2d_color


def splat_depth_to_rgb(
    depth_mm: np.ndarray,
    uv_rgb: np.ndarray,
    *,
    rgb_h: int,
    rgb_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each depth pixel, splat Z onto RGB grid with z-buffer (min depth wins).

    Args:
        depth_mm: (H, W)
        uv_rgb: (H*W, 2) pixel coordinates; invalid can be nan.
        rgb_h, rgb_w: RGB image size.

    Returns:
        aligned_z: (rgb_h, rgb_w) float32 depth in **metres**; 0 = empty.
        hits: int count of writes per pixel (for debugging).
    """
    z = depth_mm.astype(np.float64).ravel() * 0.001  # metres
    uv = np.asarray(uv_rgb, dtype=np.float64).reshape(-1, 2)
    out = np.full((rgb_h, rgb_w), np.inf, dtype=np.float32)
    hits = np.zeros((rgb_h, rgb_w), dtype=np.int32)

    for i in range(z.size):
        zi = z[i]
        if zi <= 0:
            continue
        u, v = uv[i, 0], uv[i, 1]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        ui = int(round(u))
        vi = int(round(v))
        if ui < 0 or vi < 0 or ui >= rgb_w or vi >= rgb_h:
            continue
        if zi < out[vi, ui]:
            out[vi, ui] = zi
            hits[vi, ui] = 1
        elif zi == out[vi, ui]:
            hits[vi, ui] += 1

    out_finite = np.isfinite(out)
    out_z = np.zeros_like(out, dtype=np.float32)
    out_z[out_finite] = out[out_finite]
    return out_z, hits
