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

from typing import Literal, Tuple

import numpy as np


def undistort_color_uv_to_new_k(
    uv_dist: np.ndarray,
    K_color: np.ndarray,
    dist_color: np.ndarray,
    new_K: np.ndarray,
) -> np.ndarray:
    """Map distorted Kinect color pixels to undistorted pixel coords (``P=new_K``).

    Matches ``cv2.undistort(..., P=new_K)`` / ``cv2.undistortPoints(..., P=new_K)``:
    depth splatted in this space aligns with ``rgb_undistort`` images.

    Non-finite inputs copy through as NaN so splat can skip them.
    """
    import cv2

    uv = np.asarray(uv_dist, dtype=np.float64).reshape(-1, 2)
    out = np.full_like(uv, np.nan, dtype=np.float64)
    fin = np.isfinite(uv).all(axis=1)
    if not np.any(fin):
        return out
    K = np.asarray(K_color, dtype=np.float64)
    dist = np.asarray(dist_color, dtype=np.float64).reshape(-1)
    P = np.asarray(new_K, dtype=np.float64)
    pts = uv[fin].reshape(-1, 1, 2)
    mapped = cv2.undistortPoints(pts, K, dist, P=P)
    out[fin] = mapped.reshape(-1, 2)
    return out


def _as_float_mat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def normalize_dist_coeffs_for_opencv(dist: np.ndarray) -> np.ndarray:
    """Return a 1D distortion vector that ``cv2.getOptimalNewCameraMatrix`` accepts.

    OpenCV allows **4, 5, 8, 12, or 14** coefficients. CMU Panoptic ``kcalibration``
    depth entries are often **7** floats: ``[k1,k2,p1,p2,k3, 1000, 0]`` (Kinect/MATLAB
    extras after the standard rational model). We keep the **first five** so depth
    undistort matches ``unproject_depth_pixels`` which only uses ``dist[:5]``.
    """
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    n = int(d.size)
    if n in (4, 5, 8, 12, 14):
        return d.copy()
    if n == 7:
        return d[:5].copy()
    if n > 14:
        return d[:14].copy()
    if n == 6:
        return d[:5].copy()
    # n in {1,2,3} — pad to 5
    out = np.zeros(5, dtype=np.float64)
    out[:n] = d
    return out


def undistort_depth_map_nearest(
    depth_mm: np.ndarray,
    K_depth: np.ndarray,
    dist_depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort depth in the **depth camera** image plane (``depth → undistort``).

    Uses ``cv2.initUndistortRectifyMap`` + ``cv2.remap`` with ``INTER_NEAREST`` so depth
    values are not blended across discontinuities. Returns the rectified depth map and
    ``new_K_depth`` for pinhole unprojection with **zero** distortion coeffs.

    Shape and dtype: output matches ``depth_mm`` (float32, metres scale still mm).
    """
    import cv2

    h, w = depth_mm.shape
    K = np.asarray(K_depth, dtype=np.float64)
    dist = normalize_dist_coeffs_for_opencv(dist_depth)
    new_k, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_k, (w, h), cv2.CV_32FC1
    )
    d = depth_mm.astype(np.float32)
    out = cv2.remap(
        d,
        map1,
        map2,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out, new_k


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
            (1.0 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2)
            / (1.0 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2)
        )
        delta_x = (
            2 * k[2] * x * y
            + k[3] * (r2 + 2 * x * x)
            + k[8] * r2
            + k[9] * r2 * r2
        )
        delta_y = (
            k[2] * (r2 + 2 * y * y)
            + 2 * k[3] * x * y
            + k[10] * r2
            + k[11] * r2 * r2
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
    *,
    depth_unproject_mode: Literal['opencv_undistort_pinhole', 'toolbox_iterative'] = (
        'opencv_undistort_pinhole'
    ),
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Depth → world, then optional projection to Kinect color.

    Two depth paths:

    - ``opencv_undistort_pinhole`` (default): **depth → undistort (OpenCV nearest remap)
      → unproject** with ``new_K_depth`` and **zero** distortion (pinhole). This matches
      the usual ``depth → undistort → unproject`` ordering explicitly.

    - ``toolbox_iterative``: MATLAB-style ``unproject.m`` inverse distortion iteration on
      the **raw** distorted depth grid (legacy CMU toolbox behavior).

    Returns:
        p3d: (N, 3) world coordinates (cm).
        p2d_color: (N, 2) RGB pixels (distorted), or None.
        depth_mm_for_splat: (H, W) float32 — depth values aligned with ``p3d`` raster
        order for :func:`splat_depth_to_rgb` (undistorted map in OpenCV mode, else raw).
    """
    K_depth = cam_calib['K_depth']
    M_depth = np.asarray(cam_calib['M_depth'], dtype=np.float64)
    dist_d = np.asarray(cam_calib['distCoeffs_depth'], dtype=np.float64).reshape(-1)
    h, w = depth_mm.shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    if depth_unproject_mode == 'opencv_undistort_pinhole':
        depth_u, new_k_d = undistort_depth_map_nearest(depth_mm, K_depth, dist_d)
        flat = np.stack(
            [
                grid_x.ravel(),
                grid_y.ravel(),
                depth_u.astype(np.float64).ravel(),
            ],
            axis=1,
        )
        # Pinhole unprojection in the **undistorted** depth image (no radial model).
        zerod = np.zeros(12, dtype=np.float64)
        p3d = unproject_depth_pixels(flat, new_k_d, zerod, M_depth[0:3, :])
        depth_for_splat = depth_u
    elif depth_unproject_mode == 'toolbox_iterative':
        flat = np.stack(
            [
                grid_x.ravel(),
                grid_y.ravel(),
                depth_mm.astype(np.float64).ravel(),
            ],
            axis=1,
        )
        p3d = unproject_depth_pixels(flat, K_depth, dist_d, M_depth[0:3, :])
        depth_for_splat = depth_mm.astype(np.float32)
    else:
        raise ValueError(
            f'Unknown depth_unproject_mode: {depth_unproject_mode!r}'
        )

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
    return p3d, p2d_color, depth_for_splat


def splat_depth_to_rgb(
    depth_mm: np.ndarray,
    uv_rgb: np.ndarray,
    *,
    rgb_h: int,
    rgb_w: int,
    supersample: int = 2,
    pool: str = 'min',
) -> Tuple[np.ndarray, np.ndarray]:
    """For each depth pixel, splat Z onto RGB grid with z-buffer (min depth wins).

    ``supersample`` > 1 splats onto a finer grid (``rgb * supersample``) with the same
    min-depth rule, then pools back to full resolution.

    ``pool`` (only used when ``supersample`` > 1):

    - ``min``: min over each ``ss×ss`` macro block (closest surface; default z-buffer).
    - ``mean``: mean of **finite** subcell depths per macro block (denser, smoother;
      can mix depths at edges — use for visualization / denser labels when advised).

    Args:
        depth_mm: (H, W)
        uv_rgb: (H*W, 2) pixel coordinates; invalid can be nan.
        rgb_h, rgb_w: RGB image size.
        supersample: Grid multiplier for the splat phase (1 = legacy single-cell bin).
        pool: ``min`` or ``mean`` (supersample > 1 only).

    Returns:
        aligned_z: (rgb_h, rgb_w) float32 depth in **metres**; 0 = empty.
        hits: int count of writes per fine cell (if supersample>1, sum pooled to RGB).
    """
    z = depth_mm.astype(np.float64).ravel() * 0.001  # metres
    uv = np.asarray(uv_rgb, dtype=np.float64).reshape(-1, 2)
    ss = int(supersample)
    if ss < 1:
        raise ValueError('supersample must be >= 1')
    pool = str(pool).lower().strip()
    if pool not in ('min', 'mean'):
        raise ValueError("pool must be 'min' or 'mean'")

    if ss == 1:
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

    Hf = rgb_h * ss
    Wf = rgb_w * ss
    out = np.full((Hf, Wf), np.inf, dtype=np.float32)
    hits_f = np.zeros((Hf, Wf), dtype=np.int32)

    for i in range(z.size):
        zi = z[i]
        if zi <= 0:
            continue
        u, v = uv[i, 0], uv[i, 1]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        # Subcells partition each RGB pixel into ss×ss bins (continuous u,v coords).
        ui = int(np.floor(float(u) * ss))
        vi = int(np.floor(float(v) * ss))
        if ui < 0 or vi < 0 or ui >= Wf or vi >= Hf:
            continue
        if zi < out[vi, ui]:
            out[vi, ui] = zi
            hits_f[vi, ui] = 1
        elif zi == out[vi, ui]:
            hits_f[vi, ui] += 1

    if pool == 'min':
        # use inf (not nan) for empty so all-inf blocks min to inf without warnings
        bl = np.where(np.isfinite(out), out, np.inf)
        tmp = bl.reshape(rgb_h, ss, rgb_w, ss)
        macro = np.min(tmp, axis=(1, 3))
        out_z = np.where(np.isfinite(macro) & (macro < np.inf), macro, 0.0).astype(
            np.float32
        )
        hf = np.where(np.isfinite(out), hits_f, 0).reshape(rgb_h, ss, rgb_w, ss)
        hits = hf.sum(axis=(1, 3)).astype(np.int32)
    else:
        # Mean of finite subcell depths per macro pixel (empty subcells ignored).
        bl = np.where(np.isfinite(out), out, np.nan)
        blocks = bl.reshape(rgb_h, rgb_w, ss * ss)
        valid_n = np.isfinite(blocks).sum(axis=2).astype(np.float64)
        macro = np.nansum(blocks, axis=2) / np.maximum(valid_n, 1.0)
        macro = np.where(valid_n > 0, macro, np.nan)
        out_z = np.where(np.isfinite(macro), macro, 0.0).astype(np.float32)
        fin = np.isfinite(bl.reshape(rgb_h, rgb_w, ss * ss))
        hits = fin.sum(axis=2).astype(np.int32)
    return out_z, hits
