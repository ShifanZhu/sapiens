"""Verify RGB / depth / joint alignment for both rotated and normal sequences.

Coordinate convention (right-handed):
  X = forward (depth),  Y = left,  Z = up
Projection:
  u = fx * (-Y / X) + cx
  v = fy * (-Z / X) + cy

We compare our reprojected joints_cam against the stored joints_2d as
ground truth. Median reprojection error < 1 px is expected.
We also check depth[v,u] ≈ joint X (the depth/forward axis).

Outputs are saved to verify_output/.
"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OUT_DIR   = "verify_output"
os.makedirs(OUT_DIR, exist_ok=True)

BODY_JOINTS = list(range(22))   # skip face/hand surface landmarks

CASES = [
    ("20240806_1_250_ai1101_vcam/seq_000000.npz", 10, "rotated"),
    ("20240808_1_250_ai1105_vcam/seq_000000.npz", 10, "normal"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def project(joints_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (J,3) camera-space (X-fwd, Y-left, Z-up) → (J,2) pixel [u,v].

    u = fx * (-Y / X) + cx
    v = fy * (-Z / X) + cy
    """
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    u = K[0, 0] * (-Y / X) + K[0, 2]
    v = K[1, 1] * (-Z / X) + K[1, 2]
    return np.stack([u, v], axis=1)


def draw_joints(img_bgr, pts_reproj, pts_gt, radius=6):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for (ur, vr), (ug, vg) in zip(pts_reproj, pts_gt):
        ur, vr = int(round(ur)), int(round(vr))
        ug, vg = int(round(ug)), int(round(vg))
        if 0 <= ur < w and 0 <= vr < h:
            cv2.circle(out, (ur, vr), radius, (0, 255, 0), -1)   # green: reprojected
        if 0 <= ug < w and 0 <= vg < h:
            cv2.circle(out, (ug, vg), radius // 2, (0, 0, 255), -1)  # red: GT
    return out


def depth_colormap(depth):
    valid = depth[depth > 0]
    if not len(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = valid.min(), np.percentile(valid, 99)
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ── main ─────────────────────────────────────────────────────────────────────

from decord import VideoReader, cpu

for seq_rel, frame_idx, label in CASES:
    print(f"\n{'='*60}")
    print(f"Case: {label}  ({seq_rel.split('/')[1]}, frame {frame_idx})")

    meta = np.load(f"{DATA_ROOT}/data/label/{seq_rel}", allow_pickle=True)
    folder      = str(meta["folder_name"])
    seq         = str(meta["seq_name"])
    rotate_flag = bool(meta["rotate_flag"])
    K           = meta["intrinsic_matrix"].astype(np.float64)
    joints_cam  = meta["joints_cam"][0, frame_idx].astype(np.float64)   # (127,3)
    joints_2d   = meta["joints_2d"][0, frame_idx].astype(np.float64)    # (127,2) GT pixels
    mask        = meta["joints_2d_mask"][0, frame_idx].astype(bool)     # (127,) visible

    # RGB
    vr  = VideoReader(f"{DATA_ROOT}/data/mp4/{folder}_mp4/{folder}/mp4/{seq}.mp4", ctx=cpu(0))
    rgb = vr[frame_idx * 5].asnumpy()
    if rotate_flag:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Depth (already upright)
    with np.load(f"{DATA_ROOT}/data/depth/npz/{folder}/{seq}.npz", mmap_mode="r") as f:
        depth = np.array(f["depth"][frame_idx], dtype=np.float64)

    img_h, img_w = rgb.shape[:2]
    print(f"  rotate_flag={rotate_flag}")
    print(f"  RGB shape: {rgb.shape}   Depth shape: {depth.shape}")
    print(f"  K:\n{K}")

    # ── shape check ──────────────────────────────────────────────────────
    assert rgb.shape[:2] == depth.shape, \
        f"[FAIL] Shape mismatch rgb={rgb.shape[:2]} depth={depth.shape}"
    print(f"  [OK] RGB and depth shapes match")

    # ── reprojection check vs joints_2d ──────────────────────────────────
    body_3d = joints_cam[BODY_JOINTS]
    body_gt = joints_2d[BODY_JOINTS]
    body_mask = mask[BODY_JOINTS]

    pts_repr = project(body_3d, K)   # (22, 2)

    # Only compare joints that are in front of camera (X > 0) and marked visible
    valid = body_mask & (body_3d[:, 0] > 0)
    errors = np.linalg.norm(pts_repr[valid] - body_gt[valid], axis=1)
    in_img = ((pts_repr[:, 0] >= 0) & (pts_repr[:, 0] < img_w) &
              (pts_repr[:, 1] >= 0) & (pts_repr[:, 1] < img_h))
    print(f"  Joints in front of camera (X>0) & visible: {valid.sum()}/22")
    print(f"  Joints projected inside image: {(valid & in_img).sum()}/22")
    if len(errors):
        print(f"  Reprojection error (px):  mean={errors.mean():.3f}  "
              f"median={np.median(errors):.3f}  max={errors.max():.3f}")
        if np.median(errors) < 1.0:
            print(f"  [OK] Reprojection matches joints_2d (median < 1 px)")
        else:
            print(f"  [FAIL] Large reprojection error — check formula or K")

    # ── depth consistency check ───────────────────────────────────────────
    depth_errors = []
    for ji in np.where(valid & in_img)[0]:
        u, v = int(round(pts_repr[ji, 0])), int(round(pts_repr[ji, 1]))
        d_px  = depth[v, u]     # depth at pixel = distance in metres
        x_3d  = body_3d[ji, 0]  # X is the forward/depth axis
        if d_px > 0:
            depth_errors.append(abs(d_px - x_3d))
    if depth_errors:
        de = np.array(depth_errors)
        print(f"  Depth vs joint-X error (m): mean={de.mean():.3f}  "
              f"median={np.median(de):.3f}  max={de.max():.3f}")
        if np.median(de) < 0.3:
            print(f"  [OK] Depth aligns with joints (median < 0.3 m)")
        else:
            print(f"  [WARN] Depth error large — possible depth/joint misalignment")

    # ── save visualisations ───────────────────────────────────────────────
    rgb_bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb_vis  = draw_joints(rgb_bgr, pts_repr, body_gt)
    dep_vis  = draw_joints(depth_colormap(depth), pts_repr, body_gt)

    cv2.imwrite(f"{OUT_DIR}/{label}_rgb_joints.jpg",   rgb_vis)
    cv2.imwrite(f"{OUT_DIR}/{label}_depth_joints.jpg", dep_vis)

    if rgb_vis.shape == dep_vis.shape:
        side = np.concatenate([rgb_vis, dep_vis], axis=1)
        cv2.imwrite(f"{OUT_DIR}/{label}_side_by_side.jpg", side)
        print(f"  Saved: {OUT_DIR}/{label}_side_by_side.jpg")
    print("  (green = reprojected from joints_cam, red dot = stored joints_2d GT)")

print("\nDone.")
