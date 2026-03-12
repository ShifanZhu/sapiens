"""Demo visualisation: RGB + depth + skeleton overlay.

Saves per-frame side-by-side images and a summary grid to demo_output/.
Usage:  conda run -n sapiens_gpu python demo.py
"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OUT_DIR   = "demo_output"
os.makedirs(OUT_DIR, exist_ok=True)

from data.constants import SMPLX_SKELETON
from decord import VideoReader, cpu

# ── cases: pick varied sequences ─────────────────────────────────────────────
CASES = [
    ("20240806_1_250_ai1101_vcam/seq_000000.npz",  "rotated_aerial",   [5, 15, 25, 35]),
    ("20240808_1_250_ai1105_vcam/seq_000000.npz",  "normal_walking",   [5, 15, 25, 35]),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def project(joints_3d, K):
    X, Y, Z = joints_3d[:,0], joints_3d[:,1], joints_3d[:,2]
    u = K[0,0]*(-Y/X) + K[0,2]
    v = K[1,1]*(-Z/X) + K[1,2]
    return np.stack([u, v], axis=1)

def draw_skeleton(img_bgr, pts2d, mask, img_h, img_w,
                  joint_r=5, bone_t=2):
    """Draw bones then joints on a copy of img_bgr."""
    out = img_bgr.copy()

    def inside(u, v):
        return 0 <= u < img_w and 0 <= v < img_h

    # bones
    for i, j in SMPLX_SKELETON:
        if i >= len(pts2d) or j >= len(pts2d):
            continue
        if not (mask[i] and mask[j]):
            continue
        u1,v1 = int(round(pts2d[i,0])), int(round(pts2d[i,1]))
        u2,v2 = int(round(pts2d[j,0])), int(round(pts2d[j,1]))
        if inside(u1,v1) and inside(u2,v2):
            cv2.line(out, (u1,v1), (u2,v2), (255,255,0), bone_t, cv2.LINE_AA)

    # joints
    for idx, (u,v) in enumerate(pts2d):
        if not mask[idx]:
            continue
        u,v = int(round(u)), int(round(v))
        if inside(u,v):
            cv2.circle(out, (u,v), joint_r, (0,255,0), -1, cv2.LINE_AA)
            cv2.circle(out, (u,v), joint_r, (0,180,0), 1,  cv2.LINE_AA)
    return out

def depth_colormap(depth):
    valid = depth[depth > 0]
    if not len(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = np.percentile(valid, 2)
    hi = np.percentile(valid, 98)
    d  = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def depth_rgb_blend(rgb_bgr, depth, alpha=0.5):
    dep_c = depth_colormap(depth)
    dep_c = cv2.resize(dep_c, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    return cv2.addWeighted(rgb_bgr, 1-alpha, dep_c, alpha, 0)

def add_label(img, text, pos=(10,30), scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
    return img

# ── main ──────────────────────────────────────────────────────────────────────

all_strip_rows = []

for seq_rel, name, frames in CASES:
    print(f"\n{'─'*55}")
    print(f"Sequence: {name}")

    meta = np.load(f"{DATA_ROOT}/data/label/{seq_rel}", allow_pickle=True)
    folder      = str(meta["folder_name"])
    seq         = str(meta["seq_name"])
    rotate_flag = bool(meta["rotate_flag"])
    K           = meta["intrinsic_matrix"].astype(np.float64)
    joints_cam  = meta["joints_cam"][0]          # (n_frames, 127, 3)
    joints_2d_mask = meta["joints_2d_mask"][0]   # (n_frames, 127)

    vr = VideoReader(
        f"{DATA_ROOT}/data/mp4/{folder}_mp4/{folder}/mp4/{seq}.mp4",
        ctx=cpu(0)
    )
    with np.load(f"{DATA_ROOT}/data/depth/npz/{folder}/{seq}.npz", mmap_mode="r") as f:
        depth_all = f["depth"]                   # (n_frames, H, W)

    strip_frames = []

    for fi in frames:
        # --- load ---
        rgb = vr[fi * 5].asnumpy()
        if rotate_flag:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = np.array(depth_all[fi], dtype=np.float64)
        jc    = joints_cam[fi]           # (127, 3)
        jmask = joints_2d_mask[fi].astype(bool)  # (127,)

        img_h, img_w = rgb.shape[:2]

        # project all 127 joints
        front = jc[:, 0] > 0
        pts   = np.full((127, 2), -1.0)
        pts[front] = project(jc[front], K)

        vis_mask = jmask & front  # visible + in front of camera

        # --- panels ---
        rgb_bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        skel_img = draw_skeleton(rgb_bgr, pts, vis_mask, img_h, img_w)
        blend    = draw_skeleton(depth_rgb_blend(rgb_bgr, depth), pts, vis_mask, img_h, img_w)
        dep_img  = draw_skeleton(depth_colormap(depth), pts, vis_mask, img_h, img_w)

        # labels
        add_label(skel_img, f"RGB + skeleton  [frame {fi}]")
        add_label(blend,    f"RGB + depth blend")
        add_label(dep_img,  f"Depth (TURBO)")

        # resize to fixed height for tiling
        target_h = 320
        def rh(im):
            scale = target_h / im.shape[0]
            return cv2.resize(im, (int(im.shape[1]*scale), target_h))

        row = np.concatenate([rh(skel_img), rh(blend), rh(dep_img)], axis=1)
        strip_frames.append(row)

        out_path = f"{OUT_DIR}/{name}_f{fi:03d}.jpg"
        cv2.imwrite(out_path, row, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  frame {fi:3d} → {out_path}")

    # vertical strip for this sequence
    strip = np.concatenate(strip_frames, axis=0)
    # add sequence label bar
    bar = np.zeros((40, strip.shape[1], 3), dtype=np.uint8)
    tag = f"{'[rotated] ' if rotate_flag else '[normal]  '}{name}   |   {seq_rel}"
    add_label(bar, tag, pos=(10, 28), scale=0.7, color=(0, 220, 255))
    strip = np.concatenate([bar, strip], axis=0)
    all_strip_rows.append(strip)

# ── combined grid ─────────────────────────────────────────────────────────────
max_w = max(s.shape[1] for s in all_strip_rows)
padded = []
for s in all_strip_rows:
    if s.shape[1] < max_w:
        pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), dtype=np.uint8)
        s = np.concatenate([s, pad], axis=1)
    padded.append(s)

divider = np.full((6, max_w, 3), 80, dtype=np.uint8)
grid = np.concatenate(
    [p for pair in zip(padded, [divider]*len(padded)) for p in pair],
    axis=0
)

grid_path = f"{OUT_DIR}/demo_grid.jpg"
cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nCombined grid → {grid_path}  ({grid.shape[1]}×{grid.shape[0]})")
print("Done.")
