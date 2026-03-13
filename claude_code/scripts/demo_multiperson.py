"""Multi-person 3D pose estimation demo (top-down).

Pipeline:
  1. Load RGB + depth
  2. Detect people (GT bboxes from labels, or external detector)
  3. Crop each person → resize to model input → normalize → update K
  4. Batch forward pass → root-relative (B, 127, 3)
  5. Recover absolute pelvis per person from depth map
  6. Visualize all skeletons on original image

Usage:
    # Using GT bboxes from BEDLAM2 labels:
    conda run -n sapiens_gpu python scripts/demo_multiperson.py \\
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \\
        --checkpoint runs/exp001/best.pth \\
        --output-dir runs/demo_multi

    # Specify sequences:
    conda run -n sapiens_gpu python scripts/demo_multiperson.py \\
        --data-root ... --checkpoint ... --seq-filter "seq_000000"
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.constants import (
    DEPTH_MAX_METERS,
    NUM_JOINTS,
    PELVIS_IDX,
    RGB_MEAN,
    RGB_STD,
    SMPLX_SKELETON,
)
from model import SapiensPose3D

# ── Constants ────────────────────────────────────────────────────────────────

OUT_H, OUT_W = 640, 384
COLORS_BGR = [
    (0, 255, 0),    # green
    (255, 128, 0),  # blue-ish
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 128),
    (255, 128, 128),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-person 3D pose demo")
    p.add_argument("--data-root", default="/home/hang/repos_local/MMC/BEDLAM2Datatest")
    p.add_argument("--checkpoint", default="", help="Model checkpoint path")
    p.add_argument("--output-dir", default="runs/demo_multi")
    p.add_argument("--img-h", type=int, default=OUT_H)
    p.add_argument("--img-w", type=int, default=OUT_W)
    p.add_argument("--arch", default="sapiens_0.3b")
    p.add_argument("--max-frames", type=int, default=5, help="Max frames per sequence")
    p.add_argument("--max-seqs", type=int, default=10, help="Max sequences to process")
    p.add_argument("--seq-filter", default="", help="Only process seqs containing this string")
    p.add_argument("--score-thr", type=float, default=0.5, help="Detector score threshold")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-model", action="store_true",
                   help="Skip model inference, just visualize GT crops and bboxes")
    return p.parse_args()


def crop_person(rgb, depth, K, bbox, out_h, out_w):
    """Crop and resize a single person. Returns (rgb_crop, depth_crop, K_crop).

    Args:
        rgb:   (H, W, 3) uint8
        depth: (H, W) float32 or None
        K:     (3, 3) float32
        bbox:  (4,) float32 — (x1, y1, x2, y2)

    Returns:
        rgb_crop:   (out_h, out_w, 3) uint8
        depth_crop: (out_h, out_w) float32 or None
        K_crop:     (3, 3) float32
    """
    H, W = rgb.shape[:2]

    cx_box = (bbox[0] + bbox[2]) / 2.0
    cy_box = (bbox[1] + bbox[3]) / 2.0
    w_box = max(bbox[2] - bbox[0], 1.0)
    h_box = max(bbox[3] - bbox[1], 1.0)

    # Expand to target aspect ratio
    target_aspect = out_w / out_h
    if (w_box / h_box) < target_aspect:
        w_exp = h_box * target_aspect
        h_exp = h_box
    else:
        h_exp = w_box / target_aspect
        w_exp = w_box

    x0 = cx_box - w_exp / 2.0
    y0 = cy_box - h_exp / 2.0
    x1 = cx_box + w_exp / 2.0
    y1 = cy_box + h_exp / 2.0

    # Padding
    pad_left = max(0, int(math.ceil(-x0)))
    pad_top = max(0, int(math.ceil(-y0)))
    pad_right = max(0, int(math.ceil(x1 - W)))
    pad_bottom = max(0, int(math.ceil(y1 - H)))

    # Resize depth to match RGB if needed
    if depth is not None:
        dH, dW = depth.shape[:2]
        if dH != H or dW != W:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if depth is not None:
            depth = cv2.copyMakeBorder(depth, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=0.0)
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top

    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = int(round(x1)), int(round(y1))
    crop_w = max(ix1 - ix0, 1)
    crop_h = max(iy1 - iy0, 1)
    sx = out_w / crop_w
    sy = out_h / crop_h

    rgb_crop = cv2.resize(rgb[iy0:iy1, ix0:ix1], (out_w, out_h),
                          interpolation=cv2.INTER_LINEAR)
    depth_crop = None
    if depth is not None:
        depth_crop = cv2.resize(depth[iy0:iy1, ix0:ix1], (out_w, out_h),
                                interpolation=cv2.INTER_NEAREST)

    orig_x0 = ix0 - pad_left
    orig_y0 = iy0 - pad_top
    K_crop = K.copy()
    K_crop[0, 0] *= sx
    K_crop[1, 1] *= sy
    K_crop[0, 2] = (K_crop[0, 2] - orig_x0) * sx
    K_crop[1, 2] = (K_crop[1, 2] - orig_y0) * sy

    return rgb_crop, depth_crop, K_crop


def normalize_for_model(rgb, depth, depth_max=DEPTH_MAX_METERS):
    """Normalize RGB + depth to model input tensor (4, H, W)."""
    mean = np.array(RGB_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(RGB_STD, dtype=np.float32).reshape(1, 1, 3)

    rgb_f = rgb.astype(np.float32) / 255.0
    rgb_f = (rgb_f - mean) / std
    rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_f.transpose(2, 0, 1)))  # (3, H, W)

    if depth is not None:
        depth_f = np.clip(depth, 0.0, depth_max) / depth_max
        depth_t = torch.from_numpy(depth_f[np.newaxis].astype(np.float32))  # (1, H, W)
    else:
        depth_t = torch.zeros(1, rgb.shape[0], rgb.shape[1])

    return torch.cat([rgb_t, depth_t], dim=0)  # (4, H, W)


def recover_pelvis_from_model(
    pelvis_uv_norm, pelvis_depth, K_crop, K_orig, bbox, out_h, out_w, H_img, W_img,
):
    """Recover absolute pelvis 3D position from model-predicted UV and depth.

    Args:
        pelvis_uv_norm:  (2,) predicted (u, v) normalized to [-1, 1]
        pelvis_depth:    float — predicted forward distance (X) in metres
        K_crop:          (3, 3) crop intrinsic matrix
        K_orig:          (3, 3) original image intrinsic matrix
        bbox:            (4,) (x1, y1, x2, y2)
        out_h, out_w:    crop output dimensions
        H_img, W_img:    original image dimensions

    Returns:
        pelvis_abs: (3,) — absolute 3D pelvis position, or None if invalid
    """
    if pelvis_depth < 0.01:
        return None

    # Denormalize [-1, 1] → crop pixels
    u_crop = (float(pelvis_uv_norm[0]) + 1.0) / 2.0 * out_w
    v_crop = (float(pelvis_uv_norm[1]) + 1.0) / 2.0 * out_h

    # Map crop pixel back to original image coordinates
    cx_box = (bbox[0] + bbox[2]) / 2.0
    cy_box = (bbox[1] + bbox[3]) / 2.0
    w_box = max(bbox[2] - bbox[0], 1.0)
    h_box = max(bbox[3] - bbox[1], 1.0)

    target_aspect = out_w / out_h
    if (w_box / h_box) < target_aspect:
        w_exp = h_box * target_aspect
        h_exp = h_box
    else:
        h_exp = w_box / target_aspect
        w_exp = w_box

    x0 = cx_box - w_exp / 2.0
    y0 = cy_box - h_exp / 2.0

    sx = out_w / w_exp
    sy = out_h / h_exp

    # Invert crop transform: crop_pixel → original_pixel
    u_orig = u_crop / sx + x0
    v_orig = v_crop / sy + y0

    # Unproject using original K: BEDLAM2 convention
    # X=forward(depth), Y=left, Z=up
    X_fwd = pelvis_depth
    Y_left = -(u_orig - K_orig[0, 2]) * X_fwd / K_orig[0, 0]
    Z_up = -(v_orig - K_orig[1, 2]) * X_fwd / K_orig[1, 1]

    return np.array([X_fwd, Y_left, Z_up], dtype=np.float32)


def project_joints(joints_3d, K):
    """Project 3D joints to 2D. Returns (u, v, valid)."""
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    valid = X > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(valid, K[0, 0] * (-Y / X) + K[0, 2], -1.0)
        v = np.where(valid, K[1, 1] * (-Z / X) + K[1, 2], -1.0)
    return u, v, valid


def draw_skeleton(img, u, v, valid, color=(0, 255, 0), thickness=2, radius=3):
    """Draw skeleton on BGR image in-place."""
    H, W = img.shape[:2]
    for a, b in SMPLX_SKELETON:
        if valid[a] and valid[b]:
            cv2.line(img,
                     (int(round(u[a])), int(round(v[a]))),
                     (int(round(u[b])), int(round(v[b]))),
                     color, thickness, cv2.LINE_AA)
    for j in range(len(u)):
        if valid[j]:
            pt = (int(round(u[j])), int(round(v[j])))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)


def get_gt_bboxes_and_labels(label_path, frame_idx):
    """Get GT bboxes and joints for all people in a frame.

    Returns list of dicts with keys: bbox, joints_cam, body_idx
    """
    with np.load(label_path, allow_pickle=True) as meta:
        n_body = int(meta["joints_cam"].shape[0])
        joints_cam = meta["joints_cam"].astype(np.float32)  # (n_body, n_frames, 127, 3)
        joints_2d = meta["joints_2d"].astype(np.float32)    # (n_body, n_frames, 127, 2)
        K = meta["intrinsic_matrix"].astype(np.float32)

    people = []
    for bi in range(n_body):
        kpts = joints_2d[bi, frame_idx]  # (127, 2)
        x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
        x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()
        w, h = x_max - x_min, y_max - y_min
        bbox = np.array([x_min - w * 0.1, y_min - h * 0.1,
                         x_max + w * 0.1, y_max + h * 0.1], dtype=np.float32)
        people.append({
            "bbox": bbox,
            "joints_cam": joints_cam[bi, frame_idx],
            "body_idx": bi,
        })
    return people, K


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_h, out_w = args.img_h, args.img_w

    device = torch.device(args.device)

    # ── Model ────────────────────────────────────────────────────────────
    model = None
    if not args.no_model and args.checkpoint:
        print(f"Loading model from {args.checkpoint} ...")
        model = SapiensPose3D(
            arch=args.arch,
            img_size=(out_h, out_w),
            num_joints=NUM_JOINTS,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("  Model loaded.")
    elif args.no_model:
        print("Skipping model (--no-model), visualizing GT only.")
    else:
        print("No checkpoint provided, visualizing GT only.")

    # ── Find sequences ───────────────────────────────────────────────────
    from data import get_splits
    overview = os.path.join(args.data_root, "data", "overview.txt")
    all_seqs, _, _ = get_splits(
        overview_path=overview, val_ratio=0, test_ratio=0,
        single_body_only=False, skip_missing_body=True,
        depth_required=True, mp4_required=False,
    )

    if args.seq_filter:
        all_seqs = [s for s in all_seqs if args.seq_filter in s]

    # Prefer multi-person sequences first
    seqs_sorted = sorted(all_seqs, key=lambda s: "not_single" in s, reverse=True)
    seqs_to_process = seqs_sorted[:args.max_seqs]

    print(f"\nProcessing {len(seqs_to_process)} sequences ...")

    n_saved = 0
    for seq_rel in seqs_to_process:
        label_path = os.path.join(args.data_root, "data", "label", seq_rel)
        with np.load(label_path, allow_pickle=True) as meta:
            folder = str(meta["folder_name"])
            seq_name = str(meta["seq_name"])
            n_frames = int(meta["n_frames"])
            n_body = int(meta["joints_cam"].shape[0])
            K_orig = meta["intrinsic_matrix"].astype(np.float32)

        # Check if frames exist
        frame_dir = os.path.join(args.data_root, "data", "frames", folder, seq_name)
        if not os.path.isdir(frame_dir):
            continue

        frames_to_vis = list(range(0, n_frames, max(1, n_frames // args.max_frames)))[:args.max_frames]

        print(f"\n  {seq_rel}  ({n_body} bodies, {n_frames} frames)")

        for fi in frames_to_vis:
            # Load RGB
            jpeg_path = os.path.join(frame_dir, f"{fi:05d}.jpg")
            if not os.path.exists(jpeg_path):
                continue
            rgb_bgr = cv2.imread(jpeg_path)
            if rgb_bgr is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            H_img, W_img = rgb.shape[:2]

            # Load depth
            npy_path = os.path.join(args.data_root, "data", "depth", "npy", folder, f"{seq_name}.npy")
            npz_path = os.path.join(args.data_root, "data", "depth", "npz", folder, f"{seq_name}.npz")
            depth = None
            try:
                if os.path.exists(npy_path):
                    depth = np.load(npy_path, mmap_mode="r")[fi].astype(np.float32)
                elif os.path.exists(npz_path):
                    with np.load(npz_path) as f:
                        depth = f["depth"][fi].astype(np.float32)
            except (ValueError, OSError):
                depth = None  # skip corrupt depth files

            # Get people bboxes (GT)
            people, K_orig = get_gt_bboxes_and_labels(label_path, fi)

            # Draw output image
            vis_img = rgb_bgr.copy()

            # Per-person processing
            for pi, person in enumerate(people):
                bbox = person["bbox"].copy()
                bbox[0] = max(0, min(bbox[0], W_img))
                bbox[1] = max(0, min(bbox[1], H_img))
                bbox[2] = max(0, min(bbox[2], W_img))
                bbox[3] = max(0, min(bbox[3], H_img))

                color = COLORS_BGR[pi % len(COLORS_BGR)]

                # Draw bbox
                cv2.rectangle(vis_img,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              color, 2)
                cv2.putText(vis_img, f"P{pi}",
                            (int(bbox[0]), int(bbox[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if model is not None:
                    # Crop, normalize, forward
                    rgb_crop, depth_crop, K_crop = crop_person(
                        rgb, depth, K_orig, bbox, out_h, out_w)
                    x = normalize_for_model(rgb_crop, depth_crop).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = model(x)
                        pred_rel = out["joints"][0].float().cpu().numpy()         # (127, 3)
                        pred_pelvis_depth = out["pelvis_depth"][0, 0].float().item()  # scalar
                        pred_pelvis_uv = out["pelvis_uv"][0].float().cpu().numpy()    # (2,)

                    # Recover absolute pelvis from model predictions
                    pelvis_abs = recover_pelvis_from_model(
                        pred_pelvis_uv, pred_pelvis_depth,
                        K_crop, K_orig, bbox, out_h, out_w, H_img, W_img)

                    if pelvis_abs is not None:
                        joints_abs = pred_rel + pelvis_abs[np.newaxis, :]
                        u, v, valid = project_joints(joints_abs, K_orig)
                        draw_skeleton(vis_img, u, v, valid, color=color)
                else:
                    # No model — draw GT skeleton
                    joints_gt = person["joints_cam"]
                    u, v, valid = project_joints(joints_gt, K_orig)
                    draw_skeleton(vis_img, u, v, valid, color=color)

            # Add info text
            mode = "pred" if model is not None else "GT"
            cv2.putText(vis_img, f"{seq_name} f{fi} [{mode}] {n_body}ppl",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save
            fname = f"{seq_name}_f{fi:03d}_{n_body}ppl.jpg"
            cv2.imwrite(os.path.join(args.output_dir, fname), vis_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_saved += 1

    print(f"\nSaved {n_saved} images to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
