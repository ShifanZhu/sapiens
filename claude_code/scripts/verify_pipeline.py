"""Verify the multi-person data pipeline and save visual outputs.

Saves to runs/verify_pipeline/:
  - Per-sample images showing: RGB crop with projected GT joints (root-relative + pelvis recovery)
  - Side-by-side: original full image with bbox overlay | cropped person with skeleton
  - Prints pipeline stats (index size, bbox sizes, root-relative joint ranges)

Usage:
    conda run -n sapiens_gpu python scripts/verify_pipeline.py
"""

import os
import sys
import random

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_splits, build_train_transform, build_val_transform
from data.dataset import BedlamFrameDataset
from data.constants import SMPLX_SKELETON, PELVIS_IDX, RGB_MEAN, RGB_STD

DATA_ROOT = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OVERVIEW_PATH = f"{DATA_ROOT}/data/overview.txt"
OUT_H, OUT_W = 640, 384
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "verify_pipeline")
os.makedirs(OUT_DIR, exist_ok=True)

# Colors for different people (BGR)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


def project_joints(joints_3d, K):
    """Project 3D joints to 2D using BEDLAM2 convention.
    u = fx*(-Y/X) + cx,  v = fy*(-Z/X) + cy
    """
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    valid = X > 0.01
    u = np.where(valid, K[0, 0] * (-Y / X) + K[0, 2], -1.0)
    v = np.where(valid, K[1, 1] * (-Z / X) + K[1, 2], -1.0)
    return u, v, valid


def draw_skeleton(img, u, v, valid, color=(0, 255, 0), thickness=2, radius=3):
    """Draw skeleton on image (BGR)."""
    H, W = img.shape[:2]
    for a, b in SMPLX_SKELETON:
        if valid[a] and valid[b]:
            pt_a = (int(round(u[a])), int(round(v[a])))
            pt_b = (int(round(u[b])), int(round(v[b])))
            cv2.line(img, pt_a, pt_b, color, thickness, cv2.LINE_AA)
    for j in range(len(u)):
        if valid[j]:
            pt = (int(round(u[j])), int(round(v[j])))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
    return img


def denormalize_rgb(rgb_tensor):
    """Convert (3,H,W) normalized float tensor back to (H,W,3) uint8 BGR."""
    mean = np.array(RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(RGB_STD, dtype=np.float32).reshape(3, 1, 1)
    rgb = rgb_tensor * std + mean
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    # (3,H,W) RGB -> (H,W,3) BGR
    return rgb.transpose(1, 2, 0)[:, :, ::-1].copy()


def main():
    print("=" * 60)
    print("Pipeline Verification")
    print("=" * 60)

    # ── 1. Build splits (include multi-person) ──────────────────────────
    train_seqs_multi, val_seqs_multi, _ = get_splits(
        overview_path=OVERVIEW_PATH,
        val_ratio=0.1, test_ratio=0.1, seed=2026,
        single_body_only=False,  # include multi-person
        skip_missing_body=True, depth_required=True, mp4_required=False,
    )
    train_seqs_single, val_seqs_single, _ = get_splits(
        overview_path=OVERVIEW_PATH,
        val_ratio=0.1, test_ratio=0.1, seed=2026,
        single_body_only=True,
        skip_missing_body=True, depth_required=True, mp4_required=False,
    )
    print(f"\nSingle-body sequences: train={len(train_seqs_single)}, val={len(val_seqs_single)}")
    print(f"All sequences (incl multi): train={len(train_seqs_multi)}, val={len(val_seqs_multi)}")

    # ── 2. Build raw dataset (no transform) to inspect pre-transform data
    print("\n--- Raw dataset (no transform) ---")
    raw_ds = BedlamFrameDataset(
        seq_paths=val_seqs_multi[:50],  # first 50 val seqs for speed
        data_root=DATA_ROOT,
        transform=None,
        depth_required=True,
    )
    print(f"Index size: {len(raw_ds)} samples")

    # Count body indices
    body_counts = {}
    for _, bi, _ in raw_ds.index:
        body_counts[bi] = body_counts.get(bi, 0) + 1
    print(f"Body index distribution: { {k: body_counts[k] for k in sorted(body_counts)} }")

    # Count sequences with >1 body
    from collections import defaultdict
    seq_bodies = defaultdict(set)
    for lp, bi, _ in raw_ds.index:
        seq_bodies[lp].add(bi)
    multi_seqs = sum(1 for s in seq_bodies.values() if len(s) > 1)
    print(f"Sequences with >1 body: {multi_seqs} / {len(seq_bodies)}")

    # ── 3. Build transformed dataset (val transform — no noise) ─────────
    print("\n--- Transformed dataset (val) ---")
    val_tf = build_val_transform(OUT_H, OUT_W)
    tf_ds = BedlamFrameDataset(
        seq_paths=val_seqs_multi[:50],
        data_root=DATA_ROOT,
        transform=val_tf,
        depth_required=True,
    )

    # ── 4. Visualize samples ────────────────────────────────────────────
    print(f"\nSaving visualizations to {OUT_DIR}/")

    # Pick diverse samples: some single-person, some multi-person
    # Group by (label_path) to find multi-person frames
    frame_groups = defaultdict(list)
    for i, (lp, bi, fi) in enumerate(raw_ds.index):
        frame_groups[(lp, fi)].append((i, bi))

    # Filter to frames that actually have extracted JPEGs
    def has_frames(lp, fi):
        try:
            cached = raw_ds._label_cache.get(lp)
            if cached is None:
                with np.load(lp, allow_pickle=True) as m:
                    folder = str(m["folder_name"])
                    seq = str(m["seq_name"])
            else:
                folder = cached["folder_name"]
                seq = cached["seq_name"]
            jpeg = os.path.join(DATA_ROOT, "data", "frames", folder, seq, f"{fi:05d}.jpg")
            return os.path.exists(jpeg)
        except Exception:
            return False

    multi_frames = [(k, v) for k, v in frame_groups.items() if len(v) > 1 and has_frames(*k)]
    single_frames = [(k, v) for k, v in frame_groups.items() if len(v) == 1 and has_frames(*k)]

    print(f"Multi-person frames: {len(multi_frames)}")
    print(f"Single-person frames: {len(single_frames)}")

    n_vis = min(10, len(single_frames))
    n_vis_multi = min(10, len(multi_frames))

    # --- A. Single-person samples (transformed) ---
    print(f"\nVisualizing {n_vis} single-person samples...")
    random.seed(42)
    vis_single = random.sample(single_frames, n_vis) if len(single_frames) >= n_vis else single_frames

    for vis_idx, ((lp, fi), idx_list) in enumerate(vis_single):
        flat_idx, body_idx = idx_list[0]

        # Raw sample (no transform)
        raw_sample = raw_ds[flat_idx]
        # Transformed sample
        tf_sample = tf_ds[flat_idx]

        # --- Left panel: original image with bbox ---
        raw_rgb_bgr = raw_sample["rgb"][:, :, ::-1].copy()  # RGB->BGR
        if "bbox" in raw_sample:
            bbox = raw_sample["bbox"]
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(raw_rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(raw_rgb_bgr, f"body={body_idx}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Project GT joints on original image
        joints_abs = raw_sample["joints"]  # (127, 3) absolute
        K_orig = raw_sample["intrinsic"]
        u, v, valid = project_joints(joints_abs, K_orig)
        draw_skeleton(raw_rgb_bgr, u, v, valid, color=(0, 255, 0))

        # --- Right panel: cropped + transformed with root-relative joints ---
        rgb_crop = denormalize_rgb(tf_sample["rgb"].numpy())
        joints_rel = tf_sample["joints"].numpy()       # (127, 3) root-relative
        pelvis_abs = tf_sample["pelvis_abs"].numpy()    # (3,)
        K_crop = tf_sample["intrinsic"].numpy()

        # Recover absolute joints for projection
        joints_recovered = joints_rel + pelvis_abs[np.newaxis, :]
        u_crop, v_crop, valid_crop = project_joints(joints_recovered, K_crop)
        draw_skeleton(rgb_crop, u_crop, v_crop, valid_crop, color=(0, 255, 0))

        # Add text annotations
        cv2.putText(rgb_crop, "root-rel + pelvis recovery", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(rgb_crop, f"pelvis=({pelvis_abs[0]:.2f},{pelvis_abs[1]:.2f},{pelvis_abs[2]:.2f})",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # --- Depth panel ---
        depth_vis = tf_sample["depth"].numpy()[0]  # (H, W) in [0,1]
        depth_color = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # Resize all panels to same height
        h_target = OUT_H
        raw_resized = cv2.resize(raw_rgb_bgr, (int(raw_rgb_bgr.shape[1] * h_target / raw_rgb_bgr.shape[0]), h_target))

        # Compose side-by-side: [original+bbox | crop+skeleton | depth]
        panels = [raw_resized, rgb_crop, depth_color]
        combined = np.concatenate(panels, axis=1)

        seq_name = raw_sample["seq_name"]
        fname = f"single_{vis_idx:02d}_{seq_name}_f{fi}_b{body_idx}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), combined)

    # --- B. Multi-person samples ---
    if multi_frames:
        print(f"Visualizing {n_vis_multi} multi-person frames...")
        vis_multi = random.sample(multi_frames, n_vis_multi) if len(multi_frames) >= n_vis_multi else multi_frames

        for vis_idx, ((lp, fi), idx_list) in enumerate(vis_multi):
            # Get the raw full image (same for all bodies in this frame)
            first_flat_idx = idx_list[0][0]
            try:
                raw_first = raw_ds[first_flat_idx]
            except FileNotFoundError:
                print(f"  Skipping multi frame (missing JPG): {lp} frame {fi}")
                continue
            full_img = raw_first["rgb"][:, :, ::-1].copy()  # BGR

            # Draw all people's bboxes and skeletons on the full image
            for person_i, (flat_idx, bi) in enumerate(idx_list):
                try:
                    raw_s = raw_ds[flat_idx]
                except FileNotFoundError:
                    continue
                color = COLORS[person_i % len(COLORS)]

                if "bbox" in raw_s:
                    bbox = raw_s["bbox"].astype(int)
                    cv2.rectangle(full_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(full_img, f"body={bi}", (bbox[0], bbox[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                joints_abs = raw_s["joints"]
                K_orig = raw_s["intrinsic"]
                u, v, valid = project_joints(joints_abs, K_orig)
                draw_skeleton(full_img, u, v, valid, color=color)

            # Now make per-person crop panels
            crop_panels = []
            for person_i, (flat_idx, bi) in enumerate(idx_list):
                try:
                    tf_s = tf_ds[flat_idx]
                except FileNotFoundError:
                    continue
                color = COLORS[person_i % len(COLORS)]

                rgb_crop = denormalize_rgb(tf_s["rgb"].numpy())
                joints_rel = tf_s["joints"].numpy()
                pelvis_abs = tf_s["pelvis_abs"].numpy()
                K_crop = tf_s["intrinsic"].numpy()

                joints_recovered = joints_rel + pelvis_abs[np.newaxis, :]
                u_crop, v_crop, valid_crop = project_joints(joints_recovered, K_crop)
                draw_skeleton(rgb_crop, u_crop, v_crop, valid_crop, color=color)
                cv2.putText(rgb_crop, f"body={bi}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                crop_panels.append(rgb_crop)

            # Top row: full image; bottom row: per-person crops
            h_target = OUT_H
            full_resized = cv2.resize(full_img, (int(full_img.shape[1] * h_target / full_img.shape[0]), h_target))

            # Stack crops horizontally
            if crop_panels:
                crops_row = np.concatenate(crop_panels, axis=1)
                # Pad or resize to match widths
                target_w = max(full_resized.shape[1], crops_row.shape[1])
                if full_resized.shape[1] < target_w:
                    pad = np.zeros((full_resized.shape[0], target_w - full_resized.shape[1], 3), dtype=np.uint8)
                    full_resized = np.concatenate([full_resized, pad], axis=1)
                if crops_row.shape[1] < target_w:
                    pad = np.zeros((crops_row.shape[0], target_w - crops_row.shape[1], 3), dtype=np.uint8)
                    crops_row = np.concatenate([crops_row, pad], axis=1)
                elif crops_row.shape[1] > target_w:
                    crops_row = cv2.resize(crops_row, (target_w, h_target))

                combined = np.concatenate([full_resized, crops_row], axis=0)
            else:
                combined = full_resized

            seq_name = raw_first["seq_name"]
            fname = f"multi_{vis_idx:02d}_{seq_name}_f{fi}_{len(idx_list)}ppl.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fname), combined)

    # --- C. Train transform (with noise) comparison ---
    print("\nVisualizing train transform (with bbox noise)...")
    train_tf = build_train_transform(OUT_H, OUT_W)
    train_ds = BedlamFrameDataset(
        seq_paths=val_seqs_multi[:50],
        data_root=DATA_ROOT,
        transform=train_tf,
        depth_required=True,
    )

    # Show same samples with and without noise
    for vis_idx in range(min(5, len(single_frames))):
        (lp, fi), idx_list = vis_single[vis_idx]
        flat_idx = idx_list[0][0]

        # Val (no noise)
        val_s = tf_ds[flat_idx]
        rgb_val = denormalize_rgb(val_s["rgb"].numpy())
        joints_rel = val_s["joints"].numpy()
        pelvis_abs = val_s["pelvis_abs"].numpy()
        K_crop = val_s["intrinsic"].numpy()
        joints_rec = joints_rel + pelvis_abs[np.newaxis, :]
        u, v, valid = project_joints(joints_rec, K_crop)
        draw_skeleton(rgb_val, u, v, valid, color=(0, 255, 0))
        cv2.putText(rgb_val, "val (no noise)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Train (with noise) — run 3 times to show variation
        train_panels = [rgb_val]
        for trial in range(3):
            tr_s = train_ds[flat_idx]
            rgb_tr = denormalize_rgb(tr_s["rgb"].numpy())
            joints_rel = tr_s["joints"].numpy()
            pelvis_abs = tr_s["pelvis_abs"].numpy()
            K_crop = tr_s["intrinsic"].numpy()
            joints_rec = joints_rel + pelvis_abs[np.newaxis, :]
            u, v, valid = project_joints(joints_rec, K_crop)
            draw_skeleton(rgb_tr, u, v, valid, color=(0, 255, 0))
            cv2.putText(rgb_tr, f"train noise #{trial}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            train_panels.append(rgb_tr)

        combined = np.concatenate(train_panels, axis=1)
        seq_name = val_s["seq_name"] if isinstance(val_s.get("seq_name"), str) else f"idx{flat_idx}"
        fname = f"noise_compare_{vis_idx:02d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), combined)

    # ── 5. Print stats ──────────────────────────────────────────────────
    print("\n--- Pipeline Stats ---")

    # Collect bbox sizes and joint ranges from raw dataset
    bbox_widths, bbox_heights = [], []
    pelvis_depths = []
    joint_ranges = []

    rng = random.Random(42)
    sample_indices = rng.sample(range(len(raw_ds)), min(200, len(raw_ds)))

    for idx in sample_indices:
        try:
            s = raw_ds[idx]
        except FileNotFoundError:
            continue
        if "bbox" in s:
            bbox = s["bbox"]
            bbox_widths.append(bbox[2] - bbox[0])
            bbox_heights.append(bbox[3] - bbox[1])
        joints = s["joints"]
        pelvis_depths.append(joints[PELVIS_IDX, 0])  # X = forward/depth
        joint_ranges.append(joints.max(axis=0) - joints.min(axis=0))

    if bbox_widths:
        print(f"Bbox width:  min={min(bbox_widths):.0f}  max={max(bbox_widths):.0f}  "
              f"mean={np.mean(bbox_widths):.0f}  median={np.median(bbox_widths):.0f}")
        print(f"Bbox height: min={min(bbox_heights):.0f}  max={max(bbox_heights):.0f}  "
              f"mean={np.mean(bbox_heights):.0f}  median={np.median(bbox_heights):.0f}")

    print(f"Pelvis depth (X): min={min(pelvis_depths):.2f}m  max={max(pelvis_depths):.2f}m  "
          f"mean={np.mean(pelvis_depths):.2f}m")

    jr = np.array(joint_ranges)
    print(f"Joint span X (depth):  mean={jr[:,0].mean():.3f}m  max={jr[:,0].max():.3f}m")
    print(f"Joint span Y (left):   mean={jr[:,1].mean():.3f}m  max={jr[:,1].max():.3f}m")
    print(f"Joint span Z (up):     mean={jr[:,2].mean():.3f}m  max={jr[:,2].max():.3f}m")

    # Root-relative stats
    print("\n--- Root-Relative Joint Stats ---")
    rel_ranges = []
    for idx in sample_indices:
        try:
            s = raw_ds[idx]
        except FileNotFoundError:
            continue
        joints = s["joints"]
        pelvis = joints[PELVIS_IDX:PELVIS_IDX+1]
        rel = joints - pelvis
        rel_ranges.append(np.abs(rel).max(axis=0))

    rr = np.array(rel_ranges)
    print(f"Max |rel| X: mean={rr[:,0].mean():.3f}m  max={rr[:,0].max():.3f}m")
    print(f"Max |rel| Y: mean={rr[:,1].mean():.3f}m  max={rr[:,1].max():.3f}m")
    print(f"Max |rel| Z: mean={rr[:,2].mean():.3f}m  max={rr[:,2].max():.3f}m")

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
