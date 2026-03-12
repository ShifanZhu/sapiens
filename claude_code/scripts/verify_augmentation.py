"""Verify joint projection is correct after Resize (K scaling check).

Saves a PNG grid with 4 columns per sample:
  1. Original RGB + projected 2D joints  (using original K)
  2. Original depth
  3. Resized RGB + projected 2D joints   (using K scaled by Resize)
  4. Resized depth

If the joints land on the same body part in both columns 1 and 3,
the intrinsic K is being scaled correctly by the Resize transform.

Run:
    conda run -n sapiens_gpu python verify_augmentation.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import get_splits, build_train_transform
from data.constants import RGB_MEAN, RGB_STD, SMPLX_SKELETON
from data.dataset import BedlamFrameDataset

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OUT_H, OUT_W = 384, 640
N_SAMPLES = 6        # rows in the output grid
OUT_PATH = "verify_augmentation.png"

# ── Helpers ───────────────────────────────────────────────────────────────────

_RGB_MEAN = np.array(RGB_MEAN, dtype=np.float32).reshape(1, 1, 3)
_RGB_STD  = np.array(RGB_STD,  dtype=np.float32).reshape(1, 1, 3)


def denorm_rgb(tensor) -> np.ndarray:
    """(3,H,W) float32 normalised tensor → (H,W,3) uint8."""
    arr = tensor.numpy().transpose(1, 2, 0)   # HWC
    arr = np.clip(arr * _RGB_STD + _RGB_MEAN, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def project_joints(joints: np.ndarray, K: np.ndarray):
    """Project (J,3) camera-space joints → pixel (u,v).

    Convention (BEDLAM2): u = fx*(-Y/X)+cx,  v = fy*(-Z/X)+cy
    Returns u, v arrays and boolean valid mask.
    """
    X, Y, Z = joints[:, 0], joints[:, 1], joints[:, 2]
    valid = X > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(valid, K[0, 0] * (-Y / X) + K[0, 2], -1.0)
        v = np.where(valid, K[1, 1] * (-Z / X) + K[1, 2], -1.0)
    return u, v, valid


def draw_joints_on_image(img: np.ndarray, u, v, valid, H: int, W: int) -> np.ndarray:
    """Draw skeleton bones + joint dots on a (H,W,3) uint8 image."""
    out = img.copy()
    for a, b in SMPLX_SKELETON:
        if valid[a] and valid[b]:
            pa = (int(round(u[a])), int(round(v[a])))
            pb = (int(round(u[b])), int(round(v[b])))
            cv2.line(out, pa, pb, (0, 220, 0), 1, cv2.LINE_AA)
    for j in range(len(u)):
        if valid[j]:
            pt = (int(round(u[j])), int(round(v[j])))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(out, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def depth_colormap(depth: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """(H,W) float32 → (H,W,3) uint8 plasma colormap."""
    d = np.clip(depth, 0.0, vmax or depth.max() or 1.0)
    d_norm = (d / (vmax or d.max() or 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ── Data ──────────────────────────────────────────────────────────────────────

print("Loading splits ...")
_, val_seqs, _ = get_splits(
    overview_path=f"{DATA_ROOT}/data/overview.txt",
    val_ratio=0.1, test_ratio=0.1, seed=2026,
    single_body_only=True, skip_missing_body=True,
    depth_required=True, mp4_required=False,
)

# Raw dataset — no transform so we get original numpy arrays
dataset = BedlamFrameDataset(
    seq_paths=val_seqs[:30],    # first 30 val sequences, plenty for N_SAMPLES
    data_root=DATA_ROOT,
    transform=None,
)
print(f"Raw dataset: {len(dataset)} frames from {len(val_seqs[:30])} sequences")

train_tf = build_train_transform(OUT_H, OUT_W, scale_jitter=True)

# Space samples evenly across the dataset
indices = [i * (len(dataset) // N_SAMPLES) for i in range(N_SAMPLES)]

# ── Build figure ──────────────────────────────────────────────────────────────

COL_TITLES = [
    "Original RGB + Joints  (original K)",
    "Original Depth",
    "Resized RGB + Joints  (scaled K)",
    "Resized Depth",
]

fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(18, N_SAMPLES * 4 + 1))
for col, title in enumerate(COL_TITLES):
    axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=8)

for row, idx in enumerate(indices):
    sample = dataset[idx]
    seq_label = f"{sample['seq_name']}\nframe {sample['frame_idx']}"

    rgb_orig   = sample["rgb"]        # (H, W, 3) uint8, original resolution
    depth_orig = sample["depth"]      # (H, W) float32, metres
    joints     = sample["joints"]     # (127, 3) float32, camera-space XYZ
    K_orig     = sample["intrinsic"]  # (3, 3)

    H_orig, W_orig = rgb_orig.shape[:2]

    # Project joints onto original image using original K
    u_orig, v_orig, valid_orig = project_joints(joints, K_orig)
    rgb_orig_joints = draw_joints_on_image(rgb_orig, u_orig, v_orig, valid_orig,
                                           H_orig, W_orig)
    in_orig = int(((u_orig >= 0) & (u_orig < W_orig) &
                   (v_orig >= 0) & (v_orig < H_orig) & valid_orig).sum())

    # Apply Resize → ToTensor
    sample_resized = train_tf(copy.deepcopy(sample))

    rgb_resized  = denorm_rgb(sample_resized["rgb"])       # (OUT_H, OUT_W, 3) uint8
    depth_resized = sample_resized["depth"].numpy()[0]     # (OUT_H, OUT_W) float32 [0,1]
    K_resized    = sample_resized["intrinsic"].numpy()     # (3, 3) scaled by Resize
    joints_resized = sample_resized["joints"].numpy()      # (127, 3) unchanged

    # Project same joints onto resized image using scaled K
    u_res, v_res, valid_res = project_joints(joints_resized, K_resized)
    rgb_resized_joints = draw_joints_on_image(rgb_resized, u_res, v_res, valid_res,
                                              OUT_H, OUT_W)
    in_res = int(((u_res >= 0) & (u_res < OUT_W) &
                  (v_res >= 0) & (v_res < OUT_H) & valid_res).sum())

    # Original depth display (clip at 95th percentile for contrast)
    depth_orig_disp = depth_orig if depth_orig is not None \
                      else np.zeros((H_orig, W_orig), np.float32)
    depth_vmax = float(np.percentile(depth_orig_disp[depth_orig_disp > 0], 95)) \
                 if (depth_orig_disp > 0).any() else 10.0

    # ── Plot row ──────────────────────────────────────────────────────────
    def annotate(ax, text):
        ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=8,
                color="yellow", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.2", fc="black", alpha=0.6, lw=0))

    axes[row, 0].imshow(rgb_orig_joints)
    axes[row, 0].set_ylabel(seq_label, fontsize=7, rotation=0,
                             ha="right", va="center", labelpad=60)
    axes[row, 0].set_xlabel(f"{in_orig}/127 joints in frame", fontsize=8)
    annotate(axes[row, 0], f"{W_orig}×{H_orig}")

    axes[row, 1].imshow(depth_colormap(depth_orig_disp, vmax=depth_vmax))
    annotate(axes[row, 1], f"{W_orig}×{H_orig}")

    axes[row, 2].imshow(rgb_resized_joints)
    axes[row, 2].set_xlabel(f"{in_res}/127 joints in frame", fontsize=8)
    annotate(axes[row, 2], f"{OUT_W}×{OUT_H}")

    axes[row, 3].imshow(depth_colormap(depth_resized, vmax=1.0))
    annotate(axes[row, 3], f"{OUT_W}×{OUT_H}")

    for col in range(4):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    print(f"  [{row+1}/{N_SAMPLES}] {sample['seq_name']} frame {sample['frame_idx']}"
          f"  — in-frame: orig={in_orig}/127  resized={in_res}/127")

fig.suptitle(
    "Joint Projection Verification: Original vs Resized  (Resize scales K proportionally)",
    fontsize=12, y=1.002,
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
print(f"\nSaved → {OUT_PATH}")
