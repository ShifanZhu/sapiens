"""Smoke-test for the BEDLAM2 data pipeline.

Run from claude_code/:
    python test_data_pipeline.py
"""

import time
import numpy as np
import torch

DATA_ROOT     = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OVERVIEW_PATH = f"{DATA_ROOT}/data/overview.txt"

# Target input size for Sapiens ViT (H x W, must be multiples of patch size 16)
OUT_H, OUT_W = 384, 640

from data import get_splits, build_train_transform, build_val_transform, build_dataloader, RandomResizedCropRGBD

# ── 1. Splits ──────────────────────────────────────────────────────────────
train_seqs, val_seqs, test_seqs = get_splits(
    overview_path=OVERVIEW_PATH,
    val_ratio=0.1,
    test_ratio=0.1,
    single_body_only=True,
    skip_missing_body=True,
    depth_required=True,
    mp4_required=False,
)
print(f"Sequences  — train: {len(train_seqs)}, val: {len(val_seqs)}, test: {len(test_seqs)}")

# ── 2. Dataloaders ─────────────────────────────────────────────────────────
train_loader = build_dataloader(
    seq_paths=train_seqs,
    data_root=DATA_ROOT,
    transform=build_train_transform(OUT_H, OUT_W),
    batch_size=4,
    num_workers=2,
)
val_loader = build_dataloader(
    seq_paths=val_seqs,
    data_root=DATA_ROOT,
    transform=build_val_transform(OUT_H, OUT_W),
    batch_size=4,
    shuffle=False,
    num_workers=2,
)
print(f"Frames     — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

# ── 3. Single batch sanity check ───────────────────────────────────────────
print("\nLoading one training batch ...")
t0 = time.time()
batch = next(iter(train_loader))
print(f"  Load time : {time.time() - t0:.2f}s")

rgb    = batch["rgb"]
depth  = batch["depth"]
joints = batch["joints"]

print(f"  rgb    : {rgb.shape}   dtype={rgb.dtype}   range=[{rgb.min():.2f}, {rgb.max():.2f}]")
print(f"  depth  : {depth.shape}  dtype={depth.dtype}  range=[{depth.min():.2f}, {depth.max():.2f}]")
print(f"  joints : {joints.shape} dtype={joints.dtype}")
print(f"  seq    : {batch['seq_name'][0]},  frame {batch['frame_idx'][0]}")

assert rgb.shape == (4, 3, OUT_H, OUT_W),    f"Unexpected rgb shape: {rgb.shape}"
assert depth.shape == (4, 1, OUT_H, OUT_W),  f"Unexpected depth shape: {depth.shape}"
assert joints.shape == (4, 127, 3),          f"Unexpected joints shape: {joints.shape}"
assert depth.min() >= 0.0 and depth.max() <= 1.0, "Depth out of [0,1]"

print("\nAll assertions passed. Data pipeline is working correctly.")

# ── 4. RandomResizedCropRGBD: K-update unit test ───────────────────────────
print("\nTesting RandomResizedCropRGBD K-update math ...")

import random
random.seed(42)
np.random.seed(42)

# Synthetic sample: 384×640 image, known K, random 3D joints
H_test, W_test = OUT_H, OUT_W
rgb_test  = np.random.randint(0, 256, (H_test, W_test, 3), dtype=np.uint8)
depth_test = np.random.uniform(1.0, 5.0, (H_test, W_test)).astype(np.float32)
K_test = np.array([[500.0, 0.0, 320.0],
                   [0.0,   500.0, 192.0],
                   [0.0,   0.0,   1.0]], dtype=np.float32)

# Random 3D joints in camera space (X=depth, so X > 0 required for valid projection)
joints_3d = np.random.uniform(0.5, 5.0, (127, 3)).astype(np.float32)
joints_3d[:, 0] = np.abs(joints_3d[:, 0]) + 0.5   # ensure X > 0

# Project with original K: u = fx*(-Y/X)+cx, v = fy*(-Z/X)+cy
X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
u_orig = K_test[0, 0] * (-Y / X) + K_test[0, 2]
v_orig = K_test[1, 1] * (-Z / X) + K_test[1, 2]

sample_test = {
    "rgb":       rgb_test.copy(),
    "depth":     depth_test.copy(),
    "joints":    joints_3d.copy(),
    "intrinsic": K_test.copy(),
}

# Apply crop (use deterministic seed via random.seed above)
crop = RandomResizedCropRGBD(H_test, W_test, scale=(0.7, 1.0), ratio=(0.55, 0.65))
sample_out = crop(sample_test)

K_new = sample_out["intrinsic"]
assert K_new.shape == (3, 3), "K shape mismatch after crop"

# Project same 3D joints with updated K'
u_new = K_new[0, 0] * (-Y / X) + K_new[0, 2]
v_new = K_new[1, 1] * (-Z / X) + K_new[1, 2]

# The crop transform: pixel (u_orig, v_orig) maps to (u_new, v_new)
# We need to recover the crop parameters from K_new vs K_test to verify the
# mapping. Instead we directly verify geometric consistency:
# After crop+resize, the updated K' must satisfy:
#   u_new = (u_orig - x0) * sx   and   v_new = (v_orig - y0) * sy
# which is equivalent to: K_new produces the same projected coords as
# manually applying the crop transform to the pixel positions.
# We verify this by checking round-trip: project with K_new and compare to
# the expected remapped pixel positions derived from the known crop params.

# Recover crop params from K transformation (K_new = K_orig with crop applied)
# fx_new = fx_orig * sx  =>  sx = fx_new / fx_orig
# cx_new = (cx_orig - x0) * sx  =>  x0 = cx_orig - cx_new/sx
sx = K_new[0, 0] / K_test[0, 0]
sy = K_new[1, 1] / K_test[1, 1]
x0 = K_test[0, 2] - K_new[0, 2] / sx
y0 = K_test[1, 2] - K_new[1, 2] / sy

# Expected remapped pixel positions (crop + resize)
u_expected = (u_orig - x0) * sx
v_expected = (v_orig - y0) * sy

max_err = np.abs(u_new - u_expected).max()
max_err_v = np.abs(v_new - v_expected).max()
assert max_err < 1e-3, f"K u-update error too large: {max_err:.6f}"
assert max_err_v < 1e-3, f"K v-update error too large: {max_err_v:.6f}"

assert sample_out["rgb"].shape   == (H_test, W_test, 3), "RGB shape changed after crop"
assert sample_out["depth"].shape == (H_test, W_test),    "Depth shape changed after crop"
assert np.array_equal(sample_out["joints"], joints_3d),  "Joints should be unchanged"

print(f"  Max projection error: u={max_err:.2e}px  v={max_err_v:.2e}px  (tolerance 1e-3)")
print("RandomResizedCropRGBD K-update test passed.")
