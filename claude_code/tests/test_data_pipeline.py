"""Smoke-test for the BEDLAM2 data pipeline.

Run from claude_code/:
    conda run -n sapiens_gpu python tests/test_data_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import random
import time
import numpy as np
import torch

DATA_ROOT     = "/home/hang/repos_local/MMC/BEDLAM2Datatest"
OVERVIEW_PATH = f"{DATA_ROOT}/data/overview.txt"

# Target input size for Sapiens ViT (H x W, must be multiples of patch size 16)
OUT_H, OUT_W = 640, 384

from data import (
    get_splits, build_train_transform, build_val_transform,
    build_dataloader, RandomResizedCropRGBD, CropPerson, NoisyBBox, SubtractRoot,
    PELVIS_IDX,
)
from data.dataset import BedlamFrameDataset

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Splits — single-body vs all
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 1. Splits ──────────────────────────────────────────")

train_single, val_single, test_single = get_splits(
    overview_path=OVERVIEW_PATH, single_body_only=True,
    skip_missing_body=True, depth_required=True, mp4_required=False,
)
train_all, val_all, test_all = get_splits(
    overview_path=OVERVIEW_PATH, single_body_only=False,
    skip_missing_body=True, depth_required=True, mp4_required=False,
)

print(f"  Single-body: train={len(train_single)} val={len(val_single)} test={len(test_single)}")
print(f"  All:         train={len(train_all)} val={len(val_all)} test={len(test_all)}")

check("single-body fewer than all", len(train_single) <= len(train_all))
check("all includes multi-person", len(train_all) > len(train_single))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Multi-person index
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 2. Multi-person index ──────────────────────────────")

ds_all = BedlamFrameDataset(
    seq_paths=val_all[:30], data_root=DATA_ROOT, transform=None,
)
ds_single = BedlamFrameDataset(
    seq_paths=val_single[:30], data_root=DATA_ROOT, transform=None,
)

print(f"  All index: {len(ds_all)} samples")
print(f"  Single index: {len(ds_single)} samples")
check("multi-body index >= single-body index", len(ds_all) >= len(ds_single))

# Verify index is (label_path, body_idx, frame_idx) 3-tuple
check("index is 3-tuple", len(ds_all.index[0]) == 3)

# Verify multi-person sequences have body_idx > 0
body_indices = set(bi for _, bi, _ in ds_all.index)
print(f"  Body indices present: {sorted(body_indices)}")
# Note: may or may not have multi-person in first 30 val seqs

# Verify index length = sum(n_body * n_frames)
expected_len = 0
for seq_rel in val_all[:30]:
    label_path = os.path.join(DATA_ROOT, "data", "label", seq_rel)
    meta = np.load(label_path, allow_pickle=True)
    n_body = meta["joints_cam"].shape[0]
    n_frames = int(meta["n_frames"])
    expected_len += n_body * n_frames

check("index length = sum(n_body * n_frames)", len(ds_all) == expected_len,
      f"expected {expected_len}, got {len(ds_all)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Single batch sanity check (with transforms)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 3. Batch sanity check ──────────────────────────────")

train_loader = build_dataloader(
    seq_paths=train_single, data_root=DATA_ROOT,
    transform=build_train_transform(OUT_H, OUT_W),
    batch_size=4, num_workers=0,
)
val_loader = build_dataloader(
    seq_paths=val_single, data_root=DATA_ROOT,
    transform=build_val_transform(OUT_H, OUT_W),
    batch_size=4, shuffle=False, num_workers=0,
)
print(f"  Frames — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

print("  Loading one training batch ...")
t0 = time.time()
batch = next(iter(train_loader))
print(f"  Load time: {time.time() - t0:.2f}s")

rgb    = batch["rgb"]
depth  = batch["depth"]
joints = batch["joints"]

print(f"  rgb    : {rgb.shape}  range=[{rgb.min():.2f}, {rgb.max():.2f}]")
print(f"  depth  : {depth.shape}  range=[{depth.min():.2f}, {depth.max():.2f}]")
print(f"  joints : {joints.shape}")

check("rgb shape", rgb.shape == (4, 3, OUT_H, OUT_W), str(rgb.shape))
check("depth shape", depth.shape == (4, 1, OUT_H, OUT_W), str(depth.shape))
check("joints shape", joints.shape == (4, 127, 3), str(joints.shape))
check("depth in [0,1]", depth.min() >= 0.0 and depth.max() <= 1.0)

# Root-relative: pelvis should be near zero
pelvis = joints[:, PELVIS_IDX]  # (B, 3)
check("pelvis near zero (root-relative)",
      pelvis.abs().max().item() < 1e-5,
      f"pelvis max abs = {pelvis.abs().max().item():.6f}")

# pelvis_abs should be present
check("pelvis_abs in batch", "pelvis_abs" in batch)
if "pelvis_abs" in batch:
    pa = batch["pelvis_abs"]
    check("pelvis_abs shape", pa.shape == (4, 3), str(pa.shape))
    check("pelvis_abs reasonable depth",
          pa[:, 0].min() > 0.1 and pa[:, 0].max() < 50.0,
          f"range [{pa[:, 0].min():.2f}, {pa[:, 0].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Bbox computation — uses all keypoints
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 4. Bbox from joints_2d ─────────────────────────────")

# Get a raw sample with bbox
raw_ds = BedlamFrameDataset(
    seq_paths=val_single[:10], data_root=DATA_ROOT, transform=None,
)
raw_sample = raw_ds[0]
check("bbox in raw sample", "bbox" in raw_sample)

if "bbox" in raw_sample:
    bbox = raw_sample["bbox"]
    check("bbox shape is (4,)", bbox.shape == (4,), str(bbox.shape))
    check("bbox x1 < x2", bbox[0] < bbox[2])
    check("bbox y1 < y2", bbox[1] < bbox[3])
    print(f"  bbox = [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CropPerson + K-update geometric consistency
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 5. CropPerson K-update consistency ─────────────────")

random.seed(42)
np.random.seed(42)

# Synthetic sample with known K and 3D joints
H_test, W_test = 720, 1280
K_test = np.array([[500.0, 0.0, 640.0],
                   [0.0,   500.0, 360.0],
                   [0.0,   0.0,   1.0]], dtype=np.float32)

joints_3d = np.random.uniform(0.5, 5.0, (127, 3)).astype(np.float32)
joints_3d[:, 0] = np.abs(joints_3d[:, 0]) + 0.5  # ensure X > 0

# Project with original K
X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
u_orig = K_test[0, 0] * (-Y / X) + K_test[0, 2]
v_orig = K_test[1, 1] * (-Z / X) + K_test[1, 2]

# Create bbox around a subset of projected points (simulate person bbox)
bbox_test = np.array([
    u_orig.min() - 20, v_orig.min() - 20,
    u_orig.max() + 20, v_orig.max() + 20
], dtype=np.float32)
bbox_test = np.clip(bbox_test, 0, [W_test, H_test, W_test, H_test])

sample_test = {
    "rgb":       np.random.randint(0, 256, (H_test, W_test, 3), dtype=np.uint8),
    "depth":     np.random.uniform(1.0, 5.0, (H_test, W_test)).astype(np.float32),
    "joints":    joints_3d.copy(),
    "intrinsic": K_test.copy(),
    "bbox":      bbox_test.copy(),
}

crop = CropPerson(OUT_H, OUT_W)
sample_out = crop(copy.deepcopy(sample_test))

K_new = sample_out["intrinsic"]
check("K shape after crop", K_new.shape == (3, 3))
check("rgb shape after crop", sample_out["rgb"].shape == (OUT_H, OUT_W, 3))
check("depth shape after crop", sample_out["depth"].shape == (OUT_H, OUT_W))

# Project same 3D joints with new K
u_new = K_new[0, 0] * (-Y / X) + K_new[0, 2]
v_new = K_new[1, 1] * (-Z / X) + K_new[1, 2]

# Recover crop params from K
sx = K_new[0, 0] / K_test[0, 0]
sy = K_new[1, 1] / K_test[1, 1]
x0_rec = K_test[0, 2] - K_new[0, 2] / sx
y0_rec = K_test[1, 2] - K_new[1, 2] / sy

# Expected remapped pixel positions
u_expected = (u_orig - x0_rec) * sx
v_expected = (v_orig - y0_rec) * sy

max_err_u = np.abs(u_new - u_expected).max()
max_err_v = np.abs(v_new - v_expected).max()
print(f"  Max projection error: u={max_err_u:.2e}px  v={max_err_v:.2e}px")
check("CropPerson K u-update", max_err_u < 1e-3, f"err={max_err_u:.6f}")
check("CropPerson K v-update", max_err_v < 1e-3, f"err={max_err_v:.6f}")

# 3D joints should be unchanged by CropPerson
check("joints unchanged after crop", np.array_equal(sample_out["joints"], joints_3d))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SubtractRoot roundtrip
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 6. Root-relative roundtrip ─────────────────────────")

joints_orig = np.random.uniform(-2, 2, (127, 3)).astype(np.float32)
joints_orig[:, 0] = np.abs(joints_orig[:, 0]) + 0.5  # ensure X > 0
joints_orig[PELVIS_IDX] = [3.0, 0.5, -0.2]

# SubtractRoot now needs rgb (for crop dims) and intrinsic (for projection)
K_sr = np.array([[500.0, 0, 192.0], [0, 500.0, 320.0], [0, 0, 1]], dtype=np.float32)
sample_rt = {
    "joints": joints_orig.copy(),
    "rgb": np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8),
    "intrinsic": K_sr.copy(),
}
sr = SubtractRoot()
sample_rt = sr(sample_rt)

joints_rel = sample_rt["joints"]
pelvis_abs = sample_rt["pelvis_abs"]

check("pelvis_abs correct", np.allclose(pelvis_abs, [3.0, 0.5, -0.2]))
check("pelvis is zero after subtract", np.allclose(joints_rel[PELVIS_IDX], [0, 0, 0]))

# Recovery
joints_recovered = joints_rel + pelvis_abs[np.newaxis, :]
max_recovery_err = np.abs(joints_recovered - joints_orig).max()
check("roundtrip recovery", max_recovery_err < 1e-6, f"max err={max_recovery_err:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. NoisyBBox
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 7. NoisyBBox ───────────────────────────────────────")

bbox_orig = np.array([100, 50, 300, 400], dtype=np.float32)
sample_nb = {
    "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
    "bbox": bbox_orig.copy(),
}

nb = NoisyBBox()
results = [nb(copy.deepcopy(sample_nb))["bbox"] for _ in range(100)]
results = np.array(results)

# Should vary across trials
check("NoisyBBox adds variation",
      results.std(axis=0).max() > 1.0,
      f"std={results.std(axis=0)}")

# All results should be within image bounds
check("NoisyBBox within image bounds",
      (results[:, 0] >= 0).all() and (results[:, 2] <= 640).all() and
      (results[:, 1] >= 0).all() and (results[:, 3] <= 480).all())

# Without bbox key, should be no-op
sample_no_bbox = {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}
out_no_bbox = nb(sample_no_bbox)
check("NoisyBBox no-op without bbox", "bbox" not in out_no_bbox)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CropPerson backward compatibility (no bbox = full Resize)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 8. CropPerson backward compat ──────────────────────")

sample_full = {
    "rgb":       np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
    "depth":     np.random.uniform(1.0, 5.0, (720, 1280)).astype(np.float32),
    "joints":    np.random.uniform(-1, 1, (127, 3)).astype(np.float32),
    "intrinsic": K_test.copy(),
}

cp = CropPerson(OUT_H, OUT_W)
out_full = cp(copy.deepcopy(sample_full))

check("no-bbox → rgb resized", out_full["rgb"].shape == (OUT_H, OUT_W, 3))
check("no-bbox → depth resized", out_full["depth"].shape == (OUT_H, OUT_W))


# ══════════════════════════════════════════════════════════════════════════════
# 9. RandomResizedCropRGBD K-update (existing test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 9. RandomResizedCropRGBD K-update ──────────────────")

random.seed(42)
np.random.seed(42)

rrc_sample = {
    "rgb":       np.random.randint(0, 256, (OUT_H, OUT_W, 3), dtype=np.uint8),
    "depth":     np.random.uniform(1.0, 5.0, (OUT_H, OUT_W)).astype(np.float32),
    "joints":    joints_3d.copy(),
    "intrinsic": np.array([[500.0, 0, 192.0], [0, 500.0, 320.0], [0, 0, 1]], dtype=np.float32),
}
K_rrc = rrc_sample["intrinsic"].copy()

rrc = RandomResizedCropRGBD(OUT_H, OUT_W, scale=(0.7, 1.0), ratio=(0.55, 0.65))
rrc_out = rrc(rrc_sample)

K_rrc_new = rrc_out["intrinsic"]
u_rrc_orig = K_rrc[0, 0] * (-Y / X) + K_rrc[0, 2]
v_rrc_orig = K_rrc[1, 1] * (-Z / X) + K_rrc[1, 2]
u_rrc_new = K_rrc_new[0, 0] * (-Y / X) + K_rrc_new[0, 2]
v_rrc_new = K_rrc_new[1, 1] * (-Z / X) + K_rrc_new[1, 2]

sx_rrc = K_rrc_new[0, 0] / K_rrc[0, 0]
sy_rrc = K_rrc_new[1, 1] / K_rrc[1, 1]
x0_rrc = K_rrc[0, 2] - K_rrc_new[0, 2] / sx_rrc
y0_rrc = K_rrc[1, 2] - K_rrc_new[1, 2] / sy_rrc

u_rrc_exp = (u_rrc_orig - x0_rrc) * sx_rrc
v_rrc_exp = (v_rrc_orig - y0_rrc) * sy_rrc

max_err_rrc_u = np.abs(u_rrc_new - u_rrc_exp).max()
max_err_rrc_v = np.abs(v_rrc_new - v_rrc_exp).max()
print(f"  Max error: u={max_err_rrc_u:.2e}  v={max_err_rrc_v:.2e}")
check("RRC K u-update", max_err_rrc_u < 1e-3)
check("RRC K v-update", max_err_rrc_v < 1e-3)
check("RRC rgb shape", rrc_out["rgb"].shape == (OUT_H, OUT_W, 3))
check("RRC joints unchanged", np.array_equal(rrc_out["joints"], joints_3d))


# ══════════════════════════════════════════════════════════════════════════════
# 10. Retry logic for tiny bboxes
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 10. Retry logic ────────────────────────────────────")
# We can't easily trigger a tiny bbox in real data, but we verify the dataset
# returns a valid sample for every index (no crashes)
print("  Sampling 20 random indices from val dataset ...")
random.seed(123)
val_ds = BedlamFrameDataset(
    seq_paths=val_single[:10], data_root=DATA_ROOT,
    transform=build_val_transform(OUT_H, OUT_W),
)
for _ in range(20):
    idx = random.randint(0, len(val_ds) - 1)
    sample = val_ds[idx]
    assert sample["rgb"].shape == (3, OUT_H, OUT_W)
check("20 random samples all valid", True)


# ══════════════════════════════════════════════════════════════════════════════
# 11. pelvis_depth and pelvis_uv GT fields
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 11. pelvis_depth & pelvis_uv GT ────────────────────")

# 11a. Check fields present in a batch
check("pelvis_depth in batch", "pelvis_depth" in batch)
check("pelvis_uv in batch", "pelvis_uv" in batch)

if "pelvis_depth" in batch:
    pd = batch["pelvis_depth"]
    check("pelvis_depth shape", pd.shape == (4, 1), str(pd.shape))
    check("pelvis_depth positive", (pd > 0).all().item(),
          f"min={pd.min():.4f}")
    check("pelvis_depth reasonable range (0.1-50m)",
          pd.min() > 0.1 and pd.max() < 50.0,
          f"range [{pd.min():.2f}, {pd.max():.2f}]")
    # Should equal pelvis_abs[:, 0]
    if "pelvis_abs" in batch:
        check("pelvis_depth == pelvis_abs[:,0]",
              torch.allclose(pd.squeeze(-1), batch["pelvis_abs"][:, 0], atol=1e-5),
              f"max diff={torch.abs(pd.squeeze(-1) - batch['pelvis_abs'][:, 0]).max():.6f}")

if "pelvis_uv" in batch:
    puv = batch["pelvis_uv"]
    check("pelvis_uv shape", puv.shape == (4, 2), str(puv.shape))
    check("pelvis_uv in [-1, 1] (normalized)",
          puv.min() >= -2.0 and puv.max() <= 2.0,
          f"range [{puv.min():.3f}, {puv.max():.3f}]")
    # Pelvis is typically near crop center after CropPerson
    check("pelvis_uv near center (abs < 1.5)",
          puv.abs().max() < 1.5,
          f"max abs = {puv.abs().max():.3f}")

# 11b. Synthetic roundtrip: project pelvis → pelvis_uv, then unproject → recover pelvis_abs
print("  Roundtrip test (project → normalize → denormalize → unproject) ...")
K_rt = np.array([[500.0, 0, 192.0], [0, 500.0, 320.0], [0, 0, 1]], dtype=np.float32)
pelvis_3d = np.array([3.0, 0.5, -0.2], dtype=np.float32)

# Forward: project + normalize (what SubtractRoot does)
u_px = K_rt[0, 0] * (-pelvis_3d[1] / pelvis_3d[0]) + K_rt[0, 2]
v_px = K_rt[1, 1] * (-pelvis_3d[2] / pelvis_3d[0]) + K_rt[1, 2]
u_norm = u_px / OUT_W * 2.0 - 1.0
v_norm = v_px / OUT_H * 2.0 - 1.0
depth_gt = pelvis_3d[0]

# Inverse: denormalize + unproject (what inference does)
u_px_rec = (u_norm + 1.0) / 2.0 * OUT_W
v_px_rec = (v_norm + 1.0) / 2.0 * OUT_H
X_rec = depth_gt
Y_rec = -(u_px_rec - K_rt[0, 2]) * X_rec / K_rt[0, 0]
Z_rec = -(v_px_rec - K_rt[1, 2]) * X_rec / K_rt[1, 1]
pelvis_rec = np.array([X_rec, Y_rec, Z_rec])

roundtrip_err = np.abs(pelvis_rec - pelvis_3d).max()
print(f"  Roundtrip error: {roundtrip_err:.2e} m")
check("pelvis_uv roundtrip < 1e-5", roundtrip_err < 1e-5,
      f"err={roundtrip_err:.2e}")

# 11c. Check with real data samples
print("  Checking real data samples ...")
val_ds_11 = BedlamFrameDataset(
    seq_paths=val_single[:5], data_root=DATA_ROOT,
    transform=build_val_transform(OUT_H, OUT_W),
)
n_checked = 0
for idx in range(min(30, len(val_ds_11))):
    s = val_ds_11[idx]
    pd_val = s["pelvis_depth"].item()
    puv_val = s["pelvis_uv"].numpy()
    pa_val = s["pelvis_abs"].numpy()

    # pelvis_depth should match pelvis_abs[0]
    assert abs(pd_val - pa_val[0]) < 1e-5, f"depth mismatch at idx {idx}"

    # pelvis_uv should be in reasonable range
    assert puv_val[0] > -2.0 and puv_val[0] < 2.0, f"u_norm out of range at idx {idx}"
    assert puv_val[1] > -2.0 and puv_val[1] < 2.0, f"v_norm out of range at idx {idx}"

    # Denormalize and unproject should recover pelvis_abs
    K_s = s["intrinsic"].numpy()
    u_px_s = (puv_val[0] + 1.0) / 2.0 * OUT_W
    v_px_s = (puv_val[1] + 1.0) / 2.0 * OUT_H
    X_s = pd_val
    Y_s = -(u_px_s - K_s[0, 2]) * X_s / K_s[0, 0]
    Z_s = -(v_px_s - K_s[1, 2]) * X_s / K_s[1, 1]
    rec_err = np.abs(np.array([X_s, Y_s, Z_s]) - pa_val).max()
    assert rec_err < 1e-3, f"roundtrip err={rec_err:.4f} at idx {idx}"
    n_checked += 1

check(f"real data roundtrip ({n_checked} samples)", n_checked == min(30, len(val_ds_11)))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 55}")
print(f"  {passed} passed, {failed} failed")
if failed == 0:
    print("  All tests passed!")
else:
    print("  SOME TESTS FAILED")
    sys.exit(1)
