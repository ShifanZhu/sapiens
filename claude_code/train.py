"""Training script for SapiensPose3D.

Usage:
    conda run -n sapiens_gpu python train.py \\
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \\
        --pretrain   checkpoints/sapiens_0.3b_epoch_1600_clean.pth \\
        --output-dir runs/exp001

Key hyper-parameters (override via CLI):
    --batch-size   per-GPU batch size            (default 16)
    --epochs       total training epochs         (default 50)
    --lr-backbone  backbone learning rate        (default 1e-5)
    --lr-head      head learning rate            (default 1e-4)
    --img-h/--img-w  input resolution            (default 384 × 640)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import resource
import sys
import time
from pathlib import Path

# Raise the open-file-descriptor limit to avoid "Too many open files" with
# many DataLoader workers each caching NPZ label handles.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data import get_splits, build_train_transform, build_val_transform, build_dataloader
from data.constants import RGB_MEAN, RGB_STD, SMPLX_SKELETON, NUM_JOINTS
from model import SapiensPose3D


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SapiensPose3D on BEDLAM2")

    # paths
    p.add_argument("--data-root",   required=True)
    p.add_argument("--pretrain",    required=True,
                   help="Path to Sapiens pretrain checkpoint")
    p.add_argument("--output-dir",  default="runs/exp001")
    p.add_argument("--resume",      default="", help="Resume from this checkpoint")

    # model
    p.add_argument("--arch",        default="sapiens_0.3b",
                   choices=["sapiens_0.3b", "sapiens_0.6b", "sapiens_1b", "sapiens_2b"])
    p.add_argument("--img-h",       type=int, default=640)
    p.add_argument("--img-w",       type=int, default=384)
    p.add_argument("--head-hidden", type=int, default=2048)
    p.add_argument("--drop-path",   type=float, default=0.1)
    p.add_argument("--head-dropout",type=float, default=0.2)

    # training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head",     type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float, default=0.03)
    p.add_argument("--warmup-epochs",type=int,  default=3)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--accum-steps", type=int,   default=1,
                   help="Gradient accumulation steps (effective BS = batch-size × accum-steps)")

    # data
    p.add_argument("--val-ratio",   type=float, default=0.1)
    p.add_argument("--test-ratio",  type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=2026)

    # misc
    p.add_argument("--log-interval",  type=int, default=50,  help="Print every N batches")
    p.add_argument("--save-interval", type=int, default=10,   help="Save checkpoint every N epochs")
    p.add_argument("--val-interval",  type=int, default=1,   help="Run validation every N epochs (default 5)")
    p.add_argument("--max-batches",   type=int, default=0,   help="Cap batches per epoch (0=unlimited, for quick debug)")
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      dest="amp", action="store_false")
    p.add_argument("--patience",    type=int, default=5,
                   help="Early stopping: stop if val MPJPE (body) does not improve for N "
                        "consecutive val checks (0=disabled)")
    p.add_argument("--no-scale-jitter", action="store_true", default=False,
                   help="Disable RandomResizedCropRGBD scale-jitter augmentation")
    p.add_argument("--single-body-only", action="store_true", default=False,
                   help="Only use single-person sequences (exclude multi-person)")
    p.add_argument("--lambda-depth", type=float, default=1.0,
                   help="Loss weight for pelvis depth prediction")
    p.add_argument("--lambda-uv", type=float, default=1.0,
                   help="Loss weight for pelvis UV prediction (normalized to [-1,1])")

    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

# Joint index ranges for per-group MPJPE (indices into the active joint array)
# Active ordering: body(0-21), eyes(22-23), hands(24-53), non-face surface(54-69)
BODY_IDX  = slice(0, 22)   # core kinematic joints (unchanged)
HAND_IDX  = slice(24, 54)  # left + right hand joints (was 25-54 before jaw removal)

def mpjpe(pred: torch.Tensor, target: torch.Tensor,
          idx: slice | None = None) -> torch.Tensor:
    """Mean per-joint position error in **millimetres**.

    Args:
        pred, target: ``(B, J, 3)`` in metres.
        idx:  Optional slice to restrict which joints are evaluated.
    Returns:
        Scalar tensor (mm).
    """
    if idx is not None:
        pred, target = pred[:, idx], target[:, idx]
    return (pred - target).norm(dim=-1).mean() * 1000.0


# ── Loss ──────────────────────────────────────────────────────────────────────

def pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss over all joints and coordinates.

    beta=0.05 m (5 cm) — treats errors below 5 cm as L2, above as L1.
    Units are metres so this is scale-appropriate.
    """
    return nn.functional.smooth_l1_loss(pred, target, beta=0.05)


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr_scale(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Linear warmup then cosine decay. Returns a multiplicative scale in (0,1]."""
    import math
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Logging ───────────────────────────────────────────────────────────────────

_CSV_FIELDNAMES = [
    "epoch", "lr_backbone", "lr_head",
    "train_loss", "train_loss_pose", "train_loss_depth", "train_loss_uv",
    "train_mpjpe_body",
    "val_loss", "val_loss_pose", "val_loss_depth", "val_loss_uv",
    "val_mpjpe_all", "val_mpjpe_body", "val_mpjpe_hand",
    "epoch_time",
]


class Logger:
    """Writes metrics to stdout and a CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDNAMES,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ── Visualization ─────────────────────────────────────────────────────────────

_RGB_MEAN_T = torch.tensor(RGB_MEAN, dtype=torch.float32).view(3, 1, 1)
_RGB_STD_T  = torch.tensor(RGB_STD,  dtype=torch.float32).view(3, 1, 1)

_VIS_FRAMES = 16   # frames per visualization video
_PERSON_COLORS = [          # one color per body_idx, cycling if n_body > 5
    (0,   255, 0),          # green
    (255, 128, 0),          # orange
    (0,   128, 255),        # blue
    (255,   0, 255),        # magenta
    (0,   255, 255),        # cyan
]


def draw_pose_frame(
    rgb_chw: np.ndarray,   # (3, H, W) uint8
    joints: np.ndarray,    # (NUM_JOINTS, 3) metres, root-relative or absolute
    K: np.ndarray,         # (3, 3)
    pelvis_abs: np.ndarray | None = None,  # (3,) — if given, joints are root-relative
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw all predicted joints + skeleton on one RGB frame. Returns (3,H,W) uint8."""
    import cv2

    # If root-relative, convert to absolute for projection
    if pelvis_abs is not None:
        joints = joints + pelvis_abs[np.newaxis, :]

    img = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0)[:, :, ::-1])  # HWC BGR
    H, W = img.shape[:2]

    X, Y, Z = joints[:, 0], joints[:, 1], joints[:, 2]
    valid = X > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(valid, K[0, 0] * (-Y / X) + K[0, 2], -1.0)
        v = np.where(valid, K[1, 1] * (-Z / X) + K[1, 2], -1.0)

    # Draw skeleton bones
    for a, b in SMPLX_SKELETON:
        if valid[a] and valid[b]:
            cv2.line(img,
                     (int(round(u[a])), int(round(v[a]))),
                     (int(round(u[b])), int(round(v[b]))),
                     color, 2, cv2.LINE_AA)

    # Draw all active joints as dots
    for j in range(NUM_JOINTS):
        if valid[j]:
            pt = (int(round(u[j])), int(round(v[j])))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)

    return img[:, :, ::-1].transpose(2, 0, 1)  # CHW RGB uint8


def build_val_video(
    rgb_frames: list[np.ndarray],      # list of (3, H, W) uint8
    pred_frames: list[np.ndarray],     # list of (NUM_JOINTS, 3) float32
    K_frames: list[np.ndarray],        # list of (3, 3) float32
    pelvis_frames: list[np.ndarray | None] | None = None,  # list of (3,) float32
) -> np.ndarray:
    """Return (1, T, 3, H, W) uint8 for writer.add_video()."""
    if pelvis_frames is None:
        pelvis_frames = [None] * len(rgb_frames)
    frames = [draw_pose_frame(r, p, k, pelvis_abs=pa)
              for r, p, k, pa in zip(rgb_frames, pred_frames, K_frames, pelvis_frames)]
    return np.stack(frames, axis=0)[np.newaxis]  # (1, T, 3, H, W)


def select_vis_indices(
    dataset,
    n_rotate_true: int = 1,
    n_rotate_false: int = 1,
    n_multi_person: int = 1,
) -> list[int]:
    """Return flat indices of the first frame from sequences satisfying the slot distribution.

    Reads ``rotate_flag`` and ``n_body`` from each unique sequence's label NPZ.
    Returns ``n_rotate_true + n_rotate_false + n_multi_person`` indices:
    ``[rotate_true_idx, rotate_false_idx, multi_person_idx]``.

    The multi-person slot picks the first sequence with n_body > 1; the index
    returned points to the first frame of body_idx=1 so that the secondary
    person is shown.
    """
    true_indices: list[int] = []
    false_indices: list[int] = []
    multi_indices: list[int] = []
    seen: set = set()   # (label_path, body_idx) already assigned to a slot
    seen_label: set = set()  # label_path already loaded (cache rotate_flag/n_body)
    label_meta: dict = {}    # label_path -> (rotate_flag, n_body)

    for i, (label_path, body_idx, _frame_idx) in enumerate(dataset.index):
        key = (label_path, body_idx)
        if key in seen:
            continue

        if len(true_indices) >= n_rotate_true and len(false_indices) >= n_rotate_false \
                and len(multi_indices) >= n_multi_person:
            break

        if label_path not in label_meta:
            try:
                with np.load(label_path, allow_pickle=True) as f:
                    label_meta[label_path] = (bool(f["rotate_flag"]),
                                              int(f["joints_cam"].shape[0]))
            except Exception:
                label_meta[label_path] = (False, 1)

        rotate_flag, n_body = label_meta[label_path]

        if n_body > 1 and len(multi_indices) < n_multi_person:
            multi_indices.append(i)
            seen.add(key)
        elif rotate_flag and len(true_indices) < n_rotate_true:
            true_indices.append(i)
            seen.add(key)
        elif not rotate_flag and len(false_indices) < n_rotate_false:
            false_indices.append(i)
            seen.add(key)

    return true_indices + false_indices + multi_indices


def sample_random_vis_index(dataset) -> int:
    """Return flat index of the first frame from a randomly selected sequence."""
    seq_first: dict[tuple, int] = {}
    for i, (label_path, body_idx, _frame_idx) in enumerate(dataset.index):
        key = (label_path, body_idx)
        if key not in seq_first:
            seq_first[key] = i
    key = random.choice(list(seq_first.keys()))
    return seq_first[key]


def recover_pelvis_from_pred(
    pred_depth: float,
    pred_uv: np.ndarray,          # (2,) normalized to [-1, 1]
    K: np.ndarray,                # (3, 3) crop intrinsic
    crop_hw: tuple[int, int] = (640, 384),
) -> np.ndarray:
    """Recover absolute pelvis XYZ from predicted depth and UV.

    Returns ``(3,)`` float32 ``[X, Y, Z]`` in camera space metres.
    """
    crop_h, crop_w = crop_hw
    u_crop = (pred_uv[0] + 1.0) / 2.0 * crop_w
    v_crop = (pred_uv[1] + 1.0) / 2.0 * crop_h
    X = float(pred_depth)
    fx, cx = float(K[0, 0]), float(K[0, 2])
    fy, cy = float(K[1, 1]), float(K[1, 2])
    Y = -(u_crop - cx) * X / fx
    Z = -(v_crop - cy) * X / fy
    return np.array([X, Y, Z], dtype=np.float32)


def visualize_fixed_samples(
    model: nn.Module,
    dataset,
    indices: list[int],
    device: torch.device,
    val_tf,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Render _VIS_FRAMES-frame pose-overlay videos for each starting index.

    Uses val_tf (no augmentation) regardless of the dataset's own transform.
    Returns a list of ``(vid_gt, vid_pred)`` pairs, one per starting index.
    ``vid_gt`` anchors joints at GT pelvis; ``vid_pred`` anchors at the pelvis
    recovered from ``pred_depth`` + ``pred_uv``.  Each video is
    ``(1, T, 3, H, W)`` uint8 ready for ``writer.add_video()``.
    """
    orig_transform = dataset.transform
    dataset.transform = val_tf   # deterministic, no jitter
    result = []
    try:
        for start_idx in indices:
            label_path, body_idx, _ = dataset.index[start_idx]

            # Detect multi-person sequence
            try:
                with np.load(label_path, allow_pickle=True) as f:
                    n_body = int(f["joints_cam"].shape[0])
            except Exception:
                n_body = 1
            is_multi = n_body > 1

            # For multi-person: build frame_idx → {body_idx: flat_index} lookup
            if is_multi:
                frame_body_map: dict[int, dict[int, int]] = {}
                for j, (lp, bi, fi_) in enumerate(dataset.index):
                    if lp == label_path:
                        frame_body_map.setdefault(fi_, {})[bi] = j

            rgb_frames, pred_frames, K_frames = [], [], []
            orig_rgb_frames: list[np.ndarray] = []
            orig_K_frames: list[np.ndarray] = []
            gt_pelvis_frames: list[np.ndarray | None] = []
            # per frame: list of (joints (NUM_JOINTS,3), pelvis_abs (3,), color) for each body
            multi_pred_per_frame: list[list[tuple]] = []

            i = start_idx
            while len(rgb_frames) < _VIS_FRAMES and i < len(dataset.index):
                lp, bi, fi = dataset.index[i]
                if lp != label_path or bi != body_idx:
                    break
                sample = dataset[i]
                rgb_t   = sample["rgb"]        # (3, H, W) float32 normalized
                depth_t = sample["depth"]      # (1, H, W) float32
                K_t     = sample["intrinsic"]  # (3, 3) crop K tensor

                x = torch.cat([rgb_t.unsqueeze(0), depth_t.unsqueeze(0)], dim=1).to(device)
                with torch.no_grad():
                    out = model(x)
                    pred_joints_np = out["joints"][0].float().cpu().numpy()
                    pd_val = out["pelvis_depth"][0, 0].float().cpu().item()
                    pu_val = out["pelvis_uv"][0].float().cpu().numpy()
                del x, out

                rgb_u8 = ((rgb_t * _RGB_STD_T + _RGB_MEAN_T).clamp(0, 1) * 255).byte().numpy()
                K_np   = K_t.numpy() if isinstance(K_t, torch.Tensor) else K_t
                rgb_frames.append(rgb_u8)
                pred_frames.append(pred_joints_np)
                K_frames.append(K_np)

                # Original (uncropped) RGB and K for pred_pelvis video
                cached = dataset._label_cache[lp]
                orig_rgb_hwc = dataset._read_frame(
                    cached["folder_name"], cached["seq_name"], fi, lp
                )
                orig_rgb_frames.append(orig_rgb_hwc.transpose(2, 0, 1))  # (3, H, W) uint8
                orig_K_frames.append(cached["intrinsic_matrix"])

                # GT pelvis (selected body only, used for vid_gt)
                if "pelvis_abs" in sample:
                    pa = sample["pelvis_abs"]
                    gt_pelvis_frames.append(pa.numpy() if isinstance(pa, torch.Tensor) else pa)
                else:
                    gt_pelvis_frames.append(None)

                # Collect per-body predictions for pred_pelvis video
                if is_multi:
                    frame_bodies: list[tuple] = []
                    for bk in range(n_body):
                        idx_bk = frame_body_map.get(fi, {}).get(bk)
                        if idx_bk is None:
                            continue
                        sample_bk = dataset[idx_bk]
                        K_bk = sample_bk["intrinsic"]
                        K_bk = K_bk.numpy() if isinstance(K_bk, torch.Tensor) else K_bk
                        x_bk = torch.cat([
                            sample_bk["rgb"].unsqueeze(0),
                            sample_bk["depth"].unsqueeze(0),
                        ], dim=1).to(device)
                        with torch.no_grad():
                            out_bk = model(x_bk)
                        joints_bk = out_bk["joints"][0].float().cpu().numpy()
                        pelvis_bk = recover_pelvis_from_pred(
                            out_bk["pelvis_depth"][0, 0].float().cpu().item(),
                            out_bk["pelvis_uv"][0].float().cpu().numpy(),
                            K_bk,
                        )
                        del x_bk, out_bk
                        frame_bodies.append((joints_bk, pelvis_bk, _PERSON_COLORS[bk % len(_PERSON_COLORS)]))
                    multi_pred_per_frame.append(frame_bodies)
                else:
                    pelvis_pred = recover_pelvis_from_pred(pd_val, pu_val, K_np)
                    multi_pred_per_frame.append([
                        (pred_joints_np, pelvis_pred, _PERSON_COLORS[0])
                    ])

                i += 1

            if rgb_frames:
                vid_gt = build_val_video(rgb_frames, pred_frames, K_frames, gt_pelvis_frames)

                # pred_pelvis: draw all bodies on original image
                pred_frames_out = []
                for orig_rgb, frame_bodies, orig_K in zip(
                    orig_rgb_frames, multi_pred_per_frame, orig_K_frames
                ):
                    img = orig_rgb
                    for joints, pelvis, color in frame_bodies:
                        img = draw_pose_frame(img, joints, orig_K, pelvis_abs=pelvis, color=color)
                    pred_frames_out.append(img)
                vid_pred = np.stack(pred_frames_out)[np.newaxis]  # (1, T, 3, H, W)

                result.append((vid_gt, vid_pred))
    finally:
        dataset.transform = orig_transform

    return result


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict:
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_loss_pose = 0.0
    total_loss_depth = 0.0
    total_loss_uv = 0.0
    total_mpjpe_body = 0.0
    n_batches = 0

    total = args.max_batches if args.max_batches > 0 else len(loader)
    pbar = tqdm(loader, total=total, desc=f"Epoch {epoch} [train]", leave=True, dynamic_ncols=True)

    for i, batch in enumerate(pbar):
        rgb          = batch["rgb"].to(device, non_blocking=True)           # (B, 3, H, W)
        depth        = batch["depth"].to(device, non_blocking=True)         # (B, 1, H, W)
        joints       = batch["joints"].to(device, non_blocking=True)        # (B, NUM_JOINTS, 3)
        gt_depth     = batch["pelvis_depth"].to(device, non_blocking=True)  # (B, 1)
        gt_uv        = batch["pelvis_uv"].to(device, non_blocking=True)     # (B, 2)

        x = torch.cat([rgb, depth], dim=1)                                  # (B, 4, H, W)

        with torch.amp.autocast("cuda", enabled=args.amp):
            out = model(x)
            pred_joints = out["joints"]           # (B, NUM_JOINTS, 3)
            pred_depth  = out["pelvis_depth"]     # (B, 1)
            pred_uv     = out["pelvis_uv"]        # (B, 2)

            l_pose  = pose_loss(pred_joints, joints)
            l_depth = nn.functional.smooth_l1_loss(pred_depth, gt_depth, beta=0.05)
            l_uv    = nn.functional.smooth_l1_loss(pred_uv, gt_uv, beta=0.05)
            loss = (l_pose + args.lambda_depth * l_depth + args.lambda_uv * l_uv) / args.accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % args.accum_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            total_loss       += loss.item() * args.accum_steps
            total_loss_pose  += l_pose.item()
            total_loss_depth += l_depth.item()
            total_loss_uv    += l_uv.item()
            total_mpjpe_body += mpjpe(pred_joints.float(), joints, BODY_IDX).item()
        del x, out, pred_joints, pred_depth, pred_uv, l_pose, l_depth, l_uv, loss
        n_batches += 1

        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}",
                         mpjpe_body=f"{total_mpjpe_body/n_batches:.1f}mm")

        if args.max_batches > 0 and n_batches >= args.max_batches:
            break

    pbar.close()
    n = max(1, n_batches)
    return {
        "train_loss":       total_loss       / n,
        "train_loss_pose":  total_loss_pose  / n,
        "train_loss_depth": total_loss_depth / n,
        "train_loss_uv":    total_loss_uv    / n,
        "train_mpjpe_body": total_mpjpe_body / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model.eval()

    total_loss  = 0.0
    total_loss_pose  = 0.0
    total_loss_depth = 0.0
    total_loss_uv    = 0.0
    sum_all     = 0.0
    sum_body    = 0.0
    sum_hand    = 0.0
    n_batches   = 0

    total = args.max_batches if args.max_batches > 0 else len(loader)
    pbar = tqdm(loader, total=total, desc="         [val] ", leave=True, dynamic_ncols=True)

    for batch in pbar:
        rgb      = batch["rgb"].to(device, non_blocking=True)
        depth    = batch["depth"].to(device, non_blocking=True)
        joints   = batch["joints"].to(device, non_blocking=True)
        gt_depth = batch["pelvis_depth"].to(device, non_blocking=True)
        gt_uv    = batch["pelvis_uv"].to(device, non_blocking=True)

        x = torch.cat([rgb, depth], dim=1)

        with torch.amp.autocast("cuda", enabled=args.amp):
            out = model(x)
            pred_joints = out["joints"]
            pred_depth  = out["pelvis_depth"]
            pred_uv     = out["pelvis_uv"]

        pred_joints_f = pred_joints.float()
        l_pose  = pose_loss(pred_joints_f, joints).item()
        l_depth = nn.functional.smooth_l1_loss(pred_depth.float(), gt_depth, beta=0.05).item()
        l_uv    = nn.functional.smooth_l1_loss(pred_uv.float(), gt_uv, beta=0.05).item()

        total_loss       += l_pose + args.lambda_depth * l_depth + args.lambda_uv * l_uv
        total_loss_pose  += l_pose
        total_loss_depth += l_depth
        total_loss_uv    += l_uv
        sum_all    += mpjpe(pred_joints_f, joints).item()
        sum_body   += mpjpe(pred_joints_f, joints, BODY_IDX).item()
        sum_hand   += mpjpe(pred_joints_f, joints, HAND_IDX).item()
        del x, out, pred_joints, pred_joints_f, pred_depth, pred_uv
        n_batches  += 1

        pbar.set_postfix(mpjpe_body=f"{sum_body/n_batches:.1f}mm",
                         mpjpe_all=f"{sum_all/n_batches:.1f}mm")

        if args.max_batches > 0 and n_batches >= args.max_batches:
            break

    pbar.close()

    n = max(1, n_batches)
    return {
        "val_loss":       total_loss / n,
        "val_loss_pose":  total_loss_pose / n,
        "val_loss_depth": total_loss_depth / n,
        "val_loss_uv":    total_loss_uv / n,
        "val_mpjpe_all":  sum_all    / n,
        "val_mpjpe_body": sum_body   / n,
        "val_mpjpe_hand": sum_hand   / n,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, scaler, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Handle old checkpoints that used head.mlp instead of head.trunk + branches
    ckpt_sd = ckpt["model"]
    try:
        model.load_state_dict(ckpt_sd)
    except RuntimeError:
        # Old checkpoint — load what matches, let new head layers init randomly
        model.load_state_dict(ckpt_sd, strict=False)
        print("  Warning: loaded checkpoint with strict=False (head structure changed)")
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_mpjpe  = ckpt.get("best_mpjpe", float("inf"))
    print(f"  Resumed from epoch {ckpt['epoch']}  (best MPJPE body={best_mpjpe:.1f}mm)")
    return start_epoch, best_mpjpe


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nBuilding data splits ...")
    train_seqs, val_seqs, _ = get_splits(
        overview_path=str(Path(args.data_root) / "data" / "overview.txt"),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        single_body_only=args.single_body_only,
        skip_missing_body=True,
        depth_required=True,
        mp4_required=False,
        frames_root=str(Path(args.data_root) / "data" / "frames"),
    )
    print(f"  Sequences — train: {len(train_seqs)}, val: {len(val_seqs)}")

    train_tf = build_train_transform(args.img_h, args.img_w)
    val_tf   = build_val_transform(args.img_h, args.img_w)

    train_loader = build_dataloader(
        seq_paths=train_seqs, data_root=args.data_root,
        transform=train_tf, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        seq_paths=val_seqs, data_root=args.data_root,
        transform=val_tf, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    print(f"  Frames  — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nBuilding {args.arch} ...")
    model = SapiensPose3D(
        arch=args.arch,
        img_size=(args.img_h, args.img_w),
        num_joints=NUM_JOINTS,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        drop_path_rate=args.drop_path,
    ).to(device)

    if args.pretrain and not args.resume:
        model.load_pretrained(args.pretrain)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_p:.1f}M")

    # ── Optimizer — two param groups (backbone slower than head) ──────────
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(),     "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── Optionally resume ─────────────────────────────────────────────────
    start_epoch = 0
    best_mpjpe  = float("inf")
    if args.resume:
        start_epoch, best_mpjpe = load_checkpoint(
            model, optimizer, scaler, args.resume, device
        )

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ── Logger ────────────────────────────────────────────────────────────
    logger = Logger(str(out_dir / "metrics.csv"))

    # ── Fixed visualization indices (3 per split + 1 fixed random) ─────────
    # All 4 indices are fixed for the entire run so the main-process dataset's
    # _label_cache and _depth_mmap caches stay bounded (they are never evicted
    # from the main process object; picking a new random sequence each epoch
    # would cause both caches to grow by 2 entries per epoch indefinitely).
    vis_val_fixed   = select_vis_indices(val_loader.dataset)   \
                      + [sample_random_vis_index(val_loader.dataset)]
    vis_train_fixed = select_vis_indices(train_loader.dataset) \
                      + [sample_random_vis_index(train_loader.dataset)]

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs ...\n")

    # Store base LRs before the schedule touches them
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        # LR schedule
        scale = get_lr_scale(epoch, args.epochs, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"lr_backbone={lr_bb:.2e}  lr_head={lr_hd:.2e}")

        t_epoch = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch + 1, args
        )

        # TensorBoard: train scalars every epoch
        writer.add_scalar("train/loss",       train_metrics["train_loss"],       epoch + 1)
        writer.add_scalar("train/loss_pose",  train_metrics["train_loss_pose"],  epoch + 1)
        writer.add_scalar("train/loss_depth", train_metrics["train_loss_depth"], epoch + 1)
        writer.add_scalar("train/loss_uv",    train_metrics["train_loss_uv"],    epoch + 1)
        writer.add_scalar("train/mpjpe_body", train_metrics["train_mpjpe_body"], epoch + 1)

        # Conditional validation
        val_metrics = None
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            val_metrics = validate(model, val_loader, device, args)
            writer.add_scalar("val/loss",        val_metrics["val_loss"],        epoch + 1)
            writer.add_scalar("val/loss_pose",   val_metrics["val_loss_pose"],   epoch + 1)
            writer.add_scalar("val/loss_depth",  val_metrics["val_loss_depth"],  epoch + 1)
            writer.add_scalar("val/loss_uv",     val_metrics["val_loss_uv"],     epoch + 1)
            writer.add_scalar("val/mpjpe_all",   val_metrics["val_mpjpe_all"],   epoch + 1)
            writer.add_scalar("val/mpjpe_body",  val_metrics["val_mpjpe_body"],  epoch + 1)
            writer.add_scalar("val/mpjpe_hand",  val_metrics["val_mpjpe_hand"],  epoch + 1)

            # Visualization: fixed 4 scenes per split (3 structured + 1 fixed random)
            # All indices are pre-selected at startup to keep main-process caches bounded.
            _scene_tags = ["scene_0", "scene_1", "scene_2", "scene_3_random"]
            for i, (vid_gt, vid_pred) in enumerate(
                visualize_fixed_samples(model, val_loader.dataset, vis_val_fixed, device, val_tf)
            ):
                tag = _scene_tags[i]
                writer.add_video(f"val/{tag}/gt_pelvis",   vid_gt,   global_step=epoch + 1, fps=4)
                writer.add_video(f"val/{tag}/pred_pelvis", vid_pred, global_step=epoch + 1, fps=4)
            for i, (vid_gt, vid_pred) in enumerate(
                visualize_fixed_samples(model, train_loader.dataset, vis_train_fixed, device, val_tf)
            ):
                tag = _scene_tags[i]
                writer.add_video(f"train/{tag}/gt_pelvis",   vid_gt,   global_step=epoch + 1, fps=4)
                writer.add_video(f"train/{tag}/pred_pelvis", vid_pred, global_step=epoch + 1, fps=4)
            torch.cuda.empty_cache()
            model.train()

        epoch_time = time.time() - t_epoch

        print(f"  → loss={train_metrics['train_loss']:.4f}"
              f"  mpjpe_body={train_metrics['train_mpjpe_body']:.1f}mm", end="")
        if val_metrics is not None:
            print(f"  val_mpjpe_body={val_metrics['val_mpjpe_body']:.1f}mm"
                  f"  val_mpjpe_all={val_metrics['val_mpjpe_all']:.1f}mm"
                  f"  val_mpjpe_hand={val_metrics['val_mpjpe_hand']:.1f}mm", end="")
        print(f"  ({epoch_time:.0f}s)\n")

        # Log to CSV
        row = {"epoch": epoch + 1, "lr_backbone": lr_bb, "lr_head": lr_hd,
               **train_metrics, **(val_metrics or {}), "epoch_time": epoch_time}
        logger.log(row)

        # Save best + early stopping (only when validation ran)
        if val_metrics is not None:
            val_body = val_metrics["val_mpjpe_body"]
            if val_body < best_mpjpe:
                best_mpjpe = val_body
                patience_counter = 0
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "best_mpjpe": best_mpjpe, "args": vars(args)},
                    str(out_dir / "best.pth"),
                )
                print(f"  *** New best MPJPE body = {best_mpjpe:.1f}mm ***\n")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"  Early stopping triggered at epoch {epoch + 1} "
                          f"(no improvement for {args.patience} val checks)")
                    break

        # Periodic save
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "best_mpjpe": best_mpjpe, "args": vars(args)},
                str(out_dir / f"epoch_{epoch+1:04d}.pth"),
            )

    writer.close()
    logger.close()
    print(f"Training complete. Best val MPJPE body = {best_mpjpe:.1f}mm")
    print(f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
