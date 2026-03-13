"""Profile training pipeline timing: data load, preprocess, forward, backward.

Runs N warm-up batches then M measurement batches, saving per-batch timings
and a summary to <output-dir>/timing_results.json.

Usage:
    conda run -n sapiens_gpu python profile_timing.py \
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
        --pretrain  checkpoints/sapiens_0.3b_epoch_1600_clean.pth \
        --output-dir runs/profile \
        --num-batches 50 \
        --warmup 5
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_splits, build_train_transform, build_dataloader
from model import SapiensPose3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="/home/hang/repos_local/MMC/BEDLAM2Datatest")
    p.add_argument("--pretrain",    default="checkpoints/sapiens_0.3b_epoch_1600_clean.pth")
    p.add_argument("--output-dir",  default="runs/profile")
    p.add_argument("--arch",        default="sapiens_0.3b",
                   choices=["sapiens_0.3b", "sapiens_0.6b", "sapiens_1b", "sapiens_2b"])
    p.add_argument("--img-h",       type=int, default=384)
    p.add_argument("--img-w",       type=int, default=640)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=50,  help="Batches to measure")
    p.add_argument("--warmup",      type=int, default=5,   help="Warm-up batches (not measured)")
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      dest="amp", action="store_false")
    p.add_argument("--seed",        type=int, default=2026)
    return p.parse_args()


def sync() -> float:
    """Synchronize CUDA and return current time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"AMP    : {args.amp}")
    print(f"Batches: {args.warmup} warm-up + {args.num_batches} measured")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_seqs, _, _ = get_splits(
        overview_path=str(Path(args.data_root) / "data" / "overview.txt"),
        val_ratio=0.1, test_ratio=0.1, seed=args.seed,
        single_body_only=True, skip_missing_body=True,
        depth_required=True, mp4_required=False,
    )
    train_tf = build_train_transform(args.img_h, args.img_w)
    loader = build_dataloader(
        seq_paths=train_seqs, data_root=args.data_root,
        transform=train_tf, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    print(f"Dataset: {len(loader.dataset)} frames, {len(loader)} batches/epoch\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SapiensPose3D(
        arch=args.arch, img_size=(args.img_h, args.img_w),
        num_joints=127, head_hidden=2048,
    ).to(device)
    if args.pretrain:
        model.load_pretrained(args.pretrain)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── Profiling loop ────────────────────────────────────────────────────────
    records: list[dict] = []          # one dict per measured batch
    phase_names = ["data_load", "host_to_device", "forward", "backward"]

    total_needed = args.warmup + args.num_batches
    loader_iter  = iter(loader)
    batch_count  = 0

    print(f"{'Batch':>6}  {'DataLoad':>10}  {'H2D':>10}  {'Forward':>10}  {'Backward':>10}  {'Total':>10}  (ms)")
    print("-" * 72)

    # Mark start of data-load window *before* the first next() call
    t_data_start = sync()

    while batch_count < total_needed:
        # ── 1. Data load (time waiting for prefetch workers) ─────────────────
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            t_data_start = sync()
            batch = next(loader_iter)

        t_after_data = sync()
        data_ms = (t_after_data - t_data_start) * 1e3

        # ── 2. Host → device transfer + cat ──────────────────────────────────
        rgb    = batch["rgb"].to(device, non_blocking=False)
        depth  = batch["depth"].to(device, non_blocking=False)
        joints = batch["joints"].to(device, non_blocking=False)
        x = torch.cat([rgb, depth], dim=1)

        t_after_h2d = sync()
        h2d_ms = (t_after_h2d - t_after_data) * 1e3

        # ── 3. Forward ───────────────────────────────────────────────────────
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            pred = model(x)
            loss = nn.functional.smooth_l1_loss(pred, joints, beta=0.05)

        t_after_fwd = sync()
        fwd_ms = (t_after_fwd - t_after_h2d) * 1e3

        # ── 4. Backward ──────────────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        t_after_bwd = sync()
        bwd_ms = (t_after_bwd - t_after_fwd) * 1e3

        total_ms = data_ms + h2d_ms + fwd_ms + bwd_ms

        # Mark start of next data-load window immediately after backward
        t_data_start = sync()

        batch_count += 1
        is_warmup = batch_count <= args.warmup

        if not is_warmup:
            records.append({
                "batch":       batch_count - args.warmup,
                "data_load":   data_ms,
                "host_to_device": h2d_ms,
                "forward":     fwd_ms,
                "backward":    bwd_ms,
                "total":       total_ms,
            })

        tag = "warm" if is_warmup else "    "
        print(f"{batch_count:>6}{tag}  "
              f"{data_ms:>10.1f}  {h2d_ms:>10.1f}  {fwd_ms:>10.1f}  {bwd_ms:>10.1f}  {total_ms:>10.1f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Summary over {len(records)} batches (batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}, amp={args.amp})\n")

    summary = {}
    for phase in phase_names + ["total"]:
        vals = [r[phase] for r in records]
        arr  = np.array(vals)
        summary[phase] = {
            "mean_ms":   float(arr.mean()),
            "std_ms":    float(arr.std()),
            "min_ms":    float(arr.min()),
            "max_ms":    float(arr.max()),
            "pct_total": 0.0,   # filled below
        }

    total_mean = summary["total"]["mean_ms"]
    for phase in phase_names:
        summary[phase]["pct_total"] = summary[phase]["mean_ms"] / total_mean * 100

    print(f"{'Phase':<18}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'% total':>9}")
    print("-" * 72)
    for phase in phase_names + ["total"]:
        s = summary[phase]
        pct = f"{s['pct_total']:.1f}%" if phase != "total" else ""
        print(f"{phase:<18}  {s['mean_ms']:>8.1f}ms  {s['std_ms']:>8.1f}ms  "
              f"{s['min_ms']:>8.1f}ms  {s['max_ms']:>8.1f}ms  {pct:>9}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "config": {
            "arch":        args.arch,
            "img_h":       args.img_h,
            "img_w":       args.img_w,
            "batch_size":  args.batch_size,
            "num_workers": args.num_workers,
            "amp":         args.amp,
            "warmup":      args.warmup,
            "num_batches": args.num_batches,
            "device":      str(device),
        },
        "summary":    summary,
        "per_batch":  records,
    }
    out_path = out_dir / "timing_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
