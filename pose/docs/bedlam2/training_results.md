# BEDLAM2 Training Results

Val set: 831 samples. Metric: MPJPE (mm, lower is better).
`mpjpe/body` = 22 body joints; `mpjpe/hand` = 30 hand joints; `mpjpe/all` = all 70 joints; `mpjpe/abs` = absolute (not root-relative).

---

## Run A — Transformer Head (Pose3dTransformerHead), ep1–8

**Config:** `sapiens_0.3b-50e_bedlam2-640x384-transformer.py`
**Run dir:** `runs/bedlam2_transformer_0.3b/20260318_000832/`
**Best ckpt:** `best_bedlam_mpjpe_body_epoch_8.pth`
**Note:** Pre-TensorBoard-restructure metric names (`bedlam/mpjpe/*`). No absolute MPJPE logged. Run stopped at ep8 (still improving).

| Epoch | body | hand | all |
|-------|------|------|-----|
| 1 | 204.35 | 407.44 | 331.11 |
| 2 | 156.73 | 281.08 | 237.30 |
| 3 | 130.45 | 233.88 | 197.90 |
| 4 | 112.55 | 209.07 | 175.89 |
| 5 | 110.99 | 194.42 | 166.12 |
| 6 | 108.50 | 189.83 | 162.42 |
| 7 | 95.91 | 169.15 | 145.28 |
| **8** | **89.76** | **163.76** | **139.53** |

---

## Run B/C — Regression Head (Pose3dRegressionHead), ep1–17+

**Config:** `sapiens_0.3b-50e_bedlam2-640x384.py`
**Run dirs:** `runs/bedlam2_transformer_0.3b/20260318_152401/` + `20260320_005710/` (multiple interrupted sessions)
**Best ckpt:** `best_mpjpe_body_val_epoch_16.pth`
**Note:** Post-TensorBoard-restructure metric names. Ep5–12 val logs missing (sessions killed before validation). ep18 was in progress when stopped.

| Epoch | body | hand | all | abs |
|-------|------|------|-----|-----|
| 1 | 199.26 | 454.35 | 359.26 | — |
| 2 | 174.25 | 419.19 | 325.37 | — |
| 3 | 147.34 | 330.07 | 262.03 | — |
| 4 | 137.25 | 296.62 | 238.35 | — |
| 5–12 | — | — | — | — (interrupted before val) |
| 13 | 103.73 | 184.37 | 159.87 | 222.11 |
| 14 | 102.25 | 179.81 | 156.32 | 218.98 |
| 15 | 102.44 | 178.20 | 155.95 | 225.10 |
| **16** | **98.27** | **172.35** | **150.16** | **212.69** |
| 17 | 100.30 | 175.16 | 152.44 | 224.90 |
| 18 | stopped at step 6500/6508 — no val | | | |

---

## Run D — Transformer Head (Pose3dTransformerHead), ep1–4+ (active)

**Config:** `sapiens_0.3b-50e_bedlam2-640x384-transformer.py`
**Run dir:** `runs/bedlam2_transformer_0.3b/20260320_102409/`
**Status:** Active as of 2026-03-20 ~15:35, mid-ep4 (step ~1950/6508).

| Epoch | body | hand | all | abs |
|-------|------|------|-----|-----|
| 1 | 195.76 | 421.56 | 335.50 | 531.88 |
| 2 | 159.63 | 307.59 | 253.76 | 422.14 |
| 3 | 141.17 | 246.61 | 209.74 | 301.92 |
| 4 | in progress | | | |

---

## Summary

| Model | Head | Best body MPJPE | Epoch | Checkpoint |
|-------|------|----------------|-------|-----------|
| Run A | Transformer | **89.76 mm** | 8 / 50 | `best_bedlam_mpjpe_body_epoch_8.pth` |
| Run B/C | Regression | 98.27 mm | 16 / 50 | `best_mpjpe_body_val_epoch_16.pth` |
| Run D | Transformer | 141.17 mm | 3 / 50 | (ongoing) |

- Run A (transformer) best so far at ep8=89.76mm, but was killed early — still converging.
- Run B/C (regression) reached 98.27mm at ep16 but also not finished.
- Run D (transformer, fresh) actively training; too early to compare.
- Neither run has reached 50 epochs.
