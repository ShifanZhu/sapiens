# BEDLAM2 3D Pose — Start Here

Custom RGBD 3D pose task integrated into the main `pose/` module. Predicts 70
active SMPL-X joints + pelvis depth/UV from 4-channel (RGB+D) input.

---

## Quick Links

| I want to… | Read |
|---|---|
| Train the model | [training.md](training.md) |
| Understand the architecture & files | [integration.md](integration.md) |
| See training results / compare heads | [training_results.md](training_results.md) |
| Understand model architecture + inference | [../design/pipeline.md](../design/pipeline.md) |
| Understand data format + transforms | [../design/data_transforms.md](../design/data_transforms.md) |
| Understand training config (loss, metrics, optimizer) | [../design/training_loop.md](../design/training_loop.md) |
| Understand depth preprocessing | [../design/dataload.md](../design/dataload.md) |
| Understand visualization | [../design/visualization.md](../design/visualization.md) |

---

## Recommended Reading Order

1. **[training.md](training.md)** — Set up data, train, evaluate. Practical guide.
2. **[integration.md](integration.md)** — Architecture overview, file locations, coordinate convention.
3. **[../design/pipeline.md](../design/pipeline.md)** — Deep dive into data transforms, model internals, loss, metrics.
4. **[training_results.md](training_results.md)** — Benchmark results for regression vs transformer head.

---

## Key Facts

- **Input:** 640×384 px, 4-channel (RGB + normalized depth)
- **Output:** 70 root-relative joint positions (m), pelvis depth (m), pelvis UV (normalized)
- **Coordinate system:** X=forward (depth), Y=left, Z=up (non-standard — see [integration.md](integration.md))
- **Two heads available:**
  - `Pose3dRegressionHead` — GAP + MLP (faster, simpler)
  - `Pose3dTransformerHead` — cross-attention decoder (spatial-aware, generally better)
- **Metric:** MPJPE in mm; reported for body / hand / all joint subsets
- **Environment:** `sapiens` conda env
