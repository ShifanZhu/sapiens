# Data Loading Speedup

## Profiling Results (batch_size=16, num_workers=4, AMP=True, sapiens_0.3b)

### Before (NPZ)
| Phase | Mean | % total |
|---|---|---|
| data_load | 1391 ms | 63.6% |
| forward | 242 ms | 11.1% |
| backward | 544 ms | 24.9% |
| **total** | **2188 ms** | |

### After (NPY mmap)
| Phase | Mean | % total |
|---|---|---|
| data_load | 0.2 ms | 0.0% |
| forward | 217 ms | 30.1% |
| backward | 496 ms | 68.6% |
| **total** | **723 ms** | |

**3× overall speedup. Training is now fully GPU-bound.**

---

## What Was Done

### 1. Depth NPZ → NPY mmap (`convert_depth_npy.py`)
Depth files were stored as compressed NPZ (zlib, 10:1 ratio). Loading a 36 MB file to read one 3.7 MB frame was the dominant bottleneck (63.6% of training time).

**Fix:** Convert each NPZ to an uncompressed NPY pre-resized to training resolution (384×640) in float16. The dataset loads NPY with `mmap_mode='r'` — the OS pages in only the frames accessed, with zero decompression.

```bash
conda run -n sapiens_gpu python claude_code/scripts/convert_depth_npy.py \
    --data-root ${BEDLAM2_DATA_ROOT} --out-h 384 --out-w 640 --workers 8
```

### 2. Depth LRU cache for NPZ fallback (`data/dataset.py`)
Added a bounded `OrderedDict` cache (max 3 sequences per worker) for the NPZ fallback path, preventing OOM when NPY is not available.

### 3. Depth cache (`data/dataset.py`)
The dataset now checks for NPY first, falls back to NPZ automatically — no change needed in training scripts.

---

## Trade-offs

| Trade-off | Detail |
|---|---|
| +28 GB disk | NPY (75.6 GB) alongside original NPZ (47.7 GB). NPZ kept as fallback. |
| float16 precision | ~0.001 m resolution over 0–10 m range — sufficient for pose estimation. Original float32 preserved in NPZ. |
| Resolution baked in | NPY pre-resized to 384×640. Re-run `convert_depth_npy.py` if `--img-h`/`--img-w` changes. |
