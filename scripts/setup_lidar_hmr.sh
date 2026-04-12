#!/usr/bin/env bash
# Clone LiDAR-HMR next to Sapiens for depth-token extraction and training.
# Paper/repo: https://github.com/soullessrobot/LiDAR-HMR
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${LIDAR_HMR_DEST:-$ROOT/third_party/LiDAR-HMR}"
if [[ -d "$DEST/.git" ]]; then
  echo "Already cloned: $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 https://github.com/soullessrobot/LiDAR-HMR.git "$DEST"
fi
echo "LiDAR-HMR cloned to: $DEST"
echo "Use the project conda env: conda activate sapiens  (see CLAUDE.md / _install/conda.sh)"
echo "Then: pip install -e pretrain/  if needed, plus LiDAR-HMR deps (Point Transformer V2 / pointops, SMPL under models/graphormer/data/, etc.)."
echo "CUDA + Blackwell (sm_120): PyTorch cu128 + nvcc notes in docs/update_log/2026-04-09.md"
echo "Smoke (ViT+depth, can skip LiDAR forward): python scripts/smoke_lidar_hmr_depth_vit.py --skip-lidar-hmr"
echo "Pool vert_feat offline: scripts/convert_lidar_hmr_vert_feat.py or mmpretrain pool_vert_feat_to_depth_tokens."
