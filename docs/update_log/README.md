# Update Log

Session-level changelog for the Sapiens project.

| Date | Summary |
|---|---|
| [2026-04-21](2026-04-21.md) | Panoptic Kinect1 offline preprocessor: `depthdata.dat` reader, toolbox-aligned unproject/project, HD sync, RGB-depth splat + overlays (`pose/tools/panoptic/`) |
| [2026-04-20](2026-04-20.md) | Panoptic Kinect1: optional GT COCO-19 skeleton overlay from `hdPose3d_stage1_coco19` (`--gt-pose`, `panoptic_body3d.py`) |
| [2026-03-23](2026-03-23.md) | Reduce max depth 20→10 m; filter far-person (all joints ≥ 10 m) and OOB (≥ 50% joints out of image) for train + val |
| [2026-03-21](2026-03-21.md) | Code + doc architecture review; fix broken links/stale data in docs; add bedlam2/README.md navigation hub; RFC 006 |
| [2026-03-20](2026-03-20.md) | Fix stale mpjpe key purge: before_run → before_train (runs after checkpoint load) |
| [2026-03-19](2026-03-19.md) | Fixed TrainMPJPEAveragingHook: DDP wrapper traversal + TensorBoard-only logging |
| [2026-03-18](2026-03-18.md) | Skeleton viz fix, TensorBoard logging restructure, docs reorganization, TensorBoard tag bug fixes (mpjpe/abs/val, raw mpjpe suppression) |
| [2026-03-17](2026-03-17.md) | Fixed BEDLAM2 training plateau bug, transformer decoder head (PRD + issues 1-3) |
| [2026-03-16](2026-03-16.md) | Fixed 7 BEDLAM2 training pipeline bugs, configurable data_root |
