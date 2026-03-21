# Issue 2: Training config + integration smoke test

**Type:** AFK
**Blocked by:** Issue 1 (transformer decoder head module)

## Parent PRD

`pose/docs/prd/transformer_decoder_head.md`

## What to build

Create a new config file for training the transformer decoder head on BEDLAM2. The config should be identical to the existing `sapiens_0.3b-50e_bedlam2-640x384.py` except for the `head` block and `custom_imports`. Verify the full pipeline (config load → model build → 1 training iteration) runs without error.

The training recipe must remain identical to the baseline:
- AdamW, 1e-4 head / 1e-5 backbone, weight decay 0.03
- 3-epoch linear warmup (start_factor=0.333), cosine decay to 0
- 50 epochs, batch size 16, AMP enabled
- Same data pipeline and augmentations

**Status: COMPLETE** (commit 1800b43)

## Acceptance criteria

- [x] New config file created (e.g., `configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384-transformer.py`)
- [x] Only the `head=dict(...)` block and `custom_imports` differ from the baseline config
- [x] `custom_imports` includes the new head module
- [x] Config loads successfully: `python -c "from mmengine.config import Config; Config.fromfile('<config_path>')"`
- [x] Model builds successfully from the config
- [x] Completes 1 training iteration without error (forward + backward + optimizer step)

## User stories addressed

- User story 7: switchable via config
- User story 9: identical training recipe for fair A/B comparison
