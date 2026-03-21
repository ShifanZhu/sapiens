# Issue 3: A/B training run + evaluation

**Type:** HITL (requires human to run training and interpret results)
**Blocked by:** Issue 2 (training config + smoke test)

## Parent PRD

`pose/docs/prd/transformer_decoder_head.md`

## What to build

Run a full 50-epoch training of the transformer decoder head on sapiens_0.3b and compare against the GAP+MLP baseline. Train from scratch (pretrained Sapiens backbone, randomly initialized head).

Baseline results (GAP+MLP head):

| Model | Body MPJPE | Hand MPJPE | All MPJPE |
|-------|-----------|-----------|----------|
| sapiens_0.3b | 80.6 mm | 130.5 mm | 117.6 mm |

Success criterion: body MPJPE ≤ 75mm on 0.3b (≥5mm / ~7% relative improvement).

## Acceptance criteria

- [ ] Full 50-epoch training completed on sapiens_0.3b with transformer decoder head
- [ ] Validation MPJPE (body/hand/all) recorded at best checkpoint
- [ ] Results compared against baseline in a table
- [ ] `pose/docs/design/pipeline.md` Head section updated: status changed from "NOT YET IMPLEMENTED", results documented
- [ ] Decision documented: keep transformer head, iterate, or revert based on results

## User stories addressed

- User story 9: fair A/B comparison with identical recipe
- User story 10: clear success criterion for go/no-go decision
