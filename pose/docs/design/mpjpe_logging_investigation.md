# Investigation: Observations on Invariant Training MPJPE

## Observations
In recent training runs (notably `20260319_121652`), the training metrics `mpjpe` and `mpjpe_abs` have been observed to remain constant across many iterations. This occurs even while the primary loss components are fluctuating normally, suggesting that these specific metrics might not be receiving updates from the current training loop.

## Potential Factors Under Investigation

### 1. Checkpoint Resumption and MessageHub State
The run in question was resumed from a checkpoint (`epoch_5.pth`). One possibility is that the `MessageHub` (MMEngine's internal state tracker) loaded stale values for the keys `mpjpe` and `mpjpe_abs` from that checkpoint. If the active training code does not explicitly overwrite or update these specific keys in the `MessageHub`, the logger may continue to report the last known values, leading to the observed "flat" lines in the logs.

### 2. Model Wrapping and Attribute Access
The `TrainMPJPEAveragingHook` attempts to read per-batch values from attributes stored on the model's head (e.g., `_train_mpjpe`). 
- **Hypothesis**: When using Distributed Data Parallel (DDP) or other wrappers, the structure of the model changes (e.g., the actual model is moved to `model.module`).
- **Potential Issue**: If the hook is looking for attributes on the wrapper (`runner.model`) rather than the wrapped module (`runner.model.module`), it might be reading from an uninitialized or static version of the head that does not reflect the current batch's computations.

### 3. Logging Backend Divergence
Observations suggest that the current hook writes directly to TensorBoard via its own writer, rather than updating the central `MessageHub`.
- **Potential Result**: Metrics updated this way may only appear in TensorBoard and could be missing from other outputs like `scalars.json` or the terminal log. This might explain why the expected hierarchical tags (like `mpjpe/rel/train`) were not found during file analysis, even if the averaging logic was technically executing.

## Next Steps for Verification
- Investigate whether explicitly checking for model wrappers (like `.module`) in the hook ensures access to the correct head attributes.
- Consider if using `runner.message_hub.update_scalar` would provide more consistent logging across all backends (JSON, TensorBoard, and Terminal).
- Verify if the constant values match the final values of the run that produced the resumed checkpoint.
