"""Hook to log epoch-averaged training MPJPE to TensorBoard.

Accumulates per-batch ``mpjpe`` and ``mpjpe_abs`` values from the head's
``loss()`` output, computes epoch averages, and writes them as
``mpjpe/rel/train`` and ``mpjpe/abs/train`` scalars.
"""

from __future__ import annotations

from typing import Optional

import torch
from mmengine.hooks import Hook

from mmpose.registry import HOOKS


@HOOKS.register_module()
class TrainMPJPEAveragingHook(Hook):
    """Epoch-averaged training MPJPE logger.

    Reads ``outputs['mpjpe']`` and ``outputs['mpjpe_abs']`` from each
    training iteration and logs their epoch averages to TensorBoard.
    """

    def __init__(self) -> None:
        self._mpjpe_buffer: list[float] = []
        self._mpjpe_abs_buffer: list[float] = []

    def before_train(self, runner) -> None:
        # Remove stale 'mpjpe' / 'mpjpe_abs' keys that older checkpoints
        # wrote directly into the MessageHub (as resumed HistoryBuffers).
        # Must run here (after load_or_resume), not in before_run.
        # Also clear _resumed_keys so they are not re-saved into future ckpts.
        hub = runner.message_hub
        for key in ('train/mpjpe', 'train/mpjpe_abs'):
            hub.log_scalars.pop(key, None)
            hub._resumed_keys.pop(key, None)

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict],
        outputs: dict,
    ) -> None:
        # Read per-batch MPJPE stored as attributes on the head
        # (not in the losses dict to avoid MMEngine auto-logging them).
        # Unwrap DDP wrapper if present.
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        head = getattr(model, 'head', None)
        if head is None:
            return
        val = getattr(head, '_train_mpjpe', None)
        if val is not None:
            self._mpjpe_buffer.append(
                val.item() if isinstance(val, torch.Tensor) else float(val))
        val_abs = getattr(head, '_train_mpjpe_abs', None)
        if val_abs is not None:
            self._mpjpe_abs_buffer.append(
                val_abs.item() if isinstance(val_abs, torch.Tensor)
                else float(val_abs))

    def after_train_epoch(self, runner) -> None:
        if not self._mpjpe_buffer:
            return

        avg_rel = sum(self._mpjpe_buffer) / len(self._mpjpe_buffer)
        avg_abs = (sum(self._mpjpe_abs_buffer) / len(self._mpjpe_abs_buffer)
                   if self._mpjpe_abs_buffer else 0.0)

        # Write via visualizer so all backends (TensorBoard, JSON, terminal)
        # receive the values.
        runner.visualizer.add_scalar('mpjpe/rel/train', avg_rel, runner.epoch)
        runner.visualizer.add_scalar('mpjpe/abs/train', avg_abs, runner.epoch)

        # Reset
        self._mpjpe_buffer.clear()
        self._mpjpe_abs_buffer.clear()

