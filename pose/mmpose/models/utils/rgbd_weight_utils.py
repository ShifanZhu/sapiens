# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for loading Sapiens RGB pretrained weights into a 4-channel RGBD model.

The pretrain checkpoint was trained at 1024×1024 with a CLS token and 3-channel
patch embedding.  Our model has:
  - no CLS token
  - 4-channel patch embedding (RGB + depth)
  - different spatial resolution (e.g. 640×384)

Three conversions are applied:
1. Key remapping: flat keys → ``backbone.vit.*`` prefixed keys
2. Patch-embed channel expansion: ``(C, 3, 16, 16)`` → ``(C, 4, 16, 16)``
   Depth channel initialised as the mean of the 3 RGB channels.
3. Positional-embedding interpolation: ``(1, 4097, D)`` → ``(1, H'×W', D)``
   Strip the CLS position, reshape to 2D grid, bicubic-resize, flatten.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interp_pos_embed(
    pos_embed: torch.Tensor,
    tgt_h: int,
    tgt_w: int,
    has_cls: bool = True,
) -> torch.Tensor:
    """Bicubic-interpolate a 2D positional embedding to a new grid size.

    Args:
        pos_embed: ``(1, N, D)`` — source positional embedding.
                   N = src_h * src_w  (+1 if has_cls).
        tgt_h, tgt_w: Target grid dimensions.
        has_cls: Whether the first token is a CLS token to be stripped.

    Returns:
        ``(1, tgt_h * tgt_w, D)``
    """
    if has_cls:
        pos_embed = pos_embed[:, 1:, :]   # strip CLS token

    _, N, D = pos_embed.shape
    src_h = src_w = int(N ** 0.5)
    assert src_h * src_w == N, f"Source pos_embed is not square: N={N}"

    # (1, N, D) → (1, D, src_h, src_w)
    grid = pos_embed.reshape(1, src_h, src_w, D).permute(0, 3, 1, 2).float()

    # Bicubic resize
    grid = F.interpolate(grid, size=(tgt_h, tgt_w), mode='bicubic',
                         align_corners=False)

    # (1, D, tgt_h, tgt_w) → (1, tgt_h*tgt_w, D)
    return grid.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, D)


def load_sapiens_pretrained_rgbd(
    model: nn.Module,
    ckpt_path: str,
    verbose: bool = True,
) -> None:
    """Load a Sapiens RGB pretrain checkpoint into an RGBD model.

    The function expects the model to have a ``backbone.vit`` sub-module
    (a ``mmpretrain.VisionTransformer`` with ``in_channels=4``).
    Backbone weights are loaded with patch-embed channel expansion and
    pos-embed interpolation.  The regression head is left randomly initialised.

    Args:
        model:     An mmpose model with ``backbone.vit`` (SapiensBackboneRGBD).
        ckpt_path: Path to the Sapiens pretrain checkpoint
                   (e.g. ``sapiens_0.3b_epoch_1600_clean.pth``).
        verbose:   Print a loading summary.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Unwrap if stored under 'state_dict' / 'model'
    if isinstance(raw, dict) and 'state_dict' in raw:
        src = raw['state_dict']
    elif isinstance(raw, dict) and 'model' in raw:
        src = raw['model']
    else:
        src = raw   # flat dict of tensors (the pretrain _clean.pth format)

    # ── 1. Remap keys: add 'backbone.vit.' prefix ─────────────────────────
    remapped: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        remapped[f'backbone.vit.{k}'] = v

    model_sd = model.state_dict()

    # ── 2. Patch-embed: expand 3 → 4 input channels ───────────────────────
    pe_key = 'backbone.vit.patch_embed.projection.weight'
    if pe_key in remapped:
        w_rgb = remapped[pe_key]                          # (C, 3, 16, 16)
        w_depth = w_rgb.mean(dim=1, keepdim=True)         # (C, 1, 16, 16)
        remapped[pe_key] = torch.cat([w_rgb, w_depth], dim=1)  # (C, 4, 16, 16)
        if verbose:
            print('[rgbd_weights] patch_embed: 3ch → 4ch  '
                  '(depth = mean of RGB channels)')

    # ── 3. Positional-embedding interpolation ─────────────────────────────
    pe_key = 'backbone.vit.pos_embed'
    if pe_key in remapped and pe_key in model_sd:
        src_pe = remapped[pe_key]     # (1, 4097, D)  — has CLS token
        tgt_pe = model_sd[pe_key]     # (1, H'*W', D) — no CLS token
        tgt_N = tgt_pe.shape[1]

        # Infer target grid from model's patch_resolution attribute
        try:
            pr = model.backbone.vit.patch_resolution   # (H', W')
            tgt_h, tgt_w = pr
        except AttributeError:
            # Fallback: assume square
            tgt_h = tgt_w = int(tgt_N ** 0.5)

        src_N_no_cls = src_pe.shape[1] - 1   # 4096 = 64×64
        src_g = int(src_N_no_cls ** 0.5)

        interp = _interp_pos_embed(src_pe, tgt_h, tgt_w, has_cls=True)
        remapped[pe_key] = interp.to(src_pe.dtype)
        if verbose:
            print(f'[rgbd_weights] pos_embed: {src_g}×{src_g} → {tgt_h}×{tgt_w}'
                  f'  (bicubic interpolation)')

    # ── 4. Skip cls_token (our model has no CLS token) ────────────────────
    remapped.pop('backbone.vit.cls_token', None)

    # ── 5. Load into model (backbone only) ────────────────────────────────
    load_sd: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for k, v_model in model_sd.items():
        if not k.startswith('backbone.'):
            continue   # head is randomly initialised — skip
        if k not in remapped:
            missing.append(k)
            continue
        if v_model.shape != remapped[k].shape:
            shape_mismatch.append(
                f'  {k}: model {v_model.shape} vs ckpt {remapped[k].shape}'
            )
            continue
        load_sd[k] = remapped[k]

    model.load_state_dict(load_sd, strict=False)

    if verbose:
        n_backbone = sum(1 for k in model_sd if k.startswith('backbone.'))
        print(f'[rgbd_weights] Loaded  {len(load_sd)} / {n_backbone} '
              f'backbone tensors')
        head_params = sum(1 for k in model_sd if not k.startswith('backbone.'))
        print(f'[rgbd_weights] Head ({head_params} tensors) randomly initialised')
        if missing:
            print(f'[rgbd_weights] Missing  ({len(missing)}): '
                  f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
        if shape_mismatch:
            print(f'[rgbd_weights] Shape mismatch ({len(shape_mismatch)}):')
            for s in shape_mismatch:
                print(s)
