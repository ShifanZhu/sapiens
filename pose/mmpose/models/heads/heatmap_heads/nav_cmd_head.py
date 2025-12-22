# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, Features, OptConfigType, OptSampleList, Predictions
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class NavCmdHead(BaseHead):
    """A navigation command regression head.

    This is a modified version of a heatmap head. Instead of producing K heatmaps,
    it regresses a 3-DoF command vector: (vx, vy, v_yaw).

    Output:
        pred_cmd: Tensor of shape (B, 3) in forward()
        predict(): list[InstanceData], each has field `nav_cmd` (Tensor(3,))

    Expected GT in each data sample (one of these locations):
        - sample.gt_instance_labels.nav_cmd: Tensor shape (3,)
        - sample.gt_fields.nav_cmd: Tensor shape (3,)
        - sample.metainfo['nav_cmd']: array-like shape (3,)
    """

    _version = 3  # bump to3 since we are changed from old heatmap head

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int = 3,  # keep name for compatibility; must be 3 (vx,vy,vyaw)
                 deconv_out_channels: OptIntSeq = None,
                 deconv_kernel_sizes: OptIntSeq = None,
                 conv_out_channels: OptIntSeq = (256, ),
                 conv_kernel_sizes: OptIntSeq = (3, ),
                 mlp_hidden_dims: Sequence[int] = (256, 128),
                 final_act: Optional[str] = None,
                 use_silu: bool = True,
                 # If we prefer registry-built losses, set e.g.:
                 # loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
                 loss: Optional[ConfigType] = None,
                 # Optional per-dimension scaling and weighting
                 cmd_mean: Optional[Sequence[float]] = None,
                 cmd_std: Optional[Sequence[float]] = None,
                 cmd_weights: Optional[Sequence[float]] = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(init_cfg)

        if out_channels != 3:
            raise ValueError(f'NavCmdHead must output 3 dims (vx, vy, v_yaw), got out_channels={out_channels}')

        self.in_channels = in_channels
        self.out_channels = 3 # vx, vy, v_yaw
        self.use_silu = use_silu  # instance norm + silu instead of batchnorm + relu

        # ---- optional deconv stack (usually not needed for command regression) ----
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should have the same length. '
                    f'Got {deconv_out_channels} vs {deconv_kernel_sizes}'
                )
            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        # ---- optional conv stack ----
        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should have the same length. '
                    f'Got {conv_out_channels} vs {conv_kernel_sizes}'
                )
            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes,
            )
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        # ---- global pooling + MLP to (vx, vy, v_yaw) ----
        self.pool = nn.AdaptiveAvgPool2d(1)

        mlp_layers = []
        prev = in_channels
        for h in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            if self.use_silu:
                mlp_layers.append(nn.SiLU(inplace=True))
            else:
                mlp_layers.append(nn.ReLU(inplace=True))
            prev = h
        mlp_layers.append(nn.Linear(prev, out_channels))
        self.mlp = nn.Sequential(*mlp_layers)

        self.out_act = nn.Tanh()
        self.register_buffer(
            'cmd_scale',
            torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32) # clip vx to ±1.0 m/s, vy to ±0.5 m/s, yaw rate to ±0.5 rad/s
        )

        # ---- loss ----
        # Default: SmoothL1Loss(beta=1.0)
        if loss is None:
            self.loss_module = nn.SmoothL1Loss(beta=1.0, reduction='none')
            self.loss_weight = 1.0
        else:
            # Build from registry if we configured it that way
            self.loss_module = MODELS.build(loss)
            # common pattern: loss modules may include loss_weight inside; but if not, keep 1.0
            self.loss_weight = float(loss.get('loss_weight', 1.0)) if isinstance(loss, dict) else 1.0

        # ---- normalization / weighting (optional but recommended) ----
        # cmd_norm: (cmd - mean) / std
        self.register_buffer('cmd_mean', torch.tensor(cmd_mean, dtype=torch.float32) if cmd_mean is not None else None)
        self.register_buffer('cmd_std', torch.tensor(cmd_std, dtype=torch.float32) if cmd_std is not None else None)
        self.register_buffer('cmd_weights', torch.tensor(cmd_weights, dtype=torch.float32) if cmd_weights is not None else None)

        # Register hook to handle old checkpoints gracefully (best-effort)
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    # ------------------------- building blocks -------------------------

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))

            if self.use_silu:
                layers.append(nn.InstanceNorm2d(out_channels))
                layers.append(nn.SiLU(inplace=True))
            else:
                layers.append(nn.BatchNorm2d(num_features=out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            if kernel_size == 4:
                padding, output_padding = 1, 0
            elif kernel_size == 3:
                padding, output_padding = 1, 1
            elif kernel_size == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(
                    f'Unsupported kernel size {kernel_size} for deconv layers in {self.__class__.__name__}'
                )

            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,  # upsample ×2
                padding=padding,
                output_padding=output_padding,
                bias=False
            )
            layers.append(build_upsample_layer(cfg))

            if self.use_silu:
                layers.append(nn.InstanceNorm2d(out_channels))
                layers.append(nn.SiLU(inplace=True))
            else:
                layers.append(nn.BatchNorm2d(num_features=out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels
        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        # Note: MLP layers are nn.Linear; leave them with default init,
        # or add custom init if we want.
        return [
            dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Constant', layer='InstanceNorm2d', val=1, bias=0),
        ]

    # ------------------------- helpers -------------------------

    def _get_gt_cmd(self, batch_data_samples: OptSampleList, device: torch.device) -> Tensor:
        """Extract GT command tensor of shape (B, 3) from samples."""
        gts = []
        for d in batch_data_samples:
            cmd = None

            # 1) recommended: gt_instance_labels.nav_cmd
            if hasattr(d, 'gt_instance_labels') and d.gt_instance_labels is not None:
                if hasattr(d.gt_instance_labels, 'nav_cmd'):
                    cmd = d.gt_instance_labels.nav_cmd

            # 2) alternative: gt_fields.nav_cmd
            if cmd is None and hasattr(d, 'gt_fields') and d.gt_fields is not None:
                if hasattr(d.gt_fields, 'nav_cmd'):
                    cmd = d.gt_fields.nav_cmd

            # 3) fallback: metainfo['nav_cmd']
            if cmd is None and hasattr(d, 'metainfo') and d.metainfo is not None:
                if 'nav_cmd' in d.metainfo:
                    cmd = d.metainfo['nav_cmd']

            if cmd is None:
                raise KeyError(
                    'Cannot find GT nav command in data sample. Expected one of:\n'
                    '  - sample.gt_instance_labels.nav_cmd\n'
                    '  - sample.gt_fields.nav_cmd\n'
                    "  - sample.metainfo['nav_cmd']\n"
                )

            cmd_t = torch.as_tensor(cmd, dtype=torch.float32, device=device).view(3)
            gts.append(cmd_t)

        return torch.stack(gts, dim=0)  # (B,3)

    def _normalize_cmd(self, cmd: Tensor) -> Tensor:
        """Apply optional (cmd - mean) / std."""
        if self.cmd_mean is not None:
            cmd = cmd - self.cmd_mean.to(cmd.device)
        if self.cmd_std is not None:
            cmd = cmd / (self.cmd_std.to(cmd.device) + 1e-8)
        return cmd

    def _weighted_loss_reduce(self, per_elem_loss: Tensor) -> Tensor:
        """per_elem_loss: (B,3) -> scalar with optional dim weights."""
        if self.cmd_weights is not None:
            w = self.cmd_weights.to(per_elem_loss.device).view(1, 3)
            per_elem_loss = per_elem_loss * w
        return per_elem_loss.mean()

    # ------------------------- core API -------------------------

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward.

        Args:
            feats: Tuple of multi-scale features. We use feats[-1] as (B,C,H,W).

        Returns:
            Tensor: (B,3) = (vx, vy, v_yaw)
        """
        x = feats[-1] # (B,1024,48,64) Pick the most semantic / deepest feature map from the backbone.
        x = self.deconv_layers(x) # (B,1024,48,64) Optionally upsample the feature map to get spatially richer features (we do not need here).
        x = self.conv_layers(x) # (B,256,48,64) Optionally conv layers to refine features.
        x = self.pool(x)              # (B,256,1,1) Global average pooling to get a single feature vector per sample.
        x = torch.flatten(x, 1)       # (B,256)
        cmd = self.mlp(x)             # (B,256) -> (B,256) -> (B,128) -> (B,out_channels=3)  Unconstrained
        cmd = self.out_act(cmd)       # (B,out_channels=3) (tanh to [-1,1])
        cmd = cmd * self.cmd_scale # per-dimension scaling (vx to ±1.0 m/s, vy to ±0.5 m/s, yaw rate to ±0.5 rad/s)
        return cmd

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict navigation command.

        Returns:
            List[InstanceData]: length B; each InstanceData has field `nav_cmd` (Tensor(3,))
        """
        pred_cmd = self.forward(feats)  # (B,3)

        # Optionally unnormalize outputs for logging / deployment
        # (Only do this if we *trained* in normalized space and want real units here.)
        if test_cfg.get('unnormalize', False):
            if self.cmd_std is not None:
                pred_cmd = pred_cmd * self.cmd_std.to(pred_cmd.device)
            if self.cmd_mean is not None:
                pred_cmd = pred_cmd + self.cmd_mean.to(pred_cmd.device)

        preds = []
        for i in range(pred_cmd.shape[0]):
            inst = InstanceData()
            inst.nav_cmd = pred_cmd[i]
            preds.append(inst)
        return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Compute loss.

        Returns:
            dict of losses/metrics
        """
        pred_cmd = self.forward(feats)  # (B,3)
        gt_cmd = self._get_gt_cmd(batch_data_samples, device=pred_cmd.device)  # (B,3)

        # clip gt to same range
        gt_cmd = gt_cmd.clamp(
            torch.tensor([-1.0, -0.5, -0.5], device=gt_cmd.device),
            torch.tensor([ 1.0,  0.5,  0.5], device=gt_cmd.device),
        )

        # Train in normalized space if mean/std provided
        # pred_n = self._normalize_cmd(pred_cmd)
        # gt_n = self._normalize_cmd(gt_cmd)

        losses = {}

        # If using torch.nn SmoothL1Loss with reduction='none': returns (B,3)
        # If using a registry loss that returns scalar directly, we handle both.
        loss_raw = self.loss_module(pred_cmd, gt_cmd)
        if isinstance(loss_raw, Tensor) and loss_raw.ndim == 2 and loss_raw.shape[-1] == 3:
            loss_val = self._weighted_loss_reduce(loss_raw) * self.loss_weight
        else:
            # assume scalar tensor
            loss_val = loss_raw * self.loss_weight

        losses['loss_nav'] = loss_val

        # Metrics in real units (unnormalized)
        with torch.no_grad():
            err = (pred_cmd - gt_cmd)  # (B,3)
            mae = err.abs().mean(dim=0)     # (3,)  Mean Absolute Error (focuse more on average error)
            rmse = torch.sqrt((err ** 2).mean(dim=0) + 1e-12) # (3,) Root Mean Square Error (focus more on outliers)

            losses['mae_vx'] = mae[0]
            losses['mae_vy'] = mae[1]
            losses['mae_yaw'] = mae[2]
            losses['rmse_vx'] = rmse[0]
            losses['rmse_vy'] = rmse[1]
            losses['rmse_yaw'] = rmse[2]

            # Optional "within tolerance" accuracy
            if 'tol_vx' in train_cfg or 'tol_vy' in train_cfg or 'tol_yaw' in train_cfg:
                tol = torch.tensor([
                    float(train_cfg.get('tol_vx', 0.05)),
                    float(train_cfg.get('tol_vy', 0.05)),
                    float(train_cfg.get('tol_yaw', 0.10)),
                ], device=pred_cmd.device).view(1, 3)
                within = (err.abs() <= tol).float().mean(dim=0)
                losses['acc_vx_tol'] = within[0]
                losses['acc_vy_tol'] = within[1]
                losses['acc_yaw_tol'] = within[2]

        return losses

    # ------------------------- checkpoint compatibility -------------------------

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args, **kwargs):
        """Best-effort compatibility hook.

        Old heatmap head checkpoints won't match this head. We *do not* attempt
        to convert heatmap weights into regression weights; we only try to avoid
        hard crashes by ignoring incompatible keys if we load with strict=False.

        Recommendation:
            - If we changed task (heatmap -> nav cmd), load backbone/neck only,
              or use strict=False and allow this head to be randomly initialized.
        """
        # Nothing to convert reliably here; kept for future extension.
        return
