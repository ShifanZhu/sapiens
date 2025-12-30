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
    """A navigation command regression head for robot control.

    This head converts visual features (from a backbone like Sapiens ViT) into
    robot navigation commands. Instead of producing K heatmaps like pose estimation,
    it regresses a 3-DoF velocity command vector: (vx, vy, v_yaw).

    Key Features:
        1. FiLM Conditioning: Modulate features based on reference target
        2. Reference Caching: Efficiently cache embeddings per sequence/track

    Output:
        pred_cmd: Tensor of shape (B, 3) in forward()
                  - vx: forward/backward velocity (±1.0 m/s)
                  - vy: left/right strafe velocity (±0.5 m/s)
                  - v_yaw: rotational velocity (±0.5 rad/s)
        predict(): list[InstanceData], each has field `nav_cmd` (Tensor(3,))

    ===== PoseDataSample =====

    MMPose uses three data structures to store different types of information:

    1. gt_instance_labels (InstanceData) - Per-instance annotations
       Type: InstanceData object with dynamic attributes
       Keypoints, bboxes, person-specific data, navigation commands
       Example: sample.gt_instance_labels = InstanceData(
                    nav_cmd=np.array([0.5, 0.2, 0.1]),
                    ref_emb=np.array([...]),  # 512D person embedding
                    keypoints=np.array([[x,y], ...]),
                    bbox=[x, y, w, h]
                )

    2. gt_fields (PixelData) - Dense spatial predictions
       Type: PixelData object with tensor attributes
       Heatmaps, segmentation masks, depth maps, dense fields
       Example: sample.gt_fields = PixelData(
                    heatmaps=torch.randn(17, 192, 256),  # Keypoint heatmaps
                    nav_cmd=torch.tensor([0.5, 0.2, 0.1])  # Alternative location
                )

    3. metainfo (dict) - Metadata and auxiliary information
       Type: Python dictionary
       File paths, sequence IDs, frame indices, transforms, flags
       Example: sample.metainfo = {
                    'img_path': '/data/seq_001/frame_005.jpg',
                    'sequence_id': '00042',  # For caching
                    'frame_idx': 5,
                    'is_ref_frame': False,
                    'nav_cmd': [0.5, 0.2, 0.1],  # Fallback location
                    'ref_emb': [...]  # Precomputed embedding
                }

    Expected GT in each data sample (priority order):
        - sample.gt_instance_labels.nav_cmd: Tensor shape (3,) ← Recommended
        - sample.gt_fields.nav_cmd: Tensor shape (3,)
        - sample.metainfo['nav_cmd']: array-like shape (3,)

    Expected Reference Embedding (for FiLM conditioning):
        - sample.gt_instance_labels.ref_emb: Tensor shape (ref_in_channels,) ← Recommended
        - sample.gt_fields.ref_emb: Tensor shape (ref_in_channels,)
        - sample.metainfo['ref_emb']: array-like shape (ref_in_channels,)
    """

    _version = 3  # bump to3 since we are changed from old heatmap head

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],  # Backbone output channels (1024 for Sapiens-0.3B)
                 out_channels: int = 3,  # keep name for compatibility; must be 3 (vx,vy,vyaw)

                 # ===== Optional Deconvolution Layers =====
                 # Usually not needed for regression tasks
                 deconv_out_channels: OptIntSeq = None,
                 deconv_kernel_sizes: OptIntSeq = None,

                 # ===== Convolutional Refinement Layers =====
                 # Refine backbone features before pooling
                 conv_out_channels: OptIntSeq = (256, ),  # Default: one conv layer with 256 channels
                 conv_kernel_sizes: OptIntSeq = (3, ),    # Default: 3x3 kernels

                 # ===== MLP Regression Head =====
                 # After global pooling, MLP predicts velocity commands
                 mlp_hidden_dims: Sequence[int] = (256, 128),  # Hidden dimensions: in→256→128→3

                 # ===== FiLM Embedding =====
                 ref_in_channels: Optional[int] = None,  # Dimension of reference embedding; None disables FiLM
                 film_hidden_dim: int = 256,  # ref_emb → gamma, beta

                 # ===== Optional Cached Reference Embeddings =====
                 # Useful for video sequences with frozen backbone to avoid recomputation
                 cache_ref_emb: bool = False,  # Enable CPU-based caching of reference embeddings
                 ref_cache_keys: Sequence[str] = ('sequence_id', 'seq_id', 'track_id', 'video_id'),  # Metainfo keys for cache lookup
                 ref_flag_keys: Sequence[str] = ('is_ref', 'is_ref_frame', 'is_first_frame'),  # Keys marking reference frames
                 ref_frame_idx_key: Optional[str] = 'frame_idx',  # Frame index key (frame_idx==0 → reference)
                 ref_from_feats: bool = False,  # Extract ref_emb from pooled features

                 # ===== Activation & Normalization =====
                 final_act: Optional[str] = None,  # unused
                 use_silu: bool = True,  # True: InstanceNorm+SiLU, False: BatchNorm+ReLU

                 # ===== Loss Configuration =====
                 # If we prefer registry-built losses, set e.g.:
                 # loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
                 loss: Optional[ConfigType] = None,  # Default: SmoothL1Loss(beta=1.0)

                 # ===== Optional Per-Dimension Scaling and Weighting =====
                 # Normalize commands or weight dimensions differently in loss
                 cmd_mean: Optional[Sequence[float]] = None,  # Mean for (vx, vy, v_yaw) normalization
                 cmd_std: Optional[Sequence[float]] = None,   # Std for (vx, vy, v_yaw) normalization
                 cmd_weights: Optional[Sequence[float]] = None,  # Loss weights [w_vx, w_vy, w_yaw]

                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(init_cfg)

        if out_channels != 3:
            raise ValueError(f'NavCmdHead must output 3 dims (vx, vy, v_yaw), got out_channels={out_channels}')

        self.in_channels = in_channels # feature vector channels from backbone
        self.out_channels = 3 # vx, vy, v_yaw
        self.use_silu = use_silu

        # If ref_in_channels is None, FiLM conditioning is disabled
        self.ref_in_channels = ref_in_channels

        # Cache ref embeddings per sequence/track (CPU dict) for frozen-backbone use.
        # This allows efficient video processing: compute ref_emb once for frame 0,
        # then reuse for all subsequent frames in the same sequence/track.
        self.cache_ref_emb = cache_ref_emb  # Enable caching?
        self.ref_cache_keys = tuple(ref_cache_keys)
        self.ref_flag_keys = tuple(ref_flag_keys)
        self.ref_frame_idx_key = ref_frame_idx_key
        self.ref_from_feats = ref_from_feats
        self._ref_cache = {}  # Dictionary: {(key_name, key_value): ref_emb_tensor}

        # ===== Optional Deconvolution Stack =====
        # Usually not needed for command regression
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

        # ===== Convolutional Refinement Stack =====
        # Refine features before pooling (default: one 3x3 conv with 256 channels)
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
            in_channels = conv_out_channels[-1]  # Update in_channels for FiLM and MLP
        else:
            self.conv_layers = nn.Identity()  

        # ===== Reference Embedding Projection =====
        # If we derive ref_emb from pooled features, optionally project to ref_in_channels.
        # This ensures the ref_emb dimension matches what FiLM expects.
        self.ref_from_feat_proj = None
        if self.ref_in_channels is not None and self.ref_from_feats:
            if self.ref_in_channels != in_channels:
                self.ref_from_feat_proj = nn.Linear(in_channels, self.ref_in_channels)

        # ===== FiLM Conditioning Network =====
        # Feature-wise Linear Modulation: conditions features on reference embedding
        # ref_emb (B, ref_in_channels) → FiLM network → (gamma, beta) (B, 2*in_channels)
        # Then: x_modulated = x * (1 + gamma) + beta (per-channel affine transformation)
        # This allows the model to adapt its processing based on the target
        if self.ref_in_channels is not None:
            film_layers = [
                nn.Linear(self.ref_in_channels, film_hidden_dim),  # Expand to hidden dim
                nn.SiLU(inplace=True) if self.use_silu else nn.ReLU(inplace=True),
                nn.Linear(film_hidden_dim, 2 * in_channels),  # Output: gamma (in_channels) + beta (in_channels)
            ]
            self.film = nn.Sequential(*film_layers)
        else:
            self.film = None  

        # ===== Global Average Pooling =====
        # Collapse spatial dimensions: (B, C, H, W) → (B, C, 1, 1) → flatten to (B, C)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ===== MLP Regression Head =====
        # Final layers that map pooled features to 3D velocity commands
        # Default architecture: (B,256) → Linear(256) → SiLU → Linear(128) → SiLU → Linear(3)
        mlp_layers = []
        prev = in_channels
        for h in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            if self.use_silu:
                mlp_layers.append(nn.SiLU(inplace=True))
            else:
                mlp_layers.append(nn.ReLU(inplace=True))
            prev = h
        mlp_layers.append(nn.Linear(prev, out_channels))  # Final layer: hidden_dim → 3 (vx,vy, vyaw)
        self.mlp = nn.Sequential(*mlp_layers)

        # ===== Output Activation & Scaling =====
        # Tanh: constrain raw outputs to [-1, 1]
        # Scale: apply per-dimension physical limits based on robot capabilities
        self.out_act = nn.Tanh()
        self.register_buffer(
            'cmd_scale',
            torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32) # vx: ±1.0 m/s, vy: ±0.5 m/s, v_yaw: ±0.5 rad/s
        )

        # ---- loss ----
        # Default: SmoothL1Loss(beta=1.0)
        if loss is None:
            self.loss_module = nn.SmoothL1Loss(beta=1.0, reduction='none')  # Per-element for custom weighting
            self.loss_weight = 1.0
        else:
            # Build from registry if we configured it that way
            # Example: loss=dict(type='MSELoss', loss_weight=2.0)
            self.loss_module = MODELS.build(loss)
            # common pattern: loss modules may include loss_weight inside; but if not, keep 1.0
            self.loss_weight = float(loss.get('loss_weight', 1.0)) if isinstance(loss, dict) else 1.0

        # ===== Optional Normalization & Per-Dimension Weighting =====
        # cmd_mean/std: Normalize commands to zero mean, unit variance for training stability
        # cmd_weights: Weight dimensions differently in loss
        # Registered as buffers so they're saved with the model and moved to GPU automatically
        self.register_buffer('cmd_mean', torch.tensor(cmd_mean, dtype=torch.float32) if cmd_mean is not None else None)
        self.register_buffer('cmd_std', torch.tensor(cmd_std, dtype=torch.float32) if cmd_std is not None else None)
        self.register_buffer('cmd_weights', torch.tensor(cmd_weights, dtype=torch.float32) if cmd_weights is not None else None)

        # ===== Checkpoint Compatibility Hook =====
        # Register hook to handle old checkpoints gracefully
        # Important when loading heatmap head weights into this regression head
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
        """Extract GT command tensor of shape (B, 3) from samples.

        Searches three different data structures in MMPose's PoseDataSample:

        1. gt_instance_labels (InstanceData): Per-instance annotations
           - Instance-level ground truth (keypoints, bboxes, custom annotations)

        2. gt_fields (PixelData): Dense spatial predictions
           - Heatmaps, segmentation masks, dense field annotations
           - Alternative location if treating as field data

        3. metainfo (dict): Metadata and auxiliary information
           - File paths, sequence IDs, transforms, flags, misc data
           - Fallback location, less structured
        """
        gts = []
        for d in batch_data_samples:
            cmd = None

            # 1: gt_instance_labels
            # InstanceData object containing instance-level ground truth
            # Stores: keypoints, bboxes, nav_cmd, ref_emb, etc.
            if hasattr(d, 'gt_instance_labels') and d.gt_instance_labels is not None:
                if hasattr(d.gt_instance_labels, 'nav_cmd'):
                    cmd = d.gt_instance_labels.nav_cmd

            # 2: gt_fields
            # PixelData object containing dense spatial predictions
            # Stores: heatmaps, segmentation, depth maps, etc.
            if cmd is None and hasattr(d, 'gt_fields') and d.gt_fields is not None:
                if hasattr(d.gt_fields, 'nav_cmd'):
                    cmd = d.gt_fields.nav_cmd

            # 3: metainfo (fallback for metadata storage)
            # Python dict containing metadata and auxiliary info
            # Stores: img_path, sequence_id, frame_idx, transforms, flags, etc.
            if cmd is None and hasattr(d, 'metainfo') and d.metainfo is not None:
                if 'nav_cmd' in d.metainfo:
                    cmd = d.metainfo['nav_cmd']  # Shape: (3,) array-like

            if cmd is None:
                raise KeyError(
                    'Cannot find GT nav command in data sample. Expected one of:\n'
                    '  - sample.gt_instance_labels.nav_cmd (InstanceData - recommended)\n'
                    '  - sample.gt_fields.nav_cmd (PixelData)\n'
                    "  - sample.metainfo['nav_cmd'] (dict - fallback)\n"
                )

            # Convert to tensor and ensure shape is (3,)
            cmd_t = torch.as_tensor(cmd, dtype=torch.float32, device=device).view(3)
            gts.append(cmd_t)

        return torch.stack(gts, dim=0)  # (B,3)

    # ===== Reference Embedding Extraction =====
    def _get_ref_emb(self, batch_data_samples: OptSampleList, device: torch.device) -> Optional[Tensor]:
        """Extract reference embedding tensor of shape (B, ref_in_channels) if present.

        Searches three different data structures in MMPose's PoseDataSample:

        1. gt_instance_labels (InstanceData): Per-instance annotations
        
        2. gt_fields (PixelData): Dense spatial predictions
        
        3. metainfo (dict): Metadata and auxiliary information
           - Fallback: Useful for storing computed embeddings as metadata

        Returns:
            None if no ref_emb found in any sample (all None)
            Tensor (B, ref_in_channels) if found

        Raises:
            KeyError: If some samples have ref_emb but others don't (inconsistent batch)
        """
        if self.ref_in_channels is None:
            return None  # FiLM disabled

        refs = []
        for d in batch_data_samples:
            ref = None

            # 1: gt_instance_labels
            # InstanceData: stores per-instance annotations
            if hasattr(d, 'gt_instance_labels') and d.gt_instance_labels is not None:
                if hasattr(d.gt_instance_labels, 'ref_emb'):
                    ref = d.gt_instance_labels.ref_emb  # Shape: (ref_in_channels,)

            # 2: gt_fields
            # PixelData: stores dense spatial predictions
            if ref is None and hasattr(d, 'gt_fields') and d.gt_fields is not None:
                if hasattr(d.gt_fields, 'ref_emb'):
                    ref = d.gt_fields.ref_emb  # Shape: (ref_in_channels,)

            # 3: metainfo (fallback for precomputed/cached embeddings)
            # dict: stores metadata and auxiliary info
            if ref is None and hasattr(d, 'metainfo') and d.metainfo is not None:
                if 'ref_emb' in d.metainfo:
                    ref = d.metainfo['ref_emb']  # Shape: (ref_in_channels,) array-like

            if ref is None:
                refs.append(None)
                continue

            # Convert to tensor and flatten to 1D vector
            ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device).view(-1)
            refs.append(ref_t)

        # If all samples have no ref_emb, return None (FiLM won't be applied)
        if all(r is None for r in refs):
            return None

        # If some samples have ref_emb but others don't, raise error
        # This ensures batch consistency for FiLM conditioning
        for r in refs:
            if r is None:
                raise KeyError(
                    'Reference embedding missing for some samples. Expected one of:\n'
                    '  - sample.gt_instance_labels.ref_emb (InstanceData - recommended)\n'
                    '  - sample.gt_fields.ref_emb (PixelData)\n'
                    "  - sample.metainfo['ref_emb'] (dict - fallback)\n"
                )
        return torch.stack(refs, dim=0)  # (B, ref_in_channels)

    def _get_ref_cache_key(self, data_sample) -> Optional[Tuple[str, str]]:
        """Generate cache key from data sample metainfo for reference embedding lookup.

        Searches metainfo for first matching key in priority order:
        sequence_id > seq_id > track_id > video_id

        Example: {'sequence_id': '00042', 'frame_idx': 5} → ('sequence_id', '00042')

        Returns:
            Tuple (key_name, key_value) if found, else None
        """
        if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
            return None
        # First matching key in metainfo is used as cache key (e.g., sequence_id).
        for key in self.ref_cache_keys:
            if key in data_sample.metainfo:
                return (key, str(data_sample.metainfo[key]))
        return None

    def _is_ref_sample(self, data_sample) -> bool:
        """Check if this data sample is a reference frame.

        Reference frames are identified by:
        1. Explicit flag: is_ref=True, is_ref_frame=True, or is_first_frame=True
        2. Implicit: frame_idx == 0

        Returns:
            True if this is a reference frame, False otherwise
        """
        if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
            return False
        # Method 1: Explicit flag
        for key in self.ref_flag_keys:
            if key in data_sample.metainfo:
                return bool(data_sample.metainfo[key])
        # Method 2: Frame index 0 implies reference
        if self.ref_frame_idx_key and self.ref_frame_idx_key in data_sample.metainfo:
            return int(data_sample.metainfo[self.ref_frame_idx_key]) == 0
        return False

    def _compute_ref_emb_from_x(self, x: Tensor) -> Tensor:
        """Compute reference embedding from feature map.

        Used when ref_from_feats=True to extract ref_emb from pooled features.
        Flow: (B,C,H,W) → pool → (B,C) → optional projection → (B,ref_in_channels)

        Args:
            x: Feature map (B, C, H, W) after conv_layers

        Returns:
            ref_emb: (B, ref_in_channels)
        """
        # Pool the feature map into a single vector per sample for ref_emb.
        ref = self.pool(x)  # (B, C, H, W) → (B, C, 1, 1)
        ref = torch.flatten(ref, 1)  # (B, C)

        # Project to ref_in_channels if needed
        if self.ref_from_feat_proj is not None:
            ref = self.ref_from_feat_proj(ref)  # (B, C) → (B, ref_in_channels)
        elif self.ref_in_channels is not None and ref.shape[1] != self.ref_in_channels:
            raise ValueError(
                'ref_from_feats is enabled but ref_in_channels does not match '
                f'pooled feature dim ({ref.shape[1]} != {self.ref_in_channels}).'
            )
        return ref  # (B, ref_in_channels)

    def _get_or_build_ref_emb(self, x: Tensor, batch_data_samples: OptSampleList) -> Optional[Tensor]:
        """Resolve ref embeddings via samples and/or cache.

        This is the core caching logic. Three strategies in priority order:
        1. Explicit ref_emb: Use ref_emb from data samples if provided
        2. Build from ref frame: If this is a ref frame, compute ref_emb from features
        3. Load from cache: For non-ref frames, lookup cached ref_emb by sequence ID

        Args:
            x: Feature map (B,C,H,W) after conv_layers
            batch_data_samples: List of data samples with metainfo

        Returns:
            ref_emb (B, ref_in_channels) or None
        """
        if self.ref_in_channels is None:
            return None  # FiLM disabled

        # 1. Explicit ref_emb in data samples
        # If ref_emb is explicitly provided, use it and optionally cache it
        ref_emb = self._get_ref_emb(batch_data_samples, device=x.device)
        if ref_emb is not None:
            if self.cache_ref_emb:
                # Store on CPU to avoid GPU memory growth.
                for i, d in enumerate(batch_data_samples):
                    key = self._get_ref_cache_key(d)
                    if key is not None:
                        self._ref_cache[key] = ref_emb[i].detach().cpu()
            return ref_emb

        # If caching is disabled and no explicit ref_emb, return None
        if not self.cache_ref_emb:
            return None

        # 2. Build cache from reference frame features
        # Check if any sample in batch is a reference frame that needs caching
        keys = [self._get_ref_cache_key(d) for d in batch_data_samples]
        need_build = self.ref_from_feats and any(
            self._is_ref_sample(d) and k is not None and k not in self._ref_cache
            for d, k in zip(batch_data_samples, keys)
        )
        ref_from_x = self._compute_ref_emb_from_x(x) if need_build else None

        # Store newly computed ref_emb for reference frames
        if need_build:
            for i, (d, key) in enumerate(zip(batch_data_samples, keys)):
                if key is None or not self._is_ref_sample(d):
                    continue
                self._ref_cache[key] = ref_from_x[i].detach().cpu()

        # 3. Load ref_emb from cache
        # For each sample, lookup cached ref_emb by sequence ID
        refs = []
        missing = []
        for i, key in enumerate(keys):
            if key is None or key not in self._ref_cache:
                missing.append(i)
                refs.append(None)
            else:
                refs.append(self._ref_cache[key].to(x.device))

        if missing:
            raise KeyError(
                f'Reference embedding not found in cache for samples: {missing}. '
                'Provide ref_emb in data samples or mark the ref frame with a ref flag '
                'and enable ref_from_feats.'
            )

        return torch.stack(refs, dim=0)  # (B, ref_in_channels)

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

    def forward(self,
                feats: Tuple[Tensor],
                ref_emb: Optional[Tensor] = None,
                batch_data_samples: OptSampleList = None) -> Tensor:
        """Forward pass: Image features → Navigation commands.

        Complete pipeline:
        1. Extract backbone features
        3. Conv Layer
        4. FiLM conditioning - modulate features based on target reference
        5. Global pooling
        6. MLP
        7. Tanh activation + physical scaling

        Args:
            feats: Tuple of multi-scale features from backbone. We use feats[-1] (deepest).
                   Example: feats[-1] is (B, 1024, 48, 64) for Sapiens-0.3B with 768x1024 input
            ref_emb: Optional reference embedding (B, ref_in_channels) for FiLM conditioning.
                     If None and FiLM enabled, will attempt to resolve from batch_data_samples/cache.
            batch_data_samples: Optional data samples for resolving cached ref embeddings.
                                Required if using FiLM with caching.

        Returns:
            Tensor: (B, 3) navigation commands = (vx, vy, v_yaw)
                    vx ∈ [-1.0, 1.0] m/s (forward/backward)
                    vy ∈ [-0.5, 0.5] m/s (left/right strafe)
                    v_yaw ∈ [-0.5, 0.5] rad/s (rotation)
        """
        x = feats[-1] # Final layer from backbone

        x = self.deconv_layers(x)

        x = self.conv_layers(x)

        # FiLM Conditioning
        # TODO: Experiment with applying FiLM before vs after deconv/convs
        if self.film is not None:
            # Resolve ref_emb if not explicitly provided
            if ref_emb is None and batch_data_samples is not None:
                ref_emb = self._get_or_build_ref_emb(x, batch_data_samples)

            # Apply FiLM modulation: x_new = x * (1 + γ) + β
            if ref_emb is not None:
                gamma_beta = self.film(ref_emb)  # (B, ref_in_channels) → (B, 2*256)
                gamma, beta = gamma_beta.chunk(2, dim=1)  # Split: (B,256), (B,256)
                # Broadcast and apply per-channel affine transformation
                x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                # Shape: (B,256,48,64) ⊙ (B,256,1,1) + (B,256,1,1) → (B,256,48,64)

        x = self.pool(x)              # (B,256,48,64) → (B,256,1,1) Global average pooling
        x = torch.flatten(x, 1)       # (B,256,1,1) → (B,256) Flatten to vector

        # ===== Step 6: MLP Regression =====
        cmd = self.mlp(x)             # (B,256) → (B,256) → SiLU → (B,128) → SiLU → (B,3)

        # ===== Step 7: Output Activation & Scaling =====
        cmd = self.out_act(cmd)       # Tanh: (B,3) ∈ [-1, 1]
        cmd = cmd * self.cmd_scale    # Scale: [1.0, 0.5, 0.5] → physical velocity limits
        # Final: vx∈[-1.0,1.0], vy∈[-0.5,0.5], v_yaw∈[-0.5,0.5]

        return cmd  # (B, 3)

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict navigation command.

        Returns:
            List[InstanceData]: length B; each InstanceData has field `nav_cmd` (Tensor(3,))
        """
        pred_cmd = self.forward(feats, batch_data_samples=batch_data_samples)  # (B,3) from forward pass

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
        """Compute training loss and evaluation metrics.

        Loss computation pipeline:
        1. Forward pass to get predictions
        2. Extract ground truth commands from data samples
        3. Clamp GT to valid range (match output bounds)
        4. Compute main loss (default: SmoothL1)
        5. Compute auxiliary metrics (MAE, RMSE, accuracy)

        Args:
            feats: Tuple of backbone features
            batch_data_samples: List of data samples containing GT commands
            train_cfg: Training configuration, may include tolerance thresholds

        Returns:
            dict of losses and metrics:
                - loss_nav: Main training loss (scalar)
                - mae_vx, mae_vy, mae_yaw: Mean Absolute Error per dimension
                - rmse_vx, rmse_vy, rmse_yaw: Root Mean Square Error per dimension
                - acc_vx_tol, acc_vy_tol, acc_yaw_tol: Accuracy within tolerance (optional)
        """
        pred_cmd = self.forward(feats, batch_data_samples=batch_data_samples)
        gt_cmd = self._get_gt_cmd(batch_data_samples, device=pred_cmd.device) 

        # ===== Clamp GT to Valid Range =====
        # Ensure GT matches output bounds to avoid unrealistic targets
        gt_cmd = gt_cmd.clamp(
            torch.tensor([-1.0, -0.5, -0.5], device=gt_cmd.device),  # Min
            torch.tensor([ 1.0,  0.5,  0.5], device=gt_cmd.device),  # Max
        )

        # ===== Optional: Train in Normalized Space =====
        # If cmd_mean/std are provided, can normalize both pred and GT for stability
        # pred_n = self._normalize_cmd(pred_cmd)
        # gt_n = self._normalize_cmd(gt_cmd)

        losses = {}

        # ===== Main Training Loss =====
        loss_raw = self.loss_module(pred_cmd, gt_cmd)

        if isinstance(loss_raw, Tensor) and loss_raw.ndim == 2 and loss_raw.shape[-1] == 3:
            loss_val = self._weighted_loss_reduce(loss_raw) * self.loss_weight
        else:
            loss_val = loss_raw * self.loss_weight

        losses['loss_nav'] = loss_val

        with torch.no_grad():
            err = (pred_cmd - gt_cmd)  # (B,3) Per-sample errors

            mae = err.abs().mean(dim=0)     # (3,) average over batch
            losses['mae_vx'] = mae[0]
            losses['mae_vy'] = mae[1]
            losses['mae_yaw'] = mae[2]

            rmse = torch.sqrt((err ** 2).mean(dim=0) + 1e-12) # (3,) average over batch
            losses['rmse_vx'] = rmse[0]
            losses['rmse_vy'] = rmse[1]
            losses['rmse_yaw'] = rmse[2]

            # ===== Optional: Accuracy Within Tolerance =====
            # Percentage of predictions within acceptable error bounds
            if 'tol_vx' in train_cfg or 'tol_vy' in train_cfg or 'tol_yaw' in train_cfg:
                tol = torch.tensor([
                    float(train_cfg.get('tol_vx', 0.05)),    # Default: ±0.05 m/s for vx
                    float(train_cfg.get('tol_vy', 0.05)),    # Default: ±0.05 m/s for vy
                    float(train_cfg.get('tol_yaw', 0.10)),   # Default: ±0.10 rad/s for v_yaw
                ], device=pred_cmd.device).view(1, 3)
                within = (err.abs() <= tol).float().mean(dim=0)  # % within tolerance
                losses['acc_vx_tol'] = within[0]
                losses['acc_vy_tol'] = within[1]
                losses['acc_yaw_tol'] = within[2]

        return losses, pred_cmd

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
