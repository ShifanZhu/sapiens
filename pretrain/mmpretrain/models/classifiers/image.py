# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import List, Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier


def _kwargs_for_backbone_forward(backbone: nn.Module, kw: dict) -> dict:
    """Drop kwargs the backbone's ``forward`` does not accept (e.g. depth extras)."""
    sig = inspect.signature(backbone.forward)
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kw
    names = {p.name for p in params}
    return {k: v for k, v in kw.items() if k in names}


def _point_clouds_from_data_samples(
    data_samples: Optional[List[DataSample]],
    field_name: str = 'human_points_local',
) -> Optional[torch.Tensor]:
    """Stack per-sample point clouds when every sample defines ``field_name``.

    Datasets should pack points via :class:`PackInputs` ``algorithm_keys`` (e.g.
    ``human_points_local``) so each :class:`DataSample` holds a ``(N, 3)`` tensor.
    """
    if not data_samples:
        return None
    tensors = []
    for sample in data_samples:
        if sample is None or field_name not in sample:
            return None
        pc = sample.get(field_name)
        if pc is None:
            return None
        if not isinstance(pc, torch.Tensor):
            pc = torch.as_tensor(pc, dtype=torch.float32)
        if pc.dim() != 2 or pc.size(-1) != 3:
            raise ValueError(
                f'{field_name} must be (N, 3), got {tuple(pc.shape)}')
        tensors.append(pc)
    return torch.stack(tensors, dim=0)


def _image_names_from_data_samples(
    data_samples: Optional[List[DataSample]],
) -> Optional[list]:
    """Return ``img_path`` for each sample when all define it in metainfo.

    Enables :class:`~mmpretrain.models.backbones.vision_transformer_with_depth.VisionTransformerWithDepth`
    to resolve ``depth_embed_path`` files during ``loss`` / ``predict``.
    """
    if not data_samples:
        return None
    names = []
    for sample in data_samples:
        if sample is None or not getattr(sample, 'metainfo', None):
            return None
        path = sample.metainfo.get('img_path')
        if not path:
            return None
        names.append(path)
    return names


@MODELS.register_module()
class ImageClassifier(BaseClassifier):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(ImageClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone._checkpoint_filter)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            image_names = _image_names_from_data_samples(data_samples)
            point_clouds = _point_clouds_from_data_samples(data_samples)
            if point_clouds is not None:
                point_clouds = point_clouds.to(
                    device=inputs.device, non_blocking=True)
            feats = self.extract_feat(
                inputs,
                image_names=image_names,
                point_clouds=point_clouds)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self,
                     inputs,
                     stage='neck',
                     depth_embeddings=None,
                     image_names=None,
                     point_clouds=None,
                     **kwargs):
        """Extract features from the input tensor with shape (N, C, ...).
        
        Now supports depth embeddings! 

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".
            depth_embeddings (torch.Tensor, optional): Depth embeddings of shape
                (B, num_depth_tokens, depth_embed_dim). Passed to backbone if supported.
            image_names (list[str], optional): List of image names for loading depth
                embeddings. Passed to backbone if supported.
            point_clouds (torch.Tensor, optional): ``(B, N, 3)`` for live LiDAR-HMR encoding
                when the backbone implements it (see ``VisionTransformerWithDepth``).
            **kwargs: Other keyword arguments (forwarded to backbone).

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        # Pass depth embeddings and image names to backbone if it supports them
        # Most backbones will ignore these kwargs, but VisionTransformerWithDepth will use them
        backbone_kw = dict(kwargs)
        if depth_embeddings is not None:
            backbone_kw['depth_embeddings'] = depth_embeddings
        if image_names is not None:
            backbone_kw['image_names'] = image_names
        if point_clouds is not None:
            backbone_kw['point_clouds'] = point_clouds
        backbone_kw = _kwargs_for_backbone_forward(self.backbone, backbone_kw)
        x = self.backbone(inputs, **backbone_kw)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        image_names = _image_names_from_data_samples(data_samples)
        point_clouds = _point_clouds_from_data_samples(data_samples)
        if point_clouds is not None:
            point_clouds = point_clouds.to(
                device=inputs.device, non_blocking=True)
        feats = self.extract_feat(
            inputs,
            image_names=image_names,
            point_clouds=point_clouds)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        image_names = _image_names_from_data_samples(data_samples)
        point_clouds = _point_clouds_from_data_samples(data_samples)
        if point_clouds is not None:
            point_clouds = point_clouds.to(
                device=inputs.device, non_blocking=True)
        feats = self.extract_feat(
            inputs,
            image_names=image_names,
            point_clouds=point_clouds)
        return self.head.predict(feats, data_samples, **kwargs)

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The backbone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')
