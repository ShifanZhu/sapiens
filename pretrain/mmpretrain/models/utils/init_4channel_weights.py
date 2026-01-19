
import torch
import torch.nn as nn


def init_4channel_patch_embed_from_3channel(
    patch_embed_4ch: nn.Module,
    state_dict_3ch: dict,
    prefix: str = 'patch_embed.projection.',
    init_method: str = 'mean'
) -> None:
    """Init 4-channel patch embed from 3-channel pretrained weights."""
    # get the conv layer
    if hasattr(patch_embed_4ch, 'projection'):
        proj = patch_embed_4ch.projection
    else:
        proj = patch_embed_4ch
    
    if not isinstance(proj, nn.Conv2d):
        raise ValueError(
            f"Expected Conv2d layer, got {type(proj)}. "
            "Make sure patch_embed has a 'projection' attribute."
        )
    
    if proj.in_channels != 4:
        raise ValueError(
            f"Expected 4-channel input, got {proj.in_channels} channels. "
            "Make sure the model is initialized with in_channels=4."
        )
    
    # get pretrained weights
    weight_key = prefix + 'weight'
    bias_key = prefix + 'bias'
    
    if weight_key not in state_dict_3ch:
        raise KeyError(
            f"Weight key '{weight_key}' not found in state_dict. "
            f"Available keys: {list(state_dict_3ch.keys())[:10]}..."
        )
    
    weight_3ch = state_dict_3ch[weight_key]
    bias_3ch = state_dict_3ch.get(bias_key, None)
    
    # check shapes match
    if weight_3ch.shape[0] != proj.out_channels:
        raise ValueError(
            f"Output channels don't match: pretrained={weight_3ch.shape[0]}, "
            f"model={proj.out_channels}"
        )
    
    if weight_3ch.shape[2:] != proj.kernel_size:
        raise ValueError(
            f"Kernel size doesn't match: pretrained={weight_3ch.shape[2:]}, "
            f"model={proj.kernel_size}"
        )
    
    # create 4-channel weight tensor
    weight_4ch = torch.zeros(
        proj.out_channels, 4, *proj.kernel_size,
        dtype=weight_3ch.dtype, device=weight_3ch.device
    )
    
    # copy RGB channels
    weight_4ch[:, :3, :, :] = weight_3ch
    
    # init depth channel (4th channel)
    if init_method == 'mean':
        weight_4ch[:, 3, :, :] = weight_3ch.mean(dim=1)
    elif init_method == 'copy_r':
        weight_4ch[:, 3, :, :] = weight_3ch[:, 0, :, :]
    elif init_method == 'copy_g':
        weight_4ch[:, 3, :, :] = weight_3ch[:, 1, :, :]
    elif init_method == 'copy_b':
        weight_4ch[:, 3, :, :] = weight_3ch[:, 2, :, :]
    elif init_method == 'zero':
        weight_4ch[:, 3, :, :] = 0.0
    else:
        raise ValueError(
            f"Unknown init_method: {init_method}. "
            "Choose from: 'mean', 'copy_r', 'copy_g', 'copy_b', 'zero'"
        )
    
    proj.weight.data = weight_4ch
    
    # handle bias
    if bias_3ch is not None and proj.bias is not None:
        proj.bias.data = bias_3ch
    elif proj.bias is not None:
        proj.bias.data.zero_()


def init_4channel_vit_from_3channel(
    model: nn.Module,
    pretrained_path: str,
    init_method: str = 'mean',
    strict: bool = False
) -> None:
    """Helper to init 4-channel ViT from 3-channel pretrained."""
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # find patch_embed prefix
    prefixes = [
        'backbone.patch_embed.projection.',
        'patch_embed.projection.',
        'model.patch_embed.projection.',
    ]
    
    prefix = None
    for p in prefixes:
        weight_key = p + 'weight'
        if weight_key in state_dict:
            prefix = p
            break
    
    if prefix is None:
        # search for any patch_embed weight key
        for key in state_dict.keys():
            if 'patch_embed' in key and 'weight' in key:
                prefix = key.rsplit('weight', 1)[0]
                break
    
    if prefix is None:
        raise KeyError(
            "Could not find patch_embed projection weights in checkpoint. "
            "Please specify the prefix manually."
        )
    
    # get patch_embed layer
    if hasattr(model, 'backbone'):
        patch_embed = model.backbone.patch_embed
    elif hasattr(model, 'patch_embed'):
        patch_embed = model.patch_embed
    else:
        raise AttributeError(
            "Model must have 'backbone.patch_embed' or 'patch_embed' attribute."
        )
    
    init_4channel_patch_embed_from_3channel(
        patch_embed, state_dict, prefix=prefix, init_method=init_method
    )
    
    # remove patch_embed keys and load rest
    keys_to_remove = [k for k in state_dict.keys() if 'patch_embed.projection' in k]
    for k in keys_to_remove:
        del state_dict[k]
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")



