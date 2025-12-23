

"""
Vision Transformer with Depth Embeddings Integration

So basically this extends the original VisionTransformer to work with depth embeddings.
The main idea is we concatenate RGB patch embeddings with depth embeddings before 
the transformer layers. This way RGB and depth tokens can interact through self-attention.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from ..utils import resize_pos_embed
from .vision_transformer import VisionTransformer


@MODELS.register_module()
class VisionTransformerWithDepth(VisionTransformer):
    """Vision Transformer with Depth Embeddings Support
    
    This extends VisionTransformer to add depth embeddings from Point-MAE encoder.
    The depth embeddings can have any number of tokens (e.g., 26 tokens with 1024 dimensions).
    We project them to match the RGB embedding dimension (like 1536 for sapiens_1b), 
    then concatenate with RGB patch tokens before the transformer processes everything.
    
    Key Integration Point: Line ~311 - Concatenation of RGB and depth embeddings!
    
    How it works:
    1. RGB Image (1024×1024) gets split into patches → 4096 RGB tokens (1536-dim for sapiens_1b)
    2. Depth Embeddings (num_depth_tokens, 1024-dim) → project to 1536-dim to match RGB
    3. Concatenate: [4096 RGB, num_depth_tokens depth] = (4096 + num_depth_tokens) tokens total
    4. Add CLS token: (4096 + num_depth_tokens + 1) tokens
    5. Transformer processes all tokens together so RGB and depth can attend to each other
    6. Output: RGB features but now they have depth info mixed in
    
    Args:
        depth_embed_dim (int): Dimension of depth embeddings. Defaults to 1024.
        depth_embed_path (str, optional): Path to folder with depth embedding .npy files.
            If None, you need to pass depth embeddings in forward().
        use_depth_projection (bool): Whether to project depth to match RGB dimension. Defaults to True.
        depth_projection_type (str): Type of projection - 'linear' or 'mlp'. Defaults to 'linear'.
        **kwargs: Same args as VisionTransformer.
    """
    
    def __init__(self,
                 depth_embed_dim: int = 1024,
                 depth_embed_path: Optional[str] = None,
                 use_depth_projection: bool = True,
                 depth_projection_type: str = 'linear',
                 **kwargs):
        # First call the parent VisionTransformer init
        super(VisionTransformerWithDepth, self).__init__(**kwargs)
        
        # Save the depth embedding settings
        self.depth_embed_dim = depth_embed_dim
        self.depth_embed_path = depth_embed_path
        self.use_depth_projection = use_depth_projection
        
        # Need to project depth embeddings to match RGB dimension
        # RGB embeddings are embed_dims (like 1536 for sapiens_1b)
        # Depth embeddings are depth_embed_dim (1024)
        # So we project depth from 1024 to 1536 so they can be concatenated
        if use_depth_projection:
            if depth_projection_type == 'linear':
                # Just a simple linear layer: 1024 → 1536
                self.depth_proj = nn.Linear(depth_embed_dim, self.embed_dims)
            elif depth_projection_type == 'mlp':
                # MLP with two layers: 1024 → 1536 → 1536 (with GELU in between)
                self.depth_proj = nn.Sequential(
                    nn.Linear(depth_embed_dim, self.embed_dims),
                    nn.GELU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
            else:
                raise ValueError(f"Unknown projection type: {depth_projection_type}")
        else:
            # If no projection, dimensions must already match
            assert depth_embed_dim == self.embed_dims, \
                f"Without projection, depth_embed_dim ({depth_embed_dim}) must equal embed_dims ({self.embed_dims})"
            self.depth_proj = nn.Identity()
        
        # Position embeddings need to account for depth tokens too
        # Original: num_patches + 1 (for CLS token)
        # With depth: num_patches + num_depth_tokens + 1 (CLS token)
        # We'll extend pos_embed dynamically in forward() when we need it
        
    def load_depth_embedding(self, image_name: str) -> Optional[torch.Tensor]:
        """Load depth embedding from .npy file
        
        Tries to find depth embeddings that match the image name. Looks for files like
        {image_name}.npy or {image_name}_depth.npy etc.
        
        Example:
            RGB image: "1764112925473950_rgb.png"
            Depth embedding: "1764112925473950.npy" or "1764112925473950_rgb.npy"
        
        Args:
            image_name (str): Name of the RGB image (can be full path or just filename)
            
        Returns:
            torch.Tensor or None: Depth embedding shape (num_depth_tokens, depth_embed_dim),
                or None if not found
        """
        if self.depth_embed_path is None:
            return None
        
        # Get just the filename from path (in case full path is given)
        base_name = os.path.basename(image_name)
        # Remove image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        
        # Remove '_rgb' suffix if it's there
        if base_name.endswith('_rgb'):
            base_name = base_name[:-4]
        
        # Try different possible filenames for the depth embedding
        possible_names = [
            f"{base_name}.npy",
            f"{base_name}_depth.npy",
            f"{base_name}_latent.npy",
        ]
        
        for name in possible_names:
            embed_path = os.path.join(self.depth_embed_path, name)
            if os.path.exists(embed_path):
                # Load the numpy file and convert to tensor
                depth_embed = np.load(embed_path)
                # Make sure it's 2D: (num_tokens, embed_dim)
                if depth_embed.ndim == 1:
                    depth_embed = depth_embed.reshape(1, -1)
                return torch.from_numpy(depth_embed).float()
        
        # Couldn't find it - return None (forward() will handle this)
        print(f"Warning: Depth embedding not found for {base_name} in {self.depth_embed_path}")
        return None
    
    def forward(self, x, depth_embeddings=None, image_names=None):
        """Forward pass with depth embeddings integration
        MAIN FUNCTION
        
        Stage 1: Get RGB patch embeddings (same as original VisionTransformer)
        Stage 2: Load/process depth embeddings (new part)
        Stage 3: Project depth to match RGB dimension (new part)
        Stage 4: Concatenate RGB + Depth tokens (this is the key integration point)
        Stage 5: Continue with CLS token, positions, transformer (same as original)
        
        Args:
            x (torch.Tensor): RGB images shape (B, 3, H, W)
            depth_embeddings (torch.Tensor, optional): Pre-loaded depth embeddings shape
                (B, num_depth_tokens, depth_embed_dim). If None and depth_embed_path is set,
                will try to load from disk using image_names.
            image_names (list[str], optional): List of image names for loading depth embeddings.
                Only used if depth_embeddings is None. Should match batch size.
                
        Returns:
            tuple: Output features as defined by out_type (same as VisionTransformer)
        """
        B = x.shape[0]
        
        # Stage 1: Get RGB Patch Embeddings (same as original)
        # Convert RGB image into patch tokens
        # Input: (B, 3, 1024, 1024)
        # Output: (B, 4096, embed_dim) where 4096 = (1024/16)² patches
        x_rgb, patch_resolution = self.patch_embed(x)
        num_rgb_patches = x_rgb.shape[1]  # Usually 4096 for 1024×1024 images
        
        # Stage 2: Load/Process Depth Embeddings (new part)
        if depth_embeddings is None and self.depth_embed_path is not None:
            # Try loading depth embeddings from disk using image names
            if image_names is not None:
                depth_emb_list = []
                num_depth_tokens_per_sample = None
                skip_depth = False
                
                for img_name in image_names:
                    depth_embed = self.load_depth_embedding(img_name)
                    if depth_embed is None:
                        # If not found, we need to know how many tokens to create
                        # Try to get it from first successfully loaded embedding
                        if num_depth_tokens_per_sample is None:
                            # Can't create fallback without knowing num tokens - skip depth for this batch
                            print(f"Warning: Could not load depth embedding for {img_name} and num_depth_tokens unknown. Skipping depth embeddings.")
                            skip_depth = True
                            break
                        else:
                            # Use zeros as fallback with same number of tokens as other samples
                            depth_embed = torch.zeros(num_depth_tokens_per_sample, self.depth_embed_dim)
                    else:
                        # Track the number of tokens from first successful load
                        if num_depth_tokens_per_sample is None:
                            num_depth_tokens_per_sample = depth_embed.shape[0]
                        elif depth_embed.shape[0] != num_depth_tokens_per_sample:
                            # All depth embeddings in batch should have same number of tokens
                            raise ValueError(f"Depth embedding tokens mismatch: expected {num_depth_tokens_per_sample}, got {depth_embed.shape[0]}")
                    depth_emb_list.append(depth_embed)
                
                if skip_depth or len(depth_emb_list) == 0:
                    # Couldn't load any depth embeddings, skip them
                    depth_embeddings = None
                else:
                    # Stack them into a batch: (B, num_depth_tokens, depth_embed_dim)
                    depth_embeddings = torch.stack(depth_emb_list, dim=0).to(x.device)
            else:
                # No image names given, so skip depth embeddings
                depth_embeddings = None
        
        # Stage 3: Project Depth Embeddings (new part)
        if depth_embeddings is not None:
            # Project depth embeddings to match RGB embedding dimension
            # Input: (B, num_depth_tokens, depth_embed_dim) - depth embeddings from Point-MAE
            # Output: (B, num_depth_tokens, embed_dim) - projected to match RGB dimension
            depth_embeddings = self.depth_proj(depth_embeddings)
            num_depth_tokens = depth_embeddings.shape[1]  # Get actual number from tensor
        else:
            num_depth_tokens = 0
        
        # Stage 4: Concatenate RGB + Depth Tokens (this is the key integration point)
        # Combine RGB patch tokens with depth tokens before transformer processing
        if num_depth_tokens > 0:
            # Concatenate along sequence dimension (dim=1)
            # RGB: (B, num_rgb_patches, embed_dim)
            # Depth: (B, num_depth_tokens, embed_dim)
            # Combined: (B, num_rgb_patches + num_depth_tokens, embed_dim)
            x = torch.cat([x_rgb, depth_embeddings], dim=1)
        else:
            # No depth embeddings, so just use RGB (original behavior)
            x = x_rgb
        
        # Stage 5: Add CLS Token (same as original)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            # Add CLS token at the beginning
            # Combined: (B, num_rgb_patches + num_depth_tokens + 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1)
        
        # Stage 6: Add Position Embeddings (need to extend for depth tokens)
        # Original pos_embed: (1, num_patches + 1, embed_dim)
        # Extended: (1, num_patches + num_depth_tokens + 1, embed_dim)
        
        # Check how long the sequence is now
        current_seq_len = x.shape[1]  # Includes CLS token
        original_seq_len = self.pos_embed.shape[1]  # Original pos_embed length
        
        if current_seq_len > original_seq_len:
            # Need to extend position embeddings for depth tokens
            num_extra_tokens = current_seq_len - original_seq_len
            
            # Could use zeros (simpler) or learn separate embeddings (better)
            # For now just pad with zeros and let the model learn during training
            extra_pos_embed = torch.zeros(
                1, num_extra_tokens, self.embed_dims,
                device=self.pos_embed.device,
                dtype=self.pos_embed.dtype
            )
            extended_pos_embed = torch.cat([self.pos_embed, extra_pos_embed], dim=1)
        else:
            extended_pos_embed = self.pos_embed
        
        # Resize position embeddings to match current patch resolution
        # This handles different input sizes (same as original)
        x = x + resize_pos_embed(
            extended_pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)
        
        # Stage 7: Pre-normalization (same as original)
        x = self.pre_norm(x)  # Shape: (B, num_rgb_patches + num_depth_tokens + 1, embed_dim)
        
        # Stage 8: Transformer Layers (same as original, but now processes RGB+Depth)
        # All tokens (RGB + Depth + CLS) get processed together
        # Self-attention lets RGB tokens attend to depth tokens and vice versa
        # This is where the cross-modal interaction happens
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)
            
            if i in self.out_indices:
                # Format output - only return RGB patches (exclude depth tokens)
                # This keeps compatibility with downstream tasks that expect RGB features
                outs.append(self._format_output(x, patch_resolution))
        
        return tuple(outs)
    
    def _format_output(self, x, hw):
        """Format output features (same as original, but handles extended sequence)
        
        Important: We only return RGB patch tokens, not depth tokens
        Depth tokens are used during processing but not in the final output
        This keeps compatibility with downstream tasks that expect RGB features
        """
        if self.out_type == 'raw':
            # Return all tokens including depth (useful for debugging)
            return x
        if self.out_type == 'cls_token':
            # Just return the CLS token
            return x[:, 0]
        
        # Extract only RGB patch tokens (skip CLS token and depth tokens)
        # x shape: (B, 1 + num_rgb_patches + num_depth_tokens, embed_dim)
        # We want: (B, num_rgb_patches, embed_dim)
        # hw[0] * hw[1] gives number of RGB patches (like 64*64 = 4096)
        num_rgb_patches = hw[0] * hw[1]
        patch_token = x[:, self.num_extra_tokens:self.num_extra_tokens + num_rgb_patches]
        
        if self.out_type == 'featmap':
            B = x.size(0)
            # Reshape to spatial format: (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))

