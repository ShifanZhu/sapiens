"""Smoke-test for SapiensPose3D forward pass."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SapiensPose3D

B, H, W = 2, 384, 640

print("Building SapiensPose3D (sapiens_0.3b) ...")
model = SapiensPose3D(arch="sapiens_0.3b", img_size=(H, W))
model.eval()

# count parameters
total  = sum(p.numel() for p in model.parameters())
bb     = sum(p.numel() for p in model.backbone.parameters())
hd     = sum(p.numel() for p in model.head.parameters())
print(f"  backbone : {bb/1e6:.1f}M params")
print(f"  head     : {hd/1e6:.1f}M params")
print(f"  total    : {total/1e6:.1f}M params")

# forward pass
x = torch.randn(B, 4, H, W)
print(f"\nForward pass: input {x.shape}")
with torch.no_grad():
    out = model(x)
print(f"  output: {out.shape}  (expected [{B}, 127, 3])")
assert out.shape == (B, 127, 3), f"Wrong output shape: {out.shape}"

# check patch embed is 4-channel
pe = model.backbone.vit.patch_embed.projection
assert pe.in_channels == 4, f"patch_embed in_channels should be 4, got {pe.in_channels}"
print(f"  patch_embed: {pe}")

print("\nAll assertions passed.")
