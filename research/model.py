"""SAM + LoRA model for sclera segmentation (Track 1)."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from segment_anything import sam_model_registry


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen linear layer."""
    def __init__(self, original, rank=4, alpha=1.0):
        super().__init__()
        self.original = original
        in_f = original.in_features
        out_f = original.out_features
        self.lora_down = nn.Linear(in_f, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_normal_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x)) * self.scale


class SegDecoder(nn.Module):
    """Lightweight decoder: 64x64x256 -> 256x256x1."""
    def __init__(self, in_ch=256):
        super().__init__()
        self.up = nn.Sequential(
            # 64 -> 128
            nn.ConvTranspose2d(in_ch, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 128 -> 256
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Refine
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.up(x)


class SAMLoRASeg(nn.Module):
    """SAM ViT encoder + LoRA adapters + segmentation decoder.

    Architecture:
      - SAM ViT-B image encoder (frozen)
      - LoRA adapters injected into every attention Q/K/V and projection
      - Lightweight ConvTranspose decoder for binary mask output

    Trainable params: LoRA (~220K) + Decoder (~100K) ≈ 320K total
    Frozen params: SAM encoder (~86M)
    """
    def __init__(self, checkpoint, model_type='vit_b', lora_rank=4, lora_alpha=1.0):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.encoder = sam.image_encoder
        # SAM normalization (pixel values in 0-255 range)
        self.register_buffer(
            'pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer(
            'pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Inject LoRA into attention and MLP layers
        for block in self.encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=lora_rank, alpha=lora_alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank=lora_rank, alpha=lora_alpha)
            block.mlp.lin1 = LoRALinear(block.mlp.lin1, rank=lora_rank, alpha=lora_alpha)
            block.mlp.lin2 = LoRALinear(block.mlp.lin2, rank=lora_rank, alpha=lora_alpha)
        # Decoder
        self.decoder = SegDecoder(in_ch=256)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 1024, 1024] float tensor in 0-255 range
        Returns:
            logits: [B, 1, 256, 256] raw logits (apply sigmoid for probs)
        """
        x = (x - self.pixel_mean) / self.pixel_std
        # Gradient checkpointing to save VRAM
        if self.training:
            features = grad_checkpoint(self.encoder, x, use_reentrant=False)
        else:
            features = self.encoder(x)
        return self.decoder(features)  # [B, 1, 256, 256]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
