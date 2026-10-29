"""SAM + LoRA v4 — Model with dropout regularization in decoder."""
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


class SegDecoderV4(nn.Module):
    """Decoder with dropout for regularization: 64x64x256 -> 256x256x1."""
    def __init__(self, in_ch=256, dropout=0.15):
        super().__init__()
        self.up = nn.Sequential(
            # 64 -> 128
            nn.ConvTranspose2d(in_ch, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout),
            # 128 -> 256
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout),
            # Refine
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.up(x)


class SAMLoRASegV4(nn.Module):
    """SAM ViT encoder + LoRA + regularized decoder.

    Same architecture as v3 but with:
      - Dropout2d in decoder layers
      - Compatible with SWA and TTA
    """
    def __init__(self, checkpoint, model_type='vit_b', lora_rank=4,
                 lora_alpha=1.0, decoder_dropout=0.15):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.encoder = sam.image_encoder
        self.register_buffer(
            'pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer(
            'pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Inject LoRA
        for block in self.encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=lora_rank, alpha=lora_alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank=lora_rank, alpha=lora_alpha)
            block.mlp.lin1 = LoRALinear(block.mlp.lin1, rank=lora_rank, alpha=lora_alpha)
            block.mlp.lin2 = LoRALinear(block.mlp.lin2, rank=lora_rank, alpha=lora_alpha)
        # Decoder with dropout
        self.decoder = SegDecoderV4(in_ch=256, dropout=decoder_dropout)

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        if self.training:
            features = grad_checkpoint(self.encoder, x, use_reentrant=False)
        else:
            features = self.encoder(x)
        return self.decoder(features)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
