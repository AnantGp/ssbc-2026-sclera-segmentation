"""DINOv2-based segmentation model for SSBC 2026 Track 2.

Architecture: DINOv2 ViT-B/14 (frozen, LayerNorm unfrozen) + FPN decoder
- Multi-scale features from 4 intermediate transformer blocks
- Lightweight FPN decoder with GroupNorm + GELU (~1.5M trainable params)
- No BatchNorm anywhere → clean Mean Teacher EMA (no buffer sync needed)

Design informed by:
- CSIRO 8th place: Frozen DINOv2 + lightweight head
- CSIRO 4th place: LoRA fine-tuning (optional Stage 2)
- CSIRO 1st place: Multi-layer feature extraction
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# DECODER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class ConvGnGelu(nn.Module):
    """Conv2d → GroupNorm → GELU (BatchNorm-free)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FPNDecoder(nn.Module):
    """Lightweight FPN decoder: progressive 2x upsampling.

    Input: (B, in_dim, H/14, W/14) → Output: (B, 32, ~H, ~W)
    4 upsample stages: 37→74→148→296→592 (then bilinear to exact size)
    """

    def __init__(self, in_dim=256, dims=(256, 128, 64, 32)):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = in_dim
        for out_ch in dims:
            self.stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvGnGelu(ch, out_ch),
                ConvGnGelu(out_ch, out_ch),
            ))
            ch = out_ch

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2Segmenter(nn.Module):
    """DINOv2 ViT-B/14 backbone + FPN decoder for sclera segmentation.

    Forward: image → DINOv2 multi-layer features → lateral 1x1 conv → sum → FPN → logits
    """

    def __init__(self, backbone_name='dinov2_vitb14',
                 feature_layers=(2, 5, 8, 11), decoder_dim=256,
                 unfreeze_ln=True):
        super().__init__()

        # Load DINOv2 backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', backbone_name, pretrained=True
        )
        self.feature_layers = list(feature_layers)
        self.embed_dim = self.backbone.embed_dim      # 768 for ViT-B
        self.patch_size = self.backbone.patch_size    # 14

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm for domain adaptation (CSIRO insight)
        if unfreeze_ln:
            for name, param in self.backbone.named_parameters():
                if 'norm' in name.lower() or 'ln' in name.lower():
                    param.requires_grad = True

        # Lateral projections: one 1×1 conv per feature layer
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, decoder_dim, 1, bias=False),
                nn.GroupNorm(32, decoder_dim),
                nn.GELU(),
            )
            for _ in self.feature_layers
        ])

        # FPN decoder: progressive upsampling
        self.decoder = FPNDecoder(in_dim=decoder_dim, dims=(256, 128, 64, 32))

        # Segmentation head
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract multi-layer features from DINOv2
        # Returns list of (B, embed_dim, H//14, W//14) when reshape=True
        features = self.backbone.get_intermediate_layers(
            x, n=self.feature_layers, reshape=True
        )

        # Lateral projections + sum (all same spatial resolution in ViT)
        fused = self.lateral_convs[0](features[0])
        for i in range(1, len(self.feature_layers)):
            fused = fused + self.lateral_convs[i](features[i])

        # Decode: progressive upsample
        decoded = self.decoder(fused)

        # Bilinear interpolate to exact input size
        if decoded.shape[2:] != (H, W):
            decoded = F.interpolate(
                decoded, size=(H, W), mode='bilinear', align_corners=False
            )

        # Segmentation head → logits
        logits = self.head(decoded)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_dino_model(backbone_name='dinov2_vitb14', feature_layers=(2, 5, 8, 11),
                     decoder_dim=256, unfreeze_ln=True):
    """Build DINOv2 segmentation model."""
    return DINOv2Segmenter(
        backbone_name=backbone_name,
        feature_layers=feature_layers,
        decoder_dim=decoder_dim,
        unfreeze_ln=unfreeze_ln,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FREEZE / UNFREEZE
# ══════════════════════════════════════════════════════════════════════════════

def freeze_backbone(model):
    """Freeze entire backbone including LayerNorm."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_layernorm(model):
    """Unfreeze only LayerNorm layers in backbone."""
    for name, param in model.backbone.named_parameters():
        if 'norm' in name.lower() or 'ln' in name.lower():
            param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# MEAN TEACHER (simplified — no BN buffer sync needed)
# ══════════════════════════════════════════════════════════════════════════════

def create_ema_model(model):
    """Create EMA teacher model.

    DINOv2 uses LayerNorm (no BatchNorm), so no buffer sync issues.
    """
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Update EMA model parameters.

    No buffer syncing needed (no BatchNorm in DINOv2 + GroupNorm decoder).
    GroupNorm has no running stats buffers.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def sync_teacher(teacher, student):
    """Fully sync teacher parameters with student."""
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.copy_(param.data)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_param_groups(model, base_lr=1e-3, ln_lr_scale=0.1):
    """Get parameter groups with different learning rates.

    - Decoder / lateral / head: base_lr
    - Backbone LayerNorm: base_lr * ln_lr_scale (lower for pretrained norms)
    """
    decoder_params = []
    ln_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            ln_params.append(param)
        else:
            decoder_params.append(param)

    return [
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
        {'params': ln_params, 'lr': base_lr * ln_lr_scale, 'name': 'layernorm'},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS (same as t2_model.py for consistency)
# ══════════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss."""

    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class ConsistencyLoss(nn.Module):
    """MSE consistency loss for Mean Teacher.

    Only applied to confident teacher predictions. Always FP32.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, student_logits, teacher_logits, conf_threshold=0.95):
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)

        confidence = torch.abs(teacher_probs - 0.5) * 2
        mask = (confidence > (conf_threshold - 0.5) * 2).float()

        mse = self.mse(student_probs, teacher_probs)
        masked_mse = mse * mask

        num_confident = mask.sum() + 1e-6
        loss = masked_mse.sum() / num_confident

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """Compute segmentation metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
    }
