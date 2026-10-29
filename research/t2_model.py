"""Model architecture for SSBC 2026 Track 2 (Semi-supervised).

Architecture: UNet + EfficientNet-B4 encoder (pretrained on ImageNet)

Key design choices from analysis:
- EfficientNet-B4: Good accuracy/speed tradeoff, ImageNet pretrained
- Unfreeze BatchNorm: CSIRO 4th place — allows domain adaptation
- Mean Teacher: EMA copy of student model
"""
import copy
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(encoder='efficientnet-b4', encoder_weights='imagenet'):
    """Build UNet with pretrained EfficientNet encoder.
    
    Args:
        encoder: Encoder backbone name
        encoder_weights: Pretrained weights ('imagenet' or None)
    
    Returns:
        UNet model outputting [B, 1, H, W] logits
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,  # Raw logits; apply sigmoid in loss/inference
    )
    return model


def freeze_encoder(model):
    """Freeze encoder parameters (Stage 1 training)."""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False


def unfreeze_encoder(model, unfreeze_bn=True):
    """Unfreeze encoder for Stage 2 fine-tuning.
    
    Args:
        model: The UNet model
        unfreeze_bn: If True, also unfreeze BatchNorm layers
                    (CSIRO 4th place: allows domain adaptation of batch stats)
    """
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = True
            
    # Optionally set BN to training mode even if mostly frozen
    # This allows batch statistics to adapt to sclera domain
    if unfreeze_bn:
        for module in model.encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.train()
                for p in module.parameters():
                    p.requires_grad = True


def create_ema_model(model):
    """Create EMA (Exponential Moving Average) copy for Mean Teacher.
    
    The teacher model is NOT trained via gradients — it's updated via EMA.
    """
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Update EMA model parameters AND buffers (including BN running stats).
    
    θ_teacher = decay * θ_teacher + (1 - decay) * θ_student
    
    CRITICAL: Must update buffers (running_mean, running_var) too,
    otherwise teacher BN stats go stale → extreme activations → NaN under AMP.
    """
    # EMA update for trainable parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    # EMA update for buffers (BN running_mean, running_var, num_batches_tracked)
    for (ema_name, ema_buf), (_, src_buf) in zip(
        ema_model.named_buffers(), model.named_buffers()
    ):
        if ema_buf.is_floating_point():
            # running_mean, running_var: smooth EMA update
            ema_buf.mul_(decay).add_(src_buf, alpha=1 - decay)
        else:
            # num_batches_tracked (integer): just copy
            ema_buf.copy_(src_buf)


@torch.no_grad()
def sync_teacher(teacher, student):
    """Fully sync teacher with student — parameters AND buffers.
    
    Used at stage transitions to prevent divergence.
    """
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.copy_(param.data)
    for (_, ema_buf), (_, src_buf) in zip(
        teacher.named_buffers(), student.named_buffers()
    ):
        ema_buf.copy_(src_buf)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_param_groups(model, encoder_lr_scale=0.1, base_lr=1e-4):
    """Get parameter groups with different learning rates.
    
    Encoder uses lower LR (encoder_lr_scale * base_lr)
    Decoder uses base_lr
    
    This is standard practice for fine-tuning pretrained encoders.
    """
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    param_groups = [
        {'params': decoder_params, 'lr': base_lr},
        {'params': encoder_params, 'lr': base_lr * encoder_lr_scale},
    ]
    
    return param_groups


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss (standard for segmentation).
    
    From T1 and his_approach.py — proven effective.
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE loss
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)  # Batch, H, W
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class ConsistencyLoss(nn.Module):
    """MSE consistency loss for Mean Teacher.
    
    Measures agreement between teacher and student predictions
    on unlabelled data. Only applied to confident predictions.
    
    Always computed in FP32 to avoid AMP precision loss.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, student_logits, teacher_logits, conf_threshold=0.95):
        """
        Args:
            student_logits: Student predictions [B, 1, H, W]
            teacher_logits: Teacher predictions [B, 1, H, W] (detached)
            conf_threshold: Only consider pixels where teacher is confident
        
        Returns:
            Scalar consistency loss
        """
        # Force FP32 to avoid AMP precision loss in sigmoid/MSE
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()
        
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)
        
        # Confidence mask: teacher must be confident (near 0 or near 1)
        confidence = torch.abs(teacher_probs - 0.5) * 2  # 0→1 scale
        mask = (confidence > (conf_threshold - 0.5) * 2).float()  # [B, 1, H, W]
        
        # MSE loss only on confident pixels
        mse = self.mse(student_probs, teacher_probs)
        masked_mse = mse * mask
        
        # Avoid division by zero
        num_confident = mask.sum() + 1e-6
        loss = masked_mse.sum() / num_confident
        
        # NaN guard — return 0 if loss is NaN
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
