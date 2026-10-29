"""LoRA fine-tuning stage for DINOv2 T2 model (CSIRO 4th place technique).

Loads the best student checkpoint, adds LoRA adapters to the last N
transformer blocks, and fine-tunes with a lower learning rate.

This is Stage 2 of training — runs AFTER the main decoder training is done.
"""
import sys
import math
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg
from t2_dino_model import (
    build_dino_model, create_ema_model, update_ema, sync_teacher,
    count_parameters, DiceBCELoss, ConsistencyLoss, compute_metrics
)
from t2_dataset import (
    build_labelled_pairs, build_unlabelled_paths, split_train_val,
    LabelledDataset, UnlabelledDataset, ValidationDataset
)


# ══════════════════════════════════════════════════════════════════════════════
# LoRA IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers.

    Adds low-rank A*B decomposition: output = original(x) + x @ A^T @ B^T * (alpha/rank)
    """

    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices — A initialized with Kaiming, B with zeros
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out


def inject_lora(model, target_blocks=4, rank=8, alpha=16):
    """Inject LoRA adapters into the last N transformer blocks.

    Targets: qkv projection and output projection in attention,
             and fc1/fc2 in the MLP/SwiGLU FFN.
    """
    blocks = model.backbone.blocks
    num_blocks = len(blocks)
    start_block = num_blocks - target_blocks

    lora_count = 0
    for i in range(start_block, num_blocks):
        block = blocks[i]

        # Attention QKV projection
        if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, nn.Linear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
            lora_count += 1

        # Attention output projection
        if hasattr(block.attn, 'proj') and isinstance(block.attn.proj, nn.Linear):
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha)
            lora_count += 1

        # MLP layers — DINOv2 uses SwiGLU FFN
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            for attr_name in ['w1', 'w2', 'w3', 'w12', 'fc1', 'fc2']:
                if hasattr(mlp, attr_name):
                    layer = getattr(mlp, attr_name)
                    if isinstance(layer, nn.Linear):
                        setattr(mlp, attr_name, LoRALinear(layer, rank=rank, alpha=alpha))
                        lora_count += 1

    print(f"  Injected {lora_count} LoRA adapters into blocks {start_block}-{num_blocks-1}")
    return lora_count


def get_lora_param_groups(model, lora_lr, decoder_lr, ln_lr):
    """Get parameter groups: LoRA params + decoder + LayerNorm."""
    lora_params = []
    decoder_params = []
    ln_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(param)
        elif 'backbone' in name:
            ln_params.append(param)
        else:
            decoder_params.append(param)

    return [
        {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'},
        {'params': lora_params, 'lr': lora_lr, 'name': 'lora'},
        {'params': ln_params, 'lr': ln_lr, 'name': 'layernorm'},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(student, teacher, train_loader_l, train_loader_u,
                    optimizer, criterion_sup, criterion_cons,
                    epoch, device, cons_weight, conf_threshold,
                    grad_accum_steps=4, use_bf16=True):
    student.train()
    teacher.eval()

    loss_m = AverageMeter()
    sup_m = AverageMeter()
    cons_m = AverageMeter()
    dice_m = AverageMeter()

    unlabelled_iter = iter(train_loader_u)
    pbar = tqdm(train_loader_l, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for step, (images_l, masks_l) in enumerate(pbar):
        try:
            weak_u, strong_u = next(unlabelled_iter)
        except StopIteration:
            unlabelled_iter = iter(train_loader_u)
            weak_u, strong_u = next(unlabelled_iter)

        images_l = images_l.to(device, non_blocking=True)
        masks_l = masks_l.to(device, non_blocking=True)
        weak_u = weak_u.to(device, non_blocking=True)
        strong_u = strong_u.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=amp_dtype):
            logits_l = student(images_l)
            sup_loss = criterion_sup(logits_l, masks_l)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=amp_dtype):
                teacher_logits = teacher(weak_u)

        with torch.autocast('cuda', dtype=amp_dtype):
            student_logits = student(strong_u)

        cons_loss = criterion_cons(student_logits, teacher_logits.detach(), conf_threshold)
        total_loss = (sup_loss.float() + cons_weight * cons_loss) / grad_accum_steps
        total_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            update_ema(teacher, student, decay=cfg.EMA_DECAY)

        with torch.no_grad():
            metrics = compute_metrics(logits_l.float(), masks_l)

        bs = images_l.size(0)
        loss_m.update(total_loss.item() * grad_accum_steps, bs)
        sup_m.update(sup_loss.item(), bs)
        cons_m.update(cons_loss.item(), weak_u.size(0))
        dice_m.update(metrics['dice'], bs)
        pbar.set_postfix(loss=f'{loss_m.avg:.4f}', sup=f'{sup_m.avg:.4f}',
                         cons=f'{cons_m.avg:.4f}', dice=f'{dice_m.avg:.4f}')

    return {'loss': loss_m.avg, 'sup_loss': sup_m.avg,
            'cons_loss': cons_m.avg, 'dice': dice_m.avg}


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_bf16=True):
    model.eval()
    loss_m, dice_m, iou_m, f1_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)
        metrics = compute_metrics(logits.float(), masks)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        dice_m.update(metrics['dice'], bs)
        iou_m.update(metrics['iou'], bs)
        f1_m.update(metrics['f1'], bs)

    return {'loss': loss_m.avg, 'dice': dice_m.avg, 'iou': iou_m.avg, 'f1': f1_m.avg}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lora_model_dir = cfg.MODEL_DIR / 'lora'
    lora_log_dir = cfg.LOG_DIR / 'lora'
    lora_model_dir.mkdir(parents=True, exist_ok=True)
    lora_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"LoRA rank={cfg.LORA_RANK}, alpha={cfg.LORA_ALPHA}, target_blocks={cfg.LORA_TARGET_BLOCKS}")
    print(f"LoRA LR={cfg.LORA_LR}, epochs={cfg.LORA_EPOCHS}")

    # ── Data ──
    print("\n=== Loading Data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg.T2_LABELLED)
    unlabelled_paths = build_unlabelled_paths(cfg.T2_UNLABELLED, labelled_pairs)
    train_pairs, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels, val_split=cfg.VAL_SPLIT, seed=cfg.SEED
    )
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Unlabelled: {len(unlabelled_paths)}")

    train_dataset_l = LabelledDataset(train_pairs, img_size=cfg.IMG_SIZE)
    train_dataset_u = UnlabelledDataset(unlabelled_paths, img_size=cfg.IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=cfg.IMG_SIZE)

    train_loader_l = DataLoader(train_dataset_l, batch_size=cfg.BATCH_SIZE_L,
                                shuffle=True, num_workers=cfg.NUM_WORKERS,
                                pin_memory=True, drop_last=True)
    train_loader_u = DataLoader(train_dataset_u, batch_size=cfg.BATCH_SIZE_U,
                                shuffle=True, num_workers=cfg.NUM_WORKERS,
                                pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE_L * 2,
                            shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True)

    # ── Load best student ──
    print("\n=== Loading Best Student Model ===")
    student = build_dino_model(
        backbone_name=cfg.BACKBONE, feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM, unfreeze_ln=cfg.UNFREEZE_LN,
    )
    ckpt = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=False)
    # Strip _orig_mod. prefix from torch.compile
    def strip_compile_prefix(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    if 'student_state_dict' in ckpt:
        student.load_state_dict(strip_compile_prefix(ckpt['student_state_dict']))
    elif 'model_state_dict' in ckpt:
        student.load_state_dict(strip_compile_prefix(ckpt['model_state_dict']))
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, F1={ckpt.get('best_f1', '?')}")

    # ── Inject LoRA ──
    print("\n=== Injecting LoRA ===")
    inject_lora(student, target_blocks=cfg.LORA_TARGET_BLOCKS,
                rank=cfg.LORA_RANK, alpha=cfg.LORA_ALPHA)
    student = student.to(device)

    total, trainable = count_parameters(student)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Count LoRA-specific params
    lora_p = sum(p.numel() for n, p in student.named_parameters()
                 if p.requires_grad and 'lora_' in n)
    print(f"  LoRA params: {lora_p:,}")

    # ── Teacher ──
    teacher = create_ema_model(student)
    teacher = teacher.to(device)

    # ── Validate before LoRA ──
    criterion_sup = DiceBCELoss()
    criterion_cons = ConsistencyLoss()

    print("\n=== Pre-LoRA Validation ===")
    pre_val = validate(student, val_loader, criterion_sup, device)
    print(f"  Pre-LoRA: Dice={pre_val['dice']:.4f}, F1={pre_val['f1']:.4f}")

    # ── Optimizer ──
    param_groups = get_lora_param_groups(
        student,
        lora_lr=cfg.LORA_LR,
        decoder_lr=cfg.LORA_LR * 0.5,   # Decoder at half LoRA LR
        ln_lr=cfg.LORA_LR * 0.1,        # LayerNorm even lower
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.LORA_EPOCHS, eta_min=1e-6
    )

    # ── Train ──
    best_f1 = pre_val['f1']
    patience = 15
    patience_counter = 0
    history = []

    print(f"\n=== LoRA Training: {cfg.LORA_EPOCHS} epochs ===")

    for epoch in range(1, cfg.LORA_EPOCHS + 1):
        train_log = train_one_epoch(
            student, teacher, train_loader_l, train_loader_u,
            optimizer, criterion_sup, criterion_cons,
            epoch, device, cons_weight=1.0, conf_threshold=cfg.CONF_THRESHOLD,
            grad_accum_steps=cfg.GRAD_ACCUM_STEPS, use_bf16=cfg.USE_BFLOAT16
        )

        val_log = validate(student, val_loader, criterion_sup, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{cfg.LORA_EPOCHS} | LR: {lr:.6f} | "
              f"Train Loss: {train_log['loss']:.4f} | Train Dice: {train_log['dice']:.4f} | "
              f"Val Loss: {val_log['loss']:.4f} | Val Dice: {val_log['dice']:.4f} | Val F1: {val_log['f1']:.4f}")

        history.append({
            'epoch': epoch, 'lr': lr,
            **{f'train_{k}': v for k, v in train_log.items()},
            **{f'val_{k}': v for k, v in val_log.items()},
        })

        if val_log['f1'] > best_f1:
            best_f1 = val_log['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'best_f1': best_f1,
                'best_dice': val_log['dice'],
                'lora_config': {
                    'rank': cfg.LORA_RANK,
                    'alpha': cfg.LORA_ALPHA,
                    'target_blocks': cfg.LORA_TARGET_BLOCKS,
                },
            }, lora_model_dir / 'best_lora_model.pth')
            print(f"  → New best! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    with open(lora_log_dir / 'lora_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n=== LoRA Training Complete ===")
    print(f"Pre-LoRA F1:  {pre_val['f1']:.4f}")
    print(f"Post-LoRA F1: {best_f1:.4f}")
    print(f"Improvement:  {best_f1 - pre_val['f1']:+.4f}")


if __name__ == '__main__':
    main()
