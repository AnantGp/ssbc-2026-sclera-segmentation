"""Training script for SSBC 2026 Track 2 — DINOv2 + Mean Teacher.

Key differences from t2_train.py (UNet variant):
- DINOv2 ViT-B/14 backbone (frozen, LayerNorm unfrozen)
- bfloat16 autocast (no GradScaler needed on A100)
- Single main stage (decoder + LN training) + optional LoRA stage
- No BatchNorm buffer sync needed (DINOv2 uses LayerNorm)
- torch.compile for throughput
- Gradient accumulation = 4

Training pipeline:
1. Main stage (epochs 1-150): Train decoder + LayerNorm with Mean Teacher
2. SWA (epochs 120-150): Stochastic Weight Averaging
3. Optional LoRA stage: Fine-tune last 4 transformer blocks with LoRA
"""
import os
import sys
import math
import json
import random
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg
from t2_dataset import (
    build_labelled_pairs, build_unlabelled_paths, split_train_val,
    LabelledDataset, UnlabelledDataset, ValidationDataset
)
from t2_dino_model import (
    build_dino_model, create_ema_model, update_ema, sync_teacher,
    count_parameters, get_param_groups,
    DiceBCELoss, ConsistencyLoss, compute_metrics
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sigmoid_rampup(current, rampup_length):
    """Sigmoid ramp-up from 0 to 1."""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, rampup_epochs, max_weight):
    """Get consistency loss weight for current epoch."""
    return max_weight * sigmoid_rampup(epoch, rampup_epochs)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(log_file):
    """Setup dual logging to file and stdout."""
    import logging
    logger = logging.getLogger('t2_dino')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    student, teacher,
    train_loader_l, train_loader_u,
    optimizer, criterion_sup, criterion_cons,
    epoch, device,
    consistency_weight, conf_threshold,
    grad_accum_steps=1, use_bf16=True
):
    """Train one epoch with Mean Teacher + bfloat16."""
    student.train()
    teacher.eval()

    loss_meter = AverageMeter()
    sup_loss_meter = AverageMeter()
    cons_loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    unlabelled_iter = iter(train_loader_u)
    pbar = tqdm(train_loader_l, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for step, (images_l, masks_l) in enumerate(pbar):
        # Get unlabelled batch
        try:
            weak_u, strong_u = next(unlabelled_iter)
        except StopIteration:
            unlabelled_iter = iter(train_loader_u)
            weak_u, strong_u = next(unlabelled_iter)

        images_l = images_l.to(device, non_blocking=True)
        masks_l = masks_l.to(device, non_blocking=True)
        weak_u = weak_u.to(device, non_blocking=True)
        strong_u = strong_u.to(device, non_blocking=True)

        # ── Supervised loss on labelled data ──
        with torch.autocast('cuda', dtype=amp_dtype):
            logits_l = student(images_l)
            sup_loss = criterion_sup(logits_l, masks_l)

        # ── Consistency loss on unlabelled data ──
        with torch.no_grad():
            with torch.autocast('cuda', dtype=amp_dtype):
                teacher_logits = teacher(weak_u)

        with torch.autocast('cuda', dtype=amp_dtype):
            student_logits = student(strong_u)

        # Consistency always in FP32
        cons_loss = criterion_cons(student_logits, teacher_logits.detach(), conf_threshold)

        # ── Total loss ──
        total_loss = sup_loss.float() + consistency_weight * cons_loss
        total_loss = total_loss / grad_accum_steps

        total_loss.backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Update EMA teacher
            update_ema(teacher, student, decay=cfg.EMA_DECAY)

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(logits_l.float(), masks_l)

        batch_size = images_l.size(0)
        loss_meter.update(total_loss.item() * grad_accum_steps, batch_size)
        sup_loss_meter.update(sup_loss.item(), batch_size)
        cons_loss_meter.update(cons_loss.item(), weak_u.size(0))
        dice_meter.update(metrics['dice'], batch_size)

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'sup': f'{sup_loss_meter.avg:.4f}',
            'cons': f'{cons_loss_meter.avg:.4f}',
            'dice': f'{dice_meter.avg:.4f}',
        })

    return {
        'loss': loss_meter.avg,
        'sup_loss': sup_loss_meter.avg,
        'cons_loss': cons_loss_meter.avg,
        'dice': dice_meter.avg,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_bf16=True):
    """Validate model on validation set."""
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    f1_meter = AverageMeter()

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)

        metrics = compute_metrics(logits.float(), masks)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(metrics['dice'], batch_size)
        iou_meter.update(metrics['iou'], batch_size)
        f1_meter.update(metrics['f1'], batch_size)

    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'iou': iou_meter.avg,
        'f1': f1_meter.avg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train():
    """Main training function: DINOv2 + Mean Teacher."""

    # ── Setup ──
    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = cfg.LOG_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file)
    logger.info(f"Device: {device}")
    logger.info(f"Backbone: {cfg.BACKBONE}")
    logger.info(f"IMG_SIZE: {cfg.IMG_SIZE}")
    logger.info(f"bfloat16: {cfg.USE_BFLOAT16}")
    logger.info(f"torch.compile: {cfg.USE_COMPILE}")
    logger.info(f"Grad accum: {cfg.GRAD_ACCUM_STEPS}")

    # ── Load data ──
    logger.info("\n=== Loading Data ===")
    print("Labelled datasets:")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg.T2_LABELLED)
    print(f"Total labelled: {len(labelled_pairs)}")

    print("\nUnlabelled datasets:")
    unlabelled_paths = build_unlabelled_paths(cfg.T2_UNLABELLED, labelled_pairs)
    print(f"Total unlabelled: {len(unlabelled_paths)}")

    train_pairs, val_pairs, train_labels = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg.VAL_SPLIT, seed=cfg.SEED
    )
    logger.info(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ── Create datasets (reusing t2_dataset with DINOv2 img_size) ──
    train_dataset_l = LabelledDataset(train_pairs, img_size=cfg.IMG_SIZE)
    train_dataset_u = UnlabelledDataset(unlabelled_paths, img_size=cfg.IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=cfg.IMG_SIZE)

    train_loader_l = DataLoader(
        train_dataset_l, batch_size=cfg.BATCH_SIZE_L,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    train_loader_u = DataLoader(
        train_dataset_u, batch_size=cfg.BATCH_SIZE_U,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE_L * 2,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    # ── Build models ──
    logger.info("\n=== Building DINOv2 Model ===")
    student = build_dino_model(
        backbone_name=cfg.BACKBONE,
        feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM,
        unfreeze_ln=cfg.UNFREEZE_LN,
    )
    student = student.to(device)

    # torch.compile for throughput
    if cfg.USE_COMPILE and hasattr(torch, 'compile'):
        logger.info("Applying torch.compile to student model...")
        student = torch.compile(student)

    teacher = create_ema_model(student)
    teacher = teacher.to(device)

    total, trainable = count_parameters(student)
    logger.info(f"Total params: {total:,}")
    logger.info(f"Trainable params: {trainable:,}")
    logger.info(f"Frozen backbone params: {total - trainable:,}")

    # Log param breakdown
    backbone_trainable = sum(
        p.numel() for n, p in student.named_parameters()
        if p.requires_grad and 'backbone' in n
    )
    decoder_trainable = trainable - backbone_trainable
    logger.info(f"  Backbone LayerNorm: {backbone_trainable:,} trainable")
    logger.info(f"  Decoder + Head: {decoder_trainable:,} trainable")

    # ── Loss functions ──
    criterion_sup = DiceBCELoss()
    criterion_cons = ConsistencyLoss()

    # ── Optimizer with differential LR ──
    param_groups = get_param_groups(
        student, base_lr=cfg.BASE_LR, ln_lr_scale=cfg.LN_LR_SCALE
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)

    # Warmup + cosine schedule
    warmup_epochs = 5
    def warmup_cosine_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(cfg.NUM_EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    # ── Training state ──
    best_dice = 0
    best_f1 = 0
    patience_counter = 0
    history = []

    # SWA setup
    swa_model = None
    swa_scheduler = None

    logger.info(f"\n=== Training: {cfg.NUM_EPOCHS} epochs ===")
    logger.info(f"Effective batch: L={cfg.BATCH_SIZE_L * cfg.GRAD_ACCUM_STEPS}, "
                f"U={cfg.BATCH_SIZE_U * cfg.GRAD_ACCUM_STEPS}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        cons_weight = get_current_consistency_weight(
            epoch, cfg.RAMPUP_EPOCHS, cfg.CONSISTENCY_WEIGHT
        )

        train_log = train_one_epoch(
            student, teacher, train_loader_l, train_loader_u,
            optimizer, criterion_sup, criterion_cons,
            epoch, device,
            cons_weight, cfg.CONF_THRESHOLD,
            cfg.GRAD_ACCUM_STEPS, cfg.USE_BFLOAT16
        )

        val_log = validate(student, val_loader, criterion_sup, device, cfg.USE_BFLOAT16)

        # SWA
        if epoch >= cfg.SWA_START:
            if swa_model is None:
                logger.info(f"\n>>> Starting SWA at epoch {epoch}")
                swa_model = AveragedModel(student)
                swa_scheduler = SWALR(optimizer, swa_lr=cfg.SWA_LR)
            swa_model.update_parameters(student)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch}/{cfg.NUM_EPOCHS} | "
            f"LR: {lr:.6f} | ConsW: {cons_weight:.3f} | "
            f"Train Loss: {train_log['loss']:.4f} | Train Dice: {train_log['dice']:.4f} | "
            f"Val Loss: {val_log['loss']:.4f} | Val Dice: {val_log['dice']:.4f} | "
            f"Val F1: {val_log['f1']:.4f}"
        )
        logger.info(log_msg)

        history.append({
            'epoch': epoch, 'lr': lr,
            **{f'train_{k}': v for k, v in train_log.items()},
            **{f'val_{k}': v for k, v in val_log.items()},
        })

        # Save best
        if val_log['dice'] > best_dice:
            best_dice = val_log['dice']
            best_f1 = val_log['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_f1': best_f1,
                'config': {
                    'backbone': cfg.BACKBONE,
                    'img_size': cfg.IMG_SIZE,
                    'feature_layers': cfg.FEATURE_LAYERS,
                    'decoder_dim': cfg.DECODER_DIM,
                },
            }, cfg.MODEL_DIR / 'best_model.pth')
            logger.info(f"  → New best! Dice: {best_dice:.4f}, F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                logger.info(f"\nEarly stopping at epoch {epoch} (patience={cfg.PATIENCE})")
                break

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, cfg.MODEL_DIR / f'checkpoint_ep{epoch}.pth')

    # ── Finalize SWA ──
    if swa_model is not None:
        logger.info("\n=== Finalizing SWA ===")
        torch.optim.swa_utils.update_bn(train_loader_l, swa_model, device)
        swa_val_log = validate(swa_model, val_loader, criterion_sup, device, cfg.USE_BFLOAT16)
        logger.info(f"SWA Val: Dice: {swa_val_log['dice']:.4f}, F1: {swa_val_log['f1']:.4f}")

        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'dice': swa_val_log['dice'],
            'f1': swa_val_log['f1'],
        }, cfg.MODEL_DIR / 'swa_model.pth')

        if swa_val_log['dice'] > best_dice:
            logger.info("  → SWA model is best! Saving as best_model.pth")
            torch.save({
                'model_state_dict': swa_model.module.state_dict(),
                'dice': swa_val_log['dice'],
                'f1': swa_val_log['f1'],
                'config': {
                    'backbone': cfg.BACKBONE,
                    'img_size': cfg.IMG_SIZE,
                    'feature_layers': cfg.FEATURE_LAYERS,
                    'decoder_dim': cfg.DECODER_DIM,
                },
            }, cfg.MODEL_DIR / 'best_model.pth')
            best_dice = swa_val_log['dice']
            best_f1 = swa_val_log['f1']
    else:
        # No SWA — best_model already saved during training
        pass

    # ── Save training history ──
    with open(cfg.LOG_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best Dice: {best_dice:.4f}")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to: {cfg.MODEL_DIR / 'best_model.pth'}")

    return best_dice, best_f1


if __name__ == '__main__':
    train()
