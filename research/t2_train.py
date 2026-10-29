"""Training script for SSBC 2026 Track 2 — Semi-supervised Mean Teacher.

Design informed by:
- CSIRO 1st place: Two-stage (freeze→unfreeze) + test-time self-distillation (≈ Mean Teacher)
- CSIRO 4th place: Unfreeze BatchNorm for domain adaptation
- CSIRO 6th place: Confidence thresholding to prevent pseudo-label poisoning
- Wang et al. 2025: Semi-supervised sclera segmentation baseline
- Standard Mean Teacher: Tarvainen & Valpola 2017

Training pipeline:
1. Stage 1 (epochs 1-50): Freeze encoder, train decoder + mean teacher
2. Stage 2 (epochs 51-200): Unfreeze encoder with lower LR, continue mean teacher
3. SWA (epochs 150-200): Stochastic Weight Averaging for robustness
"""
import os
import sys
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

import t2_config as cfg
from t2_dataset import (
    build_labelled_pairs, build_unlabelled_paths, split_train_val,
    LabelledDataset, UnlabelledDataset, ValidationDataset
)
from t2_model import (
    build_model, freeze_encoder, unfreeze_encoder,
    create_ema_model, update_ema, sync_teacher,
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
    """Sigmoid ramp-up from 0 to 1 over `rampup_length` epochs.
    
    Used for gradually increasing consistency loss weight.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    student, teacher, 
    train_loader_l, train_loader_u,
    optimizer, criterion_sup, criterion_cons,
    scaler, epoch, device,
    consistency_weight, conf_threshold,
    grad_accum_steps=1
):
    """Train for one epoch with Mean Teacher.
    
    Args:
        student: Student model (trainable)
        teacher: Teacher model (EMA of student)
        train_loader_l: Labelled data loader
        train_loader_u: Unlabelled data loader
        optimizer: Optimizer
        criterion_sup: Supervised loss (DiceBCE)
        criterion_cons: Consistency loss (MSE)
        scaler: Gradient scaler for AMP
        epoch: Current epoch number
        device: cuda/cpu
        consistency_weight: Current weight for consistency loss
        conf_threshold: Confidence threshold for pseudo-labels
        grad_accum_steps: Gradient accumulation steps
    
    Returns:
        Dict of average losses and metrics
    """
    student.train()
    teacher.eval()
    
    loss_meter = AverageMeter()
    sup_loss_meter = AverageMeter()
    cons_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    # Iterate through labelled and unlabelled in parallel
    unlabelled_iter = iter(train_loader_u)
    
    pbar = tqdm(train_loader_l, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
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
        
        # ── Supervised loss on labelled data (under AMP) ──
        with autocast():
            logits_l = student(images_l)
            sup_loss = criterion_sup(logits_l, masks_l)
        
        # ── Consistency loss on unlabelled data ──
        # Teacher: weak augmentation → pseudo-labels (FP32, no autocast)
        with torch.no_grad():
            teacher_logits = teacher(weak_u)
        
        # Student: strong augmentation (under AMP for speed)
        with autocast():
            student_logits = student(strong_u)
        
        # Consistency loss always in FP32 (cast happens inside ConsistencyLoss)
        cons_loss = criterion_cons(student_logits, teacher_logits.detach(), conf_threshold)
        
        # ── Total loss (FP32) ──
        total_loss = sup_loss.float() + consistency_weight * cons_loss
        total_loss = total_loss / grad_accum_steps
        
        # Backward
        scaler.scale(total_loss).backward()
        
        # Step optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping to prevent NaN after encoder unfreeze
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA teacher — params AND buffers (CRITICAL for Mean Teacher)
            update_ema(teacher, student, decay=cfg.EMA_DECAY)
        
        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(logits_l, masks_l)
        
        # Update meters
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
def validate(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    f1_meter = AverageMeter()
    
    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        
        metrics = compute_metrics(logits, masks)
        
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
    """Main training function implementing two-stage Mean Teacher."""
    
    # ── Setup ──
    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Load data ──
    print("\n=== Loading Data ===")
    print("Labelled datasets:")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg.T2_LABELLED)
    print(f"Total labelled: {len(labelled_pairs)}")
    
    print("\nUnlabelled datasets:")
    unlabelled_paths = build_unlabelled_paths(cfg.T2_UNLABELLED, labelled_pairs)
    print(f"Total unlabelled: {len(unlabelled_paths)}")
    
    # Split train/val
    train_pairs, val_pairs, train_labels = split_train_val(
        labelled_pairs, labelled_labels, 
        val_split=cfg.VAL_SPLIT, seed=cfg.SEED
    )
    print(f"\nTrain: {len(train_pairs)} | Val: {len(val_pairs)}")
    
    # ── Create datasets ──
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
    print("\n=== Building Models ===")
    student = build_model(encoder=cfg.ENCODER, encoder_weights=cfg.ENCODER_WEIGHTS)
    student = student.to(device)
    
    teacher = create_ema_model(student)
    teacher = teacher.to(device)
    
    total, trainable = count_parameters(student)
    print(f"Student params: {total:,} total, {trainable:,} trainable")
    
    # ── Loss functions ──
    criterion_sup = DiceBCELoss()
    criterion_cons = ConsistencyLoss()
    
    # ── Training state ──
    best_dice = 0
    best_f1 = 0
    patience_counter = 0
    history = []
    skip_stage1 = False
    
    # ── Resume from Stage 1 checkpoint if available ──
    stage1_ckpt = cfg.MODEL_DIR / 'best_stage1.pth'
    if stage1_ckpt.exists():
        print(f"\n=== Found Stage 1 checkpoint, resuming from Stage 2 ===")
        ckpt = torch.load(stage1_ckpt, map_location=device)
        student.load_state_dict(ckpt['student_state_dict'])
        teacher.load_state_dict(ckpt['teacher_state_dict'])
        best_dice = ckpt.get('best_dice', 0)
        best_f1 = ckpt.get('best_f1', 0)
        print(f"  Loaded: epoch {ckpt['epoch']}, best_dice={best_dice:.4f}, best_f1={best_f1:.4f}")
        skip_stage1 = True
    
    scaler = GradScaler()
    
    # ── Stage 1: Frozen encoder ──
    if not skip_stage1:
        print("\n=== Stage 1: Frozen Encoder ===")
        freeze_encoder(student)
        total, trainable = count_parameters(student)
        print(f"Stage 1 params: {trainable:,} trainable")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, student.parameters()),
            lr=cfg.STAGE1_LR, weight_decay=cfg.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.STAGE1_EPOCHS
        )
        
        for epoch in range(1, cfg.STAGE1_EPOCHS + 1):
            cons_weight = get_current_consistency_weight(
                epoch, cfg.RAMPUP_EPOCHS, cfg.CONSISTENCY_WEIGHT
            )
            
            train_log = train_one_epoch(
                student, teacher, train_loader_l, train_loader_u,
                optimizer, criterion_sup, criterion_cons,
                scaler, epoch, device,
                cons_weight, cfg.CONF_THRESHOLD, cfg.GRAD_ACCUM_STEPS
            )
            
            val_log = validate(student, val_loader, criterion_sup, device)
            scheduler.step()
            
            # Log
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{cfg.STAGE1_EPOCHS} | "
                  f"LR: {lr:.6f} | ConsW: {cons_weight:.3f} | "
                  f"Train Loss: {train_log['loss']:.4f} | Train Dice: {train_log['dice']:.4f} | "
                  f"Val Loss: {val_log['loss']:.4f} | Val Dice: {val_log['dice']:.4f} | Val F1: {val_log['f1']:.4f}")
            
            history.append({
                'epoch': epoch, 'stage': 1, 'lr': lr,
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
                }, cfg.MODEL_DIR / 'best_stage1.pth')
                print(f"  → New best! Dice: {best_dice:.4f}, F1: {best_f1:.4f}")
            else:
                patience_counter += 1
    
    # ── Stage 2: Unfrozen encoder with lower LR ──
    print("\n=== Stage 2: Unfrozen Encoder ===")
    
    # Sync teacher with student before unfreeze — params AND buffers
    sync_teacher(teacher, student)
    print("  Teacher synced with student (params + buffers)")
    
    unfreeze_encoder(student, unfreeze_bn=cfg.UNFREEZE_BN)
    total, trainable = count_parameters(student)
    print(f"Stage 2 params: {trainable:,} trainable")
    
    # New optimizer with differential LR (halved to prevent NaN)
    param_groups = get_param_groups(
        student, encoder_lr_scale=cfg.ENCODER_LR_SCALE, base_lr=cfg.STAGE2_LR
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    
    remaining_epochs = cfg.NUM_EPOCHS - cfg.STAGE1_EPOCHS
    # Warmup + cosine: 5 epochs linear warmup then cosine decay
    warmup_epochs = 5
    def warmup_cosine_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(remaining_epochs - warmup_epochs, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    
    # SWA setup
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = cfg.SWA_START
    
    for epoch in range(cfg.STAGE1_EPOCHS + 1, cfg.NUM_EPOCHS + 1):
        cons_weight = get_current_consistency_weight(
            epoch, cfg.RAMPUP_EPOCHS, cfg.CONSISTENCY_WEIGHT
        )
        
        train_log = train_one_epoch(
            student, teacher, train_loader_l, train_loader_u,
            optimizer, criterion_sup, criterion_cons,
            scaler, epoch, device,
            cons_weight, cfg.CONF_THRESHOLD, cfg.GRAD_ACCUM_STEPS
        )
        
        val_log = validate(student, val_loader, criterion_sup, device)
        
        # SWA
        if epoch >= swa_start_epoch:
            if swa_model is None:
                print(f"\n>>> Starting SWA at epoch {epoch}")
                swa_model = AveragedModel(student)
                swa_scheduler = SWALR(optimizer, swa_lr=cfg.SWA_LR)
            swa_model.update_parameters(student)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{cfg.NUM_EPOCHS} | "
              f"LR: {lr:.6f} | ConsW: {cons_weight:.3f} | "
              f"Train Loss: {train_log['loss']:.4f} | Train Dice: {train_log['dice']:.4f} | "
              f"Val Loss: {val_log['loss']:.4f} | Val Dice: {val_log['dice']:.4f} | Val F1: {val_log['f1']:.4f}")
        
        history.append({
            'epoch': epoch, 'stage': 2, 'lr': lr,
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
            }, cfg.MODEL_DIR / 'best_student.pth')
            print(f"  → New best! Dice: {best_dice:.4f}, F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save checkpoint every 10 epochs
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
        print("\n=== Finalizing SWA ===")
        # Update BN statistics
        torch.optim.swa_utils.update_bn(train_loader_l, swa_model, device)
        
        # Validate SWA model
        swa_val_log = validate(swa_model, val_loader, criterion_sup, device)
        print(f"SWA Val: Dice: {swa_val_log['dice']:.4f}, F1: {swa_val_log['f1']:.4f}")
        
        # Save SWA model
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'dice': swa_val_log['dice'],
            'f1': swa_val_log['f1'],
        }, cfg.MODEL_DIR / 'swa_model.pth')
        
        if swa_val_log['dice'] > best_dice:
            print("  → SWA model is best! Saving as best_model.pth")
            torch.save({
                'model_state_dict': swa_model.module.state_dict(),
                'dice': swa_val_log['dice'],
                'f1': swa_val_log['f1'],
            }, cfg.MODEL_DIR / 'best_model.pth')
            best_dice = swa_val_log['dice']
            best_f1 = swa_val_log['f1']
    else:
        # Copy best student as final model
        import shutil
        best_src = cfg.MODEL_DIR / 'best_student.pth'
        if not best_src.exists():
            best_src = cfg.MODEL_DIR / 'best_stage1.pth'
        if best_src.exists():
            shutil.copy(best_src, cfg.MODEL_DIR / 'best_model.pth')
        else:
            print("WARNING: No best checkpoint found to copy")
    
    # ── Save history ──
    import json
    with open(cfg.LOG_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Model saved to: {cfg.MODEL_DIR / 'best_model.pth'}")
    
    return best_dice, best_f1


if __name__ == '__main__':
    train()
