"""Full-data retraining for SSBC 2026 Track 2 submission.

Trains on ALL labelled data (no validation holdout) for final submission models.
Since there's no val set, we:
- Disable early stopping
- Train for fixed epochs (same as original run that produced best model)
- Save checkpoints every 10 epochs + final epoch + SWA
- Use train Dice as progress indicator only

Usage:
    python t2_fulltrain.py            # ViT-B (saves to t2_dino_full/)
    python t2_fulltrain.py --large    # ViT-L (saves to t2_dino_large_full/)

Background:
    Original ViT-B run: best val F1=0.9540 at ~epoch 95, SWA final F1=0.9537
    Original ViT-L run: best val F1=0.9510 at ~epoch 90, SWA final F1=~0.9508
    Ensemble (ViT-B+L student, ms_hflip): F1=0.9564 — current submission config
"""
import os
import sys
import math
import json
import random
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


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


def setup_logging(log_file):
    import logging
    logger = logging.getLogger('t2_fulltrain')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger


def train_one_epoch(
    student, teacher,
    train_loader_l, train_loader_u,
    optimizer, criterion_sup, criterion_cons,
    epoch, num_epochs, device,
    consistency_weight, conf_threshold,
    grad_accum_steps=1, use_bf16=True
):
    student.train()
    teacher.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    unlabelled_iter = iter(train_loader_u)
    pbar = tqdm(train_loader_l, desc=f"Epoch {epoch}/{num_epochs}")
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
        total_loss = sup_loss.float() + consistency_weight * cons_loss
        total_loss = total_loss / grad_accum_steps
        total_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            from t2_dino_model import update_ema
            update_ema(teacher, student, decay=cfg.EMA_DECAY)

        with torch.no_grad():
            from t2_dino_model import compute_metrics
            metrics = compute_metrics(logits_l.float(), masks_l)

        batch_size = images_l.size(0)
        loss_meter.update(total_loss.item() * grad_accum_steps, batch_size)
        dice_meter.update(metrics['dice'], batch_size)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'dice': f'{dice_meter.avg:.4f}'})

    return {'loss': loss_meter.avg, 'dice': dice_meter.avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true', help='Use ViT-L instead of ViT-B')
    args = parser.parse_args()

    global cfg
    if args.large:
        import t2_dino_large_config as cfg
        full_model_dir = Path('/home/jupyter/sclera/models/t2_dino_large_full')
        full_output_dir = Path('/home/jupyter/sclera/outputs/t2_dino_large_full')
        variant = 'ViT-L'
        # ViT-L: original best was around epoch 90-100, total 120 epochs
        TRAIN_EPOCHS = 120
        SWA_START = 90
    else:
        import t2_dino_config as cfg
        full_model_dir = Path('/home/jupyter/sclera/models/t2_dino_full')
        full_output_dir = Path('/home/jupyter/sclera/outputs/t2_dino_full')
        variant = 'ViT-B'
        # ViT-B: original best was around epoch 95-100, total 150 epochs
        TRAIN_EPOCHS = 150
        SWA_START = 120

    full_model_dir.mkdir(parents=True, exist_ok=True)
    full_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = full_output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file = log_dir / f'fulltrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file)
    logger.info(f"=== Full-Data Retrain: {variant} ===")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {TRAIN_EPOCHS}, SWA starts: {SWA_START}")
    logger.info(f"Model dir: {full_model_dir}")
    logger.info("No validation — training on ALL labelled data for submission")

    # ── Load ALL labelled data (no val split) ──
    from t2_dataset import (
        build_labelled_pairs, build_unlabelled_paths, split_train_val,
        LabelledDataset, UnlabelledDataset
    )

    logger.info("\n=== Loading Data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg.T2_LABELLED)
    unlabelled_paths = build_unlabelled_paths(cfg.T2_UNLABELLED, labelled_pairs)

    # Use ALL labelled data — no val holdout
    train_pairs = labelled_pairs
    logger.info(f"Train (ALL labelled): {len(train_pairs)}")
    logger.info(f"Unlabelled: {len(unlabelled_paths)}")

    train_dataset_l = LabelledDataset(train_pairs, img_size=cfg.IMG_SIZE)
    train_dataset_u = UnlabelledDataset(unlabelled_paths, img_size=cfg.IMG_SIZE)

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

    # ── Build model ──
    logger.info("\n=== Building Model ===")
    from t2_dino_model import (
        build_dino_model, create_ema_model, count_parameters,
        get_param_groups, DiceBCELoss, ConsistencyLoss
    )

    student = build_dino_model(
        backbone_name=cfg.BACKBONE,
        feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM,
        unfreeze_ln=cfg.UNFREEZE_LN,
    ).to(device)

    if cfg.USE_COMPILE and hasattr(torch, 'compile'):
        logger.info("Applying torch.compile...")
        student = torch.compile(student)

    teacher = create_ema_model(student).to(device)

    total, trainable = count_parameters(student)
    logger.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    criterion_sup = DiceBCELoss()
    criterion_cons = ConsistencyLoss()

    param_groups = get_param_groups(student, base_lr=cfg.BASE_LR, ln_lr_scale=cfg.LN_LR_SCALE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)

    warmup_epochs = 5
    def warmup_cosine_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(TRAIN_EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    # ── Training ──
    logger.info(f"\n=== Training: {TRAIN_EPOCHS} epochs ===")
    logger.info(f"Effective batch: L={cfg.BATCH_SIZE_L * cfg.GRAD_ACCUM_STEPS}, "
                f"U={cfg.BATCH_SIZE_U * cfg.GRAD_ACCUM_STEPS}")

    history = []
    swa_model = None
    swa_scheduler = None
    best_train_dice = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        cons_weight = cfg.CONSISTENCY_WEIGHT * sigmoid_rampup(epoch, cfg.RAMPUP_EPOCHS)

        train_log = train_one_epoch(
            student, teacher, train_loader_l, train_loader_u,
            optimizer, criterion_sup, criterion_cons,
            epoch, TRAIN_EPOCHS, device,
            cons_weight, cfg.CONF_THRESHOLD,
            cfg.GRAD_ACCUM_STEPS, cfg.USE_BFLOAT16
        )

        # SWA
        if epoch >= SWA_START:
            if swa_model is None:
                logger.info(f"\n>>> Starting SWA at epoch {epoch}")
                swa_model = AveragedModel(student)
                swa_scheduler = SWALR(optimizer, swa_lr=cfg.SWA_LR)
            swa_model.update_parameters(student)
            swa_scheduler.step()
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{TRAIN_EPOCHS} | LR={lr:.6f} | ConsW={cons_weight:.3f} | "
            f"Train Loss={train_log['loss']:.4f} | Train Dice={train_log['dice']:.4f}"
        )

        history.append({'epoch': epoch, 'lr': lr, **{f'train_{k}': v for k, v in train_log.items()}})

        # Track best train dice (just for logging)
        if train_log['dice'] > best_train_dice:
            best_train_dice = train_log['dice']

        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_dice': train_log['dice'],
                'config': {
                    'backbone': cfg.BACKBONE,
                    'img_size': cfg.IMG_SIZE,
                    'feature_layers': cfg.FEATURE_LAYERS,
                    'decoder_dim': cfg.DECODER_DIM,
                },
            }, full_model_dir / f'checkpoint_ep{epoch}.pth')

    # ── Save final student model ──
    logger.info("\n=== Saving Final Models ===")
    torch.save({
        'epoch': TRAIN_EPOCHS,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'train_dice': best_train_dice,
        'config': {
            'backbone': cfg.BACKBONE,
            'img_size': cfg.IMG_SIZE,
            'feature_layers': cfg.FEATURE_LAYERS,
            'decoder_dim': cfg.DECODER_DIM,
        },
    }, full_model_dir / 'final_model.pth')
    logger.info(f"Final student saved: final_model.pth")

    # ── Finalize SWA ──
    if swa_model is not None:
        logger.info("Finalizing SWA (update BN stats)...")
        # DINOv2 uses LayerNorm, not BatchNorm — update_bn is a no-op but call anyway
        torch.optim.swa_utils.update_bn(train_loader_l, swa_model, device)

        # Strip compile prefix for SWA module
        from t2_dino_eval import strip_compile_prefix
        swa_state = strip_compile_prefix(swa_model.module.state_dict())

        torch.save({
            'student_state_dict': swa_state,
            'config': {
                'backbone': cfg.BACKBONE,
                'img_size': cfg.IMG_SIZE,
                'feature_layers': cfg.FEATURE_LAYERS,
                'decoder_dim': cfg.DECODER_DIM,
            },
        }, full_model_dir / 'swa_model.pth')
        logger.info(f"SWA model saved: swa_model.pth")

        # Use SWA as the primary submission model (consistent with original training)
        torch.save({
            'student_state_dict': swa_state,
            'config': {
                'backbone': cfg.BACKBONE,
                'img_size': cfg.IMG_SIZE,
                'feature_layers': cfg.FEATURE_LAYERS,
                'decoder_dim': cfg.DECODER_DIM,
            },
        }, full_model_dir / 'best_model.pth')
        logger.info(f"SWA saved as best_model.pth (primary submission model)")
    else:
        # No SWA — save final as best
        import shutil
        shutil.copy(full_model_dir / 'final_model.pth', full_model_dir / 'best_model.pth')
        logger.info("No SWA — final model saved as best_model.pth")

    # ── Save history ──
    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n=== Full-Data Retrain Complete ({variant}) ===")
    logger.info(f"Best train Dice: {best_train_dice:.4f}")
    logger.info(f"Models saved to: {full_model_dir}")
    logger.info(f"  best_model.pth — primary submission model (SWA if available)")
    logger.info(f"  swa_model.pth  — SWA averaged weights")
    logger.info(f"  final_model.pth — final epoch student weights")


if __name__ == '__main__':
    main()
