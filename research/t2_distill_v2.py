"""Pseudo-label distillation v2 — fine-tune from pre-trained weights with balanced sampling.

Fixes from v1:
1. Loads pre-trained ViT-B student weights (not from scratch)
2. Balanced sampling: ~50/50 real/pseudo per batch via WeightedRandomSampler
3. Higher confidence threshold (0.95) — drops noisy MOBIUS labels
4. Much lower LR (5e-5) for fine-tuning
5. Fewer epochs (30, patience 10)
6. Lower pseudo-label loss weight (0.3)

Usage:
  python t2_distill_v2.py             # train (uses existing pseudo-labels from v1)
  python t2_distill_v2.py --conf 0.90 # custom confidence threshold
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

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg_b
from t2_dino_model import (
    build_dino_model, count_parameters, get_param_groups,
    DiceBCELoss, compute_metrics
)
from t2_dino_eval import strip_compile_prefix
from t2_dataset import (
    build_labelled_pairs, split_train_val,
    ValidationDataset, get_val_transforms
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths ──
PSEUDO_DIR = Path("/home/jupyter/sclera/data/T2_pseudo_labels")
MODEL_DIR = Path("/home/jupyter/sclera/models/t2_distill")
PRETRAINED_PATH = cfg_b.MODEL_DIR / 'best_model.pth'  # Pre-trained ViT-B student
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_distill")
LOG_DIR = OUTPUT_DIR / "logs"

# ── Config ──
IMG_SIZE = 518
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4        # Effective batch = 32
NUM_WORKERS = 4
BASE_LR = 5e-5              # Much lower — fine-tuning
LN_LR_SCALE = 0.1
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
PATIENCE = 10
SEED = 42
USE_BFLOAT16 = True
SWA_START = 20
SWA_LR = 1e-5

# Pseudo-label config
CONFIDENCE_THRESHOLD = 0.95  # High quality only (drops MOBIUS)
PSEUDO_LOSS_WEIGHT = 0.3     # Lower weight for pseudo-labels


class DistillDataset(Dataset):
    """Dataset combining real + pseudo-labelled data with per-sample weights."""

    def __init__(self, real_pairs, pseudo_pairs, img_size=518, augment=True):
        self.pairs = []
        self.weights = []
        self.is_real = []

        for img_p, mask_p in real_pairs:
            self.pairs.append((str(img_p), str(mask_p)))
            self.weights.append(1.0)
            self.is_real.append(True)

        for img_p, pseudo_p in pseudo_pairs:
            self.pairs.append((str(img_p), str(pseudo_p)))
            self.weights.append(PSEUDO_LOSS_WEIGHT)
            self.is_real.append(False)

        self.img_size = img_size
        if augment:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                A.CenterCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
                A.GaussNoise(var_limit=(5, 25), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                A.CenterCrop(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        weight = self.weights[idx]

        image = cv2.imread(img_path)
        if image is None:
            # Fallback: return a random valid sample
            return self.__getitem__(random.randint(0, len(self) - 1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask = mask.astype(np.float32) / 255.0

        out = self.transform(image=image, mask=mask)
        image_t = out['image']
        mask_t = out['mask'].unsqueeze(0).float()

        return image_t, mask_t, torch.tensor(weight, dtype=torch.float32)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


class WeightedDiceBCELoss(nn.Module):
    """DiceBCE loss with per-sample weighting."""
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, weights=None):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        bce = bce.mean(dim=(1, 2, 3))

        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        loss_per_sample = self.bce_weight * bce + self.dice_weight * dice_loss
        if weights is not None:
            loss_per_sample = loss_per_sample * weights
        return loss_per_sample.mean()


def train(device, conf_threshold=CONFIDENCE_THRESHOLD):
    """Fine-tune pre-trained ViT-B with labelled + high-conf pseudo-labelled data."""
    print(f"\n=== Distillation v2: Fine-tune with balanced sampling ===")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  Learning rate: {BASE_LR}")
    print(f"  Pseudo loss weight: {PSEUDO_LOSS_WEIGHT}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pseudo-label metadata ──
    meta_path = PSEUDO_DIR / 'metadata.json'
    if not meta_path.exists():
        print("ERROR: No pseudo-labels found. Run t2_distill.py --generate first.")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    # Filter by confidence
    all_pseudo = meta['pairs']
    filtered = [p for p in all_pseudo if p['confidence'] >= conf_threshold]
    pseudo_pairs = [(p['image'], p['pseudo_label']) for p in filtered]
    print(f"\n  Pseudo-labels: {len(filtered)}/{len(all_pseudo)} pass conf>{conf_threshold}")

    # Per-dataset breakdown
    from collections import Counter
    ds_counts = Counter(p['dataset'] for p in filtered)
    for ds in sorted(ds_counts):
        print(f"    {ds}: {ds_counts[ds]}")

    # ── Real labelled data ──
    print("\n  Real labelled data:")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    train_pairs, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg_b.VAL_SPLIT, seed=SEED
    )
    print(f"    Train: {len(train_pairs)}")
    print(f"    Val: {len(val_pairs)}")

    # ── Dataset with balanced sampling ──
    train_dataset = DistillDataset(train_pairs, pseudo_pairs, img_size=IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=IMG_SIZE)

    n_real = len(train_pairs)
    n_pseudo = len(pseudo_pairs)
    n_total = n_real + n_pseudo

    # WeightedRandomSampler: give real samples much higher sampling weight
    # so each batch is ~50% real, ~50% pseudo
    sample_weights = []
    for i in range(n_total):
        if train_dataset.is_real[i]:
            # Real samples: weight = n_pseudo / n_real (oversample)
            sample_weights.append(n_pseudo / n_real)
        else:
            # Pseudo samples: weight = 1.0
            sample_weights.append(1.0)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_total,  # Full epoch = n_total samples
        replacement=True,     # Must be True (we oversample real)
    )

    print(f"\n  Total training samples per epoch: {n_total}")
    print(f"  Real oversampling factor: {n_pseudo/n_real:.1f}x")
    print(f"  Expected batch composition: ~50% real, ~50% pseudo")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model: load pre-trained ViT-B student ──
    print(f"\n  Building DINOv2 ViT-B model...")
    model = build_dino_model(
        backbone_name=cfg_b.BACKBONE,
        feature_layers=cfg_b.FEATURE_LAYERS,
        decoder_dim=cfg_b.DECODER_DIM,
        unfreeze_ln=True,
    )

    print(f"  Loading pre-trained weights from {PRETRAINED_PATH}")
    ckpt = torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=False)
    state_dict = strip_compile_prefix(ckpt['student_state_dict'])
    model.load_state_dict(state_dict)
    pretrain_f1 = ckpt.get('best_f1', 'N/A')
    print(f"  Pre-trained model F1: {pretrain_f1}")

    model = model.to(device)

    if hasattr(torch, 'compile'):
        print("  Applying torch.compile...")
        model = torch.compile(model)

    total, trainable = count_parameters(model)
    print(f"  Params: {total:,} total, {trainable:,} trainable")

    # ── Loss / Optimizer ──
    criterion = WeightedDiceBCELoss()
    val_criterion = DiceBCELoss()

    param_groups = get_param_groups(model, base_lr=BASE_LR, ln_lr_scale=LN_LR_SCALE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 2
    def warmup_cosine(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(NUM_EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

    # ── Training loop ──
    best_f1 = 0
    patience_counter = 0
    swa_model = None
    swa_scheduler = None

    print(f"\n  Starting training: {NUM_EPOCHS} epochs")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Steps per epoch: {len(train_loader)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for step, (images, masks, weights) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits.float(), masks, weights)

            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                metrics = compute_metrics(logits.float(), masks)

            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(metrics['dice'], images.size(0))
            pbar.set_postfix(loss=f'{loss_meter.avg:.4f}', dice=f'{dice_meter.avg:.4f}')

        # ── Validate ──
        model.eval()
        val_loss_m = AverageMeter()
        val_f1_m = AverageMeter()
        val_dice_m = AverageMeter()

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images)
                    loss = val_criterion(logits.float(), masks)
                m = compute_metrics(logits.float(), masks)
                val_loss_m.update(loss.item(), images.size(0))
                val_f1_m.update(m['f1'], images.size(0))
                val_dice_m.update(m['dice'], images.size(0))

        # SWA
        if epoch >= SWA_START:
            if swa_model is None:
                print(f"\n>>> Starting SWA at epoch {epoch}")
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch} | LR={lr:.6f} | "
              f"Train: loss={loss_meter.avg:.4f} dice={dice_meter.avg:.4f} | "
              f"Val: loss={val_loss_m.avg:.4f} f1={val_f1_m.avg:.4f} dice={val_dice_m.avg:.4f}")

        # ── Checkpointing ──
        if val_f1_m.avg > best_f1:
            best_f1 = val_f1_m.avg
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'student_state_dict': model.state_dict(),  # Use student_state_dict for compat
                'best_f1': best_f1,
                'best_dice': val_dice_m.avg,
                'config': {
                    'conf_threshold': conf_threshold,
                    'pseudo_loss_weight': PSEUDO_LOSS_WEIGHT,
                    'base_lr': BASE_LR,
                    'n_real': n_real,
                    'n_pseudo': n_pseudo,
                }
            }, MODEL_DIR / 'best_model.pth')
            print(f"  ★ New best F1={best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Save SWA model
    if swa_model is not None:
        torch.save({
            'student_state_dict': swa_model.module.state_dict(),
            'epoch': epoch,
        }, MODEL_DIR / 'swa_model.pth')
        print(f"  SWA model saved")

    print(f"\n=== Training Complete ===")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Models saved to: {MODEL_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for pseudo-labels')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    train(device, conf_threshold=args.conf)
