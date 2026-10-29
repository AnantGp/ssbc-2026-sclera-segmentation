"""Pseudo-label self-distillation for SSBC 2026 Track 2.

Strategy (inspired by CSIRO 1st place test-time self-distillation):
1. Use best ensemble (ViT-B + ViT-L student, ms_hflip TTA) to generate
   soft pseudo-labels for ALL unlabelled images
2. Retrain ViT-B from scratch with labelled + pseudo-labelled data
   (purely supervised — no Mean Teacher needed since all data has labels)

This converts our semi-supervised setup (357 labelled + ~28K unlabelled)
into a fully supervised setup (~28K+ images with labels).

Usage:
  # Step 1: Generate pseudo-labels
  python t2_distill.py --generate

  # Step 2: Train with pseudo-labels
  python t2_distill.py --train

  # Both steps
  python t2_distill.py --generate --train
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
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg_b
import t2_dino_large_config as cfg_l
from t2_dino_model import (
    build_dino_model, create_ema_model, update_ema,
    count_parameters, get_param_groups,
    DiceBCELoss, compute_metrics
)
from t2_dino_eval import strip_compile_prefix
from t2_dataset import (
    build_labelled_pairs, build_unlabelled_paths, split_train_val,
    ValidationDataset, get_val_transforms
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths ──
PSEUDO_DIR = Path("/home/jupyter/sclera/data/T2_pseudo_labels")
MODEL_DIR = Path("/home/jupyter/sclera/models/t2_distill")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_distill")
LOG_DIR = OUTPUT_DIR / "logs"

# ── Training config ──
IMG_SIZE = 518
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4        # Effective batch = 32
NUM_WORKERS = 4
BASE_LR = 1e-3
LN_LR_SCALE = 0.1
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 20
SEED = 42
USE_BFLOAT16 = True
SWA_START = 80
SWA_LR = 5e-5

# Pseudo-label filtering
CONFIDENCE_THRESHOLD = 0.7  # Only use pseudo-labels where model is confident
MIN_SCLERA_RATIO = 0.001    # Skip images with almost no sclera predicted

# Supervision weighting
PSEUDO_LOSS_WEIGHT = 0.5    # Weight for pseudo-labelled data (real labels = 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: GENERATE PSEUDO-LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble_models(device):
    """Load the ViT-B + ViT-L ensemble."""
    models = []

    # ViT-B student
    vitb_path = cfg_b.MODEL_DIR / 'best_model.pth'
    if vitb_path.exists():
        print("  Loading ViT-B student...")
        m = build_dino_model(
            backbone_name=cfg_b.BACKBONE,
            feature_layers=cfg_b.FEATURE_LAYERS,
            decoder_dim=cfg_b.DECODER_DIM,
            unfreeze_ln=False,
        )
        ckpt = torch.load(vitb_path, map_location='cpu', weights_only=False)
        m.load_state_dict(strip_compile_prefix(ckpt['student_state_dict']))
        m = m.to(device).eval()
        models.append((m, cfg_b.IMG_SIZE))

    # ViT-L student
    vitl_path = cfg_l.MODEL_DIR / 'best_model.pth'
    if vitl_path.exists():
        print("  Loading ViT-L student...")
        m = build_dino_model(
            backbone_name=cfg_l.BACKBONE,
            feature_layers=cfg_l.FEATURE_LAYERS,
            decoder_dim=cfg_l.DECODER_DIM,
            unfreeze_ln=False,
        )
        ckpt = torch.load(vitl_path, map_location='cpu', weights_only=False)
        m.load_state_dict(strip_compile_prefix(ckpt['student_state_dict']))
        m = m.to(device).eval()
        models.append((m, cfg_l.IMG_SIZE))

    print(f"  Loaded {len(models)} models for ensemble")
    return models


@torch.no_grad()
def predict_with_ms_tta(models, image_rgb, device):
    """Predict with multi-scale + HFlip TTA, return probability map at original resolution."""
    orig_h, orig_w = image_rgb.shape[:2]
    all_model_probs = []

    for model, img_size in models:
        transform = get_val_transforms(img_size)
        out = transform(image=image_rgb)
        x = out['image'].unsqueeze(0).to(device)

        views = []
        for scale in [0.75, 1.0, 1.25]:
            target_size = max(14, (int(img_size * scale) // 14) * 14)
            x_scaled = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=False)
            # Original
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x_scaled)
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            views.append(torch.sigmoid(logits.float()))

            # HFlip
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits_f = model(torch.flip(x_scaled, [3]))
            logits_f = F.interpolate(logits_f, size=x.shape[2:], mode='bilinear', align_corners=False)
            views.append(torch.flip(torch.sigmoid(logits_f.float()), [3]))

        model_prob = torch.stack(views).mean(dim=0)
        all_model_probs.append(model_prob)

    # Average across models
    ensemble_prob = torch.stack(all_model_probs).mean(dim=0)
    # Resize to original image size
    prob = F.interpolate(ensemble_prob, size=(orig_h, orig_w),
                          mode='bilinear', align_corners=False)
    return prob.squeeze().cpu().numpy()


def generate_pseudo_labels(device):
    """Generate pseudo-labels for all unlabelled data."""
    print("\n=== STEP 1: Generating Pseudo-Labels ===")
    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

    # Load ensemble
    models = load_ensemble_models(device)

    # Get all unlabelled paths
    print("\nLabelled datasets:")
    labelled_pairs, _ = build_labelled_pairs(cfg_b.T2_LABELLED)
    print("\nUnlabelled datasets:")
    unlabelled_paths = build_unlabelled_paths(cfg_b.T2_UNLABELLED, labelled_pairs)
    print(f"\nTotal unlabelled images to process: {len(unlabelled_paths)}")

    stats = {'total': 0, 'kept': 0, 'skipped_low_conf': 0, 'skipped_no_sclera': 0}
    metadata = []

    for img_path in tqdm(unlabelled_paths, desc="Generating pseudo-labels"):
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prob = predict_with_ms_tta(models, image_rgb, device)
        stats['total'] += 1

        # Compute confidence: how sure the model is (close to 0 or 1)
        confidence = np.abs(prob - 0.5) * 2  # 0 = uncertain, 1 = confident
        mean_conf = confidence.mean()

        # Sclera ratio
        sclera_ratio = (prob > 0.5).mean()

        # Filter
        if mean_conf < CONFIDENCE_THRESHOLD:
            stats['skipped_low_conf'] += 1
            continue
        if sclera_ratio < MIN_SCLERA_RATIO:
            stats['skipped_no_sclera'] += 1
            continue

        # Save soft pseudo-label as 8-bit PNG
        # Determine output path: PSEUDO_DIR / dataset_name / image_stem.png
        # Infer dataset from path
        parts = img_path.parts
        for i, p in enumerate(parts):
            if p in ('MOBIUS', 'MASD', 'SBVPI', 'SynCROI'):
                ds_name = p
                break
        else:
            ds_name = 'unknown'

        out_dir = PSEUDO_DIR / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (img_path.stem + '.png')

        # Save as uint8 (0-255 probability)
        prob_u8 = np.clip(prob * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), prob_u8)

        metadata.append({
            'image': str(img_path),
            'pseudo_label': str(out_path),
            'dataset': ds_name,
            'confidence': float(mean_conf),
            'sclera_ratio': float(sclera_ratio),
        })
        stats['kept'] += 1

    print(f"\n  Total processed: {stats['total']}")
    print(f"  Kept: {stats['kept']} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"  Skipped (low confidence): {stats['skipped_low_conf']}")
    print(f"  Skipped (no sclera): {stats['skipped_no_sclera']}")

    # Save metadata
    meta_path = PSEUDO_DIR / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump({'stats': stats, 'pairs': metadata}, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Cleanup
    for m, _ in models:
        del m
    torch.cuda.empty_cache()

    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# DATASET FOR DISTILLATION TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class DistillDataset(Dataset):
    """Dataset that combines real labelled + pseudo-labelled data.

    Returns (image, mask, weight) where weight=1.0 for real labels
    and weight=PSEUDO_LOSS_WEIGHT for pseudo-labels.
    """
    def __init__(self, real_pairs, pseudo_pairs, img_size=518):
        """
        real_pairs: [(img_path, mask_path), ...]
        pseudo_pairs: [(img_path, pseudo_mask_path), ...]
        """
        self.pairs = []
        self.weights = []

        for img_p, mask_p in real_pairs:
            self.pairs.append((str(img_p), str(mask_p)))
            self.weights.append(1.0)

        for img_p, pseudo_p in pseudo_pairs:
            self.pairs.append((str(img_p), str(pseudo_p)))
            self.weights.append(PSEUDO_LOSS_WEIGHT)

        self.img_size = img_size
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        weight = self.weights[idx]

        image = cv2.imread(img_path)
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN WITH PSEUDO-LABELS
# ══════════════════════════════════════════════════════════════════════════════

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
        # BCE per sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        bce = bce.mean(dim=(1, 2, 3))  # per-sample

        # Dice per sample
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        loss_per_sample = self.bce_weight * bce + self.dice_weight * dice_loss

        if weights is not None:
            loss_per_sample = loss_per_sample * weights

        return loss_per_sample.mean()


def train_distill(device):
    """Train with labelled + pseudo-labelled data."""
    print("\n=== STEP 2: Training with Pseudo-Labels ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ──
    meta_path = PSEUDO_DIR / 'metadata.json'
    if not meta_path.exists():
        print("ERROR: No pseudo-labels found. Run --generate first.")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    pseudo_pairs = [(p['image'], p['pseudo_label']) for p in meta['pairs']]
    print(f"  Pseudo-labelled images: {len(pseudo_pairs)}")

    # ── Real labelled data ──
    print("\nLabelled datasets:")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    train_pairs, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg_b.VAL_SPLIT, seed=SEED
    )
    print(f"  Real train: {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")

    # ── Datasets ──
    train_dataset = DistillDataset(train_pairs, pseudo_pairs, img_size=IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=IMG_SIZE)

    print(f"  Total training samples: {len(train_dataset)} "
          f"({len(train_pairs)} real + {len(pseudo_pairs)} pseudo)")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ──
    print("\n  Building DINOv2 ViT-B model...")
    model = build_dino_model(
        backbone_name=cfg_b.BACKBONE,
        feature_layers=cfg_b.FEATURE_LAYERS,
        decoder_dim=cfg_b.DECODER_DIM,
        unfreeze_ln=True,
    )
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

    warmup_epochs = 3
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
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'best_dice': val_dice_m.avg,
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
            'model_state_dict': swa_model.module.state_dict(),
            'epoch': epoch,
        }, MODEL_DIR / 'swa_model.pth')
        print(f"  SWA model saved")

    print(f"\n=== Training Complete ===")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Models saved to: {MODEL_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T2 Pseudo-Label Self-Distillation')
    parser.add_argument('--generate', action='store_true', help='Generate pseudo-labels')
    parser.add_argument('--train', action='store_true', help='Train with pseudo-labels')
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.error("Specify --generate and/or --train")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.generate:
        generate_pseudo_labels(device)

    if args.train:
        train_distill(device)
