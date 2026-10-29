"""Mentor's UNet+EfficientNetV2-S baseline — adapted for GCP A100.

Based on his_approach.py with these changes:
  - Multi-dataset loading (SynCROI_1, SBVPI, MASD_1) with SBVPI mask naming
  - Mask-image dimension mismatch fix
  - Paths updated for GCP instance
  - img_size=768, batch_size=32, epochs=50 for A100
  - CLI --mode synthetic|mixed
  - Headless (no matplotlib)
"""

import argparse
import os
import cv2
import gc
import time
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")


# =========================
# 1. CONFIG
# =========================
# Paths — GCP instance
DATA_ROOT = Path("/home/jupyter/sclera/data/T1 Training")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/baseline")
MODEL_DIR = OUTPUT_DIR / "models"

# Dataset configs
DATASETS = {
    'synthetic': [
        {'name': 'SynCROI_1', 'root': DATA_ROOT / 'SynCROI_1', 'is_real': 0},
    ],
    'mixed': [
        {'name': 'SynCROI_1', 'root': DATA_ROOT / 'SynCROI_1', 'is_real': 0},
        {'name': 'SBVPI',     'root': DATA_ROOT / 'SBVPI',     'is_real': 1},
        {'name': 'MASD_1',    'root': DATA_ROOT / 'MASD_1',    'is_real': 1},
    ],
}

# Training — tuned for A100 40GB
SEED = 42
IMG_SIZE = 768
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
VALID_SIZE = 0.1       # match SAM experiment's split
ENCODER_NAME = "tu-tf_efficientnetv2_s"
ENCODER_WEIGHTS = "imagenet"
USE_AMP = True
BIN_THRESH = 0.5

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# =========================
# 2. UTILITIES
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask to binary float32. Handles grayscale input."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.float32)


# =========================
# 3. MULTI-DATASET DATAFRAME BUILDER
# =========================
def get_mask_path(img_path: Path, dataset_name: str, mask_dir: Path) -> Path:
    """Get mask path, handling SBVPI's _sclera suffix."""
    if 'SBVPI' in dataset_name:
        mask_name = img_path.stem + '_sclera.png'
    else:
        mask_name = img_path.stem + '.png'
    return mask_dir / mask_name


def build_dataframe(dataset_configs: list) -> pd.DataFrame:
    """Build training dataframe from multiple dataset directories."""
    rows = []
    for ds in dataset_configs:
        name = ds['name']
        root = Path(ds['root'])
        is_real = ds['is_real']
        img_dir = root / 'Images'
        mask_dir = root / 'Masks'

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping")
            continue

        count = 0
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = get_mask_path(img_file, name, mask_dir)
            if not mask_path.exists():
                continue
            rows.append({
                "image_path": str(img_file),
                "mask_path": str(mask_path),
                "image_name": img_file.name,
                "dataset_name": name,
                "is_real": is_real,
                "has_mask": 1,
                "is_pseudo": 0,
            })
            count += 1
        print(f"  {name}: {count} image-mask pairs")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError("No training image-mask pairs found.")
    return df


# =========================
# 4. AUGMENTATIONS  (mentor's rich set, unchanged)
# =========================
def get_train_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=12,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=2.0),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        A.ColorJitter(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# =========================
# 5. DATASET
# =========================
class ScleraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.iloc[idx]
        img = cv2.imread(rec.image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {rec.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if rec.has_mask == 1 and str(rec.mask_path):
            mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {rec.mask_path}")
            mask = ensure_binary_mask(mask)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.float32)

        # Fix: ensure mask matches image dimensions before transforms
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"].float().unsqueeze(0)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img, mask


# =========================
# 6. MODEL
# =========================
def build_model():
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
    )


# =========================
# 7. LOSS + METRICS
# =========================
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.dice_weight * dice_loss + self.bce_weight * bce


def compute_stats(logits, targets, threshold=0.5, eps=1e-7):
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

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# =========================
# 8. TRAIN / VALIDATE
# =========================
def get_sampler(train_df: pd.DataFrame):
    """WeightedRandomSampler to balance datasets equally."""
    counts = Counter(train_df["dataset_name"])
    weights = [1.0 / counts[row.dataset_name] for _, row in train_df.iterrows()]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    running_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        imgs, masks = batch[:2]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(imgs)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        stats = compute_stats(logits.detach(), masks)
        for k, v in stats.items():
            running_metrics[k] += v

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{stats['dice']:.4f}")

    n = max(len(loader), 1)
    out = {"loss": running_loss / n}
    for k in running_metrics:
        out[k] = running_metrics[k] / n
    return out


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc="  Valid", leave=False)
    for batch in pbar:
        imgs, masks = batch[:2]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(imgs)
            loss = criterion(logits, masks)

        running_loss += loss.item()
        stats = compute_stats(logits, masks)
        for k, v in stats.items():
            running_metrics[k] += v

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{stats['dice']:.4f}")

    n = max(len(loader), 1)
    out = {"loss": running_loss / n}
    for k in running_metrics:
        out[k] = running_metrics[k] / n
    return out


# =========================
# 9. MAIN 
# =========================
def main():
    parser = argparse.ArgumentParser(description="Baseline UNet training")
    parser.add_argument('--mode', choices=['synthetic', 'mixed'], required=True,
                        help='Training mode: synthetic-only or mixed (all datasets)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories
    save_dir = MODEL_DIR / f"baseline_{args.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BASELINE UNet+EfficientNetV2-S — Mode: {args.mode}")
    print(f"img_size={args.img_size}  batch_size={args.batch_size}  "
          f"epochs={args.epochs}  lr={args.lr}")
    print(f"{'='*60}")

    # ── Data ──
    print("\nBuilding dataset...")
    dataset_configs = DATASETS[args.mode]
    df = build_dataframe(dataset_configs)
    print(f"Total: {len(df)} image-mask pairs")

    # Stratified split by dataset name
    train_df, valid_df = train_test_split(
        df,
        test_size=VALID_SIZE,
        random_state=SEED,
        stratify=df["dataset_name"],
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(valid_df)}")

    # Datasets & loaders
    train_ds = ScleraDataset(train_df, transforms=get_train_transforms(args.img_size))
    valid_ds = ScleraDataset(valid_df, transforms=get_valid_transforms(args.img_size))

    if args.mode == 'mixed':
        sampler = get_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ──
    print(f"\nBuilding model: UNet + {ENCODER_NAME}...")
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DiceBCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ── Training loop ──
    best_dice = -1.0
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        valid_log = validate_one_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train {train_log['loss']:.4f} | Val {valid_log['loss']:.4f} | "
              f"Dice {valid_log['dice']:.4f} | IoU {valid_log['iou']:.4f} | "
              f"P {valid_log['precision']:.4f} R {valid_log['recall']:.4f} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s")

        if valid_log["dice"] > best_dice:
            best_dice = valid_log["dice"]
            ckpt_path = str(save_dir / "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_dice": best_dice,
                "epoch": epoch,
                "mode": args.mode,
            }, ckpt_path)
            print(f"  ✓ New best dice={best_dice:.4f} → saved {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    print(f"Model saved at: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
