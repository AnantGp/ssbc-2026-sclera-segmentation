"""Training script for SSBC 2026 Track 1 - SAM + LoRA sclera segmentation."""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from config import *
from dataset import ScleraDataset, build_pairs
from model import SAMLoRASeg


# ── Loss functions ──

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return 0.3 * bce + 0.7 * dice


# ── Metrics ──

@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * prec * rec / (prec + rec + 1e-8))
    iou = (tp / (tp + fp + fn + 1e-8)).item()
    return {'f1': f1, 'precision': prec, 'recall': rec, 'iou': iou}


# ── Train / Validate ──

ACCUM_STEPS = 4  # Effective batch = BATCH_SIZE * ACCUM_STEPS

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, masks) in enumerate(tqdm(loader, desc='  Train', leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        # Resize masks to match decoder output
        masks = F.interpolate(masks, size=MASK_OUT_SIZE, mode='nearest')
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = combined_loss(logits, masks) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * ACCUM_STEPS
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    agg = {'f1': 0, 'precision': 0, 'recall': 0, 'iou': 0}
    n = 0
    for images, masks in tqdm(loader, desc='  Val', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        masks = F.interpolate(masks, size=MASK_OUT_SIZE, mode='nearest')
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = combined_loss(logits, masks)
        total_loss += loss.item()
        m = compute_metrics(logits, masks)
        for k in agg:
            agg[k] += m[k]
        n += 1
    return total_loss / len(loader), {k: v / n for k, v in agg.items()}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'mixed'], required=True,
                        help='Training regime: synthetic-only or mixed (all datasets)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    parser.add_argument('--lora-alpha', type=float, default=LORA_ALPHA)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--reset-epoch', action='store_true',
                        help='Reset epoch counter & best F1 when resuming (for sequential training)')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda')

    # ── Data ──
    print(f"\n{'='*60}")
    print(f"SSBC 2026 Track 1 — SAM+LoRA — Mode: {args.mode}")
    print(f"{'='*60}")
    print("Building dataset...")
    datasets = T1_DATASETS[args.mode]
    pairs, labels = build_pairs(datasets)
    print(f"Total pairs: {len(pairs)}")

    # Train/val split (stratified by dataset)
    indices = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = ScleraDataset(train_pairs, img_size=IMG_SIZE, is_train=True)
    val_ds = ScleraDataset(val_pairs, img_size=IMG_SIZE, is_train=False)

    # Balanced sampling for mixed mode (equal dataset representation)
    if args.mode == 'mixed':
        from collections import Counter
        counts = Counter(train_labels)
        weights = [1.0 / counts[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_pairs), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ──
    print("\nLoading SAM + injecting LoRA...")
    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    total, trainable = model.count_params()
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e3:.1f}K trainable "
          f"({100*trainable/total:.2f}%)")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_f1 = 0

    # Resume if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt['model_state_dict']
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(state)
        if args.reset_epoch:
            # Sequential training: keep weights, restart everything else
            print(f"Loaded weights from {args.resume} (epoch {ckpt['epoch']+1}, F1: {ckpt['best_f1']:.4f})")
            print(f"Reset epoch→0, best_f1→0 for sequential training")
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_f1 = ckpt['best_f1']
            print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ── Directories ──
    save_dir = MODEL_DIR / f"t1_{args.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, metrics = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"F1 {metrics['f1']:.4f} | IoU {metrics['iou']:.4f} | "
              f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s")

        # Save best
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'metrics': metrics,
                'args': vars(args),
            }, save_dir / 'best_model.pth')
            print(f"  ★ New best model saved (F1: {best_f1:.4f})")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, save_dir / f'checkpoint_epoch{epoch+1}.pth')

    print(f"\n{'='*60}")
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Model saved at: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
