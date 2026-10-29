"""
Track 1 — SAM+LoRA v4 Training
================================
Improvements over v3:
  1. Much stronger augmentations (Elastic, GridDistortion, CoarseDropout, etc.)
  2. Boundary-aware loss (BCE + Dice + Boundary)
  3. Real-data oversampling in mixed mode (5× repeat)
  4. Early stopping with patience
  5. SWA (Stochastic Weight Averaging) in final epochs
  6. Gradient accumulation (effective batch 16)
  7. Cosine schedule with warm restarts
  8. Dropout in decoder for regularization
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from config import *
from dataset_v4 import ScleraDatasetV4, build_pairs
from model_v4 import SAMLoRASegV4


# ── Loss functions ──

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def boundary_loss(logits, targets, theta0=3, theta=5):
    """
    Boundary-aware loss: extra penalty near mask edges.
    Computes distance-weighted BCE focused on boundary pixels.
    """
    # Compute boundary via Laplacian-like filter
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=targets.dtype, device=targets.device).view(1, 1, 3, 3)
    boundary = F.conv2d(targets, kernel, padding=1).abs()
    boundary = (boundary > 0.1).float()
    # Dilate boundary region
    dilate_k = torch.ones(1, 1, theta, theta, device=targets.device)
    boundary_wide = F.conv2d(boundary, dilate_k, padding=theta // 2)
    boundary_wide = (boundary_wide > 0).float()
    # Weighted BCE on boundary region (use with_logits for autocast safety)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    boundary_bce = (bce * boundary_wide).sum() / (boundary_wide.sum() + 1e-8)
    return boundary_bce


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Focal loss to focus on hard pixels."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def combined_loss_v4(logits, targets):
    """BCE + Dice + Boundary + Focal combined loss."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    bdry = boundary_loss(logits, targets)
    fcl = focal_loss(logits, targets)
    return 0.2 * bce + 0.5 * dice + 0.15 * bdry + 0.15 * fcl


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

ACCUM_STEPS = 4  # Effective batch = BATCH_SIZE * ACCUM_STEPS = 16

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, masks) in enumerate(tqdm(loader, desc='  Train', leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        masks = F.interpolate(masks, size=MASK_OUT_SIZE, mode='nearest')
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = combined_loss_v4(logits, masks) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
            loss = combined_loss_v4(logits, masks)
        total_loss += loss.item()
        m = compute_metrics(logits, masks)
        for k in agg:
            agg[k] += m[k]
        n += 1
    return total_loss / len(loader), {k: v / n for k, v in agg.items()}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'mixed'], required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    parser.add_argument('--lora-alpha', type=float, default=LORA_ALPHA)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset-epoch', action='store_true')
    parser.add_argument('--patience', type=int, default=12,
                        help='Early stopping patience')
    parser.add_argument('--swa-start', type=int, default=45,
                        help='Epoch to start SWA')
    parser.add_argument('--real-oversample', type=int, default=5,
                        help='How many times to repeat real data in mixed mode')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda')

    # ── Data ──
    print(f"\n{'='*60}")
    print(f"SSBC 2026 Track 1 — SAM+LoRA v4 — Mode: {args.mode}")
    print(f"{'='*60}")
    print("Building dataset...")
    datasets = T1_DATASETS[args.mode]
    pairs, labels = build_pairs(datasets)
    print(f"Total pairs: {len(pairs)}")

    # Oversample real data in mixed mode
    if args.mode == 'mixed' and args.real_oversample > 1:
        real_pairs = [(p, l) for p, l in zip(pairs, labels) if l != 'SynCROI_1']
        extra = real_pairs * (args.real_oversample - 1)
        for p, l in extra:
            pairs.append(p)
            labels.append(l)
        print(f"After {args.real_oversample}× real oversampling: {len(pairs)} total pairs")

    # Train/val split
    indices = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = ScleraDatasetV4(train_pairs, img_size=IMG_SIZE, is_train=True)
    val_ds = ScleraDatasetV4(val_pairs, img_size=IMG_SIZE, is_train=False)

    # Balanced sampling for mixed mode
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
    print("\nLoading SAM + injecting LoRA (v4 with dropout decoder)...")
    model = SAMLoRASegV4(SAM_CHECKPOINT, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    total, trainable = model.count_params()
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e3:.1f}K trainable "
          f"({100*trainable/total:.2f}%)")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── Optimizer: AdamW with decoupled weight decay ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=1e-3, betas=(0.9, 0.999))
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # SWA model for final epoch averaging
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

    start_epoch = 0
    best_f1 = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt['model_state_dict']
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(state, strict=False)
        if args.reset_epoch:
            print(f"Loaded weights from {args.resume} (epoch {ckpt['epoch']+1}, F1: {ckpt['best_f1']:.4f})")
            print(f"Reset epoch→0, best_f1→0 for v4 training")
        else:
            start_epoch = ckpt['epoch'] + 1
            best_f1 = ckpt['best_f1']
            print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ── Directories ──
    save_dir = MODEL_DIR / f"t1_{args.mode}_v4"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop with early stopping ──
    no_improve = 0
    use_swa = False

    print(f"\nStarting v4 training for {args.epochs} epochs...")
    print(f"  Patience: {args.patience}, SWA start: {args.swa_start}")
    print(f"  Loss: BCE(0.2) + Dice(0.5) + Boundary(0.15) + Focal(0.15)")
    print(f"  Optimizer: AdamW, LR: {args.lr}, WD: 1e-3")
    print(f"  Augmentations: Strong (Elastic, GridDistort, CoarseDropout, ...)")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Switch to SWA after swa_start epochs
        if epoch >= args.swa_start and not use_swa:
            use_swa = True
            print(f"  ↳ Activating SWA from epoch {epoch+1}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, metrics = validate(model, val_loader, device)

        if use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        swa_tag = " [SWA]" if use_swa else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"F1 {metrics['f1']:.4f} | IoU {metrics['iou']:.4f} | "
              f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s{swa_tag}")

        # Save best
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            no_improve = 0
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
        else:
            no_improve += 1

        # Early stopping (only before SWA phase)
        if no_improve >= args.patience and not use_swa:
            if epoch < args.swa_start:
                print(f"\n  ✗ Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                print(f"  ↳ Jumping to SWA phase...")
                # Jump to SWA instead of stopping
                use_swa = True
                no_improve = 0
                args.swa_start = epoch + 1
                args.epochs = min(args.epochs, epoch + 16)  # 15 more SWA epochs
                print(f"  ↳ Will run SWA until epoch {args.epochs}")
            else:
                print(f"\n  ✗ Stopping (SWA phase exhausted)")
                break

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, save_dir / f'checkpoint_epoch{epoch+1}.pth')

    # Save SWA model
    if use_swa:
        print("\nUpdating SWA batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_base = swa_model.module
        if isinstance(swa_base, nn.DataParallel):
            swa_base = swa_base.module
        # Validate SWA model
        swa_loss, swa_metrics = validate(swa_model, val_loader, device)
        print(f"SWA model — F1: {swa_metrics['f1']:.4f} | IoU: {swa_metrics['iou']:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': swa_base.state_dict(),
            'best_f1': swa_metrics['f1'],
            'metrics': swa_metrics,
            'args': vars(args),
            'is_swa': True,
        }, save_dir / 'swa_model.pth')
        print(f"SWA model saved at: {save_dir / 'swa_model.pth'}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Models saved at: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
