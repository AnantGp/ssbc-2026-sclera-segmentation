"""Cross-dataset evaluation for UNet+EfficientNetV2-S baseline.

Evaluates a baseline model on each dataset independently.

Usage:
  python3 baseline_evaluate.py --checkpoint outputs/baseline/models/baseline_synthetic/best_model.pth
  python3 baseline_evaluate.py --checkpoint outputs/baseline/models/baseline_mixed/best_model.pth
"""
import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

# ── Config (must match baseline_train.py) ──
DATA_ROOT = Path("/home/jupyter/sclera/data/T1 Training")
SEED = 42
IMG_SIZE = 768
BATCH_SIZE = 32
NUM_WORKERS = 4
ENCODER_NAME = "tu-tf_efficientnetv2_s"
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

EVAL_DATASETS = {
    'SynCROI_1': {'root': DATA_ROOT / 'SynCROI_1'},
    'SBVPI':     {'root': DATA_ROOT / 'SBVPI'},
    'MASD_1':    {'root': DATA_ROOT / 'MASD_1'},
}


# ── Helpers ──
def get_mask_path(img_path, dataset_name, mask_dir):
    if 'SBVPI' in dataset_name:
        return mask_dir / (img_path.stem + '_sclera.png')
    return mask_dir / (img_path.stem + '.png')


def build_pairs(dataset_name, root):
    root = Path(root)
    img_dir, mask_dir = root / 'Images', root / 'Masks'
    pairs = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        mp = get_mask_path(f, dataset_name, mask_dir)
        if mp.exists():
            pairs.append((str(f), str(mp)))
    return pairs


def ensure_binary_mask(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.float32)


def get_eval_transforms(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# ── Dataset ──
class EvalDataset(Dataset):
    def __init__(self, pairs, transforms):
        self.pairs = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = ensure_binary_mask(mask)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        out = self.transforms(image=img, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


# ── Evaluation ──
@torch.no_grad()
def evaluate_dataset(model, loader, device):
    model.eval()
    total_loss = 0
    tp = fp = fn = 0
    n = 0
    for images, masks in tqdm(loader, desc='    Eval', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            inter = (probs * masks).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice_l = 1 - ((2 * inter + 1) / (union + 1)).mean()
            loss = 0.3 * bce + 0.7 * dice_l   # match baseline_train DiceBCE weights
        total_loss += loss.item()
        preds = (probs > 0.5).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
        n += 1
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return dict(loss=total_loss / max(n, 1), f1=f1, iou=iou, dice=dice,
                precision=prec, recall=rec, n_images=len(loader.dataset))


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--datasets', nargs='+', default=['SynCROI_1', 'SBVPI', 'MASD_1'])
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    print(f"\nLoading baseline model from: {args.checkpoint}")
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None,
                      in_channels=3, classes=1, activation=None)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    ckpt_name = Path(args.checkpoint).parent.name
    print(f"Model: {ckpt_name}")
    for k in ('best_dice', 'best_f1', 'epoch'):
        if k in ckpt:
            print(f"  {k}: {ckpt[k]}")
    print()

    transforms = get_eval_transforms(IMG_SIZE)

    results = {}
    print(f"{'='*70}")
    print(f"  Baseline Cross-Dataset Evaluation — {ckpt_name}")
    print(f"{'='*70}")

    for ds_name in args.datasets:
        if ds_name not in EVAL_DATASETS:
            print(f"  WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        root = EVAL_DATASETS[ds_name]['root']
        pairs = build_pairs(ds_name, root)
        if not pairs:
            print(f"  {ds_name}: no pairs found, skipping")
            continue

        print(f"\n  {ds_name}: {len(pairs)} images")
        ds = EvalDataset(pairs, transforms)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        t0 = time.time()
        m = evaluate_dataset(model, loader, device)
        elapsed = time.time() - t0
        results[ds_name] = m
        print(f"    F1: {m['f1']:.4f} | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f} | "
              f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
              f"Loss: {m['loss']:.4f} | {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary — {ckpt_name}")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12} {'F1':>8} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'Loss':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ds_name, m in results.items():
        print(f"  {ds_name:<12} {m['f1']:>8.4f} {m['dice']:>8.4f} {m['iou']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['loss']:>8.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
