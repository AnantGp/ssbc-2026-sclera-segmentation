"""Cross-dataset evaluation for SSBC 2026 Track 1.

Evaluates a trained model on each dataset independently to measure
same-dataset vs cross-dataset generalization.

Usage:
  python evaluate.py --checkpoint /path/to/best_model.pth
  python evaluate.py --checkpoint /path/to/best_model.pth --datasets SynCROI_1 SBVPI MASD_1
"""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import ScleraDataset, build_pairs
from model import SAMLoRASeg


@torch.no_grad()
def evaluate_dataset(model, loader, device):
    """Evaluate model on a dataset, return per-batch metrics."""
    model.eval()
    total_loss = 0
    tp_all = fp_all = fn_all = 0
    n = 0
    for images, masks in tqdm(loader, desc='    Eval', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        masks = F.interpolate(masks, size=MASK_OUT_SIZE, mode='nearest')
        with torch.amp.autocast('cuda'):
            logits = model(images)
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice_s = 1.0
            intersection = (probs * masks).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice_s = 1 - ((2 * intersection + 1) / (union + 1)).mean()
            loss = bce + dice_s
        total_loss += loss.item()
        preds = (probs > 0.5).float()
        tp_all += (preds * masks).sum().item()
        fp_all += (preds * (1 - masks)).sum().item()
        fn_all += ((1 - preds) * masks).sum().item()
        n += 1
    # Global metrics (pixel-level across entire dataset)
    prec = tp_all / (tp_all + fp_all + 1e-8)
    rec = tp_all / (tp_all + fn_all + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp_all / (tp_all + fp_all + fn_all + 1e-8)
    return {
        'loss': total_loss / max(n, 1),
        'f1': f1,
        'iou': iou,
        'precision': prec,
        'recall': rec,
        'n_images': n * BATCH_SIZE,  # approximate
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--datasets', nargs='+', default=['SynCROI_1', 'SBVPI', 'MASD_1'],
                        help='Datasets to evaluate on')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    parser.add_argument('--lora-alpha', type=float, default=LORA_ALPHA)
    parser.add_argument('--split', choices=['all', 'val'], default='all',
                        help='Evaluate on all data or only validation split')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    ckpt_name = Path(args.checkpoint).parent.name  # e.g. "t1_synthetic" or "t1_mixed"
    print(f"Model: {ckpt_name}")
    if 'best_f1' in ckpt:
        print(f"Training best F1: {ckpt['best_f1']:.4f}")
    print()

    # Dataset name -> config mapping
    ds_map = {
        'SynCROI_1': {'name': 'SynCROI_1', 'root': T1_ROOT / 'SynCROI_1'},
        'SBVPI': {'name': 'SBVPI', 'root': T1_ROOT / 'SBVPI'},
        'MASD_1': {'name': 'MASD_1', 'root': T1_ROOT / 'MASD_1'},
    }

    # Evaluate each dataset
    results = {}
    print(f"{'='*70}")
    print(f"  Cross-Dataset Evaluation — {ckpt_name}")
    print(f"{'='*70}")

    for ds_name in args.datasets:
        if ds_name not in ds_map:
            print(f"  WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        print(f"\n  Evaluating on: {ds_name}")
        ds_config = [ds_map[ds_name]]
        pairs, labels = build_pairs(ds_config)

        if not pairs:
            print(f"    No pairs found, skipping")
            continue

        if args.split == 'val':
            # Use same split logic as training
            indices = np.random.permutation(len(pairs))
            n_val = max(1, int(len(pairs) * VAL_SPLIT))
            val_idx = indices[:n_val]
            pairs = [pairs[i] for i in val_idx]
            print(f"    Using val split: {len(pairs)} images")
        else:
            print(f"    Using all data: {len(pairs)} images")

        dataset = ScleraDataset(pairs, img_size=IMG_SIZE, is_train=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        t0 = time.time()
        metrics = evaluate_dataset(model, loader, device)
        elapsed = time.time() - t0

        results[ds_name] = metrics
        print(f"    F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f} | "
              f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | "
              f"Loss: {metrics['loss']:.4f} | {elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Summary — {ckpt_name}")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'Loss':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ds_name, m in results.items():
        print(f"  {ds_name:<12} {m['f1']:>8.4f} {m['iou']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['loss']:>8.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
