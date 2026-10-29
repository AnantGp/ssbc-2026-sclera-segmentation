"""Generate visual prediction samples for sharing.

Creates side-by-side images: Input | Ground Truth | Prediction | Overlay
Picks random samples from each dataset.

Usage:
  python visualize.py --checkpoint models/t1_mixed/best_model.pth
  python visualize.py --checkpoint models/t1_mixed/best_model.pth --n 5 --lora-rank 16 --lora-alpha 32
"""
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from config import *
from dataset import build_pairs, load_mask
from model import SAMLoRASeg
import albumentations as A


def make_grid(image_rgb, gt_mask, pred_mask, img_size=1024):
    """Create a single row: Input | GT | Prediction | Overlay."""
    h, w = image_rgb.shape[:2]

    # Resize for display (max 400px wide)
    scale = 400 / max(h, w)
    dh, dw = int(h * scale), int(w * scale)

    img_disp = cv2.resize(image_rgb, (dw, dh))

    # GT mask → green overlay
    gt_color = np.zeros_like(img_disp)
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    gt_color[gt_resized > 0] = [0, 255, 0]

    # Pred mask → blue overlay
    pred_color = np.zeros_like(img_disp)
    pred_resized = cv2.resize(pred_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    pred_color[pred_resized > 0] = [0, 120, 255]

    # Overlay: image + pred mask blend
    overlay = img_disp.copy()
    overlay[pred_resized > 0] = (0.6 * overlay[pred_resized > 0] + 0.4 * np.array([0, 120, 255])).astype(np.uint8)

    # GT as grayscale image
    gt_gray = np.stack([gt_resized * 255] * 3, axis=-1).astype(np.uint8)
    # Pred as grayscale image
    pred_gray = np.stack([pred_resized * 255] * 3, axis=-1).astype(np.uint8)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, label in [(img_disp, "Input"), (gt_gray, "Ground Truth"),
                       (pred_gray, "Prediction"), (overlay, "Overlay")]:
        cv2.putText(img, label, (5, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (5, 20), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # 2px white separator
    sep = np.ones((dh, 2, 3), dtype=np.uint8) * 255
    row = np.concatenate([img_disp, sep, gt_gray, sep, pred_gray, sep, overlay], axis=1)
    return row


@torch.no_grad()
def predict_single(model, image_rgb, device, img_size=1024):
    """Run inference on a single image, return binary mask at original resolution."""
    h, w = image_rgb.shape[:2]

    # Preprocess like validation
    tfm = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(height=img_size, width=img_size),
    ])
    aug = tfm(image=image_rgb, mask=np.zeros((h, w), dtype=np.uint8))
    img_t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.amp.autocast('cuda'):
        logits = model(img_t)
    probs = torch.sigmoid(logits)[0, 0].cpu().float().numpy()
    probs = np.ascontiguousarray(probs, dtype=np.float32)

    # Upscale from decoder output (256x256) to img_size
    pred_full = cv2.resize(probs, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Undo center crop + pad: figure out the crop/pad offsets
    # The transform does LongestMaxSize then Pad then CenterCrop
    # For simplicity, just resize pred to original image dimensions
    pred_orig = cv2.resize(pred_full, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (pred_orig > 0.5).astype(np.uint8)
    return binary


def main():
    parser = argparse.ArgumentParser(description='Generate visual predictions')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n', type=int, default=4, help='Samples per dataset')
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    parser.add_argument('--lora-alpha', type=float, default=LORA_ALPHA)
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR / 'visualizations'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_configs = {
        'SynCROI_1': {'name': 'SynCROI_1', 'root': T1_ROOT / 'SynCROI_1'},
        'SBVPI': {'name': 'SBVPI', 'root': T1_ROOT / 'SBVPI'},
        'MASD_1': {'name': 'MASD_1', 'root': T1_ROOT / 'MASD_1'},
    }

    ckpt_name = Path(args.checkpoint).stem

    for ds_name, ds_cfg in ds_configs.items():
        print(f"\n--- {ds_name} ---")
        pairs, _ = build_pairs([ds_cfg])
        if not pairs:
            print(f"  No pairs found, skipping")
            continue

        # Random sample
        samples = random.sample(pairs, min(args.n, len(pairs)))
        rows = []

        for i, (img_path, mask_path) in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] {img_path.name}")
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt = load_mask(mask_path)
            if gt is None:
                gt = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

            pred = predict_single(model, image_rgb, device)
            row = make_grid(image_rgb, gt, pred)
            rows.append(row)

        # Pad all rows to same width, then stack vertically
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)

        sep_h = np.ones((4, max_w, 3), dtype=np.uint8) * 200
        combined_rows = []
        for j, r in enumerate(padded):
            combined_rows.append(r)
            if j < len(padded) - 1:
                combined_rows.append(sep_h)

        grid = np.concatenate(combined_rows, axis=0)
        # Save as BGR for cv2
        save_path = out_dir / f"{ckpt_name}_{ds_name}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {save_path}")

    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == '__main__':
    main()
