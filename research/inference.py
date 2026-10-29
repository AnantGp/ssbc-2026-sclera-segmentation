"""Generate submission masks for SSBC 2026."""
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import *
from model import SAMLoRASeg
import albumentations as A


def generate_masks(model, img_dir, pred_dir, bin_dir, device,
                   img_size=1024, eval_w=400, eval_h=300):
    """Generate probabilistic and binarised masks for all images in a directory."""
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    transform = A.Compose([A.Resize(img_size, img_size)])
    img_files = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    print(f"  Processing {len(img_files)} images from {img_dir.name}...")
    for img_file in tqdm(img_files, leave=False):
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = transform(image=image)
        tensor = torch.from_numpy(out['image']).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(tensor)
        # Resize to competition size
        logits = F.interpolate(logits, size=(eval_h, eval_w), mode='bilinear', align_corners=False)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        # Probabilistic mask (0-255 grayscale)
        prob_u8 = (prob * 255).clip(0, 255).astype(np.uint8)
        out_name = img_file.stem + '.png'
        cv2.imwrite(str(pred_dir / out_name), prob_u8)
        # Binary mask (strict black/white)
        binary = ((prob > 0.5) * 255).astype(np.uint8)
        cv2.imwrite(str(bin_dir / out_name), binary)


def main():
    parser = argparse.ArgumentParser(description='Generate SSBC 2026 submission')
    parser.add_argument('--mode', choices=['synthetic', 'mixed'], required=True)
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Root of test data (should contain MOBIUS/, SMD+SLD/, Synthetic/ subdirs)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: auto from mode)')
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_root = Path(args.test_dir)

    # Load model
    ckpt_path = args.checkpoint or str(MODEL_DIR / f"t1_{args.mode}" / "best_model.pth")
    print(f"Loading model from {ckpt_path}")
    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=args.lora_rank)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"  Loaded (epoch {ckpt['epoch']+1}, F1={ckpt['best_f1']:.4f})")

    # Submission folder name maps to training regime
    regime = 'Synthetic' if args.mode == 'synthetic' else 'Mixed'

    # Test datasets
    test_datasets = ['MOBIUS', 'SMD+SLD', 'Synthetic']

    output_root = OUTPUT_DIR / 'submission' / regime
    print(f"\nGenerating masks → {output_root}")

    for ds_name in test_datasets:
        img_dir = test_root / ds_name / 'Images'
        if not img_dir.exists():
            # Try without Images subfolder
            img_dir = test_root / ds_name
        if not img_dir.exists():
            print(f"  ⚠ {ds_name}: not found at {img_dir}, skipping")
            continue
        pred_dir = output_root / ds_name / 'Predictions'
        bin_dir = output_root / ds_name / 'Binarised'
        generate_masks(model, img_dir, pred_dir, bin_dir, device,
                       img_size=IMG_SIZE, eval_w=EVAL_W, eval_h=EVAL_H)
        print(f"  ✓ {ds_name} done")

    print(f"\nSubmission files at: {output_root}")
    print("To create ZIP: cd outputs/submission && zip -r SAMLoRA.zip Synthetic/ Mixed/")


if __name__ == '__main__':
    main()
