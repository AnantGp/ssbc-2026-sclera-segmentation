"""TTA + Online Adaptation inference for SSBC 2026 Track 1.

This implements what Gourav Bhaiya suggested as "online training":
  1. Test-Time Augmentation (TTA): Run inference with multiple geometric/color
     augmentations and average predictions for smoother, more robust masks.
  2. Online Adaptation: Brief fine-tuning on each test image using a self-supervised
     entropy minimization objective — the model adapts to each image's distribution.

Usage:
  # TTA only (fast, no GPU training needed at test time)
  python tta_inference.py --mode mixed --checkpoint models/t1_mixed_v4/best_model.pth

  # TTA + Online adaptation (slower, best quality)
  python tta_inference.py --mode mixed --checkpoint models/t1_mixed_v4/best_model.pth --online

  # Generate submission
  python tta_inference.py --mode mixed --checkpoint models/t1_mixed_v4/best_model.pth --online \
      --test-dir /path/to/test --submission
"""
import argparse
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import *
from model_v4 import SAMLoRASegV4
from dataset_v4 import build_pairs, load_mask, get_val_transforms
import albumentations as A


# ── TTA transforms ──

def tta_transforms(image, img_size=1024):
    """Generate TTA variants of an image.

    Returns list of (tensor, undo_fn) tuples.
    undo_fn reverses the geometric transform on the prediction.
    """
    base_tfm = get_val_transforms(img_size)
    h, w = image.shape[:2]
    dummy_mask = np.zeros((h, w), dtype=np.uint8)

    variants = []

    # 1. Original
    aug = base_tfm(image=image, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: x))

    # 2. Horizontal flip
    img_h = np.fliplr(image).copy()
    aug = base_tfm(image=img_h, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: torch.flip(x, dims=[3])))

    # 3. Vertical flip
    img_v = np.flipud(image).copy()
    aug = base_tfm(image=img_v, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: torch.flip(x, dims=[2])))

    # 4. Slight scale up (1.1×)
    scale_tfm = A.Compose([
        A.LongestMaxSize(max_size=int(img_size * 1.1)),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(height=img_size, width=img_size),
    ])
    aug = scale_tfm(image=image, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: x))

    # 5. Slight scale down (0.9×)
    scale_tfm2 = A.Compose([
        A.LongestMaxSize(max_size=int(img_size * 0.9)),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(height=img_size, width=img_size),
    ])
    aug = scale_tfm2(image=image, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: x))

    # 6. Brightness variant
    bright_tfm = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(height=img_size, width=img_size),
        A.RandomBrightnessContrast(brightness_limit=(0.15, 0.15),
                                   contrast_limit=(0.1, 0.1), p=1.0),
    ])
    aug = bright_tfm(image=image, mask=dummy_mask)
    t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    variants.append((t, lambda x: x))

    return variants


@torch.no_grad()
def predict_with_tta(model, image_rgb, device, img_size=1024):
    """Run TTA inference on a single image. Returns probability map at original resolution."""
    h, w = image_rgb.shape[:2]
    variants = tta_transforms(image_rgb, img_size)

    all_probs = []
    for tensor, undo_fn in variants:
        tensor = tensor.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
        probs = torch.sigmoid(logits)
        probs = undo_fn(probs)
        all_probs.append(probs.cpu())

    # Average all TTA predictions
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    prob_map = avg_probs[0, 0].numpy()
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)

    # Resize to original
    prob_orig = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob_orig


# ── Online Adaptation (Test-Time Training) ──

def online_adapt(model, image_tensor, device, steps=5, lr=1e-5):
    """Adapt the model briefly on a single test image using entropy minimization.

    The idea: minimize the prediction entropy so the model becomes more confident
    on this particular image's distribution. Only adapts LoRA parameters.
    """
    # Clone model to avoid polluting the original
    adapted = copy.deepcopy(model)
    adapted.train()

    # Only optimize LoRA parameters (fast and safe)
    lora_params = []
    for name, param in adapted.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    if not lora_params:
        return adapted

    optimizer = torch.optim.Adam(lora_params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = adapted(image_tensor.to(device))
            probs = torch.sigmoid(logits)
            # Entropy minimization: encourage confident predictions
            entropy = -(probs * torch.log(probs + 1e-8) +
                       (1 - probs) * torch.log(1 - probs + 1e-8))
            loss = entropy.mean()
        loss.backward()
        optimizer.step()

    adapted.eval()
    return adapted


def predict_with_online_tta(model, image_rgb, device, img_size=1024,
                             adapt_steps=5, adapt_lr=1e-5):
    """Online adaptation + TTA prediction."""
    h, w = image_rgb.shape[:2]

    # First, do a quick entropy-minimization adaptation
    base_tfm = get_val_transforms(img_size)
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    aug = base_tfm(image=image_rgb, mask=dummy_mask)
    base_tensor = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0)

    adapted_model = online_adapt(model, base_tensor, device,
                                  steps=adapt_steps, lr=adapt_lr)

    # Then run TTA with the adapted model
    prob_map = predict_with_tta(adapted_model, image_rgb, device, img_size)

    # Clean up
    del adapted_model
    torch.cuda.empty_cache()

    return prob_map


# ── Submission generation ──

def generate_submission_masks(model, img_dir, pred_dir, bin_dir, device,
                               img_size=1024, eval_w=400, eval_h=300,
                               use_tta=True, use_online=False):
    """Generate probabilistic and binarised masks for submission."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    print(f"  Processing {len(img_files)} images...")

    for img_file in tqdm(img_files, leave=False):
        image = cv2.imread(str(img_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if use_online:
            prob = predict_with_online_tta(model, image_rgb, device, img_size)
        elif use_tta:
            prob = predict_with_tta(model, image_rgb, device, img_size)
        else:
            # Simple single-pass inference
            tfm = get_val_transforms(img_size)
            aug = tfm(image=image_rgb, mask=np.zeros(image_rgb.shape[:2], dtype=np.uint8))
            t = torch.from_numpy(aug['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(t)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Resize to competition size
        prob_resized = cv2.resize(prob, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        # Probabilistic mask (0-255 grayscale)
        prob_u8 = (prob_resized * 255).clip(0, 255).astype(np.uint8)
        out_name = img_file.stem + '.png'
        cv2.imwrite(str(pred_dir / out_name), prob_u8)

        # Binary mask (strict black/white)
        binary = ((prob_resized > 0.5) * 255).astype(np.uint8)
        cv2.imwrite(str(bin_dir / out_name), binary)


# ── Visualization ──

def make_grid(image_rgb, gt_mask, pred_prob, img_size=1024):
    """Create a row: Input | GT | Prediction | Overlay."""
    h, w = image_rgb.shape[:2]
    scale = 400 / max(h, w)
    dh, dw = int(h * scale), int(w * scale)

    img_disp = cv2.resize(image_rgb, (dw, dh))
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    pred_resized = cv2.resize((pred_prob > 0.5).astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)

    gt_gray = np.stack([gt_resized * 255] * 3, axis=-1).astype(np.uint8)
    pred_gray = np.stack([pred_resized * 255] * 3, axis=-1).astype(np.uint8)

    overlay = img_disp.copy()
    overlay[pred_resized > 0] = (0.6 * overlay[pred_resized > 0] + 0.4 * np.array([0, 120, 255])).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, label in [(img_disp, "Input"), (gt_gray, "Ground Truth"),
                       (pred_gray, "Prediction"), (overlay, "Overlay")]:
        cv2.putText(img, label, (5, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (5, 20), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    sep = np.ones((dh, 2, 3), dtype=np.uint8) * 255
    row = np.concatenate([img_disp, sep, gt_gray, sep, pred_gray, sep, overlay], axis=1)
    return row


def main():
    parser = argparse.ArgumentParser(description='TTA + Online Adaptation Inference')
    parser.add_argument('--mode', choices=['synthetic', 'mixed'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--online', action='store_true', help='Enable online adaptation')
    parser.add_argument('--adapt-steps', type=int, default=5)
    parser.add_argument('--adapt-lr', type=float, default=1e-5)
    parser.add_argument('--lora-rank', type=int, default=LORA_RANK)
    parser.add_argument('--lora-alpha', type=float, default=LORA_ALPHA)
    # Visualization mode
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--n', type=int, default=6, help='Samples per dataset for visualization')
    parser.add_argument('--output-dir', type=str, default=None)
    # Submission mode
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--test-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = SAMLoRASegV4(SAM_CHECKPOINT, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    mode_tag = "TTA+Online" if args.online else "TTA"
    print(f"Mode: {mode_tag}")
    if 'best_f1' in ckpt:
        print(f"Checkpoint F1: {ckpt['best_f1']:.4f}")

    if args.visualize:
        out_dir = Path(args.output_dir or str(OUTPUT_DIR / 'visualizations_v4'))
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
                continue

            samples = random.sample(pairs, min(args.n, len(pairs)))
            rows = []
            for i, (img_path, mask_path) in enumerate(samples):
                print(f"  [{i+1}/{len(samples)}] {img_path.name}")
                image = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gt = load_mask(mask_path)
                if gt is None:
                    gt = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

                if args.online:
                    prob = predict_with_online_tta(model, image_rgb, device,
                                                    adapt_steps=args.adapt_steps,
                                                    adapt_lr=args.adapt_lr)
                else:
                    prob = predict_with_tta(model, image_rgb, device)

                row = make_grid(image_rgb, gt, prob)
                rows.append(row)

            max_w = max(r.shape[1] for r in rows)
            padded = []
            for r in rows:
                if r.shape[1] < max_w:
                    pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
                    r = np.concatenate([r, pad], axis=1)
                padded.append(r)

            sep_h = np.ones((4, max_w, 3), dtype=np.uint8) * 200
            combined = []
            for j, r in enumerate(padded):
                combined.append(r)
                if j < len(padded) - 1:
                    combined.append(sep_h)

            grid = np.concatenate(combined, axis=0)
            save_path = out_dir / f"{ckpt_name}_{ds_name}_{mode_tag.lower()}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {save_path}")

        print(f"\nAll visualizations saved to: {out_dir}")

    if args.submission:
        if not args.test_dir:
            print("ERROR: --test-dir required for submission mode")
            return
        test_root = Path(args.test_dir)
        regime = 'Synthetic' if args.mode == 'synthetic' else 'Mixed'
        test_datasets = ['MOBIUS', 'SMD+SLD', 'Synthetic']
        output_root = OUTPUT_DIR / 'submission_v4' / regime

        print(f"\nGenerating {mode_tag} submission → {output_root}")
        for ds_name in test_datasets:
            img_dir = test_root / ds_name / 'Images'
            if not img_dir.exists():
                img_dir = test_root / ds_name
            if not img_dir.exists():
                print(f"  ⚠ {ds_name}: not found, skipping")
                continue
            pred_dir = output_root / ds_name / 'Predictions'
            bin_dir = output_root / ds_name / 'Binarised'
            generate_submission_masks(model, img_dir, pred_dir, bin_dir, device,
                                      img_size=IMG_SIZE, eval_w=EVAL_W, eval_h=EVAL_H,
                                      use_tta=True, use_online=args.online)
            print(f"  ✓ {ds_name}")

        print(f"\nSubmission at: {output_root}")


if __name__ == '__main__':
    main()
