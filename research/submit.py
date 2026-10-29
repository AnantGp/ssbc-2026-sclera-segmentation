"""Unified submission pipeline for SSBC 2026 — Track 1 + Track 2.

Generates all submission files in the required format:
  {Track}/
    {Synthetic,Mixed}/
      {MOBIUS,SMD+SLD,SynMOBIUS}/
        Predictions/    ← probabilistic grayscale PNGs (400x300)
        Binarised/      ← binary PNGs at threshold (400x300)

Usage:
  python submit.py --test-dir /path/to/T1_Test --track 1
  python submit.py --test-dir /path/to/T2_Test --track 2
  python submit.py --test-dir /path/to/T1_Test --track 1 --track2-test-dir /path/to/T2_Test

Models used (best configs from eval):
  T1: SAM ViT-B + LoRA v3 (synthetic), threshold=0.50, no TTA
  T2: Ensemble (DINOv2 ViT-B student + ViT-L student), threshold=0.50, TTA (HFlip)
"""
import os
import sys
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── Paths ──
DATA_ROOT = Path("/home/jupyter/sclera")
MODEL_ROOT = DATA_ROOT / "models"
OUTPUT_ROOT = DATA_ROOT / "outputs" / "submission"

EVAL_W, EVAL_H = 400, 300

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_images(directory):
    """Recursively find all images in a directory."""
    directory = Path(directory)
    images = []
    for f in sorted(directory.rglob('*')):
        if f.suffix.lower() in IMAGE_EXTS and f.is_file():
            images.append(f)
    return images


def save_masks(prob, pred_path, bin_path, threshold):
    """Save probabilistic and binarised masks."""
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        prob = np.squeeze(prob)
    prob = np.ascontiguousarray(prob)

    # Resize to submission size
    prob_resized = cv2.resize(prob, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)
    # Probabilistic
    prob_u8 = np.clip(prob_resized * 255, 0, 255).astype(np.uint8)
    Image.fromarray(prob_u8).save(pred_path)
    # Binarised
    bin_u8 = ((prob_resized > threshold) * 255).astype(np.uint8)
    Image.fromarray(bin_u8).save(bin_path)


# ══════════════════════════════════════════════════════════════════════════════
# TRACK 1: SAM ViT-B + LoRA v3
# ══════════════════════════════════════════════════════════════════════════════

def load_t1_model(checkpoint_path, device):
    """Load T1 SAMLoRASeg v3 model."""
    from config import SAM_CHECKPOINT, LORA_RANK
    from model import SAMLoRASeg

    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=LORA_RANK)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    f1 = ckpt.get('best_f1', ckpt.get('metrics', {}).get('f1', '?'))
    print(f"  T1 model loaded: epoch={epoch}, F1={f1}")
    return model


@torch.no_grad()
def t1_predict(model, image_bgr, device, img_size=1024):
    """T1 inference — no TTA (TTA slightly hurts T1)."""
    import albumentations as A
    transform = A.Compose([A.Resize(img_size, img_size)])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = transform(image=image_rgb)
    tensor = torch.from_numpy(out['image']).permute(2, 0, 1).float().unsqueeze(0).to(device)
    with torch.amp.autocast('cuda'):
        logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return prob


def run_t1(test_dir, device, threshold=0.50):
    """Run Track 1 inference for both Synthetic and Mixed regimes."""
    test_dir = Path(test_dir)

    # T1 uses same model for both Synthetic and Mixed submissions
    # (synthetic model is best: F1=0.9786)
    models_config = {
        'Synthetic': MODEL_ROOT / 't1_synthetic' / 'best_model.pth',
        'Mixed': MODEL_ROOT / 't1_mixed' / 'best_model.pth',
    }

    # Test dataset subdirs
    test_datasets = ['MOBIUS', 'SMD+SLD', 'SynMOBIUS']

    for regime, ckpt_path in models_config.items():
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping {regime}")
            continue

        print(f"\n--- T1 {regime} ---")
        model = load_t1_model(ckpt_path, device)

        for ds_name in test_datasets:
            # Try common directory structures
            img_dir = None
            for candidate in [
                test_dir / ds_name / 'Images',
                test_dir / ds_name,
                test_dir / ds_name.replace('+', '_'),
            ]:
                if candidate.exists():
                    img_dir = candidate
                    break

            if img_dir is None:
                print(f"  {ds_name}: not found, skipping")
                continue

            images = find_images(img_dir)
            if not images:
                print(f"  {ds_name}: no images found")
                continue

            pred_dir = OUTPUT_ROOT / 'T1' / regime / ds_name / 'Predictions'
            bin_dir = OUTPUT_ROOT / 'T1' / regime / ds_name / 'Binarised'
            pred_dir.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)

            print(f"  {ds_name}: {len(images)} images")
            for img_path in tqdm(images, desc=f"    {ds_name}", leave=False):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                prob = t1_predict(model, image, device)
                out_name = img_path.stem + '.png'
                save_masks(prob, pred_dir / out_name, bin_dir / out_name, threshold)

        del model
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# TRACK 2: DINOv2 Ensemble (ViT-B + ViT-L)
# ══════════════════════════════════════════════════════════════════════════════

def strip_compile_prefix(state_dict):
    """Remove _orig_mod. prefix from torch.compile."""
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def load_t2_model(cfg, ckpt_path, key, device):
    """Load a DINOv2 model."""
    from t2_dino_model import build_dino_model

    model = build_dino_model(
        backbone_name=cfg['backbone'],
        feature_layers=cfg['feature_layers'],
        decoder_dim=cfg['decoder_dim'],
        unfreeze_ln=False,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if key in ckpt:
        model.load_state_dict(strip_compile_prefix(ckpt[key]))
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(strip_compile_prefix(ckpt['model_state_dict']))
    else:
        model.load_state_dict(strip_compile_prefix(ckpt))
    model = model.to(device)
    model.eval()
    print(f"  Loaded: {ckpt_path.name} ({key})")
    return model


@torch.no_grad()
def _predict_at_scale(model, x, model_img_size, scale=1.0):
    """Run prediction at a given scale, return probs at original resolution."""
    B, C, H, W = x.shape
    target_size = max(14, (int(model_img_size * scale) // 14) * 14)
    if target_size != H or target_size != W:
        x_scaled = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)
    else:
        x_scaled = x
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x_scaled)
    if logits.shape[2:] != (H, W):
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    return torch.sigmoid(logits.float())


@torch.no_grad()
def t2_predict_ensemble(models_with_sizes, image_bgr, device, tta_mode='ms_hflip'):
    """T2 ensemble inference with multi-scale + HFlip TTA.

    tta_mode:
      'none'      — no TTA
      'hflip'     — horizontal flip only (2 views)
      'ms_hflip'  — multi-scale (0.75, 1.0, 1.25) × hflip (6 views, BEST)
    """
    from t2_dataset import get_val_transforms

    all_probs = []

    for model, img_size in models_with_sizes:
        transform = get_val_transforms(img_size)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out = transform(image=image_rgb)
        x = out['image'].unsqueeze(0).to(device)

        views = []

        # Original
        views.append(_predict_at_scale(model, x, img_size, scale=1.0))

        if tta_mode in ('hflip', 'ms_hflip'):
            # Horizontal flip at scale 1.0
            p = _predict_at_scale(model, torch.flip(x, [3]), img_size, scale=1.0)
            views.append(torch.flip(p, [3]))

        if tta_mode == 'ms_hflip':
            # Multi-scale: 0.75x and 1.25x, each with HFlip
            for scale in [0.75, 1.25]:
                views.append(_predict_at_scale(model, x, img_size, scale=scale))
                p = _predict_at_scale(model, torch.flip(x, [3]), img_size, scale=scale)
                views.append(torch.flip(p, [3]))

        probs = torch.stack(views).mean(dim=0)
        all_probs.append(probs)

    # Average across models (handle different output sizes by interpolating)
    ref_size = all_probs[0].shape[2:]
    aligned = []
    for p in all_probs:
        if p.shape[2:] != ref_size:
            p = F.interpolate(p, size=ref_size, mode='bilinear', align_corners=False)
        aligned.append(p)

    avg_prob = torch.stack(aligned).mean(dim=0)
    return avg_prob.squeeze().cpu().numpy()


def run_t2(test_dir, device, threshold=0.50, use_ensemble=True):
    """Run Track 2 inference — ensemble or single model."""
    test_dir = Path(test_dir)

    # Model configurations
    VITB_CFG = {
        'backbone': 'dinov2_vitb14',
        'feature_layers': [2, 5, 8, 11],
        'decoder_dim': 256,
        'img_size': 518,
    }
    VITL_CFG = {
        'backbone': 'dinov2_vitl14',
        'feature_layers': [5, 11, 17, 23],
        'decoder_dim': 256,
        'img_size': 518,
    }

    # Best ensemble: ViT-B student + ViT-L student (F1=0.9564 on val)
    # Prefer full-data retrained models (t2_dino_full / t2_dino_large_full) if available,
    # fall back to val-split models (t2_dino / t2_dino_large).
    models_to_load = []

    # ViT-B: prefer full-data retrained model
    vitb_full_path = MODEL_ROOT / 't2_dino_full' / 'best_model.pth'
    vitb_path = MODEL_ROOT / 't2_dino' / 'best_model.pth'
    if vitb_full_path.exists():
        print("\n  Loading ViT-B (full-data retrained)...")
        m = load_t2_model(VITB_CFG, vitb_full_path, 'student_state_dict', device)
    elif vitb_path.exists():
        print("\n  Loading ViT-B student (val-split trained)...")
        m = load_t2_model(VITB_CFG, vitb_path, 'student_state_dict', device)
    else:
        m = None
    if m is not None:
        models_to_load.append((m, VITB_CFG['img_size']))

    # ViT-L: prefer full-data retrained model
    if use_ensemble:
        vitl_full_path = MODEL_ROOT / 't2_dino_large_full' / 'best_model.pth'
        vitl_path = MODEL_ROOT / 't2_dino_large' / 'best_model.pth'
        if vitl_full_path.exists():
            print("  Loading ViT-L (full-data retrained)...")
            m = load_t2_model(VITL_CFG, vitl_full_path, 'student_state_dict', device)
        elif vitl_path.exists():
            print("  Loading ViT-L student (val-split trained)...")
            m = load_t2_model(VITL_CFG, vitl_path, 'student_state_dict', device)
        else:
            m = None
        if m is not None:
            models_to_load.append((m, VITL_CFG['img_size']))

    if not models_to_load:
        print("  ERROR: No T2 models found!")
        return

    tta_mode = 'ms_hflip'  # Best: multi-scale (0.75, 1.0, 1.25) + HFlip = 6 views/model
    n_views = 6
    print(f"  Ensemble: {len(models_to_load)} models, TTA={tta_mode} ({n_views} views), threshold={threshold}")

    # T2 submission: both Synthetic and Mixed (same model for both)
    test_datasets = ['MOBIUS', 'SMD+SLD', 'SynMOBIUS']

    for regime in ['Synthetic', 'Mixed']:
        print(f"\n--- T2 {regime} ---")
        for ds_name in test_datasets:
            img_dir = None
            for candidate in [
                test_dir / ds_name / 'Images',
                test_dir / ds_name,
            ]:
                if candidate.exists():
                    img_dir = candidate
                    break

            if img_dir is None:
                print(f"  {ds_name}: not found, skipping")
                continue

            images = find_images(img_dir)
            if not images:
                print(f"  {ds_name}: no images found")
                continue

            pred_dir = OUTPUT_ROOT / 'T2' / regime / ds_name / 'Predictions'
            bin_dir = OUTPUT_ROOT / 'T2' / regime / ds_name / 'Binarised'
            pred_dir.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)

            print(f"  {ds_name}: {len(images)} images")
            for img_path in tqdm(images, desc=f"    {ds_name}", leave=False):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                prob = t2_predict_ensemble(models_to_load, image, device, tta_mode=tta_mode)
                out_name = img_path.stem + '.png'
                save_masks(prob, pred_dir / out_name, bin_dir / out_name, threshold)

    # Cleanup
    for m, _ in models_to_load:
        del m
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# ZIP CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_zip(track_dir, zip_name):
    """Create submission ZIP file."""
    track_dir = Path(track_dir)
    zip_path = track_dir.parent / zip_name
    print(f"\nCreating ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(track_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, track_dir)
                zf.write(abs_path, rel_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ZIP size: {size_mb:.1f} MB")
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SSBC 2026 Submission Pipeline')
    parser.add_argument('--track', type=int, choices=[1, 2], required=True,
                        help='Track number (1 or 2)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to test data root (should contain MOBIUS/, SMD+SLD/, Synthetic/)')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='Binarisation threshold (default: 0.50)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='T2: use single ViT-B model instead of ensemble')
    parser.add_argument('--no-zip', action='store_true',
                        help='Skip ZIP creation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Track: {args.track}")
    print(f"Test dir: {args.test_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {OUTPUT_ROOT}")

    if args.track == 1:
        print("\n=== Track 1: SAM ViT-B + LoRA v3 ===")
        print("  Strategy: synthetic model for Synthetic, mixed model for Mixed")
        print("  No TTA (TTA slightly hurts T1)")
        run_t1(args.test_dir, device, threshold=args.threshold)
        if not args.no_zip:
            create_zip(OUTPUT_ROOT / 'T1', 'T1_Submission.zip')

    elif args.track == 2:
        print("\n=== Track 2: DINOv2 Ensemble (ViT-B + ViT-L) ===")
        print("  Strategy: ensemble of ViT-B student + ViT-L student")
        print("  TTA: multi-scale (0.75x, 1.0x, 1.25x) + HFlip (6 views/model)")
        run_t2(args.test_dir, device, threshold=args.threshold,
               use_ensemble=not args.no_ensemble)
        if not args.no_zip:
            create_zip(OUTPUT_ROOT / 'T2', 'T2_Submission.zip')

    print("\n=== Done ===")
    print(f"Submission files at: {OUTPUT_ROOT}")

    # Print summary of expected scores
    print("\n--- Expected Performance (from validation) ---")
    if args.track == 1:
        print("  T1 Synthetic model: F1=0.9786 (syn val), t=0.50, no TTA")
        print("  T1 Mixed model:     F1=0.9762 (mix val), t=0.50, no TTA")
    else:
        print("  T2 Ensemble (ViT-B+ViT-L): F1=0.9564 (val), t=0.50, TTA=ms_hflip (6 views)")
        print("  T2 Single (ViT-B):         F1=0.9552 (val), t=0.55, TTA=ms_hflip")


if __name__ == '__main__':
    main()
