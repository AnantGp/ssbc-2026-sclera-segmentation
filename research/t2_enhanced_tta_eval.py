"""Enhanced TTA evaluation for T2 ensemble.

Tests additional TTA strategies beyond HFlip:
  1. HFlip + VFlip
  2. HFlip + VFlip + 90/270 rotations (4-fold)
  3. Multi-scale (0.75x, 1.0x, 1.25x) + HFlip
  4. Full: multi-scale + all geometric augmentations

Compares against baseline (HFlip only) to find the best TTA strategy.
"""
import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg_b
import t2_dino_large_config as cfg_l
from t2_dino_model import build_dino_model
from t2_dino_eval import strip_compile_prefix
from t2_dataset import build_labelled_pairs, split_train_val, ValidationDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_enhanced_tta")


def load_model(cfg, ckpt_path, key='student_state_dict'):
    model = build_dino_model(
        backbone_name=cfg.BACKBONE,
        feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM,
        unfreeze_ln=False,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if key in ckpt:
        model.load_state_dict(strip_compile_prefix(ckpt[key]))
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(strip_compile_prefix(ckpt['model_state_dict']))
    else:
        model.load_state_dict(strip_compile_prefix(ckpt))
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_at_scale(model, images, model_img_size, scale=1.0, use_bf16=True):
    """Run prediction at a given scale, return probs at original resolution."""
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    B, C, H, W = images.shape

    # Compute target size for this scale
    target_size = int(model_img_size * scale)
    # Make divisible by 14 (patch size)
    target_size = max(14, (target_size // 14) * 14)

    x = F.interpolate(images, size=(target_size, target_size),
                       mode='bilinear', align_corners=False)

    with torch.autocast('cuda', dtype=amp_dtype):
        logits = model(x)

    # Resize back to common size
    if logits.shape[2:] != (H, W):
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

    return torch.sigmoid(logits.float())


@torch.no_grad()
def predict_with_enhanced_tta(model, images, model_img_size, tta_mode='hflip'):
    """Predict with various TTA strategies.

    tta_mode:
      'none'      — no TTA
      'hflip'     — horizontal flip (baseline)
      'hvflip'    — horizontal + vertical flip (3 views)
      'geo4'      — hflip + vflip + 90° + 270° (5 views)
      'ms_hflip'  — multi-scale (0.75, 1.0, 1.25) × hflip (6 views)
      'full'      — multi-scale × geo4 (15 views)
    """
    # Original prediction
    probs = predict_at_scale(model, images, model_img_size, scale=1.0)
    views = [probs]

    if tta_mode == 'none':
        return probs

    # Horizontal flip
    if tta_mode in ('hflip', 'hvflip', 'geo4', 'ms_hflip', 'full'):
        p = predict_at_scale(model, torch.flip(images, [3]), model_img_size)
        views.append(torch.flip(p, [3]))

    # Vertical flip
    if tta_mode in ('hvflip', 'geo4', 'full'):
        p = predict_at_scale(model, torch.flip(images, [2]), model_img_size)
        views.append(torch.flip(p, [2]))

    # 90° rotation (rot90 k=1: rotate CCW)
    if tta_mode in ('geo4', 'full'):
        x_rot90 = torch.rot90(images, k=1, dims=[2, 3])
        p = predict_at_scale(model, x_rot90, model_img_size)
        # Undo: rotate CW (k=3)
        p_back = torch.rot90(p, k=3, dims=[2, 3])
        # Resize back if aspect ratio changed
        if p_back.shape[2:] != probs.shape[2:]:
            p_back = F.interpolate(p_back, size=probs.shape[2:],
                                    mode='bilinear', align_corners=False)
        views.append(p_back)

    # 270° rotation (rot90 k=3: rotate CW)
    if tta_mode in ('geo4', 'full'):
        x_rot270 = torch.rot90(images, k=3, dims=[2, 3])
        p = predict_at_scale(model, x_rot270, model_img_size)
        p_back = torch.rot90(p, k=1, dims=[2, 3])
        if p_back.shape[2:] != probs.shape[2:]:
            p_back = F.interpolate(p_back, size=probs.shape[2:],
                                    mode='bilinear', align_corners=False)
        views.append(p_back)

    # Multi-scale
    if tta_mode in ('ms_hflip', 'full'):
        for scale in [0.75, 1.25]:
            p = predict_at_scale(model, images, model_img_size, scale=scale)
            views.append(p)
            # HFlip at this scale
            p_flip = predict_at_scale(model, torch.flip(images, [3]),
                                       model_img_size, scale=scale)
            views.append(torch.flip(p_flip, [3]))

        if tta_mode == 'full':
            # Multi-scale + vflip + rotations
            for scale in [0.75, 1.25]:
                # VFlip
                p = predict_at_scale(model, torch.flip(images, [2]),
                                      model_img_size, scale=scale)
                views.append(torch.flip(p, [2]))
                # Rot90
                x_rot = torch.rot90(images, k=1, dims=[2, 3])
                p = predict_at_scale(model, x_rot, model_img_size, scale=scale)
                p_back = torch.rot90(p, k=3, dims=[2, 3])
                if p_back.shape[2:] != probs.shape[2:]:
                    p_back = F.interpolate(p_back, size=probs.shape[2:],
                                            mode='bilinear', align_corners=False)
                views.append(p_back)

    return torch.stack(views).mean(dim=0)


def eval_probs(probs, targets, thresholds):
    results = {}
    for t in thresholds:
        preds = (probs > t).float()
        tgt = (targets > 0.5).float()
        eps = 1e-7
        tp = (preds * tgt).sum(dim=(1, 2, 3))
        fp = (preds * (1 - tgt)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * tgt).sum(dim=(1, 2, 3))
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        results[t] = {
            'f1': f1.mean().item(),
            'dice': dice.mean().item(),
            'iou': iou.mean().item(),
        }
    return results


def main():
    print(f"Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Validation data ──
    print("\n=== Loading Validation Data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    _, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg_b.VAL_SPLIT, seed=cfg_b.SEED
    )
    print(f"Validation samples: {len(val_pairs)}")

    ref_size = max(cfg_b.IMG_SIZE, cfg_l.IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=ref_size)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                            num_workers=4, pin_memory=True)

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]

    # ── Load models ──
    models = {}
    print("\n=== Loading Models ===")

    vitb_path = cfg_b.MODEL_DIR / 'best_model.pth'
    if vitb_path.exists():
        print("  Loading ViT-B student...")
        models['vitb'] = (load_model(cfg_b, vitb_path, 'student_state_dict'), cfg_b.IMG_SIZE)

    vitl_path = cfg_l.MODEL_DIR / 'best_model.pth'
    if vitl_path.exists():
        print("  Loading ViT-L student...")
        models['vitl'] = (load_model(cfg_l, vitl_path, 'student_state_dict'), cfg_l.IMG_SIZE)

    tta_modes = ['none', 'hflip', 'hvflip', 'geo4', 'ms_hflip', 'full']

    # ── Collect predictions with all TTA modes ──
    # For each model, run each TTA mode and store predictions
    all_results = {}

    for model_name, (model, img_size) in models.items():
        for tta_mode in tta_modes:
            key = f"{model_name}_{tta_mode}"
            n_views = {'none': 1, 'hflip': 2, 'hvflip': 3, 'geo4': 5,
                       'ms_hflip': 6, 'full': 15}[tta_mode]
            print(f"\n  {key} ({n_views} views)...")

            probs_list = []
            targets_list = []

            for images, masks in tqdm(val_loader, desc=f"    {key}", leave=False):
                images = images.to(DEVICE, non_blocking=True)

                probs = predict_with_enhanced_tta(model, images, img_size, tta_mode)
                probs_list.append(probs.cpu())
                targets_list.append(masks.cpu())

            all_probs = torch.cat(probs_list)
            targets = torch.cat(targets_list)

            res = eval_probs(all_probs, targets, thresholds)
            best_t = max(res, key=lambda t: res[t]['f1'])
            all_results[key] = {
                'best_t': best_t,
                'f1': res[best_t]['f1'],
                'dice': res[best_t]['dice'],
                'iou': res[best_t]['iou'],
                'probs': all_probs,  # Keep for ensemble
            }
            print(f"    → t={best_t:.2f} | F1={res[best_t]['f1']:.4f}")

    # ── Ensemble with enhanced TTA ──
    print("\n" + "=" * 90)
    print("ENSEMBLE RESULTS WITH ENHANCED TTA")
    print("=" * 90)

    ensemble_results = {}

    for tta_mode in tta_modes:
        vitb_key = f"vitb_{tta_mode}"
        vitl_key = f"vitl_{tta_mode}"
        if vitb_key not in all_results or vitl_key not in all_results:
            continue

        combo_name = f"vitb+vitl_{tta_mode}"
        avg_probs = (all_results[vitb_key]['probs'] + all_results[vitl_key]['probs']) / 2
        res = eval_probs(avg_probs, targets, thresholds)
        best_t = max(res, key=lambda t: res[t]['f1'])
        ensemble_results[combo_name] = {
            'best_t': best_t,
            'f1': res[best_t]['f1'],
            'dice': res[best_t]['dice'],
            'iou': res[best_t]['iou'],
        }
        n_views = {'none': 1, 'hflip': 2, 'hvflip': 3, 'geo4': 5,
                   'ms_hflip': 6, 'full': 15}[tta_mode]
        print(f"  {combo_name:35s}: t={best_t:.2f} | F1={res[best_t]['f1']:.4f} | views={n_views}")

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("SUMMARY: INDIVIDUAL MODELS")
    print("=" * 90)
    print(f"  {'Config':35s} {'Threshold':>10s} {'F1':>8s} {'Dice':>8s} {'IoU':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {key:35s} {r['best_t']:10.2f} {r['f1']:8.4f} {r['dice']:8.4f} {r['iou']:8.4f}")

    print(f"\n{'=' * 90}")
    print("SUMMARY: ENSEMBLES")
    print("=" * 90)
    print(f"  {'Config':35s} {'Threshold':>10s} {'F1':>8s} {'Dice':>8s} {'IoU':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for key in sorted(ensemble_results.keys()):
        r = ensemble_results[key]
        print(f"  {key:35s} {r['best_t']:10.2f} {r['f1']:8.4f} {r['dice']:8.4f} {r['iou']:8.4f}")

    # Find best overall
    best_single = max(all_results, key=lambda k: all_results[k]['f1'])
    best_ensemble = max(ensemble_results, key=lambda k: ensemble_results[k]['f1'])
    print(f"\n  BEST single:   {best_single:35s} F1={all_results[best_single]['f1']:.4f}")
    print(f"  BEST ensemble: {best_ensemble:35s} F1={ensemble_results[best_ensemble]['f1']:.4f}")

    # Save results (without tensors)
    save_results = {
        'individual': {k: {kk: vv for kk, vv in v.items() if kk != 'probs'}
                       for k, v in all_results.items()},
        'ensemble': ensemble_results,
        'best_single': best_single,
        'best_ensemble': best_ensemble,
    }
    with open(OUTPUT_DIR / 'enhanced_tta_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'enhanced_tta_results.json'}")


if __name__ == '__main__':
    main()
