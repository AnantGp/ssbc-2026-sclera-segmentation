"""T2 post-processing & threshold optimization.

Tests:
1. Threshold sweep (0.3-0.7) for each model combo
2. Morphological post-processing (remove small components, fill holes)
3. Expanded ensemble: ViT-B student + ViT-B SWA + ViT-L student + ViT-L SWA
4. CRF-like bilateral filtering for boundary refinement

Uses ms_hflip TTA (our best TTA mode).
"""
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg_b
import t2_dino_large_config as cfg_l
from t2_dino_model import build_dino_model, compute_metrics
from t2_dino_eval import strip_compile_prefix
from t2_dataset import (
    build_labelled_pairs, split_train_val, get_val_transforms
)

# ── Paths ──
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_optim")

# ── Models to load ──
MODEL_REGISTRY = {
    'vitb_student': {
        'config': cfg_b,
        'path': cfg_b.MODEL_DIR / 'best_model.pth',
        'key': 'student_state_dict',
    },
    'vitb_swa': {
        'config': cfg_b,
        'path': cfg_b.MODEL_DIR / 'swa_model.pth',
        'key': 'student_state_dict',  # SWA saved with this key
    },
    'vitl_student': {
        'config': cfg_l,
        'path': cfg_l.MODEL_DIR / 'best_model.pth',
        'key': 'student_state_dict',
    },
    'vitl_swa': {
        'config': cfg_l,
        'path': cfg_l.MODEL_DIR / 'swa_model.pth',
        'key': 'student_state_dict',
    },
}


def load_model(name, device):
    """Load a model from registry."""
    info = MODEL_REGISTRY[name]
    cfg = info['config']
    path = info['path']

    if not path.exists():
        print(f"  WARNING: {name} not found at {path}")
        return None

    m = build_dino_model(
        backbone_name=cfg.BACKBONE,
        feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM,
        unfreeze_ln=False,
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Try different keys
    state = None
    for key in [info['key'], 'model_state_dict', 'student_state_dict']:
        if key in ckpt:
            state = strip_compile_prefix(ckpt[key])
            break
    if state is None:
        # Maybe the checkpoint IS the state dict
        state = strip_compile_prefix(ckpt)

    m.load_state_dict(state)
    m = m.to(device).eval()
    print(f"  Loaded {name} from {path.name}")
    return m


@torch.no_grad()
def predict_ms_hflip(model, img_size, x_base, device):
    """Predict with multi-scale + HFlip TTA. Returns probability map at x_base size."""
    views = []
    for scale in [0.75, 1.0, 1.25]:
        target_size = max(14, (int(img_size * scale) // 14) * 14)
        x_scaled = F.interpolate(x_base, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x_scaled)
        logits = F.interpolate(logits, size=x_base.shape[2:], mode='bilinear', align_corners=False)
        views.append(torch.sigmoid(logits.float()))

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits_f = model(torch.flip(x_scaled, [3]))
        logits_f = F.interpolate(logits_f, size=x_base.shape[2:], mode='bilinear', align_corners=False)
        views.append(torch.flip(torch.sigmoid(logits_f.float()), [3]))

    return torch.stack(views).mean(dim=0)


def morph_postprocess(binary_mask, min_area=50, kernel_size=3):
    """Apply morphological post-processing:
    1. Remove small connected components
    2. Fill small holes
    3. Slight morphological closing
    """
    mask = binary_mask.astype(np.uint8)

    # Morphological closing (fills small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    # Fill small holes (invert, remove small components, invert back)
    inv = 1 - mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            inv[labels == i] = 0
    mask = 1 - inv

    return mask


def evaluate_config(prob_maps, gt_masks, threshold, use_morph=False, min_area=50, kernel_size=3):
    """Evaluate a configuration on all validation images."""
    f1_sum = 0
    n = 0
    for prob, gt in zip(prob_maps, gt_masks):
        pred = (prob > threshold).astype(np.uint8)
        if use_morph:
            pred = morph_postprocess(pred, min_area=min_area, kernel_size=kernel_size)

        # Compute F1
        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_sum += f1
        n += 1

    return f1_sum / n if n > 0 else 0


def main():
    import json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    print("\n=== Loading models ===")
    models = {}
    for name in MODEL_REGISTRY:
        m = load_model(name, device)
        if m is not None:
            models[name] = m

    print(f"  Loaded {len(models)} models")

    # ── Load validation data ──
    print("\n=== Loading validation data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    _, val_pairs, _ = split_train_val(labelled_pairs, labelled_labels,
                                       val_split=cfg_b.VAL_SPLIT, seed=42)
    print(f"  Validation images: {len(val_pairs)}")

    # ── Generate per-model probability maps for val set ──
    # Evaluate at 518×518 model resolution (same as previous evals)
    print("\n=== Generating probability maps (at model resolution) ===")
    model_probs = {name: [] for name in models}
    gt_masks = []

    transform_b = get_val_transforms(cfg_b.IMG_SIZE)
    transform_l = get_val_transforms(cfg_l.IMG_SIZE)

    # Use ViT-B transform for GT masks (both models use 518)
    gt_transform = get_val_transforms(cfg_b.IMG_SIZE)

    for img_path, mask_path in tqdm(val_pairs, desc="Predicting"):
        # Load image and GT mask
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_float = gt.astype(np.float32) / 255.0

        # Apply same transform to GT mask (at 518×518)
        gt_out = gt_transform(image=image_rgb, mask=gt_float)
        gt_mask = gt_out['mask']
        if hasattr(gt_mask, 'numpy'):
            gt_mask = gt_mask.numpy()
        gt_518 = (gt_mask > 0.5).astype(np.uint8)
        gt_masks.append(gt_518)

        # Generate predictions for each model
        for name, model in models.items():
            cfg = MODEL_REGISTRY[name]['config']
            transform = transform_b if 'vitb' in name else transform_l
            img_size = cfg.IMG_SIZE

            out = transform(image=image_rgb)
            x = out['image'].unsqueeze(0).to(device)

            prob = predict_ms_hflip(model, img_size, x, device)
            # Keep at model resolution (518×518) — no resize
            model_probs[name].append(prob.squeeze().cpu().numpy())

    print(f"  Generated {len(gt_masks)} predictions per model")

    # ── Define ensemble combinations ──
    ensembles = {
        'vitb_student': ['vitb_student'],
        'vitl_student': ['vitl_student'],
        'vitb_swa': ['vitb_swa'],
        'vitl_swa': ['vitl_swa'],
        'vitb_s+l_s': ['vitb_student', 'vitl_student'],
        'vitb_swa+l_s': ['vitb_swa', 'vitl_student'],
        'vitb_s+l_swa': ['vitb_student', 'vitl_swa'],
        'vitb_swa+l_swa': ['vitb_swa', 'vitl_swa'],
        'all4': ['vitb_student', 'vitb_swa', 'vitl_student', 'vitl_swa'],
        'vitb_both+l_s': ['vitb_student', 'vitb_swa', 'vitl_student'],
        'vitb_s+l_both': ['vitb_student', 'vitl_student', 'vitl_swa'],
    }

    # Filter to available models
    ensembles = {k: [m for m in v if m in models] for k, v in ensembles.items()}
    ensembles = {k: v for k, v in ensembles.items() if v}

    # ── Compute ensemble probability maps ──
    print("\n=== Computing ensemble predictions ===")
    ensemble_probs = {}
    for ens_name, model_names in ensembles.items():
        probs = []
        for i in range(len(gt_masks)):
            avg = np.mean([model_probs[m][i] for m in model_names], axis=0)
            probs.append(avg)
        ensemble_probs[ens_name] = probs

    # ── Threshold sweep ──
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n=== Threshold sweep (no post-processing) ===")
    results = {}
    for ens_name, probs in ensemble_probs.items():
        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            f1 = evaluate_config(probs, gt_masks, t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        results[ens_name] = {'f1': best_f1, 'threshold': best_t}
        print(f"  {ens_name:20s} | best t={best_t:.2f} | F1={best_f1:.4f}")

    # ── Threshold sweep with morphological post-processing ──
    morph_configs = [
        {'min_area': 30, 'kernel_size': 3},
        {'min_area': 50, 'kernel_size': 3},
        {'min_area': 100, 'kernel_size': 3},
        {'min_area': 50, 'kernel_size': 5},
    ]

    print("\n=== Threshold sweep + morphological post-processing ===")
    morph_results = {}
    for ens_name, probs in ensemble_probs.items():
        best_f1 = 0
        best_config = {}
        for mc in morph_configs:
            for t in thresholds:
                f1 = evaluate_config(probs, gt_masks, t, use_morph=True, **mc)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {'threshold': t, **mc, 'f1': best_f1}
        morph_results[ens_name] = best_config
        print(f"  {ens_name:20s} | t={best_config['threshold']:.2f} "
              f"area={best_config['min_area']} kern={best_config['kernel_size']} | "
              f"F1={best_config['f1']:.4f}")

    # ── Summary: best overall ──
    print("\n" + "=" * 60)
    print("=== SUMMARY: Top 5 configurations ===")
    print("=" * 60)

    all_configs = []
    for ens_name, r in results.items():
        all_configs.append({
            'ensemble': ens_name, 'threshold': r['threshold'],
            'morph': False, 'f1': r['f1']
        })
    for ens_name, r in morph_results.items():
        all_configs.append({
            'ensemble': ens_name, 'threshold': r['threshold'],
            'morph': True, 'min_area': r['min_area'],
            'kernel_size': r['kernel_size'], 'f1': r['f1']
        })

    all_configs.sort(key=lambda x: x['f1'], reverse=True)
    for i, c in enumerate(all_configs[:10]):
        morph_str = f"morph(area={c.get('min_area','')},k={c.get('kernel_size','')})" if c['morph'] else "no morph"
        print(f"  #{i+1}: {c['ensemble']:20s} | t={c['threshold']:.2f} | {morph_str:25s} | F1={c['f1']:.4f}")

    # ── Save full results ──
    results_path = OUTPUT_DIR / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'threshold_results': {k: {'f1': float(v['f1']), 'threshold': v['threshold']}
                                   for k, v in results.items()},
            'morph_results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                   for kk, vv in v.items()}
                              for k, v in morph_results.items()},
            'top10': [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in c.items()} for c in all_configs[:10]],
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Cleanup
    for m in models.values():
        del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
