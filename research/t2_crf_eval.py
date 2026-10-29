"""T2 CRF post-processing evaluation.

Tests Dense CRF refinement on top of our best ensemble (ViT-B + ViT-L student)
with ms_hflip TTA. CRF uses original image appearance to refine boundaries.

Also tests guided filtering as a lightweight alternative.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_config as cfg_b
import t2_dino_large_config as cfg_l
from t2_dino_model import build_dino_model
from t2_dino_eval import strip_compile_prefix
from t2_dataset import build_labelled_pairs, split_train_val, get_val_transforms

OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_crf")


def load_model(name, device):
    """Load a T2 model."""
    registry = {
        'vitb_student': (cfg_b, cfg_b.MODEL_DIR / 'best_model.pth', 'student_state_dict'),
        'vitl_student': (cfg_l, cfg_l.MODEL_DIR / 'best_model.pth', 'student_state_dict'),
    }
    cfg, path, key = registry[name]
    m = build_dino_model(
        backbone_name=cfg.BACKBONE,
        feature_layers=cfg.FEATURE_LAYERS,
        decoder_dim=cfg.DECODER_DIM,
        unfreeze_ln=False,
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = None
    for k in [key, 'model_state_dict', 'student_state_dict']:
        if k in ckpt:
            state = strip_compile_prefix(ckpt[k])
            break
    if state is None:
        state = strip_compile_prefix(ckpt)
    m.load_state_dict(state)
    m = m.to(device).eval()
    print(f"  Loaded {name} from {path.name}")
    return m


@torch.no_grad()
def predict_ms_hflip(model, img_size, x_base, device):
    """Multi-scale + HFlip TTA → probability map."""
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


def apply_dense_crf(image_rgb, prob_map, sxy_gauss=3, compat_gauss=3,
                    sxy_bilat=80, srgb_bilat=13, compat_bilat=10, n_iters=5):
    """Apply Dense CRF to refine a probability map using image appearance.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        prob_map: (H, W) float32 probability of foreground [0, 1]
        sxy_gauss: spatial sigma for Gaussian pairwise
        compat_gauss: compatibility weight for Gaussian
        sxy_bilat: spatial sigma for bilateral pairwise
        srgb_bilat: color sigma for bilateral pairwise
        compat_bilat: compatibility weight for bilateral
        n_iters: CRF inference iterations
    Returns:
        refined_prob: (H, W) float32 refined probability
    """
    H, W = prob_map.shape

    # Build 2-class unary potentials from our model probabilities
    probs_2c = np.stack([1.0 - prob_map, prob_map], axis=0).astype(np.float32)
    # Clip for numerical stability
    probs_2c = np.clip(probs_2c, 1e-6, 1.0 - 1e-6)
    probs_2c /= probs_2c.sum(axis=0, keepdims=True)
    unary = unary_from_softmax(probs_2c)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)

    # Gaussian pairwise (spatial smoothness)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

    # Bilateral pairwise (appearance-based)
    d.addPairwiseBilateral(
        sxy=sxy_bilat, srgb=srgb_bilat, rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilat
    )

    # Inference
    Q = d.inference(n_iters)
    refined = np.array(Q).reshape(2, H, W)
    return refined[1]  # foreground probability


def apply_guided_filter(image, prob_map, radius=8, eps=0.01):
    """Apply guided filtering to refine probability map using image edges.

    Lightweight alternative to CRF — uses image as guide to preserve edges
    while smoothing the probability map.
    """
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = prob_map.astype(np.float32)
    # OpenCV's guidedFilter
    result = cv2.ximgproc.guidedFilter(guide, p, radius, eps)
    return np.clip(result, 0, 1)


def compute_f1(pred, gt):
    """Compute F1 score."""
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n=== Loading models ===")
    vitb = load_model('vitb_student', device)
    vitl = load_model('vitl_student', device)

    # Load validation data
    print("\n=== Loading validation data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    _, val_pairs, _ = split_train_val(labelled_pairs, labelled_labels,
                                       val_split=cfg_b.VAL_SPLIT, seed=42)
    print(f"  Validation images: {len(val_pairs)}")

    transform_b = get_val_transforms(cfg_b.IMG_SIZE)
    transform_l = get_val_transforms(cfg_l.IMG_SIZE)
    gt_transform = get_val_transforms(cfg_b.IMG_SIZE)

    # Generate ensemble prob maps + store original images for CRF
    print("\n=== Generating ensemble probability maps ===")
    prob_maps = []
    gt_masks = []
    images_518 = []  # images at 518×518 for CRF

    for img_path, mask_path in tqdm(val_pairs, desc="Predicting"):
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # GT at 518×518
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_float = gt.astype(np.float32) / 255.0
        gt_out = gt_transform(image=image_rgb, mask=gt_float)
        gt_mask = gt_out['mask']
        if hasattr(gt_mask, 'numpy'):
            gt_mask = gt_mask.numpy()
        gt_518 = (gt_mask > 0.5).astype(np.uint8)
        gt_masks.append(gt_518)

        # Store the transformed image (518×518) for CRF
        img_out = gt_transform(image=image_rgb)
        img_tensor = img_out['image']
        if hasattr(img_tensor, 'numpy'):
            img_518 = img_tensor.numpy()
        else:
            img_518 = img_tensor
        # Undo normalization for CRF: image needs to be uint8 RGB
        # The transform applies Normalize + ToTensorV2, so we need original resized image
        img_resized = cv2.resize(image_rgb, (cfg_b.IMG_SIZE, cfg_b.IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        images_518.append(img_resized)

        # Ensemble prediction (ViT-B + ViT-L)
        views = []
        for model, transform, img_size in [
            (vitb, transform_b, cfg_b.IMG_SIZE),
            (vitl, transform_l, cfg_l.IMG_SIZE),
        ]:
            out = transform(image=image_rgb)
            x = out['image'].unsqueeze(0).to(device)
            prob = predict_ms_hflip(model, img_size, x, device)
            views.append(prob.squeeze().cpu().numpy())

        # Average ensemble
        ensemble_prob = np.mean(views, axis=0)
        prob_maps.append(ensemble_prob)

    print(f"  Generated {len(prob_maps)} ensemble predictions")

    # Baseline (no CRF, t=0.50)
    baseline_f1s = []
    for prob, gt in zip(prob_maps, gt_masks):
        pred = (prob > 0.50).astype(np.uint8)
        baseline_f1s.append(compute_f1(pred, gt))
    baseline_f1 = np.mean(baseline_f1s)
    print(f"\n  Baseline (no CRF, t=0.50): F1={baseline_f1:.4f}")

    # ── Dense CRF parameter sweep ──
    print("\n=== Dense CRF parameter sweep ===")
    crf_configs = [
        # (sxy_gauss, compat_gauss, sxy_bilat, srgb_bilat, compat_bilat, n_iters)
        # Light CRF
        (3, 3, 40, 10, 5, 5),
        (3, 3, 60, 13, 5, 5),
        (3, 3, 80, 13, 10, 5),
        # Medium CRF
        (3, 3, 80, 13, 5, 10),
        (3, 5, 80, 13, 10, 5),
        (3, 3, 50, 10, 10, 5),
        # Strong CRF
        (5, 5, 100, 15, 10, 10),
        (3, 3, 120, 15, 10, 5),
        # Very light (minimal smoothing)
        (1, 1, 30, 8, 3, 3),
        (2, 2, 50, 10, 5, 3),
        # Focused bilateral
        (1, 1, 60, 13, 10, 5),
        (1, 1, 80, 13, 15, 5),
    ]

    crf_results = []
    for cfg_idx, (sxy_g, c_g, sxy_b, srgb_b, c_b, n_iter) in enumerate(crf_configs):
        config_name = f"g({sxy_g},{c_g})_b({sxy_b},{srgb_b},{c_b})_i{n_iter}"

        # Apply CRF to each image
        refined_probs = []
        for img, prob in zip(images_518, prob_maps):
            refined = apply_dense_crf(img, prob.astype(np.float32),
                                       sxy_gauss=sxy_g, compat_gauss=c_g,
                                       sxy_bilat=sxy_b, srgb_bilat=srgb_b,
                                       compat_bilat=c_b, n_iters=n_iter)
            refined_probs.append(refined)

        # Test thresholds on CRF-refined probs
        best_f1 = 0
        best_t = 0.5
        for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
            f1s = []
            for prob, gt in zip(refined_probs, gt_masks):
                pred = (prob > t).astype(np.uint8)
                f1s.append(compute_f1(pred, gt))
            f1 = np.mean(f1s)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        delta = best_f1 - baseline_f1
        crf_results.append({
            'config': config_name,
            'params': {'sxy_gauss': sxy_g, 'compat_gauss': c_g,
                       'sxy_bilat': sxy_b, 'srgb_bilat': srgb_b,
                       'compat_bilat': c_b, 'n_iters': n_iter},
            'best_t': best_t,
            'f1': best_f1,
            'delta': delta,
        })
        sign = '+' if delta >= 0 else ''
        print(f"  [{cfg_idx+1:2d}/{len(crf_configs)}] {config_name:35s} | t={best_t:.2f} | F1={best_f1:.4f} ({sign}{delta:.4f})")

    # ── Guided filter (if ximgproc available) ──
    gf_results = []
    has_ximgproc = hasattr(cv2, 'ximgproc')
    if has_ximgproc:
        print("\n=== Guided filter sweep ===")
        gf_configs = [
            (4, 0.001), (4, 0.01), (4, 0.1),
            (8, 0.001), (8, 0.01), (8, 0.1),
            (16, 0.01), (16, 0.1),
        ]
        for radius, eps in gf_configs:
            refined_probs = []
            for img, prob in zip(images_518, prob_maps):
                refined = apply_guided_filter(img, prob.astype(np.float32), radius, eps)
                refined_probs.append(refined)

            best_f1 = 0
            best_t = 0.5
            for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
                f1s = []
                for prob, gt in zip(refined_probs, gt_masks):
                    pred = (prob > t).astype(np.uint8)
                    f1s.append(compute_f1(pred, gt))
                f1 = np.mean(f1s)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            delta = best_f1 - baseline_f1
            sign = '+' if delta >= 0 else ''
            gf_results.append({
                'radius': radius, 'eps': eps,
                'best_t': best_t, 'f1': best_f1, 'delta': delta,
            })
            print(f"  r={radius:2d} eps={eps:.3f} | t={best_t:.2f} | F1={best_f1:.4f} ({sign}{delta:.4f})")
    else:
        print("\n  cv2.ximgproc not available — skipping guided filter")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)
    print(f"  Baseline (no CRF): F1={baseline_f1:.4f}")

    all_results = sorted(crf_results + gf_results, key=lambda x: x['f1'], reverse=True)
    print(f"\n  Top 5 post-processing configs:")
    for i, r in enumerate(all_results[:5]):
        sign = '+' if r['delta'] >= 0 else ''
        if 'config' in r:
            print(f"    #{i+1}: CRF {r['config']} | t={r['best_t']:.2f} | F1={r['f1']:.4f} ({sign}{r['delta']:.4f})")
        else:
            print(f"    #{i+1}: GF r={r['radius']} eps={r['eps']} | t={r['best_t']:.2f} | F1={r['f1']:.4f} ({sign}{r['delta']:.4f})")

    # Save results
    results_path = OUTPUT_DIR / 'crf_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'baseline_f1': float(baseline_f1),
            'crf_results': crf_results,
            'guided_filter_results': gf_results,
            'top5': [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                      for k, v in r.items()} for r in all_results[:5]],
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Cleanup
    del vitb, vitl
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
