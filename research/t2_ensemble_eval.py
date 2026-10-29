"""Cross-architecture ensemble eval: ViT-B + ViT-L for T2.

Averages predictions from multiple DINOv2 models (ViT-B and ViT-L)
and evaluates with threshold sweep + TTA.
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

import t2_dino_config as cfg_b        # ViT-B config
import t2_dino_large_config as cfg_l  # ViT-L config
from t2_dino_model import build_dino_model
from t2_dino_eval import strip_compile_prefix
from t2_dataset import (
    build_labelled_pairs, split_train_val, ValidationDataset
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_ensemble")


def load_model(cfg, ckpt_path, key='student_state_dict'):
    """Load a DINOv2 model from checkpoint."""
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
def get_predictions(model, images, img_size, use_bf16=True):
    """Get probability predictions, handling different input sizes."""
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    B, C, H, W = images.shape

    # Resize to model's expected size if needed
    if H != img_size or W != img_size:
        x = F.interpolate(images, size=(img_size, img_size), mode='bilinear', align_corners=False)
    else:
        x = images

    with torch.autocast('cuda', dtype=amp_dtype):
        logits = model(x)

    # Resize logits back to common size
    if logits.shape[2:] != (H, W):
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

    return torch.sigmoid(logits.float())


def main():
    print(f"Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Val data at ViT-B resolution (518) — we'll resize per model ──
    print("\n=== Loading Validation Data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg_b.T2_LABELLED)
    _, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg_b.VAL_SPLIT, seed=cfg_b.SEED
    )
    print(f"Validation samples: {len(val_pairs)}")

    # Use largest img_size as reference
    ref_size = max(cfg_b.IMG_SIZE, cfg_l.IMG_SIZE)
    val_dataset = ValidationDataset(val_pairs, img_size=ref_size)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=True)

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]

    # ── Define model configs ──
    model_configs = {
        'vitb_student': (cfg_b, cfg_b.MODEL_DIR / 'best_model.pth', 'student_state_dict', cfg_b.IMG_SIZE),
        'vitb_swa': (cfg_b, cfg_b.MODEL_DIR / 'swa_model.pth', 'model_state_dict', cfg_b.IMG_SIZE),
        'vitl_student': (cfg_l, cfg_l.MODEL_DIR / 'best_model.pth', 'student_state_dict', cfg_l.IMG_SIZE),
        'vitl_swa': (cfg_l, cfg_l.MODEL_DIR / 'swa_model.pth', 'model_state_dict', cfg_l.IMG_SIZE),
    }

    # ── Collect predictions from all models ──
    all_preds = {}       # name → (probs, probs_tta)

    for name, (cfg, ckpt_path, key, img_size) in model_configs.items():
        if not ckpt_path.exists():
            print(f"\n  {name}: checkpoint not found, skipping")
            continue
        print(f"\n=== {name} ===")
        model = load_model(cfg, ckpt_path, key)

        probs_list = []
        probs_tta_list = []
        targets_list = []

        for images, masks in tqdm(val_loader, desc=f"  {name}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            probs = get_predictions(model, images, img_size)
            probs_flip = get_predictions(model, torch.flip(images, dims=[3]), img_size)
            probs_flip = torch.flip(probs_flip, dims=[3])
            probs_avg = (probs + probs_flip) / 2

            probs_list.append(probs.cpu())
            probs_tta_list.append(probs_avg.cpu())
            targets_list.append(masks.cpu())

        all_preds[name] = (torch.cat(probs_list), torch.cat(probs_tta_list))
        targets = torch.cat(targets_list)

        del model
        torch.cuda.empty_cache()

    # ── Evaluate individual + ensemble combinations ──
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
                'precision': precision.mean().item(),
                'recall': recall.mean().item(),
            }
        return results

    # Individual results
    print("\n" + "=" * 90)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 90)

    summary = {}
    for name in all_preds:
        probs, probs_tta = all_preds[name]
        res = eval_probs(probs, targets, thresholds)
        res_tta = eval_probs(probs_tta, targets, thresholds)
        best_t = max(res, key=lambda t: res[t]['f1'])
        best_t_tta = max(res_tta, key=lambda t: res_tta[t]['f1'])
        print(f"  {name:20s} (No TTA): t={best_t:.2f} | F1={res[best_t]['f1']:.4f}")
        print(f"  {name:20s} (TTA   ): t={best_t_tta:.2f} | F1={res_tta[best_t_tta]['f1']:.4f}")
        summary[name] = {
            'no_tta': {'best_t': best_t, **res[best_t]},
            'tta': {'best_t': best_t_tta, **res_tta[best_t_tta]},
        }

    # ── Ensemble combinations ──
    print("\n" + "=" * 90)
    print("ENSEMBLE RESULTS (TTA)")
    print("=" * 90)

    ensemble_combos = [
        ('vitb_student+vitl_student', ['vitb_student', 'vitl_student']),
        ('vitb_swa+vitl_swa', ['vitb_swa', 'vitl_swa']),
        ('vitb_student+vitb_swa', ['vitb_student', 'vitb_swa']),
        ('vitl_student+vitl_swa', ['vitl_student', 'vitl_swa']),
        ('vitb_student+vitb_swa+vitl_student', ['vitb_student', 'vitb_swa', 'vitl_student']),
        ('vitb_student+vitb_swa+vitl_student+vitl_swa', ['vitb_student', 'vitb_swa', 'vitl_student', 'vitl_swa']),
        ('best3: vitb_s+vitb_swa+vitl_swa', ['vitb_student', 'vitb_swa', 'vitl_swa']),
    ]

    for combo_name, model_names in ensemble_combos:
        valid = [n for n in model_names if n in all_preds]
        if len(valid) < 2:
            continue

        # Average TTA predictions
        avg_probs = torch.stack([all_preds[n][1] for n in valid]).mean(dim=0)
        res = eval_probs(avg_probs, targets, thresholds)
        best_t = max(res, key=lambda t: res[t]['f1'])
        print(f"  {combo_name:45s}: t={best_t:.2f} | F1={res[best_t]['f1']:.4f} | "
              f"Dice={res[best_t]['dice']:.4f} | IoU={res[best_t]['iou']:.4f}")
        summary[f"ensemble_{combo_name}"] = {'tta': {'best_t': best_t, **res[best_t]}}

    print("=" * 90)

    # Save results
    json_results = {}
    for k, v in summary.items():
        json_results[k] = {}
        for mode, data in v.items():
            json_results[k][mode] = {str(kk): vv for kk, vv in data.items()}

    with open(OUTPUT_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'ensemble_results.json'}")


if __name__ == '__main__':
    main()
