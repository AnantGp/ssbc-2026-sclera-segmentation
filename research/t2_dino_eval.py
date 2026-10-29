"""Comprehensive evaluation: Teacher vs Student vs SWA + threshold sweep.

Evaluates all saved DINOv2 models on the validation set:
1. Student (best_model.pth)
2. Teacher (from best_model.pth checkpoint)
3. SWA (swa_model.pth)

For each model, sweeps binarisation threshold from 0.30 to 0.70
to find the optimal threshold for F1 score.
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

import t2_dino_config as cfg
from t2_dino_model import build_dino_model, compute_metrics
from t2_dataset import (
    build_labelled_pairs, split_train_val, ValidationDataset
)


@torch.no_grad()
def evaluate_model(model, val_loader, device, thresholds, use_bf16=True):
    """Evaluate model across multiple thresholds. Returns per-threshold metrics."""
    model.eval()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Collect all predictions and targets
    all_probs = []
    all_targets = []

    for images, masks in tqdm(val_loader, desc="  Inference", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=amp_dtype):
            logits = model(images)

        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_targets.append(masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Also do TTA (horizontal flip) and average
    all_probs_tta = []
    for images, masks in tqdm(val_loader, desc="  TTA Inference", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype):
            logits = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))
        probs = torch.sigmoid(logits.float())
        probs_flip = torch.sigmoid(logits_flip.float())
        probs_flip = torch.flip(probs_flip, dims=[3])
        probs_avg = (probs + probs_flip) / 2
        all_probs_tta.append(probs_avg.cpu())

    all_probs_tta = torch.cat(all_probs_tta, dim=0)

    # Sweep thresholds
    results = {}
    results_tta = {}
    for t in thresholds:
        preds = (all_probs > t).float()
        targets = (all_targets > 0.5).float()
        eps = 1e-7
        tp = (preds * targets).sum(dim=(1, 2, 3))
        fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        results[t] = {
            'dice': dice.mean().item(),
            'iou': iou.mean().item(),
            'f1': f1.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
        }

        # TTA version
        preds_tta = (all_probs_tta > t).float()
        tp = (preds_tta * targets).sum(dim=(1, 2, 3))
        fp = (preds_tta * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - preds_tta) * targets).sum(dim=(1, 2, 3))
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        results_tta[t] = {
            'dice': dice.mean().item(),
            'iou': iou.mean().item(),
            'f1': f1.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
        }

    return results, results_tta


def strip_compile_prefix(state_dict):
    """Remove _orig_mod. prefix added by torch.compile."""
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('_orig_mod.', '')] = v
    return new_sd


def load_model(backbone_name, feature_layers, decoder_dim):
    """Build a fresh model (no weights loaded)."""
    return build_dino_model(
        backbone_name=backbone_name,
        feature_layers=feature_layers,
        decoder_dim=decoder_dim,
        unfreeze_ln=False,
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load validation data ──
    print("\n=== Loading Validation Data ===")
    labelled_pairs, labelled_labels = build_labelled_pairs(cfg.T2_LABELLED)
    _, val_pairs, _ = split_train_val(
        labelled_pairs, labelled_labels,
        val_split=cfg.VAL_SPLIT, seed=cfg.SEED
    )
    print(f"Validation samples: {len(val_pairs)}")

    val_dataset = ValidationDataset(val_pairs, img_size=cfg.IMG_SIZE)
    val_loader = DataLoader(
        val_dataset, batch_size=8,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Thresholds to sweep
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    print(f"Threshold sweep: {thresholds}")

    all_results = {}

    # ── 1. Student Model (best_model.pth) ──
    best_ckpt = cfg.MODEL_DIR / 'best_model.pth'
    if best_ckpt.exists():
        print("\n=== Evaluating Student (best_model.pth) ===")
        model = load_model(cfg.BACKBONE, cfg.FEATURE_LAYERS, cfg.DECODER_DIM)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

        if 'student_state_dict' in ckpt:
            model.load_state_dict(strip_compile_prefix(ckpt['student_state_dict']))
            print(f"  Loaded student from epoch {ckpt.get('epoch', '?')}")
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(strip_compile_prefix(ckpt['model_state_dict']))
        else:
            model.load_state_dict(strip_compile_prefix(ckpt))

        model = model.to(device)
        results, results_tta = evaluate_model(model, val_loader, device, thresholds)
        all_results['student'] = {'no_tta': results, 'tta': results_tta}

        best_t = max(results, key=lambda t: results[t]['f1'])
        best_t_tta = max(results_tta, key=lambda t: results_tta[t]['f1'])
        print(f"  Best (no TTA): t={best_t:.2f} → F1={results[best_t]['f1']:.4f}, Dice={results[best_t]['dice']:.4f}")
        print(f"  Best (TTA):    t={best_t_tta:.2f} → F1={results_tta[best_t_tta]['f1']:.4f}, Dice={results_tta[best_t_tta]['dice']:.4f}")
        del model
        torch.cuda.empty_cache()

    # ── 2. Teacher Model (from best_model.pth) ──
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        if 'teacher_state_dict' in ckpt:
            print("\n=== Evaluating Teacher (from best_model.pth) ===")
            model = load_model(cfg.BACKBONE, cfg.FEATURE_LAYERS, cfg.DECODER_DIM)
            model.load_state_dict(strip_compile_prefix(ckpt['teacher_state_dict']))
            model = model.to(device)
            results, results_tta = evaluate_model(model, val_loader, device, thresholds)
            all_results['teacher'] = {'no_tta': results, 'tta': results_tta}

            best_t = max(results, key=lambda t: results[t]['f1'])
            best_t_tta = max(results_tta, key=lambda t: results_tta[t]['f1'])
            print(f"  Best (no TTA): t={best_t:.2f} → F1={results[best_t]['f1']:.4f}, Dice={results[best_t]['dice']:.4f}")
            print(f"  Best (TTA):    t={best_t_tta:.2f} → F1={results_tta[best_t_tta]['f1']:.4f}, Dice={results_tta[best_t_tta]['dice']:.4f}")
            del model
            torch.cuda.empty_cache()
        else:
            print("\n  No teacher_state_dict in checkpoint, skipping teacher eval")

    # ── 3. SWA Model (swa_model.pth) ──
    swa_ckpt = cfg.MODEL_DIR / 'swa_model.pth'
    if swa_ckpt.exists():
        print("\n=== Evaluating SWA (swa_model.pth) ===")
        model = load_model(cfg.BACKBONE, cfg.FEATURE_LAYERS, cfg.DECODER_DIM)
        ckpt = torch.load(swa_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(strip_compile_prefix(ckpt['model_state_dict']))
        model = model.to(device)
        results, results_tta = evaluate_model(model, val_loader, device, thresholds)
        all_results['swa'] = {'no_tta': results, 'tta': results_tta}

        best_t = max(results, key=lambda t: results[t]['f1'])
        best_t_tta = max(results_tta, key=lambda t: results_tta[t]['f1'])
        print(f"  Best (no TTA): t={best_t:.2f} → F1={results[best_t]['f1']:.4f}, Dice={results[best_t]['dice']:.4f}")
        print(f"  Best (TTA):    t={best_t_tta:.2f} → F1={results_tta[best_t_tta]['f1']:.4f}, Dice={results_tta[best_t_tta]['dice']:.4f}")
        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY — All Models × Thresholds × TTA")
    print("=" * 80)
    for model_name in ['student', 'teacher', 'swa']:
        if model_name not in all_results:
            continue
        for tta_mode in ['no_tta', 'tta']:
            res = all_results[model_name][tta_mode]
            best_t = max(res, key=lambda t: res[t]['f1'])
            tta_label = "TTA" if tta_mode == 'tta' else "No TTA"
            print(f"  {model_name:8s} ({tta_label:6s}): best_t={best_t:.2f} | "
                  f"F1={res[best_t]['f1']:.4f} | Dice={res[best_t]['dice']:.4f} | "
                  f"IoU={res[best_t]['iou']:.4f} | "
                  f"P={res[best_t]['precision']:.4f} R={res[best_t]['recall']:.4f}")
    print("=" * 80)

    # ── Detailed threshold table for best model ──
    print("\nDetailed threshold sweep for Student + TTA:")
    print(f"  {'Threshold':>9s} | {'F1':>7s} | {'Dice':>7s} | {'IoU':>7s} | {'Prec':>7s} | {'Recall':>7s}")
    print(f"  {'-'*9} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")
    if 'student' in all_results:
        for t in thresholds:
            r = all_results['student']['tta'][t]
            print(f"  {t:9.2f} | {r['f1']:7.4f} | {r['dice']:7.4f} | {r['iou']:7.4f} | {r['precision']:7.4f} | {r['recall']:7.4f}")

    # Save results to JSON
    output_file = cfg.OUTPUT_DIR / 'eval_results.json'
    # Convert float keys to strings for JSON
    json_results = {}
    for model_name, modes in all_results.items():
        json_results[model_name] = {}
        for mode, res in modes.items():
            json_results[model_name][mode] = {str(k): v for k, v in res.items()}

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
