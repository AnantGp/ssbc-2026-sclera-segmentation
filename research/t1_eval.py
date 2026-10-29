"""T1 Evaluation: threshold sweep + TTA + ensemble for SAM+LoRA v3 models.

Evaluates:
1. t1_synthetic (v3) — F1=0.9791
2. t1_mixed (v3) — F1=0.9784
3. SWA models if available
4. Ensemble of synthetic + mixed predictions

TTA variants: original, HFlip, VFlip, original+HFlip+VFlip average
Threshold sweep: 0.30 to 0.70 step 0.05
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

from config import *
from dataset import ScleraDataset, build_pairs
from model import SAMLoRASeg


def strip_dp_prefix(state_dict):
    """Remove module. prefix from DataParallel and _orig_mod. from compile."""
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        new_sd[k] = v
    return new_sd


def load_t1_model(checkpoint_path, device, lora_rank=16, lora_alpha=32):
    """Load a T1 SAM+LoRA v3 model."""
    model = SAMLoRASeg(SAM_CHECKPOINT, lora_rank=lora_rank, lora_alpha=lora_alpha)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    sd = strip_dp_prefix(sd)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    f1 = ckpt.get('best_f1', '?')
    print(f"  Loaded {checkpoint_path.name}: epoch={epoch}, train_best_f1={f1:.4f}")
    return model


@torch.no_grad()
def collect_predictions(model, val_loader, device, do_tta=False):
    """Collect probability maps for all val samples. Returns [N,1,H,W] tensor."""
    all_probs = []
    for images, masks in tqdm(val_loader, desc="  Inference", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits.float())

        if do_tta:
            # HFlip
            with torch.amp.autocast('cuda'):
                logits_hf = model(torch.flip(images, dims=[3]))
            probs_hf = torch.sigmoid(logits_hf.float())
            probs_hf = torch.flip(probs_hf, dims=[3])

            # VFlip
            with torch.amp.autocast('cuda'):
                logits_vf = model(torch.flip(images, dims=[2]))
            probs_vf = torch.sigmoid(logits_vf.float())
            probs_vf = torch.flip(probs_vf, dims=[2])

            probs = (probs + probs_hf + probs_vf) / 3.0

        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


def collect_targets(val_loader):
    """Collect all ground truth masks."""
    all_targets = []
    for _, masks in val_loader:
        masks = F.interpolate(masks, size=MASK_OUT_SIZE, mode='nearest')
        all_targets.append(masks)
    return torch.cat(all_targets, dim=0)


def sweep_thresholds(probs, targets, thresholds):
    """Compute metrics across threshold range."""
    results = {}
    targets_bin = (targets > 0.5).float()
    eps = 1e-7
    for t in thresholds:
        preds = (probs > t).float()
        tp = (preds * targets_bin).sum(dim=(1, 2, 3))
        fp = (preds * (1 - targets_bin)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * targets_bin).sum(dim=(1, 2, 3))
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        results[t] = {
            'dice': dice.mean().item(),
            'f1': f1.mean().item(),
            'iou': iou.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
        }
    return results


def print_best(name, results, results_tta=None):
    """Print best threshold for a model."""
    best_t = max(results, key=lambda t: results[t]['f1'])
    r = results[best_t]
    print(f"  {name:25s} (No TTA): t={best_t:.2f} | F1={r['f1']:.4f} | Dice={r['dice']:.4f} | IoU={r['iou']:.4f}")
    if results_tta:
        best_t_tta = max(results_tta, key=lambda t: results_tta[t]['f1'])
        r = results_tta[best_t_tta]
        print(f"  {name:25s} (TTA   ): t={best_t_tta:.2f} | F1={r['f1']:.4f} | Dice={r['dice']:.4f} | IoU={r['iou']:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Build validation set (same split as training) ──
    # Use mixed mode datasets to get the full val set
    print("\n=== Building Validation Set ===")

    # Replicate the exact same random split as training
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # We need to evaluate on synthetic val + mixed val
    # But the split depends on which datasets were used
    # Let's eval on BOTH val sets and report separately

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    all_results = {}

    # ── Synthetic val set ──
    print("\n--- Synthetic validation set ---")
    np.random.seed(SEED)
    syn_pairs, syn_labels = build_pairs(T1_DATASETS['synthetic'])
    syn_indices = np.random.permutation(len(syn_pairs))
    n_val_syn = max(1, int(len(syn_pairs) * VAL_SPLIT))
    syn_val_idx = syn_indices[:n_val_syn]
    syn_val_pairs = [syn_pairs[i] for i in syn_val_idx]
    print(f"Synthetic val: {len(syn_val_pairs)} images")

    syn_val_ds = ScleraDataset(syn_val_pairs, img_size=IMG_SIZE, is_train=False)
    syn_val_loader = DataLoader(syn_val_ds, batch_size=4, shuffle=False,
                                num_workers=4, pin_memory=True)
    syn_targets = collect_targets(syn_val_loader)

    # ── Mixed val set ──
    print("\n--- Mixed validation set ---")
    np.random.seed(SEED)
    mix_pairs, mix_labels = build_pairs(T1_DATASETS['mixed'])
    mix_indices = np.random.permutation(len(mix_pairs))
    n_val_mix = max(1, int(len(mix_pairs) * VAL_SPLIT))
    mix_val_idx = mix_indices[:n_val_mix]
    mix_val_pairs = [mix_pairs[i] for i in mix_val_idx]
    print(f"Mixed val: {len(mix_val_pairs)} images")

    mix_val_ds = ScleraDataset(mix_val_pairs, img_size=IMG_SIZE, is_train=False)
    mix_val_loader = DataLoader(mix_val_ds, batch_size=4, shuffle=False,
                                num_workers=4, pin_memory=True)
    mix_targets = collect_targets(mix_val_loader)

    # ── Evaluate models ──
    model_paths = {
        't1_synthetic_v3': MODEL_DIR / 't1_synthetic' / 'best_model.pth',
        't1_mixed_v3': MODEL_DIR / 't1_mixed' / 'best_model.pth',
        't1_synthetic_v3_swa': MODEL_DIR / 't1_synthetic' / 'swa_model.pth',
        't1_mixed_v3_swa': MODEL_DIR / 't1_mixed' / 'swa_model.pth',
    }

    predictions = {}  # store predictions for ensemble

    for model_name, ckpt_path in model_paths.items():
        if not ckpt_path.exists():
            print(f"\n  {model_name}: {ckpt_path} not found, skipping")
            continue

        print(f"\n=== Evaluating {model_name} ===")
        model = load_t1_model(ckpt_path, device)

        # Choose appropriate val set
        if 'synthetic' in model_name and 'mixed' not in model_name:
            val_loader = syn_val_loader
            targets = syn_targets
            val_name = 'syn_val'
        else:
            val_loader = mix_val_loader
            targets = mix_targets
            val_name = 'mix_val'

        # No TTA
        probs = collect_predictions(model, val_loader, device, do_tta=False)
        results = sweep_thresholds(probs, targets, thresholds)

        # TTA
        probs_tta = collect_predictions(model, val_loader, device, do_tta=True)
        results_tta = sweep_thresholds(probs_tta, targets, thresholds)

        all_results[f"{model_name}_{val_name}"] = {'no_tta': results, 'tta': results_tta}
        predictions[model_name] = {'probs': probs, 'probs_tta': probs_tta}

        print_best(model_name, results, results_tta)
        del model
        torch.cuda.empty_cache()

    # ── Cross-evaluation: synthetic model on mixed val ──
    if 't1_synthetic_v3' in predictions:
        print(f"\n=== Cross-eval: synthetic model on mixed val ===")
        model = load_t1_model(model_paths['t1_synthetic_v3'], device)
        probs = collect_predictions(model, mix_val_loader, device, do_tta=False)
        probs_tta = collect_predictions(model, mix_val_loader, device, do_tta=True)
        results = sweep_thresholds(probs, mix_targets, thresholds)
        results_tta = sweep_thresholds(probs_tta, mix_targets, thresholds)
        all_results['syn_model_on_mix_val'] = {'no_tta': results, 'tta': results_tta}
        predictions['syn_on_mix'] = {'probs': probs, 'probs_tta': probs_tta}
        print_best('syn→mix_val', results, results_tta)
        del model
        torch.cuda.empty_cache()

    # ── Ensemble: synthetic + mixed on mixed val ──
    if 'syn_on_mix' in predictions and 't1_mixed_v3' in predictions:
        print(f"\n=== Ensemble: (synthetic + mixed) / 2 on mixed val ===")
        ens_probs = (predictions['syn_on_mix']['probs'] + predictions['t1_mixed_v3']['probs']) / 2
        ens_probs_tta = (predictions['syn_on_mix']['probs_tta'] + predictions['t1_mixed_v3']['probs_tta']) / 2
        results = sweep_thresholds(ens_probs, mix_targets, thresholds)
        results_tta = sweep_thresholds(ens_probs_tta, mix_targets, thresholds)
        all_results['ensemble_mix_val'] = {'no_tta': results, 'tta': results_tta}
        print_best('ensemble(syn+mix)', results, results_tta)

    # ── Summary ──
    print("\n" + "=" * 90)
    print("SUMMARY — T1 Models")
    print("=" * 90)
    for key in all_results:
        for mode in ['no_tta', 'tta']:
            res = all_results[key][mode]
            best_t = max(res, key=lambda t: res[t]['f1'])
            r = res[best_t]
            label = 'TTA' if mode == 'tta' else 'No TTA'
            print(f"  {key:35s} ({label:6s}): t={best_t:.2f} | F1={r['f1']:.4f} | "
                  f"Dice={r['dice']:.4f} | IoU={r['iou']:.4f} | P={r['precision']:.4f} R={r['recall']:.4f}")
    print("=" * 90)

    # Save
    output_path = OUTPUT_DIR / 't1_eval_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_results = {}
    for key, modes in all_results.items():
        json_results[key] = {}
        for mode, res in modes.items():
            json_results[key][mode] = {str(k): v for k, v in res.items()}
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
