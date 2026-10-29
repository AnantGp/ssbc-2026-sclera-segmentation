"""Inference and submission generation for SSBC 2026 Track 2.

Generates predictions in the required submission format:
- Probabilistic masks (grayscale PNG)
- Binarised masks (black/white PNG)
- 400x300 WxH
- Organized by training data (Synthetic/Mixed) and eval dataset
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm
import zipfile

sys.path.insert(0, str(Path(__file__).parent))

import t2_config as cfg
from t2_model import build_model
from t2_dataset import get_val_transforms


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(model, image, device, do_tta=True):
    """Predict with Test-Time Augmentation.
    
    TTA: Original + HFlip (average)
    """
    model.eval()
    
    # Original
    x = image.unsqueeze(0).to(device)
    with autocast():
        logits = model(x)
    probs = torch.sigmoid(logits)
    
    if not do_tta:
        return probs[0, 0].cpu().numpy()
    
    # Horizontal flip
    x_flip = torch.flip(x, dims=[3])
    with autocast():
        logits_flip = model(x_flip)
    probs_flip = torch.sigmoid(logits_flip)
    probs_flip = torch.flip(probs_flip, dims=[3])
    
    # Average
    probs_avg = (probs + probs_flip) / 2
    return probs_avg[0, 0].cpu().numpy()


def resize_to_submission(prob, width=400, height=300):
    """Resize probability map to submission size (400x300 WxH)."""
    return cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)


def save_probabilistic_mask(prob, path):
    """Save probabilistic mask as grayscale PNG."""
    arr = np.clip(prob * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_binarised_mask(prob, path, threshold=0.5):
    """Save binarised mask as black/white PNG."""
    arr = ((prob > threshold) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def infer_folder(model, image_dir, output_pred_dir, output_bin_dir, 
                 device, img_size=512, threshold=0.5, do_tta=True):
    """Run inference on a folder of images."""
    transform = get_val_transforms(img_size)
    
    os.makedirs(output_pred_dir, exist_ok=True)
    os.makedirs(output_bin_dir, exist_ok=True)
    
    image_dir = Path(image_dir)
    image_files = sorted([f for f in image_dir.iterdir() 
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    
    if len(image_files) == 0:
        print(f"  WARNING: No images found in {image_dir}")
        return
    
    for img_path in tqdm(image_files, desc=f"  {image_dir.name}"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        out = transform(image=image)
        x = out['image']
        
        # Predict
        prob = predict_with_tta(model, x, device, do_tta=do_tta)
        
        # Resize to submission size
        prob = resize_to_submission(prob, cfg.EVAL_W, cfg.EVAL_H)
        
        # Save
        out_name = img_path.stem + '.png'
        save_probabilistic_mask(prob, os.path.join(output_pred_dir, out_name))
        save_binarised_mask(prob, os.path.join(output_bin_dir, out_name), threshold)


# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_submission(model_path, eval_roots, submission_dir, 
                        threshold=0.5, do_tta=True, zip_name="T2_Submission.zip"):
    """Generate submission following SSBC format.
    
    Args:
        model_path: Path to trained model checkpoint
        eval_roots: Dict of evaluation image directories
            {
                "Mixed": {"MOBIUS": "...", "SMD+SLD": "...", "Synthetic": "..."},
                "Synthetic": {"MOBIUS": "...", "SMD+SLD": "...", "Synthetic": "..."}
            }
        submission_dir: Output directory
        threshold: Binarisation threshold
        do_tta: Whether to use TTA
        zip_name: Name of output ZIP file
    
    Submission structure:
        Submission.zip
        ├── Synthetic/
        │   ├── MOBIUS/
        │   │   ├── Predictions/
        │   │   └── Binarised/
        │   ├── SMD+SLD/
        │   └── Synthetic/
        └── Mixed/
            ├── MOBIUS/
            ├── SMD+SLD/
            └── Synthetic/
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = build_model(encoder=cfg.ENCODER, encoder_weights=None)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Checkpoint dice: {checkpoint.get('dice', 'N/A')}")
    print(f"Checkpoint F1: {checkpoint.get('f1', checkpoint.get('best_f1', 'N/A'))}")
    
    # Create submission structure
    submission_dir = Path(submission_dir)
    for train_group in ["Synthetic", "Mixed"]:
        for eval_name in ["MOBIUS", "SMD+SLD", "Synthetic"]:
            (submission_dir / train_group / eval_name / "Predictions").mkdir(parents=True, exist_ok=True)
            (submission_dir / train_group / eval_name / "Binarised").mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print("\nGenerating predictions...")
    for train_group, datasets in eval_roots.items():
        print(f"\n{train_group}:")
        for eval_name, image_dir in datasets.items():
            if not image_dir or not Path(image_dir).exists():
                print(f"  {eval_name}: SKIPPED (no path)")
                continue
            
            pred_dir = submission_dir / train_group / eval_name / "Predictions"
            bin_dir = submission_dir / train_group / eval_name / "Binarised"
            
            infer_folder(
                model, image_dir, pred_dir, bin_dir,
                device, cfg.IMG_SIZE, threshold, do_tta
            )
    
    # Create ZIP
    zip_path = submission_dir.parent / zip_name
    print(f"\nCreating ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, submission_dir)
                zf.write(abs_path, rel_path)
    
    print(f"\n=== Submission Complete ===")
    print(f"ZIP: {zip_path}")
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Example usage - update paths when test data is available
    EVAL_ROOTS = {
        "Mixed": {
            "MOBIUS": "",      # Fill with test data path
            "SMD+SLD": "",
            "Synthetic": "",
        },
        "Synthetic": {
            "MOBIUS": "",
            "SMD+SLD": "",
            "Synthetic": "",
        }
    }
    
    model_path = cfg.MODEL_DIR / 'best_model.pth'
    submission_dir = cfg.OUTPUT_DIR / 'submission'
    
    if model_path.exists():
        generate_submission(
            model_path=model_path,
            eval_roots=EVAL_ROOTS,
            submission_dir=submission_dir,
            threshold=cfg.BIN_THRESHOLD,
            do_tta=True,
            zip_name="T2_MeanTeacher_Submission.zip"
        )
    else:
        print(f"Model not found: {model_path}")
        print("Train the model first with: python t2_train.py")
