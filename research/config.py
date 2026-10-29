"""Configuration for SSBC 2026 Track 1 - SAM + LoRA sclera segmentation."""
from pathlib import Path

# ── Paths ──
DATA_ROOT = Path("/home/jupyter/sclera/data")
T1_ROOT = DATA_ROOT / "T1 Training"
T2_ROOT = DATA_ROOT / "T2 Training"
MODEL_DIR = Path("/home/jupyter/sclera/models")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs")
SAM_CHECKPOINT = MODEL_DIR / "sam_vit_b_01ec64.pth"

# ── Dataset configs ──
T1_DATASETS = {
    'synthetic': [
        {'name': 'SynCROI_1', 'root': T1_ROOT / 'SynCROI_1'},
    ],
    'mixed': [
        {'name': 'SynCROI_1', 'root': T1_ROOT / 'SynCROI_1'},
        {'name': 'SBVPI', 'root': T1_ROOT / 'SBVPI'},
        {'name': 'MASD_1', 'root': T1_ROOT / 'MASD_1'},
    ],
}

# ── Training ──
IMG_SIZE = 1024        # SAM input size
MASK_OUT_SIZE = 256    # Decoder output size
BATCH_SIZE = 4         # A100 40GB; use grad accum for effective batch 8
NUM_WORKERS = 4
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
LORA_RANK = 16
LORA_ALPHA = 32
VAL_SPLIT = 0.1
SEED = 42

# ── Evaluation ──
EVAL_W, EVAL_H = 400, 300  # Competition output size
