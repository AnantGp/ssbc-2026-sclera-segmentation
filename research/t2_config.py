"""Configuration for SSBC 2026 Track 2 — Semi-supervised sclera segmentation.

Design informed by:
- CSIRO 1st/2nd/4th/6th/8th place solutions (two-stage, CLAHE, SWA, self-distillation)
- Wang et al. 2025 (semi-supervised sclera segmentation)
- SSBC 2026 competition rules (eval@400x300, submit Synthetic+Mixed, primarily real test images)
"""
from pathlib import Path

# ── Paths ──
DATA_ROOT = Path("/home/jupyter/sclera/data")
T2_ROOT = DATA_ROOT / "T2 Training"
MODEL_DIR = Path("/home/jupyter/sclera/models/t2")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2")
LOG_DIR = OUTPUT_DIR / "logs"

# ── T2 Datasets ──
# Labelled: MASD(82), SBVPI(55), SynCROI(220) = 357 total
# Unlabelled: MOBIUS(13158) + maskless images from MASD/SBVPI/SynCROI = ~28K
T2_LABELLED = [
    {'name': 'MASD',    'root': T2_ROOT / 'MASD'},
    {'name': 'SBVPI',   'root': T2_ROOT / 'SBVPI'},
    {'name': 'SynCROI', 'root': T2_ROOT / 'SynCROI'},
]
T2_UNLABELLED = [
    {'name': 'MOBIUS',  'root': T2_ROOT / 'MOBIUS'},
    {'name': 'MASD',    'root': T2_ROOT / 'MASD'},
    {'name': 'SBVPI',   'root': T2_ROOT / 'SBVPI'},
    {'name': 'SynCROI', 'root': T2_ROOT / 'SynCROI'},
]

# ── Model ──
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
IMG_SIZE = 512

# ── Training ──
BATCH_SIZE_L = 4           # Labelled batch
BATCH_SIZE_U = 8           # Unlabelled batch (larger)
NUM_WORKERS = 4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
PATIENCE = 30
VAL_SPLIT = 0.15           # ~54 for val
SEED = 42
GRAD_ACCUM_STEPS = 2       # Effective batch: L=8, U=16
UNFREEZE_BN = True          # Unfreeze BatchNorm for domain adaptation (CSIRO 4th)

# ── Two-Stage Training (all top CSIRO solutions use freeze→unfreeze) ──
STAGE1_EPOCHS = 50          # Freeze encoder, train decoder + SSL
STAGE1_LR = 1e-3
STAGE2_LR = 5e-5            # Unfreeze entire encoder, lower LR (was 1e-4, reduced for stability)
ENCODER_LR_SCALE = 0.1      # Encoder LR = STAGE2_LR * 0.1

# ── Mean Teacher (validated by CSIRO 1st place self-distillation strategy) ──
EMA_DECAY = 0.999
RAMPUP_EPOCHS = 30          # Sigmoid ramp-up for consistency weight
CONSISTENCY_WEIGHT = 1.0
CONF_THRESHOLD = 0.95       # Filter noisy pseudo-labels (CSIRO 6th shakeup lesson)

# ── SWA (CSIRO 1st + 8th both use it) ──
SWA_START = 150
SWA_LR = 5e-5

# ── Evaluation (competition: 400x300 WxH, PNG, both probabilistic + binarised) ──
EVAL_W, EVAL_H = 400, 300
BIN_THRESHOLD = 0.5
