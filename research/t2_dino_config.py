"""Configuration for SSBC 2026 Track 2 — DINOv2 backbone variant.

Upgrade from UNet+EfficientNet-B4 to DINOv2 ViT-B/14 backbone.
Enhanced with insights from CSIRO top solutions:
- Frozen backbone + unfrozen LayerNorm (CSIRO 8th: frozen DINOv2 + head)
- bfloat16 for A100 stability (no GradScaler needed)
- Gradient accumulation = 4 for effective larger batch
- Optional LoRA on last 4 blocks (CSIRO 4th: LoRA fine-tuning)
"""
from pathlib import Path

# ── Paths ──
DATA_ROOT = Path("/home/jupyter/sclera/data")
T2_ROOT = DATA_ROOT / "T2 Training"
MODEL_DIR = Path("/home/jupyter/sclera/models/t2_dino")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_dino")
LOG_DIR = OUTPUT_DIR / "logs"

# ── T2 Datasets (same as t2_config) ──
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
BACKBONE = 'dinov2_vitb14'
IMG_SIZE = 518              # 518 / 14 = 37 patches per side (divisible by patch_size=14)
FEATURE_LAYERS = [2, 5, 8, 11]  # Block indices for multi-scale features (layers 3,6,9,12)
DECODER_DIM = 256
UNFREEZE_LN = True          # Unfreeze LayerNorm for domain adaptation

# ── Training ──
BATCH_SIZE_L = 4            # Labelled batch
BATCH_SIZE_U = 8            # Unlabelled batch
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4         # Lower for frozen backbone
NUM_EPOCHS = 150
PATIENCE = 25
VAL_SPLIT = 0.15
SEED = 42
GRAD_ACCUM_STEPS = 4        # Effective batch: L=16, U=32 (CSIRO-informed)
USE_BFLOAT16 = True         # bfloat16 on A100 — no GradScaler needed
USE_COMPILE = True           # torch.compile for throughput

# ── Learning Rate ──
BASE_LR = 1e-3              # Decoder + lateral convs
LN_LR_SCALE = 0.1           # LayerNorm LR = BASE_LR * 0.1

# ── Optional Stage 2: LoRA (CSIRO 4th place style) ──
ENABLE_LORA = False         # Set True to enable LoRA fine-tuning after main training
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_BLOCKS = 4      # LoRA on last N transformer blocks
LORA_LR = 5e-5
LORA_EPOCHS = 50

# ── Mean Teacher ──
EMA_DECAY = 0.999
RAMPUP_EPOCHS = 20          # Faster ramp-up (fewer trainable params → faster convergence)
CONSISTENCY_WEIGHT = 1.0
CONF_THRESHOLD = 0.95       # Filter noisy pseudo-labels (CSIRO 6th shakeup lesson)

# ── SWA (CSIRO 1st + 8th) ──
SWA_START = 120
SWA_LR = 5e-5

# ── Evaluation ──
EVAL_W, EVAL_H = 400, 300
BIN_THRESHOLD = 0.5
