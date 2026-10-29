"""Configuration for T2 DINOv2 ViT-L/14 — larger backbone variant.

Inherits all settings from t2_dino_config and overrides ViT-L specifics.
ViT-L/14: 24 blocks, embed_dim=1024, ~300M params (vs ViT-B: 12 blocks, 768, ~86M).
"""
from t2_dino_config import *  # noqa: F401,F403
from pathlib import Path

# ── Override for ViT-L ──
BACKBONE = 'dinov2_vitl14'
FEATURE_LAYERS = [5, 11, 17, 23]   # 24 blocks, evenly spaced (was [2,5,8,11] for 12-block ViT-B)
DECODER_DIM = 256                    # keep same decoder capacity

# ── Separate output dirs ──
MODEL_DIR = Path("/home/jupyter/sclera/models/t2_dino_large")
OUTPUT_DIR = Path("/home/jupyter/sclera/outputs/t2_dino_large")
LOG_DIR = OUTPUT_DIR / "logs"

# ── Reduced batch sizes for larger model (A100 40GB) ──
BATCH_SIZE_L = 2        # was 4 — ViT-L uses ~2x memory per sample
BATCH_SIZE_U = 4        # was 8
GRAD_ACCUM_STEPS = 8    # was 4 — effective batch stays same: L=16, U=32

# ── Training adjustments ──
BASE_LR = 5e-4          # slightly lower for larger backbone features
NUM_EPOCHS = 120        # larger backbone converges faster
SWA_START = 90          # proportionally earlier
PATIENCE = 20
