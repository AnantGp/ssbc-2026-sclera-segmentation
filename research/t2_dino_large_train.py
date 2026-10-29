"""Train DINOv2 ViT-L/14 for T2 sclera segmentation.

Thin wrapper that swaps in the ViT-L config before importing the training script.
All training logic is reused from t2_dino_train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Monkey-patch: make t2_dino_train see the large config
import t2_dino_large_config
sys.modules['t2_dino_config'] = t2_dino_large_config

from t2_dino_train import train

if __name__ == '__main__':
    train()
