"""Eval DINOv2 ViT-L/14 models — swaps in large config then runs t2_dino_eval."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import t2_dino_large_config
sys.modules['t2_dino_config'] = t2_dino_large_config

from t2_dino_eval import main

if __name__ == '__main__':
    main()
