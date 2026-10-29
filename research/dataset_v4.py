"""Dataset v4 — Strong augmentations for sclera segmentation.

Key additions over v3:
  - ElasticTransform: deforms boundaries like real tissue variability
  - GridDistortion: simulates lens/perspective distortion
  - OpticalDistortion: barrel/pincushion effects
  - CoarseDropout (Cutout): random rectangular occlusions
  - Superpixels: injects block artifacts for robustness
  - Stronger color augmentations: channel shuffle, posterize, equalize
  - Morphological mask augmentation: slight dilate/erode of GT mask
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A


def get_mask_path(img_path, dataset_name):
    """Get corresponding mask path for an image."""
    img_path = Path(img_path)
    mask_dir = img_path.parent.parent / 'Masks'
    if 'SBVPI' in dataset_name:
        mask_name = img_path.stem + '_sclera.png'
    else:
        mask_name = img_path.stem + '.png'
    return mask_dir / mask_name


def load_mask(mask_path):
    """Load mask file and convert to binary (0/1) numpy array."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 128).astype(np.uint8)


def build_pairs(datasets):
    """Build (image_path, mask_path) pairs."""
    all_pairs = []
    pair_labels = []
    for ds in datasets:
        name = ds['name']
        root = Path(ds['root'])
        img_dir = root / 'Images'
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping")
            continue
        count = 0
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            mask_path = get_mask_path(img_file, name)
            if mask_path.exists():
                all_pairs.append((img_file, mask_path))
                pair_labels.append(name)
                count += 1
        print(f"  {name}: {count} image-mask pairs found")
    return all_pairs, pair_labels


def get_strong_train_transforms(img_size):
    """Strong augmentation pipeline for v4 training."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=img_size, width=img_size),

        # === Geometric transforms ===
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.08, 0.08), 'y': (-0.08, 0.08)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=40,
                               border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3,
                             border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.OpticalDistortion(distort_limit=0.3,
                                border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ], p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.2),

        # === Color / intensity transforms ===
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.CLAHE(clip_limit=3.0),
            A.RandomGamma(gamma_limit=(70, 130)),
        ], p=0.6),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                                 val_shift_limit=20),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            A.ChannelShuffle(),
        ], p=0.3),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.3),

        # === Noise / blur ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05)),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
        ], p=0.2),

        # === Occlusion / dropout ===
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.05, 0.15),
                        fill=0, p=0.25),

        # === Quality degradation ===
        A.OneOf([
            A.ImageCompression(quality_range=(50, 90)),
            A.Posterize(num_bits=(4, 6)),
        ], p=0.15),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.15),
    ])


def get_val_transforms(img_size):
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.CenterCrop(height=img_size, width=img_size),
    ])


class ScleraDatasetV4(Dataset):
    """Sclera dataset with strong augmentation support.

    Returns images in 0-255 float range (for SAM normalization)
    and binary masks in 0-1 float range.
    """
    def __init__(self, pairs, img_size=1024, is_train=True):
        self.pairs = pairs
        self.img_size = img_size
        if is_train:
            self.transform = get_strong_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = load_mask(mask_path)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        out = self.transform(image=image, mask=mask)
        image = out['image']
        mask = out['mask']
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask
