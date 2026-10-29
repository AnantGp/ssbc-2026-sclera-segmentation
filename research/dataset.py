"""Dataset loading for SSBC 2026 sclera segmentation."""
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
    # RGBA — convert to grayscale (ignore alpha; sclera = white RGB region)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    # BGR — convert to gray
    elif len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Threshold to binary
    return (mask > 128).astype(np.uint8)


def build_pairs(datasets):
    """Build list of (image_path, mask_path) from dataset configs.

    Args:
        datasets: list of dicts with 'name' and 'root' keys
    Returns:
        list of (Path, Path) tuples and dataset labels
    """
    all_pairs = []
    pair_labels = []  # track which dataset each pair belongs to
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


class ScleraDataset(Dataset):
    """Sclera segmentation dataset.

    Returns images in 0-255 float range (for SAM normalization)
    and binary masks in 0-1 float range.
    """
    def __init__(self, pairs, img_size=1024, is_train=True):
        self.pairs = pairs
        self.img_size = img_size
        if is_train:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                         scale=(0.90, 1.10), rotate=(-12, 12),
                         border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2),
                    A.CLAHE(clip_limit=2.0),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=3),
                ], p=0.2),
                A.ColorJitter(p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                A.CenterCrop(height=img_size, width=img_size),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        # Load image BGR -> RGB, keep 0-255
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = load_mask(mask_path)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Ensure mask matches image spatial dims (some datasets differ)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        # Apply transforms
        out = self.transform(image=image, mask=mask)
        image = out['image']
        mask = out['mask']
        # To tensors: image [3,H,W] float 0-255, mask [1,H,W] float 0-1
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask
