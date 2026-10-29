"""Dataset and augmentation for SSBC 2026 Track 2 (Semi-supervised).

Augmentation design informed by:
- CSIRO 1st/2nd/4th place: CLAHE is universal for all top solutions
- Wang et al. 2025: CLAHE + gamma correction for eye images
- Mean Teacher: weak aug for teacher, strong aug for student
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ══════════════════════════════════════════════════════════════════════════════
# MASK PATH HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def get_mask_path_t2(img_path, dataset_name):
    """Get mask path for T2 datasets.
    
    IMPORTANT: T2 SBVPI masks do NOT have '_sclera' suffix (unlike T1).
    All T2 datasets: mask_name = stem + '.png'
    """
    img_path = Path(img_path)
    mask_dir = img_path.parent.parent / 'Masks'
    # T2: all datasets use stem.png (no _sclera suffix)
    mask_name = img_path.stem + '.png'
    return mask_dir / mask_name


def load_mask(mask_path):
    """Load mask and convert to binary 0/1."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    # Handle multi-channel masks
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:  # RGBA
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:  # BGR
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 128).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PAIR BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_labelled_pairs(datasets):
    """Build (image_path, mask_path) pairs for labelled data."""
    pairs = []
    labels = []  # dataset name for each pair
    for ds in datasets:
        name = ds['name']
        root = Path(ds['root'])
        img_dir = root / 'Images'
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found")
            continue
        count = 0
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            mask_path = get_mask_path_t2(img_file, name)
            if mask_path.exists():
                pairs.append((img_file, mask_path))
                labels.append(name)
                count += 1
        print(f"  {name}: {count} labelled pairs")
    return pairs, labels


def build_unlabelled_paths(datasets, labelled_pairs):
    """Build list of unlabelled image paths (images without masks)."""
    labelled_stems = {Path(p[0]).stem for p in labelled_pairs}
    paths = []
    for ds in datasets:
        name = ds['name']
        root = Path(ds['root'])
        img_dir = root / 'Images'
        if not img_dir.exists():
            continue
        count = 0
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            # Skip if this image has a mask (it's labelled)
            if img_file.stem in labelled_stems:
                continue
            paths.append(img_file)
            count += 1
        print(f"  {name}: {count} unlabelled images")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# Key insight from CSIRO analysis: CLAHE appears in ALL top solutions
# Mean Teacher: weak aug for teacher, strong aug for student
# ══════════════════════════════════════════════════════════════════════════════

def get_weak_transforms(img_size):
    """Weak augmentation for teacher branch.
    
    Keep it simple: resize + normalize + light CLAHE only.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        ),
        A.CenterCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        # CLAHE: universal in CSIRO top solutions + Wang et al.
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_strong_transforms(img_size):
    """Strong augmentation for student branch.
    
    Includes all techniques from CSIRO winners + Wang et al.:
    - CLAHE (universal)
    - Gamma correction (Wang et al.)
    - Color jitter (CSIRO 1st)
    - Geometric transforms
    - Blur (T1 baseline)
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        ),
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        # Geometric (Wang et al.: rotation ±5°, translation ±20px)
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        # Photometric - CSIRO 1st place + Wang et al.
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),  # Stronger CLAHE
            A.RandomGamma(gamma_limit=(80, 120)),           # Wang et al.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.7),
        # Color - CSIRO 1st place
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        # Blur - T1 baseline
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        # Noise - CSIRO 1st
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    """Validation transforms - deterministic."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        ),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class LabelledDataset(Dataset):
    """Dataset for labelled images with masks."""
    
    def __init__(self, pairs, img_size=512, transform=None):
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform or get_strong_transforms(img_size)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image (BGR→RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = load_mask(mask_path)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Resize mask if dimensions don't match
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        out = self.transform(image=image, mask=mask)
        image = out['image']  # Already tensor from ToTensorV2
        mask = out['mask'].unsqueeze(0).float()  # [1, H, W]
        
        return image, mask


class UnlabelledDataset(Dataset):
    """Dataset for unlabelled images (for Mean Teacher).
    
    Returns both weak and strong augmented versions.
    """
    
    def __init__(self, paths, img_size=512):
        self.paths = paths
        self.img_size = img_size
        self.weak_transform = get_weak_transforms(img_size)
        self.strong_transform = get_strong_transforms(img_size)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply weak aug (for teacher)
        weak_out = self.weak_transform(image=image)
        weak_img = weak_out['image']
        
        # Apply strong aug (for student)
        strong_out = self.strong_transform(image=image)
        strong_img = strong_out['image']
        
        return weak_img, strong_img


class ValidationDataset(Dataset):
    """Validation dataset with deterministic transforms."""
    
    def __init__(self, pairs, img_size=512):
        self.pairs = pairs
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
        mask = out['mask'].unsqueeze(0).float()
        
        return image, mask


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def split_train_val(pairs, labels, val_split=0.15, seed=42):
    """Split labelled data into train/val, stratified by dataset."""
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(pairs)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed,
        stratify=labels
    )
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    
    return train_pairs, val_pairs, train_labels
