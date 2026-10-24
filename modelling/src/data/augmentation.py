"""
Data augmentation pipelines using albumentations.

Two pipelines:
  - train_transform:  strong augmentations for training
  - val_transform:    resize + normalize only
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline for training.

    Designed for fetal US images:
      - Geometric: flip, rotate, small crop (no large distortions - anatomy matters)
      - Photometric: brightness/contrast (simulate different scanners)
      - Noise: gaussian noise (simulate low-quality African scanners)
      - CLAHE: handled in preprocessing pipeline only (not here, to avoid double-application)
      - NOT: large shear/zoom that could confuse anatomical landmarks
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6,
        ),
        A.GaussNoise(std_range=(0.04, 0.22), p=0.3),   # updated API: var_limit -> std_range
        A.CoarseDropout(                                 # updated API: max_holes etc -> range tuples
            num_holes_range=(1, 4),
            hole_height_range=(5, int(image_size * 0.05)),
            hole_width_range=(5, int(image_size * 0.05)),
            p=0.2,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    """Minimal transform for validation and test."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class AlbumentationsWrapper:
    """Wrap an albumentations Compose to accept PIL Images."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        result = self.transform(image=img)
        return result["image"]


def build_transforms(image_size: int = 224, split: str = "train"):
    """
    Factory: return the appropriate AlbumentationsWrapper for a given split.

    split: "train" | "val" | "test"
    """
    if split == "train":
        t = get_train_transform(image_size)
    else:
        t = get_val_transform(image_size)
    return AlbumentationsWrapper(t)
