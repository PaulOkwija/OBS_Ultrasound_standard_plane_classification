"""
Image Preprocessing
-------------------
Handles all transforms needed before model inference.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

from app.core.config import settings


def get_transform(image_size: int = None) -> transforms.Compose:
    """Return the standard ImageNet-style preprocessing pipeline."""
    size = image_size or settings.IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=settings.NORMALIZE_MEAN,
            std=settings.NORMALIZE_STD,
        ),
    ])


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes (e.g. from an uploaded file)."""
    img = Image.open(BytesIO(data)).convert("RGB")
    return img


def load_image_from_path(path: Union[str, Path]) -> Image.Image:
    """Load a PIL Image from a file path."""
    img = Image.open(path).convert("RGB")
    return img


def preprocess_image(
    image: Image.Image,
    image_size: int = None,
) -> torch.Tensor:
    """
    Preprocess a single PIL image into a model-ready tensor.
    Returns shape: [1, C, H, W]
    """
    transform = get_transform(image_size)
    tensor = transform(image)          # [C, H, W]
    return tensor.unsqueeze(0)         # [1, C, H, W]


def preprocess_batch(
    images: list[Image.Image],
    image_size: int = None,
) -> torch.Tensor:
    """
    Preprocess a list of PIL images into a batched tensor.
    Returns shape: [N, C, H, W]
    """
    transform = get_transform(image_size)
    tensors = [transform(img) for img in images]
    return torch.stack(tensors)        # [N, C, H, W]
