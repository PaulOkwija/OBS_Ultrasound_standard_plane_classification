"""
tests/test_preprocessing.py
Tests for image loading and preprocessing utilities.
"""

import io

import pytest
import torch
from PIL import Image

from app.utils.preprocessing import (
    get_transform,
    load_image_from_bytes,
    preprocess_batch,
    preprocess_image,
)


def _make_pil(width=300, height=250, mode="RGB") -> Image.Image:
    return Image.new(mode, (width, height), color=(128, 64, 192))


def _make_bytes(width=300, height=250, fmt="JPEG") -> bytes:
    img = _make_pil(width, height)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# Transform

def test_get_transform_returns_compose():
    t = get_transform(224)
    assert t is not None


def test_transform_output_shape():
    t = get_transform(224)
    img = _make_pil(300, 250)
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_transform_normalized_range():
    """After normalization values should leave [0,1] range."""
    t = get_transform(224)
    img = _make_pil(300, 250)
    tensor = t(img)
    # At least some values should be outside raw [0,1] range after normalization
    assert tensor.min().item() < 0 or tensor.max().item() > 1


# Load from bytes

def test_load_image_from_bytes_jpeg():
    raw = _make_bytes(fmt="JPEG")
    img = load_image_from_bytes(raw)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_from_bytes_png():
    raw = _make_bytes(fmt="PNG")
    img = load_image_from_bytes(raw)
    assert img.mode == "RGB"


def test_load_image_from_bytes_rgba_converted():
    """RGBA images must be converted to RGB."""
    rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    img = load_image_from_bytes(buf.getvalue())
    assert img.mode == "RGB"


def test_load_image_from_bytes_invalid_raises():
    with pytest.raises(Exception):
        load_image_from_bytes(b"not an image")


# preprocess_image

def test_preprocess_image_shape():
    img = _make_pil(400, 300)
    tensor = preprocess_image(img, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_dtype():
    img = _make_pil()
    tensor = preprocess_image(img)
    assert tensor.dtype == torch.float32


def test_preprocess_image_custom_size():
    img = _make_pil(500, 500)
    tensor = preprocess_image(img, image_size=128)
    assert tensor.shape == (1, 3, 128, 128)


# preprocess_batch

def test_preprocess_batch_shape():
    imgs = [_make_pil(300, 250) for _ in range(4)]
    tensor = preprocess_batch(imgs, image_size=224)
    assert tensor.shape == (4, 3, 224, 224)


def test_preprocess_batch_single():
    imgs = [_make_pil()]
    tensor = preprocess_batch(imgs, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_batch_large():
    imgs = [_make_pil(200 + i, 200 + i) for i in range(16)]
    tensor = preprocess_batch(imgs, image_size=224)
    assert tensor.shape == (16, 3, 224, 224)
