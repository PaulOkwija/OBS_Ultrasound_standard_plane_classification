"""Tests for preprocessing pipeline."""
import pytest
import numpy as np
from PIL import Image
from src.data.preprocessing import CLAHEPreprocessor, USHeaderRemover, compute_image_quality_score, build_preprocessing_pipeline


def make_test_image(h=224, w=224):
    arr = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_clahe_pil_input():
    clahe = CLAHEPreprocessor(clip_limit=2.0)
    img = make_test_image()
    result = clahe(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_header_remover():
    remover = USHeaderRemover(top_crop_fraction=0.1, bottom_crop_fraction=0.1)
    img = make_test_image(300, 300)
    result = remover(img)
    assert isinstance(result, Image.Image)
    # Should be smaller
    assert result.size[1] < 300


def test_quality_score():
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    q = compute_image_quality_score(arr)
    assert "mean_brightness" in q
    assert "sharpness" in q
    assert q["contrast"] >= 0


def test_pipeline_clahe():
    pipeline = build_preprocessing_pipeline(use_clahe=True)
    img = make_test_image()
    result = pipeline(img)
    assert isinstance(result, Image.Image)


def test_pipeline_none():
    pipeline = build_preprocessing_pipeline(use_clahe=False, remove_headers=False)
    assert pipeline is None
