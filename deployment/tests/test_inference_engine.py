"""
tests/test_inference_engine.py
Tests for the core inference engine using a tiny dummy model.
"""

import pytest
import torch
import torch.nn as nn

from app.core.model_manager import LoadedModel
from app.utils.inference_engine import run_inference
from app.utils.preprocessing import preprocess_batch


# Helpers

NUM_CLASSES = 5
CLASSES = ["cat", "dog", "bird", "fish", "horse"]


class TinyClassifier(nn.Module):
    """Minimal model: pool -> linear. Fast for tests."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


def _make_loaded_model(classes=None) -> LoadedModel:
    model = TinyClassifier(num_classes=len(classes or CLASSES))
    model.eval()
    config = {
        "format": "state_dict",
        "architecture": "tiny",
        "num_classes": len(classes or CLASSES),
        "classes": classes or CLASSES,
    }
    return LoadedModel(
        name="test_model",
        version="v1",
        model=model,
        config=config,
    )


def _make_pil(w=300, h=250):
    from PIL import Image
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def _make_tensor(n=1):
    imgs = [_make_pil() for _ in range(n)]
    return preprocess_batch(imgs, image_size=224)


# Basic inference

def test_run_inference_returns_list():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor)
    assert isinstance(results, list)
    assert len(results) == 1


def test_run_inference_single_image_top_k():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=3)
    assert len(results[0]) == 3


def test_run_inference_batch():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(4)
    results = run_inference(model_obj, tensor, top_k=2)
    assert len(results) == 4
    for r in results:
        assert len(r) == 2


def test_run_inference_prediction_keys():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    pred = run_inference(model_obj, tensor, top_k=1)[0][0]
    assert "label" in pred
    assert "class_index" in pred
    assert "confidence" in pred


def test_run_inference_confidence_is_probability():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES)
    confs = [p["confidence"] for p in results[0]]
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_run_inference_confidences_sum_to_one():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES)
    total = sum(p["confidence"] for p in results[0])
    assert abs(total - 1.0) < 1e-4


def test_run_inference_labels_from_classes():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES)
    returned_labels = {p["label"] for p in results[0]}
    assert returned_labels.issubset(set(CLASSES))


def test_run_inference_descending_confidence():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES)
    confs = [p["confidence"] for p in results[0]]
    assert confs == sorted(confs, reverse=True)


# Threshold filtering

def test_run_inference_threshold_1_returns_empty():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES, threshold=1.0)
    # No class can have confidence == 1.0 (softmax of multiple classes)
    assert results[0] == []


def test_run_inference_threshold_0_returns_top_k():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=NUM_CLASSES, threshold=0.0)
    assert len(results[0]) == NUM_CLASSES


# No classes (index fallback)

def test_run_inference_no_classes_uses_index():
    model = TinyClassifier(num_classes=3)
    model.eval()
    config = {"format": "state_dict", "num_classes": 3, "classes": []}
    model_obj = LoadedModel("m", "v1", model, config)
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=3)
    labels = [p["label"] for p in results[0]]
    assert all(label.isdigit() for label in labels)


# top_k capped at num_classes

def test_run_inference_top_k_capped():
    model_obj = _make_loaded_model()
    tensor = _make_tensor(1)
    results = run_inference(model_obj, tensor, top_k=100)   # far exceeds 5
    assert len(results[0]) <= NUM_CLASSES
