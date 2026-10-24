"""
tests/test_api.py
Integration tests for the FastAPI routes.
Uses a real in-memory model - no disk registry needed.
"""

import io
import json
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from app.core.model_manager import LoadedModel, model_manager
from app.main import app

client = TestClient(app)

NUM_CLASSES = 3
CLASSES = ["cat", "dog", "bird"]
MODEL_NAME = "test_api_model"
MODEL_VERSION = "v1"


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.pool(x).squeeze(-1).squeeze(-1))


@pytest.fixture(autouse=True)
def register_test_model():
    """Pre-load a tiny model into the manager for all tests."""
    model = TinyModel()
    model.eval()
    config = {
        "format": "state_dict",
        "architecture": "tiny",
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
    }
    loaded = LoadedModel(MODEL_NAME, MODEL_VERSION, model, config)
    key = f"{MODEL_NAME}:{MODEL_VERSION}"
    model_manager._models[key] = loaded
    yield
    model_manager._models.pop(key, None)


def _make_image_bytes(fmt="JPEG") -> bytes:
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# /health

def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["loaded_models"], list)


# /api/v1/inference/predict

def test_predict_single_image():
    img_bytes = _make_image_bytes()
    r = client.post(
        f"/api/v1/inference/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == MODEL_NAME
    assert len(data["predictions"]) > 0
    pred = data["predictions"][0]
    assert "label" in pred
    assert "confidence" in pred
    assert "class_index" in pred


def test_predict_returns_top_k():
    img_bytes = _make_image_bytes()
    r = client.post(
        f"/api/v1/inference/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}&top_k=2",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_png_image():
    img_bytes = _make_image_bytes(fmt="PNG")
    r = client.post(
        f"/api/v1/inference/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200


def test_predict_model_not_found():
    img_bytes = _make_image_bytes()
    r = client.post(
        "/api/v1/inference/predict?model_name=nonexistent_model",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 404


def test_predict_invalid_image():
    r = client.post(
        f"/api/v1/inference/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}",
        files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
    )
    assert r.status_code == 422


def test_predict_confidence_threshold_high():
    img_bytes = _make_image_bytes()
    r = client.post(
        f"/api/v1/inference/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}&threshold=0.99",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    # May return 0 predictions above 99% threshold
    assert isinstance(r.json()["predictions"], list)


# /api/v1/batch/predict

def test_batch_predict_multiple_images():
    files = [
        ("files", (f"img{i}.jpg", _make_image_bytes(), "image/jpeg"))
        for i in range(3)
    ]
    r = client.post(
        f"/api/v1/batch/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}",
        files=files,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3


def test_batch_predict_single_image():
    files = [("files", ("img.jpg", _make_image_bytes(), "image/jpeg"))]
    r = client.post(
        f"/api/v1/batch/predict?model_name={MODEL_NAME}&model_version={MODEL_VERSION}",
        files=files,
    )
    assert r.status_code == 200
    assert r.json()["total"] == 1


# /api/v1/models

def test_list_loaded_models():
    r = client.get("/api/v1/models/loaded")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert MODEL_NAME in names


def test_list_available_models():
    r = client.get("/api/v1/models/available")
    assert r.status_code == 200
    assert "models" in r.json()


def test_unload_and_reload():
    # Unload
    r = client.post(
        "/api/v1/models/unload",
        json={"name": MODEL_NAME, "version": MODEL_VERSION},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "unloaded"

    # Confirm unloaded
    r = client.get("/api/v1/models/loaded")
    names = [m["name"] for m in r.json()["models"]]
    assert MODEL_NAME not in names
