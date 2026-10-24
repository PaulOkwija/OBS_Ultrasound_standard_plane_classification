"""Tests for model building and benchmarking."""
import pytest
import torch
from src.models import build_model, count_parameters, SUPPORTED_MODELS


@pytest.mark.parametrize("arch", ["mobilenetv3_small", "efficientnet_b0", "resnet50"])
def test_build_model_forward(arch):
    model = build_model(arch, num_classes=4, pretrained=False)   # 4 classes: brain, femur, abdomen, other
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4)


def test_count_parameters():
    model = build_model("mobilenetv3_small", num_classes=4, pretrained=False)
    info = count_parameters(model)
    assert info["total"] > 0
    assert info["trainable"] <= info["total"]


def test_freeze_backbone():
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False, fine_tune_layers=0)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    # Most parameters should be frozen
    assert trainable < total * 0.1
