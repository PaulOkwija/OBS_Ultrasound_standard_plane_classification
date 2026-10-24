"""
Model zoo - all architectures used in the comparison experiment.

All models are loaded via `timm` which provides:
  - Consistent API
  - ImageNet pretrained weights
  - Easy feature extraction / fine-tuning control

Usage:
    model = build_model("efficientnet_b0", num_classes=3, pretrained=True)
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import timm
from typing import Optional



SUPPORTED_MODELS = {
    "mobilenetv2":             "mobilenetv2_100",
    "mobilenetv3_small":       "mobilenetv3_small_100",
    "efficientnet_b0":         "efficientnet_b0",
    "efficientnet_b2":         "efficientnet_b2",
    "efficientformerv2_s0":    "efficientformerv2_s0",
    "resnet50":                "resnet50",
    "densenet169":             "densenet169",
    "xception":                "xception",
    "convnext_tiny":           "convnext_tiny",
}


def build_model(
    architecture: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
    fine_tune_layers: int = -1,
    mlp_head: bool = False,
) -> nn.Module:
    """
    Build a classification model.

    Args:
        architecture:     key from SUPPORTED_MODELS or a raw timm model name.
        num_classes:      number of output classes.
        pretrained:       load ImageNet pretrained weights.
        dropout:          dropout rate before final classifier.
        fine_tune_layers: -1 = all layers trainable;
                           0 = freeze entire backbone (linear probe);
                           N = unfreeze last N layers + head.
        mlp_head:         if True, replace the default single-linear head with
                          a two-layer MLP (in -> 256 -> ReLU -> Dropout -> num_classes).
                          Recommended when fine_tune_layers=0 (frozen backbone).

    Returns:
        nn.Module ready for training.
    """
    timm_name = SUPPORTED_MODELS.get(architecture, architecture)

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )

    if mlp_head:
        _replace_with_mlp_head(model, num_classes, dropout)

    if fine_tune_layers == 0:
        _freeze_backbone(model)
    elif fine_tune_layers > 0:
        _partial_unfreeze(model, fine_tune_layers)
    # fine_tune_layers == -1: all trainable (default)

    return model


def _replace_with_mlp_head(model: nn.Module, num_classes: int, dropout: float) -> None:
    """
    Replace timm's default single-linear head with a two-layer MLP:
        in_features -> 256 -> ReLU -> Dropout -> num_classes

    Works for timm models that expose a `.head`, `.classifier`, or `.fc` attribute.
    """
    for attr in ("head", "classifier", "fc"):
        head = getattr(model, attr, None)
        if head is None:
            continue
        # Resolve in_features from a Linear layer (handles Compose heads too)
        in_features: Optional[int] = None
        if isinstance(head, nn.Linear):
            in_features = head.in_features
        else:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    break
        if in_features is None:
            continue
        mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        setattr(model, attr, mlp)
        return
    raise ValueError(
        f"Could not locate a head attribute (head/classifier/fc) on {type(model).__name__}."
    )


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final classification head.

    Uses timm's get_classifier() to find the head, so layers that merely
    contain "head" in their name (e.g. EfficientNet's conv_head) stay frozen.
    """
    for param in model.parameters():
        param.requires_grad = False
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
        return
    # Fallback for non-timm models: match on the usual head attribute names.
    for name, param in model.named_parameters():
        if name.startswith(("head.", "classifier.", "fc.")):
            param.requires_grad = True


def _partial_unfreeze(model: nn.Module, n_layers: int) -> None:
    """
    Freeze all layers, then unfreeze the last n_layers + head.

    Heuristic: any parameter whose name contains "head", "classifier", "fc"
    or "last" is treated as head (this includes EfficientNet's conv_head),
    and the last n_layers * 10 backbone parameter tensors are unfrozen.
    All reported LOCO results were produced with this exact behaviour, so it
    is kept as-is for reproducibility.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Collect all named parameters in order
    all_params = list(model.named_parameters())

    # Always unfreeze head
    for name, param in all_params:
        if any(k in name for k in ["head", "classifier", "fc", "last"]):
            param.requires_grad = True

    # Unfreeze last n_layers of backbone parameters
    backbone_params = [
        (n, p) for n, p in all_params
        if not any(k in n for k in ["head", "classifier", "fc"])
    ]
    for name, param in backbone_params[-n_layers * 10:]:  # ~10 params per layer
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}



# FPS / Latency benchmark


def benchmark_fps(
    model: nn.Module,
    image_size: int = 224,
    device: str = "cpu",
    batch_size: int = 1,
    warmup_runs: int = 10,
    eval_runs: int = 100,
) -> dict:
    """
    Measure inference speed (FPS) for a model.

    Returns:
        dict with keys: fps, latency_ms_mean, latency_ms_std
    """
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(dummy)

        # Timed runs
        latencies = []
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(eval_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

    latencies_ms = [l * 1000 for l in latencies]
    fps = 1000.0 / (sum(latencies_ms) / len(latencies_ms))

    return {
        "fps": round(fps, 1),
        "latency_ms_mean": round(sum(latencies_ms) / len(latencies_ms), 2),
        "latency_ms_std": round(float(torch.tensor(latencies_ms).std().item()), 2),
    }
