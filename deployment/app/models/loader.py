"""
Model Loader
------------
Supports two loading modes determined by config.json:

  1. "format": "torchscript"  - loads a TorchScript .pt file directly.
  2. "format": "state_dict"   - builds the architecture (timm first, then
                                 torchvision as a fallback) and loads the
                                 weights from the .pt file.

config.json example (state_dict, e.g. a checkpoint from modelling/):
{
    "format": "state_dict",
    "architecture": "efficientnet_b0",
    "num_classes": 3,
    "classes": ["brain", "femur", "abdomen"]
}

config.json example (torchscript):
{
    "format": "torchscript",
    "classes": ["abdomen", "brain", "femur"]
}

The order of "classes" must match the output index order the model was
trained with. A wrong order silently produces wrong labels.
"""

from pathlib import Path

import torch
import timm
import torchvision.models as tv_models


def load_model_from_config(
    model_path: Path, config: dict, device: torch.device
) -> torch.nn.Module:
    fmt = config.get("format")

    if fmt == "torchscript":
        model = torch.jit.load(str(model_path), map_location=device)

    elif fmt == "state_dict":
        arch = config.get("architecture", "efficientnet_b0")
        num_classes = config.get("num_classes", 1000)
        state = torch.load(str(model_path), map_location=device, weights_only=False)
        # Handle both raw state_dict and checkpoint dicts
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        try:
            model = _build_timm_model(arch, num_classes)
            model.load_state_dict(state)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                model = _build_torchvision_model(arch, num_classes)
                model.load_state_dict(state)
            else:
                raise e
                
        model = model.to(device)

    else:
        raise ValueError(f"Unknown model format: {fmt!r}. Use 'torchscript' or 'state_dict'.")

    model.eval()
    return model


def _build_timm_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Instantiate a timm model matching the training architecture."""
    try:
        model = timm.create_model(
            arch,
            pretrained=False,
            num_classes=num_classes,
        )
        return model
    except Exception as e:
        raise ValueError(
            f"Failed to build timm architecture: '{arch}'. Error: {e}"
        )


def _build_torchvision_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Instantiate a TorchVision model and replace the final classification head."""
    try:
        builder = getattr(tv_models, arch)
    except AttributeError:
        raise ValueError(
            f"Unknown TorchVision architecture: '{arch}'. "
            f"Examples: resnet50, efficientnet_b0, vit_b_16, mobilenet_v3_large"
        )

    # Build with default weights structure (no pretrained download - offline)
    model = builder(weights=None)

    # Replace classification head for non-standard num_classes
    if num_classes != 1000:
        import torch.nn as nn
        # ResNet / DenseNet style
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        # EfficientNet / MobileNet / VGG style
        elif hasattr(model, "classifier"):
            last = model.classifier[-1]
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        # ViT style
        elif hasattr(model, "heads"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    return model
