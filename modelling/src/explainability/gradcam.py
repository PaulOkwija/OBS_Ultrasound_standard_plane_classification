"""
Grad-CAM implementation for fetal US classification explainability.

Supports any model loaded via timm by automatically finding the last
convolutional layer.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from typing import Optional, List, Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Reference:
        Selvaraju et al., 2017 - Grad-CAM: Visual Explanations from Deep
        Networks via Gradient-Based Localization. ICCV.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._find_last_conv(model)
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    # Hook setup

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    # Core computation

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) - single image tensor (not batched).
            class_idx:    target class; if None uses the predicted class.

        Returns:
            heatmap: (H, W) float32 array in [0, 1].
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.unsqueeze(0).to(device) if input_tensor.dim() == 3 else input_tensor.to(device)

        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM weights: global average pooling of gradients
        # Shape: (C,) - one weight per feature map
        weights = self._gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weighted sum of activation maps
        # activations: (1, C, h, w)
        cam = (weights[:, None, None] * self._activations[0]).sum(dim=0)  # (h, w)

        # ReLU + normalise
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def generate_overlay(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image: (H, W, 3) uint8 RGB image.
            cam:            (h, w) float32 heatmap in [0, 1].
            alpha:          blending weight for heatmap.
            colormap:       OpenCV colormap.

        Returns:
            (H, W, 3) uint8 RGB overlay image.
        """
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = (cam_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_rgb + (1 - alpha) * original_image).astype(np.uint8)
        return overlay

    # Utilities

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """
        Traverse the model and return the last Conv2d layer.
        Falls back to the last BatchNorm2d if no Conv2d found.
        """
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    last_conv = module
        if last_conv is None:
            raise ValueError("Could not find a Conv2d layer in the model.")
        return last_conv



# Convenience function


def visualise_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    device: str = "cuda",
    max_images: int = 5,
    imagenet_mean: List[float] = [0.485, 0.456, 0.406],
    imagenet_std:  List[float] = [0.229, 0.224, 0.225],
) -> List[dict]:
    """
    Generate Grad-CAM overlays for a batch of images.

    Returns list of dicts: {image_rgb, cam, overlay, pred_class, true_class, correct}
    """
    cam_engine = GradCAM(model)
    model = model.to(device).eval()
    results = []

    mean = np.array(imagenet_mean)
    std  = np.array(imagenet_std)

    for i in range(min(max_images, images.size(0))):
        img_tensor = images[i]
        label = int(labels[i].item())

        # Grad-CAM
        cam = cam_engine(img_tensor.clone(), class_idx=None)

        # Denormalise for display
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Predicted class
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred = int(logits.argmax(dim=1).item())

        overlay = cam_engine.generate_overlay(img_np, cam)

        results.append({
            "image_rgb":  img_np,
            "cam":        cam,
            "overlay":    overlay,
            "pred_class": class_names[pred] if pred < len(class_names) else str(pred),
            "true_class": class_names[label] if label < len(class_names) else str(label),
            "correct":    pred == label,
        })

    cam_engine.remove_hooks()
    return results
