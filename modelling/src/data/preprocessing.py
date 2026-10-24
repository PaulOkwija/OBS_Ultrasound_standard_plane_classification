"""
Preprocessing utilities: CLAHE, header/overlay removal, normalisation.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is particularly useful
for fetal US images from low-resource African settings where image contrast and
brightness vary significantly across scanners and operators.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional



# CLAHE


class CLAHEPreprocessor:
    """
    Apply CLAHE to a PIL Image or numpy array.

    US images are grayscale in nature; we convert to LAB (if colour) or
    apply directly to the single channel.

    Args:
        clip_limit:  threshold for contrast limiting (higher = more contrast).
        tile_grid_size: size of grid for histogram equalization.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, img) -> Image.Image:
        """Accept PIL Image or numpy array, return PIL Image."""
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()

        if img_np.ndim == 2:
            # Grayscale
            enhanced = self._clahe.apply(img_np)
            result = Image.fromarray(enhanced).convert("RGB")
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            # Convert to LAB, apply CLAHE to L channel only
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_enhanced = self._clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            result = Image.fromarray(rgb_enhanced)
        else:
            result = Image.fromarray(img_np)

        return result

    def apply_to_numpy(self, img_np: np.ndarray) -> np.ndarray:
        """Apply CLAHE to numpy array (H, W) or (H, W, 3), return numpy array."""
        pil_result = self(Image.fromarray(img_np))
        return np.array(pil_result)



# Header / overlay removal


class USHeaderRemover:
    """
    Crop out machine-specific UI elements (logos, calipers, text annotations)
    from ultrasound images.

    Strategy: detect the main bright elliptical region of interest (ROI)
    and crop to a bounding box that excludes the surrounding header/footer.

    This is a heuristic approach. Threshold-based cropping works well for
    the FetalPlanes DB images where the scanner header is typically dark
    or at the borders.
    """

    def __init__(
        self,
        top_crop_fraction: float = 0.05,
        bottom_crop_fraction: float = 0.05,
        left_crop_fraction: float = 0.02,
        right_crop_fraction: float = 0.02,
    ):
        self.top = top_crop_fraction
        self.bottom = bottom_crop_fraction
        self.left = left_crop_fraction
        self.right = right_crop_fraction

    def __call__(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()

        h, w = img_np.shape[:2]
        t = int(h * self.top)
        b = int(h * (1 - self.bottom))
        l = int(w * self.left)
        r = int(w * (1 - self.right))

        cropped = img_np[t:b, l:r]

        if isinstance(img, Image.Image):
            return Image.fromarray(cropped)
        return cropped



# Artifact detection / quality filter


def compute_image_quality_score(img: np.ndarray) -> dict:
    """
    Compute basic quality metrics for a US image.

    Returns:
        dict with keys: mean_brightness, contrast, sharpness, snr_estimate
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    else:
        gray = img.astype(float)

    mean_brightness = gray.mean()
    contrast = gray.std()

    # Sharpness: Laplacian variance
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    sharpness = laplacian.var()

    # SNR estimate: signal (mean of bright pixels) / noise (std of dark pixels)
    bright_mask = gray > gray.mean()
    dark_mask = ~bright_mask
    signal = gray[bright_mask].mean() if bright_mask.any() else 0
    noise_std = gray[dark_mask].std() if dark_mask.any() else 1e-6
    snr = signal / (noise_std + 1e-6)

    return {
        "mean_brightness": float(mean_brightness),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
        "snr_estimate": float(snr),
    }


def is_valid_image(img: np.ndarray, min_sharpness: float = 5.0, min_contrast: float = 10.0) -> bool:
    """Filter out blank or severely degraded images."""
    metrics = compute_image_quality_score(img)
    return (
        metrics["sharpness"] >= min_sharpness
        and metrics["contrast"] >= min_contrast
        and metrics["mean_brightness"] > 0.5
    )



# Preprocessing pipeline factory


def build_preprocessing_pipeline(
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    contrast_threshold: float = 35.0,   # only apply CLAHE below this contrast
    remove_headers: bool = True,
) -> Optional[callable]:
    """
    Build and return a callable preprocessing pipeline.

    CLAHE is applied conditionally - only to images whose contrast falls
    below contrast_threshold. This avoids over-enhancing already good images.

    Returns None if no preprocessing is configured.
    """
    steps = []

    if remove_headers:
        steps.append(USHeaderRemover())

    if use_clahe:
        clahe = CLAHEPreprocessor(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_size,
        )

        def conditional_clahe(img):
            arr = np.array(img) if isinstance(img, Image.Image) else img
            q = compute_image_quality_score(arr)
            if q["contrast"] < contrast_threshold:
                return clahe(img)
            return img if isinstance(img, Image.Image) else Image.fromarray(img)

        steps.append(conditional_clahe)

    if not steps:
        return None

    def pipeline(img):
        for step in steps:
            img = step(img)
        return img

    return pipeline
