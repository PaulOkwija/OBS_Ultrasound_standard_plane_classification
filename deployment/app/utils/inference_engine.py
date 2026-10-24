"""
Inference Engine
----------------
Runs forward pass and postprocesses logits into ranked predictions.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.core.model_manager import LoadedModel


def run_inference(
    model_obj: LoadedModel,
    tensor: torch.Tensor,          # [N, C, H, W]
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> list[list[dict]]:
    """
    Run inference on a preprocessed tensor batch.

    Args:
        model_obj: A LoadedModel instance.
        tensor:    Preprocessed image batch [N, C, H, W].
        top_k:     Number of top predictions per image.
        threshold: Minimum confidence score to include.

    Returns:
        List of N result lists. Each result list contains dicts:
            {"label": str, "class_index": int, "confidence": float}
    """
    top_k = top_k or settings.TOP_K
    threshold = threshold if threshold is not None else settings.CONFIDENCE_THRESHOLD

    tensor = tensor.to(model_obj.device)

    with torch.no_grad():
        logits = model_obj.model(tensor)           # [N, num_classes]
        probs = F.softmax(logits, dim=-1)          # [N, num_classes]

    top_k = min(top_k, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

    results = []
    classes = model_obj.classes

    for i in range(tensor.shape[0]):
        image_results = []
        for rank in range(top_k):
            conf = top_probs[i, rank].item()
            if conf < threshold:
                break
            idx = top_indices[i, rank].item()
            label = classes[idx] if classes and idx < len(classes) else str(idx)
            image_results.append({
                "label": label,
                "class_index": idx,
                "confidence": round(conf, 6),
            })
        results.append(image_results)

    return results
