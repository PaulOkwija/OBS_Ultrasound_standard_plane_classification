"""
Domain shift analysis utilities.

Functions for:
  - Evaluating a pre-trained model on a new domain
  - Comparing performance across domains (Fetal DB vs African)
  - Visualising domain shift via t-SNE of feature embeddings
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import timm

from .metrics import full_evaluation_report
from ..utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_on_domain(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate a model on a given dataloader (any domain).
    Returns full_evaluation_report dict.
    """
    model = model.to(device).eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return full_evaluation_report(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        class_names=class_names,
    )


def evaluate_across_domains(
    model: nn.Module,
    domain_loaders: Dict[str, DataLoader],
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Evaluate model on multiple named domain loaders.

    domain_loaders keys should describe the domain, e.g.:
        {"fetal_test": loader, "african_test": loader}

    Returns dict: domain_name -> metrics dict
    """
    results = {}
    for domain_name, loader in domain_loaders.items():
        logger.info(f"Evaluating on {domain_name}...")
        metrics = evaluate_on_domain(model, loader, device=device, class_names=class_names)
        results[domain_name] = metrics
        logger.info(f"  {domain_name}: f1={metrics['f1']:.4f}  pr_auc={metrics.get('pr_auc', 0):.4f}")
    return results


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    model_name: str = "efficientnet_b0",   # was layer_name
) -> tuple:
    """
    Extract feature embeddings from the penultimate layer.

    Args:
        model:      trained model to extract features from
        loader:     dataloader for the domain
        device:     cuda or cpu
        model_name: timm model string matching the architecture (e.g. 'efficientnet_b0')

    Returns:
        (features: np.ndarray [N, D], labels: np.ndarray [N])
    """
    model = model.to(device).eval()

    feature_model = timm.create_model(model_name, pretrained=False, num_classes=0)
    try:
        feature_model.load_state_dict(
            {k: v for k, v in model.state_dict().items()
             if "head" not in k and "classifier" not in k},
            strict=False,
        )
        logger.info(f"Loaded weights into feature extractor: {model_name}")
    except Exception as e:
        logger.warning(f"Could not load weights into feature extractor: {e}")

    feature_model = feature_model.to(device).eval()

    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(device)
            feats = feature_model(imgs)
            if feats.dim() > 2:
                feats = feats.mean(dim=[2, 3])
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels)


def compute_domain_gap_score(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """
    Simple domain gap score: Euclidean distance between feature distribution means.
    Lower = smaller domain gap.
    """
    src_mean = source_features.mean(axis=0)
    tgt_mean = target_features.mean(axis=0)
    return float(np.linalg.norm(src_mean - tgt_mean))