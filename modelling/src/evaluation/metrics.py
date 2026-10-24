"""
Evaluation metrics: accuracy, AUC, precision, recall, F1 per class and macro.
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Optional, List
from ..utils.logger import get_logger
logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true:      ground truth labels (N,)
        y_pred:      predicted labels (N,)
        y_probs:     predicted probabilities (N, C) - needed for AUC
        class_names: list of class names for the report

    Returns:
        dict of scalar metric values
    """
    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    if y_probs is not None:
        num_classes = y_probs.shape[1]
        try:
            # Ensure probs sum to 1 along class axis (guard against float32 numerical drift)
            # Scikit-learn has a strict tolerance (atol=1e-7) which float32 often fails.
            y_probs_norm = np.array(y_probs, dtype=np.float64)
            row_sums = y_probs_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            y_probs_norm /= row_sums
            
            if num_classes == 2:
                pr_auc = average_precision_score(y_true, y_probs_norm[:, 1])
                auc = roc_auc_score(y_true, y_probs_norm[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                pr_auc = average_precision_score(
                    y_true_bin, y_probs_norm,
                    average="macro"
                )
                auc = roc_auc_score(
                    y_true, y_probs_norm,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(num_classes))
                )
            metrics["pr_auc"] = float(pr_auc)
            metrics["auc"] = float(auc)
        except Exception as e:
            logger.warning(f"AUC/PR-AUC computation failed: {e}")
            metrics["pr_auc"] = 0.0
            metrics["auc"] = 0.0

    return metrics


def full_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Extended evaluation report including per-class metrics and confusion matrix.
    """
    from ..data import CLASSES
    class_names = class_names or CLASSES

    metrics = compute_metrics(y_true, y_pred, y_probs, class_names)

    # Per-class metrics
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0)

    present_classes = sorted(np.unique(y_true))
    for i, cls_idx in enumerate(present_classes):
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        if i < len(prec_per):
            metrics[f"precision_{name}"] = float(prec_per[i])
            metrics[f"recall_{name}"]    = float(rec_per[i])
            metrics[f"f1_{name}"]        = float(f1_per[i])

    # Confusion matrix (as nested list for JSON serialisation)
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Text report
    # Build target_names robustly: handle cases where class_names is shorter than present_classes
    if class_names:
        target_names = [
            class_names[i] if i < len(class_names) else f"class_{i}"
            for i in present_classes
        ]
    else:
        target_names = None
    
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        labels=list(present_classes),
        target_names=target_names,
        zero_division=0,
    )

    return metrics
