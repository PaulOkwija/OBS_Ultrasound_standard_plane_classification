"""
Lightweight split utilities - kept for reference and testing purposes.
Dataset splits are pre-built by scripts/prepare_datasets.py and should
not be re-generated at training time.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


def get_split_stats(data_root: Path, classes: List[str]) -> dict:
    """
    Print image counts per split and class for a pre-built dataset directory.
    Useful for sanity-checking after prepare_datasets.py has run.
    """
    stats = {}
    for split in ["train", "val", "test"]:
        stats[split] = {}
        for cls in classes:
            d = data_root / split / cls
            if d.exists():
                count = len(list(d.glob("*.png")) + list(d.glob("*.jpg")))
                stats[split][cls] = count
            else:
                stats[split][cls] = 0

    for split, class_counts in stats.items():
        total = sum(class_counts.values())
        print(f"  {split:5s} ({total:5d} total): {class_counts}")

    return stats