"""Tests for dataset classes."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from src.data.dataset import FetalDataset, CLASSES


def make_dummy_dataset(root: Path, n_per_class: int = 5):
    """Create a tiny dummy dataset for testing."""
    for cls in CLASSES[:3]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(d / f"patient_00{i}_img_001.png")


def test_fetal_dataset_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_dummy_dataset(root)
        ds = FetalDataset(root_dir=str(root), classes=CLASSES[:3])
        assert len(ds) == 15  # 3 classes x 5 images
        img, label = ds[0]
        assert label in range(3)


def test_fetal_dataset_class_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_dummy_dataset(root, n_per_class=5)
        ds = FetalDataset(root_dir=str(root), classes=CLASSES[:3])
        weights = ds.get_class_weights()
        assert weights.shape[0] == 3
        assert abs(weights.sum().item() - 1.0) < 1e-4


def test_fetal_dataset_max_per_class():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_dummy_dataset(root, n_per_class=10)
        ds = FetalDataset(root_dir=str(root), classes=CLASSES[:3], max_per_class=3)
        assert len(ds) == 9  # 3 classes x 3 images
