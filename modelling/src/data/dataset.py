"""
Dataset classes for FetalPlanes DB and African multi-centre datasets.

Expected directory layout (both datasets share the same structure):
  data/
    fetal_db/
      train/
        brain/    *.png
        femur/    *.png
        abdomen/  *.png
        other/    *.png
      val/
      test/
    african/
      train/
        brain/    *.png
        femur/    *.png
        abdomen/  *.png
        other/    *.png
      val/
      test/
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Keep both constants available
CLASSES_3C = ["brain", "femur", "abdomen"]      # non-alpha, matches 3-class checkpoint convention
CLASSES_4C = ["abdomen", "brain", "femur", "other"]  # alphabetical - matches 4-class checkpoint folder iteration order
CLASSES    = CLASSES_4C   # default is 4-class; override explicitly if 3-class is needed
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}


class FetalDataset(Dataset):
    """
    Generic fetal ultrasound dataset - works for both Fetal DB and African
    since both share the same split/class/image folder layout.

    Args:
        root_dir:      path to a split directory, e.g. data/fetal_db/train
        transform:     torchvision/albumentations transform pipeline.
        classes:       list of class names to include (defaults to CLASSES).
        max_per_class: subsample to this many images per class.
        preprocessing: optional callable applied BEFORE transform (e.g. CLAHE).
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        classes: Optional[List[str]] = None,
        max_per_class: Optional[int] = None,
        preprocessing=None,
    ):
        self.root_dir     = Path(root_dir)
        self.transform    = transform
        self.preprocessing = preprocessing
        self.classes      = classes or CLASSES
        self.class2idx    = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            imgs = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg"))
            if max_per_class is not None:
                imgs = imgs[:max_per_class]
            for img_path in imgs:
                self.samples.append((img_path, self.class2idx[cls]))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {root_dir} for classes {self.classes}. "
                f"Check your path and class folder names."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1

        weights = np.zeros_like(counts)
        mask = counts > 0
        weights[mask] = 1.0 / counts[mask]
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler to handle class imbalance."""
        class_weights  = self.get_class_weights()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True,
        )


class AfricanDataset(FetalDataset):
    """
    African dataset - identical layout to FetalDataset (split/class/image).
    Subclassed for explicit naming and the max_patients convenience argument.

    Args:
        root_dir:     path to a split directory, e.g. data/african/train
        max_patients: subsample by patients rather than raw image count.
                      Approximates ~4 images per patient per class.
        **kwargs:     forwarded to FetalDataset.
    """

    def __init__(
        self,
        root_dir: str,
        max_patients: Optional[int] = None,
        **kwargs,
    ):
        # Convert max_patients -> max_per_class before calling super
        if max_patients is not None:
            kwargs["max_per_class"] = max_patients * 4
        super().__init__(root_dir=root_dir, **kwargs)


class AfricanDatasetFromCSV(Dataset):
    """
    African dataset built from a pre-filtered pandas DataFrame.

    Unlike AfricanDataset (which scans a directory tree), this class accepts
    a DataFrame so that arbitrary country-level splits can be constructed at
    runtime - required for Leave-One-Country-Out cross-validation.

    Expected DataFrame columns (configurable via constructor args):
        filename_col : image filename (relative to img_dir, or full path)
        label_col    : class label string (e.g. "brain", "femur", "abdomen")

    Images are located by file name anywhere under img_dir (recursive search),
    so the split/class folder an image was curated into does not matter.
    Rows whose label is not in `classes` are skipped; rows whose image cannot
    be found are skipped with a warning.

    Args:
        df:           pandas DataFrame with one row per image.
        img_dir:      root directory where images live.
        label_col:    column name containing the class string.
        filename_col: column name containing the image filename / path.
        transform:    torchvision/albumentations transform pipeline.
        preprocessing: optional callable applied BEFORE transform.
        classes:      list of class names to include (defaults to CLASSES, the
                      4-class list). Pass CLASSES_3C for the 3-class setup.
    """

    def __init__(
        self,
        df,
        img_dir: str,
        label_col: str = "Plane",
        filename_col: str = "Filename",
        transform=None,
        preprocessing=None,
        classes=None,
    ):
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        self.img_dir       = Path(img_dir)
        self.transform     = transform
        self.preprocessing = preprocessing
        self.classes       = classes or CLASSES
        self.class2idx     = {c: i for i, c in enumerate(self.classes)}

        # Scan all subdirectories to find images regardless of split/class folder
        file_index = {p.name: p for p in Path(img_dir).rglob("*.png")}
        logger.info(f"Found {len(file_index)} images in index under {img_dir}")

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            label_raw = str(row[label_col]).strip().lower()
            if label_raw not in self.class2idx:
                continue
            fname    = str(row[filename_col]).strip()
            img_path = file_index.get(fname)
            if img_path is None:
                missing += 1
                continue
            self.samples.append((img_path, self.class2idx[label_raw]))

        if missing > 0:
            logger.warning(f"{missing} images not found under {img_dir} - skipped")

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found. Check '{label_col}' and '{filename_col}' "
                f"column names and that images exist under {img_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss."""
        counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = np.zeros_like(counts)
        mask = counts > 0
        weights[mask] = 1.0 / counts[mask]
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler for class-imbalance handling."""
        class_weights  = self.get_class_weights()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True,
        )


class CombinedDataset(Dataset):
    """Merge two or more Dataset instances into one."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets  = datasets
        self._lengths  = [len(d) for d in datasets]
        self._offsets  = [0] + list(np.cumsum(self._lengths[:-1]))

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        for ds_idx, (offset, length) in enumerate(zip(self._offsets, self._lengths)):
            if idx < offset + length:
                return self.datasets[ds_idx][idx - offset]
        raise IndexError(idx)


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders."""

    sampler = None
    if use_weighted_sampler and hasattr(train_dataset, "get_sampler"):
        sampler = train_dataset.get_sampler()

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders