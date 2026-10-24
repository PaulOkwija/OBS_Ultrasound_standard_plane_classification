"""
Dataset curation: turn raw Zenodo-sourced images + CSVs into the
train/val/test/{class}/ folder layout the rest of the pipeline expects.

Called by scripts/prepare_datasets.py after src.data.hf_sync.fetch_raw_datasets()
has downloaded and extracted the raw archives.

African dataset  - african/{split}/{class}/{images}
                   - train/val: patient-level 70/30 split of the 4 non-test
                     countries (test country held out entirely)
                   - test: held-out country only, all its images

Fetal DB dataset - fetal_db/{split}/{class}/{images}
                   - test:  CSV train==0 rows (patient-safe boundary)
                   - train: CSV train==1 rows, patient-level 70% of pool
                   - val:   CSV train==1 rows, patient-level 30% of pool

Both splits use the same RANDOM_STATE so every previously-reported number
(image counts, F1 scores trained against these splits) stays reproducible.
"""

from __future__ import annotations
import shutil
import pandas as pd
from pathlib import Path
from typing import Set
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42

# Classes kept as-is; anything else (e.g. "Fetal thorax", "Maternal cervix")
# collapses into "other". One shared constant for both datasets.
VALID_CLASSES: Set[str] = {"brain", "femur", "abdomen"}


def remap_class(raw_class: str, valid: Set[str] = VALID_CLASSES) -> str:
    """Lowercase, take the last word (e.g. 'Fetal brain' -> 'brain'), and
    return it if valid, otherwise 'other'."""
    c = raw_class.strip().lower().split(" ")[-1]
    return c if c in valid else "other"


def safe_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst, creating parent dirs as needed. Skip if src missing."""
    if not src.exists():
        logger.warning(f"Source not found, skipping: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _report(split_name: str, df: pd.DataFrame, class_col: str) -> dict:
    counts = df[class_col].value_counts().to_dict()
    logger.info(f"  {split_name:20s}: {len(df):5d} images | {counts}")
    return {"n_images": len(df), "class_counts": counts}


def prepare_african(
    root: Path,
    csv_path: Path,
    out_dir: Path,
    *,
    val_size: float = 0.30,
    random_state: int = RANDOM_STATE,
    valid_classes: Set[str] = VALID_CLASSES,
    test_country: str = "Malawi",
    img_col: str = "Filename",
    class_col: str = "Plane",
    country_col: str = "Center",
    patient_col: str = "Patient_num",
) -> dict:
    """
    Curate the African dataset into out_dir/{train,val,test}/{class}/.

    test_country is held out entirely (all its images, regardless of split).
    The remaining countries are pooled and split patient-level val_size/1-val_size
    via sklearn.train_test_split (NOT using the CSV's own 'Train' column, which
    the original curation never used for this dataset either).

    Returns a dict with train/val/test stats (see _report()).
    """
    root, csv_path, out_dir = Path(root), Path(csv_path), Path(out_dir)
    logger.info("=" * 60)
    logger.info("AFRICAN DATASET")
    logger.info("=" * 60)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV: {len(df)} rows | columns: {df.columns.tolist()}")

    df["label"] = df[class_col].apply(lambda x: remap_class(str(x), valid_classes))
    logger.info(f"Class distribution after remapping: {df['label'].value_counts().to_dict()}")

    df_test = df[df[country_col].str.strip() == test_country].copy()
    df_trainval = df[df[country_col].str.strip() != test_country].copy()

    logger.info(f"Countries in train/val pool : {sorted(df_trainval[country_col].unique().tolist())}")
    logger.info(f"Test country                : {test_country}")

    unique_patients = df_trainval[patient_col].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_size, random_state=random_state,
    )
    train_patients, val_patients = set(train_patients), set(val_patients)

    df_train = df_trainval[df_trainval[patient_col].isin(train_patients)].copy()
    df_val = df_trainval[df_trainval[patient_col].isin(val_patients)].copy()

    logger.info(f"Patient-based split: train patients={len(train_patients)}  val patients={len(val_patients)}")
    stats = {
        "train": _report("Train", df_train, "label"),
        "val": _report("Val", df_val, "label"),
        "test": _report(f"Test ({test_country})", df_test, "label"),
    }

    def copy_split(subset_df: pd.DataFrame, split_name: str) -> None:
        ok = skip = 0
        for _, row in subset_df.iterrows():
            country = str(row[country_col]).strip()
            cls = row["label"]
            img_name = str(row[img_col]).strip() + ".png"
            src = root / country / img_name
            dst = out_dir / split_name / cls / img_name
            if src.exists():
                safe_copy(src, dst)
                ok += 1
            else:
                skip += 1
        logger.info(f"  Copied {split_name}: {ok}  |  Skipped (missing): {skip}")

    copy_split(df_train, "train")
    copy_split(df_val, "val")
    copy_split(df_test, "test")

    return stats


def prepare_fetal(
    root: Path,
    csv_path: Path,
    out_dir: Path,
    *,
    val_size: float = 0.30,
    random_state: int = RANDOM_STATE,
    valid_classes: Set[str] = VALID_CLASSES,
    img_col: str = "Image_name",
    class_col: str = "Plane",
    train_col: str = "Train",
    patient_col: str = "Patient_num",
    csv_sep: str = ";",
    images_subdir: str = "Images",
) -> dict:
    """
    Curate the Spanish FetalPlanes DB into out_dir/{train,val,test}/{class}/.

    test: CSV train_col==0 rows (patient-safe boundary, not a ratio split).
    train/val: CSV train_col==1 rows, patient-level val_size/1-val_size split.

    Returns a dict with train/val/test stats (see _report()).
    """
    root, csv_path, out_dir = Path(root), Path(csv_path), Path(out_dir)
    img_dir = root / images_subdir
    logger.info("=" * 60)
    logger.info("FETAL DB DATASET")
    logger.info("=" * 60)

    df = pd.read_csv(csv_path, sep=csv_sep)
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded CSV: {len(df)} rows | columns: {df.columns.tolist()}")

    df["label"] = df[class_col].apply(lambda x: remap_class(str(x), valid_classes))
    logger.info(f"Class distribution after remapping: {df['label'].value_counts().to_dict()}")

    df_pool = df[df[train_col] == 1].copy()
    df_test = df[df[train_col] == 0].copy()
    logger.info(f"Pool (train+val): {len(df_pool)} images | {df_pool[patient_col].nunique()} patients")
    logger.info(f"Test pool:        {len(df_test)} images | {df_test[patient_col].nunique()} patients")

    unique_patients = df_pool[patient_col].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_size, random_state=random_state,
    )
    train_patients, val_patients = set(train_patients), set(val_patients)

    df_train = df_pool[df_pool[patient_col].isin(train_patients)].copy()
    df_val = df_pool[df_pool[patient_col].isin(val_patients)].copy()

    logger.info(f"Patient-based split: train patients={len(train_patients)}  val patients={len(val_patients)}")
    stats = {
        "train": _report("Train", df_train, "label"),
        "val": _report("Val", df_val, "label"),
        "test": _report("Test", df_test, "label"),
    }

    def copy_split(subset_df: pd.DataFrame, split_name: str) -> None:
        ok = skip = 0
        for _, row in subset_df.iterrows():
            img_name = str(row[img_col]).strip() + ".png"
            cls = row["label"]
            src = img_dir / img_name
            dst = out_dir / split_name / cls / img_name
            if src.exists():
                safe_copy(src, dst)
                ok += 1
            else:
                skip += 1
        logger.info(f"  Copied {split_name}: {ok}  |  Skipped (missing): {skip}")

    copy_split(df_train, "train")
    copy_split(df_val, "val")
    copy_split(df_test, "test")

    return stats
