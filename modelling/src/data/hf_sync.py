"""
Download and extract the raw source datasets from the Hugging Face Hub mirror
that scripts/publish_to_huggingface.py seeds from the original Zenodo records:

  FETAL_PLANES_DB  - https://zenodo.org/records/3904280 (Burgos-Artizzu et al.)
  African dataset  - https://zenodo.org/records/7540448 (Sendra-Balcells et al.)

Both are CC BY 4.0. src.data.curate.prepare_african()/prepare_fetal() consume
the paths this module returns to build the train/val/test folder layout.
"""

from __future__ import annotations
import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_REPO_ID = "Okwija/afro-fetalnet-data"

AFRICAN_ARCHIVE = "Zenodo_dataset.tar.xz"
FETAL_ARCHIVE = "FETAL_PLANES_ZENODO.zip"

# Vendored copy of the African CSV, already committed to the repo. Used as a
# sanity check that the HF-hosted mirror hasn't silently drifted from what
# every previously-reported result was computed against.
_VENDORED_AFRICAN_CSV = Path(__file__).resolve().parents[2] / "raw_data" / "African_planes_database.csv"


def _extract_tar_xz(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:xz") as tf:
        tf.extractall(dest_dir)


def _extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest_dir)


def _find_file(root: Path, *, name_contains: str, suffix: str) -> Optional[Path]:
    """Search recursively for a file matching a substring + suffix. Defensive
    against the archive nesting an extra folder level, which is a common
    Zenodo/Kaggle re-upload pattern we can't verify without downloading the
    full 2GB Fetal DB zip."""
    name_contains = name_contains.lower()
    for p in root.rglob(f"*{suffix}"):
        if name_contains in p.name.lower():
            return p
    # Fall back to the first matching suffix if the name hint didn't match -
    # better to proceed with a loud downstream error than silently do nothing.
    matches = list(root.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def _find_images_dir(root: Path) -> Optional[Path]:
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "images":
            if any(p.glob("*.png")):
                return p
    return None


def _verify_african_csv(csv_path: Path) -> None:
    if not _VENDORED_AFRICAN_CSV.exists():
        logger.warning(
            f"No vendored CSV at {_VENDORED_AFRICAN_CSV} to verify against - skipping check."
        )
        return

    fresh = pd.read_csv(csv_path)
    vendored = pd.read_csv(_VENDORED_AFRICAN_CSV)
    if len(fresh) != len(vendored):
        raise ValueError(
            f"Downloaded African CSV has {len(fresh)} rows but the vendored copy "
            f"({_VENDORED_AFRICAN_CSV}) has {len(vendored)}. The HF mirror may have "
            f"drifted from what every previously-reported result was computed against - "
            f"investigate before curating."
        )
    fresh_hash = hashlib.sha256(pd.util.hash_pandas_object(fresh, index=False).values).hexdigest()
    vendored_hash = hashlib.sha256(pd.util.hash_pandas_object(vendored, index=False).values).hexdigest()
    if fresh_hash != vendored_hash:
        raise ValueError(
            "Downloaded African CSV has the same row count as the vendored copy but "
            "different content (hash mismatch). Investigate before curating."
        )
    logger.info("African CSV matches the vendored copy (row count + content hash).")


def fetch_raw_datasets(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: Path = Path("data/raw"),
    *,
    skip_verify: bool = False,
) -> Dict[str, Path]:
    """
    Download both raw archives from the HF dataset repo and extract them.

    Returns:
        dict with keys: african_root, african_csv, fetal_root, fetal_csv
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {AFRICAN_ARCHIVE} from {repo_id} ...")
    african_archive = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=AFRICAN_ARCHIVE,
        local_dir=local_dir,
    )
    african_extract_dir = local_dir / "african_extracted"
    _extract_tar_xz(Path(african_archive), african_extract_dir)

    african_csv = _find_file(african_extract_dir, name_contains="african_planes", suffix=".csv")
    if african_csv is None:
        raise FileNotFoundError(f"Could not locate African_planes_database.csv under {african_extract_dir}")
    african_root = african_csv.parent
    if not skip_verify:
        _verify_african_csv(african_csv)

    logger.info(f"Downloading {FETAL_ARCHIVE} from {repo_id} ...")
    fetal_archive = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=FETAL_ARCHIVE,
        local_dir=local_dir,
    )
    fetal_extract_dir = local_dir / "fetal_extracted"
    _extract_zip(Path(fetal_archive), fetal_extract_dir)

    fetal_images_dir = _find_images_dir(fetal_extract_dir)
    if fetal_images_dir is None:
        raise FileNotFoundError(
            f"Could not locate an 'Images/' directory containing .png files under "
            f"{fetal_extract_dir}. The FETAL_PLANES_ZENODO.zip internal layout may differ "
            f"from what src.data.curate.prepare_fetal() expects - inspect {fetal_extract_dir} "
            f"and adjust images_subdir accordingly."
        )
    fetal_root = fetal_images_dir.parent

    fetal_csv = _find_file(fetal_extract_dir, name_contains="fetal_planes_db", suffix=".csv")
    if fetal_csv is None:
        raise FileNotFoundError(f"Could not locate the FETAL_PLANES_DB CSV under {fetal_extract_dir}")

    logger.info(f"African: root={african_root}  csv={african_csv}")
    logger.info(f"Fetal DB: root={fetal_root}  csv={fetal_csv}")

    return {
        "african_root": african_root,
        "african_csv": african_csv,
        "fetal_root": fetal_root,
        "fetal_csv": fetal_csv,
    }
