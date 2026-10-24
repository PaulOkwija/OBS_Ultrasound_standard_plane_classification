#!/usr/bin/env python3
"""
One-time bootstrap: seed the two Hugging Face Hub repos this project's
notebooks pull from (src/data/hf_sync.py, scripts/fetch_checkpoints.py).

You only need to run this once, whenever the upstream Zenodo sources or the
local checkpoints/ tree change. Requires an HF token: either run
`huggingface-cli login` first, set the HF_TOKEN env var, or (if running on
Kaggle) add an HF_TOKEN Kaggle secret.

Usage:
    python scripts/publish_to_huggingface.py --target data
    python scripts/publish_to_huggingface.py --target checkpoints
    python scripts/publish_to_huggingface.py --target both --dry-run
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

from src.utils import get_logger

logger = get_logger("publish_to_huggingface")

DATA_REPO_ID = "Okwija/afro-fetalnet-data"
CHECKPOINTS_REPO_ID = "Okwija/afro-fetalnet-checkpoints"

FETAL_RECORD_ID = "3904280"
FETAL_FILENAME = "FETAL_PLANES_ZENODO.zip"
AFRICAN_RECORD_ID = "7540448"
AFRICAN_FILENAME = "Zenodo_dataset.tar.xz"

DATASET_CARD = """---
license: cc-by-4.0
---

# Afro-FetalNet raw data mirror

Direct mirror of two CC BY 4.0 Zenodo records, re-hosted here for fast, token-free
downloads inside `scripts/prepare_datasets.py`. No content has been modified - these
are the original archive files, re-uploaded as-is.

## Contents

- `{fetal_filename}` - FETAL_PLANES_DB: Common maternal-fetal ultrasound images.
  Burgos-Artizzu et al., *Evaluation of deep convolutional neural networks for
  automatic classification of common maternal fetal ultrasound planes*, Scientific
  Reports (2020). Source: https://zenodo.org/records/{fetal_record}
- `{african_filename}` - Maternal fetal ultrasound planes from low-resource imaging
  settings in five African countries. Sendra-Balcells et al., *Generalisability of
  fetal ultrasound deep learning models to low-resource imaging settings in five
  African countries*, Scientific Reports (2023). Source:
  https://zenodo.org/records/{african_record}

## License

Both source datasets are licensed CC BY 4.0. This mirror preserves that license;
please cite the original papers above (not this repo) when using the data.

## Usage

Downloaded and curated automatically by `src/data/hf_sync.fetch_raw_datasets()` +
`src/data/curate.prepare_african()`/`prepare_fetal()` - see
`modelling/scripts/prepare_datasets.py` in
https://github.com/PaulOkwija/US_standard_plane_classification.
""".format(
    fetal_filename=FETAL_FILENAME, fetal_record=FETAL_RECORD_ID,
    african_filename=AFRICAN_FILENAME, african_record=AFRICAN_RECORD_ID,
)

MODEL_CARD = """---
license: cc-by-4.0
---

# Afro-FetalNet checkpoints

Model checkpoints for the Afro-FetalNet fetal ultrasound plane classifier
(EfficientNet-B0 / ConvNeXt-Tiny / EfficientFormerV2-S0 backbones, LOCO-CV
fine-tuned on African fetal ultrasound data, with weight-space "model soup"
averaging across held-out-country folds).

Downloaded automatically by `scripts/fetch_checkpoints.py` /
`scripts/download_weights.sh` - see
https://github.com/PaulOkwija/US_standard_plane_classification.

Training data: see `Okwija/afro-fetalnet-data` (CC BY 4.0, re-hosted from
Zenodo records 3904280 and 7540448).
"""


def _download_zenodo_file(record_id: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists():
        logger.info(f"{dest_path} already downloaded, skipping.")
        return dest_path

    meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30).json()
    file_entry = next((f for f in meta.get("files", []) if f["key"] == filename), None)
    if file_entry is None:
        raise FileNotFoundError(f"{filename} not found in Zenodo record {record_id}")
    url = file_entry["links"]["self"]

    logger.info(f"Downloading {filename} from Zenodo record {record_id} ({file_entry['size']/1e6:.1f} MB) ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    logger.info(f"Saved to {dest_path}")
    return dest_path


def publish_data(dry_run: bool = False, work_dir: Path = None) -> None:
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="afro_fetalnet_zenodo_"))
    fetal_path = _download_zenodo_file(FETAL_RECORD_ID, FETAL_FILENAME, work_dir)
    african_path = _download_zenodo_file(AFRICAN_RECORD_ID, AFRICAN_FILENAME, work_dir)

    if dry_run:
        logger.info(f"[dry-run] Would create dataset repo {DATA_REPO_ID} and upload {fetal_path.name}, {african_path.name}, README.md")
        return

    create_repo(DATA_REPO_ID, repo_type="dataset", exist_ok=True)
    for path in (fetal_path, african_path):
        logger.info(f"Uploading {path.name} -> {DATA_REPO_ID} ...")
        upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=DATA_REPO_ID, repo_type="dataset")

    card_path = work_dir / "README.md"
    card_path.write_text(DATASET_CARD, encoding="utf-8")
    upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=DATA_REPO_ID, repo_type="dataset")
    logger.info(f"Done. https://huggingface.co/datasets/{DATA_REPO_ID}")


def publish_checkpoints(checkpoints_dir: Path = Path("checkpoints"), dry_run: bool = False) -> None:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"{checkpoints_dir} does not exist. Populate it first (e.g. by training) "
            f"before publishing."
        )
    pt_files = list(checkpoints_dir.rglob("*.pt")) + list(checkpoints_dir.rglob("*.pth"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt/.pth files found under {checkpoints_dir}.")

    logger.info(f"Found {len(pt_files)} checkpoint file(s) under {checkpoints_dir}:")
    for f in sorted(pt_files):
        logger.info(f"  {f.stat().st_size/1e6:7.1f} MB  {f.relative_to(checkpoints_dir)}")

    if dry_run:
        logger.info(f"[dry-run] Would create model repo {CHECKPOINTS_REPO_ID} and upload the tree above + README.md")
        return

    create_repo(CHECKPOINTS_REPO_ID, repo_type="model", exist_ok=True)
    logger.info(f"Uploading {checkpoints_dir} -> {CHECKPOINTS_REPO_ID} ...")
    upload_folder(
        folder_path=str(checkpoints_dir), repo_id=CHECKPOINTS_REPO_ID, repo_type="model",
        allow_patterns=["*.pt", "*.pth"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(MODEL_CARD, encoding="utf-8")
        upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=CHECKPOINTS_REPO_ID, repo_type="model")
    logger.info(f"Done. https://huggingface.co/{CHECKPOINTS_REPO_ID}")


def main():
    parser = argparse.ArgumentParser(description="Seed the Hugging Face Hub repos for this project")
    parser.add_argument("--target", choices=["data", "checkpoints", "both"], default="both")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Pick up HF_TOKEN from a Kaggle secret if present.
    if "HF_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass  # falls back to `huggingface-cli login` cache, or fails loudly on upload

    if args.target in ("data", "both"):
        publish_data(dry_run=args.dry_run)
    if args.target in ("checkpoints", "both"):
        publish_checkpoints(checkpoints_dir=Path(args.checkpoints_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
