#!/usr/bin/env python3
"""
CLI entry point for dataset preparation: download raw sources from the
Hugging Face Hub mirror, then curate into data/african/ and data/fetal_db/.

Usage:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --skip-download   # reuse data/raw/
    python scripts/prepare_datasets.py --force            # re-curate even if data/ exists
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger
from src.data.hf_sync import fetch_raw_datasets, DEFAULT_REPO_ID
from src.data.curate import prepare_african, prepare_fetal
from src.data.splits import get_split_stats

logger = get_logger("prepare_datasets")

CLASSES_3C = ["brain", "femur", "abdomen"]
CLASSES_4C = ["abdomen", "brain", "femur", "other"]


def main():
    parser = argparse.ArgumentParser(description="Prepare African + Fetal DB datasets from HF-mirrored raw sources")
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--skip-download", action="store_true", help="Reuse an already-populated --raw-dir")
    parser.add_argument("--force", action="store_true", help="Re-curate even if output splits already exist")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the African-CSV drift check")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    african_out = out_dir / "african"
    fetal_out = out_dir / "fetal_db"

    if not args.force and african_out.exists() and fetal_out.exists():
        logger.info(f"{african_out} and {fetal_out} already exist - skipping (use --force to re-curate).")
        get_split_stats(african_out, CLASSES_4C)
        get_split_stats(fetal_out, CLASSES_4C)
        return

    if args.skip_download:
        african_csv = raw_dir / "african_extracted" / "Zenodo_dataset" / "African_planes_database.csv"
        paths = {
            "african_root": african_csv.parent,
            "african_csv": african_csv,
            "fetal_root": None,
            "fetal_csv": None,
        }
        # Re-derive fetal paths the same way hf_sync would have left them.
        from src.data.hf_sync import _find_images_dir, _find_file
        fetal_extract_dir = raw_dir / "fetal_extracted"
        fetal_images_dir = _find_images_dir(fetal_extract_dir)
        paths["fetal_root"] = fetal_images_dir.parent if fetal_images_dir else None
        paths["fetal_csv"] = _find_file(fetal_extract_dir, name_contains="fetal_planes_db", suffix=".csv")
        if not paths["african_csv"].exists() or paths["fetal_root"] is None:
            logger.error(f"--skip-download given but {raw_dir} doesn't look populated. Run without --skip-download first.")
            sys.exit(1)
    else:
        paths = fetch_raw_datasets(repo_id=args.hf_repo_id, local_dir=raw_dir, skip_verify=args.skip_verify)

    logger.info("=" * 60)
    logger.info("Curating African dataset")
    logger.info("=" * 60)
    prepare_african(root=paths["african_root"], csv_path=paths["african_csv"], out_dir=african_out)

    logger.info("=" * 60)
    logger.info("Curating Fetal DB dataset")
    logger.info("=" * 60)
    prepare_fetal(root=paths["fetal_root"], csv_path=paths["fetal_csv"], out_dir=fetal_out)

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("African dataset:")
    get_split_stats(african_out, CLASSES_4C)
    logger.info("Fetal DB dataset:")
    get_split_stats(fetal_out, CLASSES_4C)


if __name__ == "__main__":
    main()
