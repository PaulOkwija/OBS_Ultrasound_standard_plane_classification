#!/usr/bin/env python3
"""
CLI entry point for fetching pre-trained checkpoints from the Hugging Face Hub.

Usage:
    python scripts/fetch_checkpoints.py
    python scripts/fetch_checkpoints.py --hf-repo-id Okwija/afro-fetalnet-checkpoints --out-dir checkpoints
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

from src.utils import get_logger

logger = get_logger("fetch_checkpoints")

DEFAULT_REPO_ID = "Okwija/afro-fetalnet-checkpoints"


def main():
    parser = argparse.ArgumentParser(description="Fetch model checkpoints from Hugging Face Hub")
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading checkpoints from {args.hf_repo_id} -> {out_dir} ...")
    snapshot_download(
        repo_id=args.hf_repo_id,
        repo_type="model",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    pt_files = sorted(out_dir.rglob("*.pt")) + sorted(out_dir.rglob("*.pth"))
    logger.info(f"Downloaded {len(pt_files)} checkpoint file(s):")
    for f in pt_files:
        size_mb = f.stat().st_size / 1e6
        logger.info(f"  {size_mb:7.1f} MB  {f.relative_to(out_dir)}")


if __name__ == "__main__":
    main()
