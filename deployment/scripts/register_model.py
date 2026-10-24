#!/usr/bin/env python3
"""
register_model.py
-----------------
Helper script to add a model to the local registry.

Usage (state_dict checkpoint produced by modelling/):
    python scripts/register_model.py \
        --name efficientnet_b0_soup \
        --version v1 \
        --weights ../modelling/checkpoints/averaged/efficientnet_b0_loco_averaged.pt \
        --architecture efficientnet_b0 \
        --num-classes 3 \
        --classes brain femur abdomen \
        --format state_dict

For TorchScript models:
    python scripts/register_model.py \
        --name my_model \
        --version v1 \
        --weights /path/to/model.pt \
        --format torchscript \
        --classes abdomen brain femur

--classes must be listed in the model's output index order. The modelling
pipeline's 3-class checkpoints use ["brain", "femur", "abdomen"].
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Register a model into the local registry.")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--version", required=True, help="Version string e.g. v1")
    parser.add_argument("--weights", required=True, help="Path to .pt weights file")
    parser.add_argument(
        "--format",
        choices=["state_dict", "torchscript"],
        default="state_dict",
        help="Model format",
    )
    parser.add_argument("--architecture", default="efficientnet_b0", help="timm or torchvision architecture name (state_dict only)")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--classes", nargs="*", default=[], help="Class label list")
    parser.add_argument(
        "--registry",
        default="model_registry",
        help="Path to registry directory",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    dest_dir = Path(args.registry) / args.name / args.version
    if dest_dir.exists() and not args.overwrite:
        print(f"[ERROR] {dest_dir} already exists. Use --overwrite to replace.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights
    src = Path(args.weights)
    if not src.exists():
        print(f"[ERROR] Weights file not found: {src}")
        return

    shutil.copy2(src, dest_dir / "model.pt")
    print(f"Copied weights -> {dest_dir / 'model.pt'}")

    # Write config
    config = {
        "format": args.format,
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "classes": args.classes or [],
    }
    cfg_path = dest_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"Wrote config  -> {cfg_path}")
    print(f"\nModel '{args.name}' version '{args.version}' registered successfully.")


if __name__ == "__main__":
    main()
