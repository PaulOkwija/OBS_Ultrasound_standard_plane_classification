#!/usr/bin/env python3
"""
CLI entry point for Leave-One-Country-Out (LOCO) CV Transfer Learning.

Usage:
    python scripts/run_loco_cv.py \
      --model efficientnet_b0 \
      --checkpoint checkpoints/efficientnetb0_base_3c.pt \
      --african_csv raw_data/African_planes_database.csv
"""

import argparse
import sys
import os
import yaml
import json
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_logger
from src.data.loco_cv import run_loco_cv
from src.data import build_preprocessing_pipeline, build_transforms

logger = get_logger("run_loco_cv")

CLASSES_3C = ["brain", "femur", "abdomen"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LOCO-CV Transfer Learning")
    parser.add_argument("--model", type=str, required=True, 
                        help="Architecture name (e.g. efficientnet_b0, convnext_tiny)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained weights (.pt file)")
    parser.add_argument("--african_csv", type=str, required=True,
                        help="Path to African_planes_database.csv")
    parser.add_argument("--african_dir", type=str, default="data/african",
                        help="Root folder of african images")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--tracker", type=str, default="none", choices=["wandb", "mlflow", "both", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="loco_cv")
    args = parser.parse_args()

    set_seed(args.seed)
    base_cfg = load_config(args.config)

    logger.info(f"Running LOCO-CV for model: {args.model}")
    logger.info(f"Pretrained weights     : {args.checkpoint}")

    # Build transforms & preprocessing
    data_cfg = base_cfg["data"]
    image_size = data_cfg.get("image_size", 224)
    train_tf = build_transforms(image_size, split="train")
    val_tf   = build_transforms(image_size, split="val")

    preprocessing = build_preprocessing_pipeline(
        use_clahe=False,  # default baselines are no-clahe usually
        clahe_clip_limit=base_cfg["preprocessing"].get("clahe_clip_limit", 2.0),
        clahe_tile_size=tuple(base_cfg["preprocessing"].get("clahe_tile_size", [8, 8])),
        contrast_threshold=base_cfg["preprocessing"].get("contrast_threshold", 35),
        remove_headers=base_cfg["preprocessing"].get("remove_headers", True),
    )

    # Train config
    train_cfg = {
        **base_cfg["training"],
        **base_cfg.get("logging", {}),
        "project_name": base_cfg["project_name"],
        "architecture": args.model,
        "class_names": CLASSES_3C,   # stored in every fold checkpoint
    }

    # Apply transfer learning specific settings
    tl_cfg = base_cfg.get("transfer_learning", {})
    fine_tune_layers = tl_cfg.get("fine_tune_layers", 4)
    train_cfg["learning_rate"] = tl_cfg.get("fine_tune_lr", train_cfg["learning_rate"] / 2)
    train_cfg["num_epochs"]    = tl_cfg.get("fine_tune_epochs", 20)

    # Run
    results = run_loco_cv(
        csv_path=args.african_csv,
        img_dir=args.african_dir,
        architecture=args.model,
        pretrained_checkpoint=args.checkpoint,
        train_tf=train_tf,
        val_tf=val_tf,
        preprocessing=preprocessing,
        train_cfg=train_cfg,
        experiment_name=f"{args.experiment}_{args.model}",
        classes=CLASSES_3C,
        fine_tune_layers=fine_tune_layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tracker=args.tracker
    )

    # Print summary
    logger.info("\n========== LOCO-CV Transfer Learning Results ==========")
    for k, v in results["mean_cv_metrics"].items():
        std = results["std_cv_metrics"].get(k, 0)
        logger.info(f"CV {k:12s}: {v:.4f} +/- {std:.4f}")
        
    logger.info("\n========== Malawi Test Set Results ==========")
    for k, v in results["mean_test_metrics"].items():
        logger.info(f"Test {k:10s}: {v:.4f}")
        
    # Save results to json
    out_dir = Path("results/loco_cv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_loco_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
