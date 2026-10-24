#!/usr/bin/env python3
"""
CLI evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/<run_dir>/best_pr_auc.pt \
        --test_data fetal_test

    python scripts/evaluate.py --checkpoint checkpoints/<run_dir>/best_pr_auc.pt \
        --test_data african_test --fps --output results/african_eval.json

Trainer appends a random 4-digit suffix to each run directory
(e.g. checkpoints/baseline_fetal_4821/), so check checkpoints/ for the exact name.
The class list comes from configs/base_config.yaml and must match the checkpoint.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import torch
from pathlib import Path

from src.utils import set_seed, get_logger
from src.data import (
    FetalDataset, AfricanDataset, build_dataloaders,
    build_preprocessing_pipeline, build_transforms, CLASSES
)
from src.models import build_model, benchmark_fps
from src.evaluation import evaluate_on_domain, evaluate_across_domains, full_evaluation_report

logger = get_logger("evaluate")


def make_loader(ds, batch_size: int = 32, num_workers: int = 4):
    """Build a simple evaluation DataLoader (no sampler, no shuffle)."""
    from torch.utils.data import DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetal Ultrasound - Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--test_data",  type=str, default="fetal_test",
                        choices=["fetal_test", "african_test", "both"],
                        help="Which test set(s) to evaluate on")
    parser.add_argument("--use_clahe",  action="store_true",
                        help="Apply conditional CLAHE preprocessing")
    parser.add_argument("--fps",        action="store_true",
                        help="Run FPS / latency benchmark after evaluation")
    parser.add_argument("--config",     type=str, default="configs/base_config.yaml")
    parser.add_argument("--output",     type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    set_seed(42)
    
    # Handle optional 'other' class
    if not base_cfg["data"].get("include_other_class", True):
        base_cfg["data"]["classes"] = [c for c in base_cfg["data"].get("classes", []) if c != "other"]
        base_cfg["data"]["num_classes"] = len(base_cfg["data"]["classes"])

    # Load model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_cfg = ckpt.get("config", {})
    arch        = saved_cfg.get("architecture", base_cfg["model"]["architecture"])
    num_classes = base_cfg["data"]["num_classes"]   # Update based on config

    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded {arch} from {args.checkpoint}")

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = base_cfg["data"].get("image_size", 224)

    preprocessing = build_preprocessing_pipeline(
        use_clahe=args.use_clahe,
        clahe_clip_limit=base_cfg["preprocessing"].get("clahe_clip_limit", 2.0),
        clahe_tile_size=tuple(base_cfg["preprocessing"].get("clahe_tile_size", [8, 8])),
        contrast_threshold=base_cfg["preprocessing"].get("contrast_threshold", 35),
        remove_headers=base_cfg["preprocessing"].get("remove_headers", True),
    )
    val_tf = build_transforms(image_size, split="val")

    # Build test loaders
    fetal_dir   = Path(base_cfg["data"]["fetal_db_dir"])
    african_dir = Path(base_cfg["data"]["african_dir"])

    loaders_dict = {}

    if args.test_data in ("fetal_test", "both"):
        try:
            ds = FetalDataset(
                root_dir=str(fetal_dir / "test"),
                transform=val_tf,
                preprocessing=preprocessing,
                classes=base_cfg["data"]["classes"],
            )
            loaders_dict["fetal_test"] = make_loader(ds)
            logger.info(f"Fetal test set: {len(ds)} images")
        except ValueError as e:
            logger.warning(f"Could not load fetal test set: {e}")

    if args.test_data in ("african_test", "both"):
        try:
            ds = AfricanDataset(
                root_dir=str(african_dir / "test"),
                transform=val_tf,
                preprocessing=preprocessing,
                classes=base_cfg["data"]["classes"],
            )
            loaders_dict["african_test"] = make_loader(ds)
            logger.info(f"African test set: {len(ds)} images")
        except ValueError as e:
            logger.warning(f"Could not load African test set: {e}")

    if not loaders_dict:
        logger.error("No test data loaded - check your paths in base_config.yaml")
        sys.exit(1)

    # Evaluate
    results = {}
    for domain, loader in loaders_dict.items():
        logger.info(f"\nEvaluating on: {domain}")
        metrics = evaluate_on_domain(model, loader, device=device, class_names=base_cfg["data"]["classes"])
        results[domain] = metrics
        logger.info(f"  F1       : {metrics['f1']:.4f}")
        logger.info(f"  PR-AUC   : {metrics.get('pr_auc', 0):.4f}")
        if "classification_report" in metrics:
            print(metrics["classification_report"])

    # FPS benchmark
    if args.fps:
        logger.info("\nBenchmarking FPS...")
        fps_result = benchmark_fps(model, image_size=image_size, device=device)
        results["fps"] = fps_result
        logger.info(f"  FPS: {fps_result['fps']}  Latency: {fps_result['latency_ms_mean']:.1f} ms")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_results = {
            k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
            if isinstance(v, dict) else v
            for k, v in results.items()
        }
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
