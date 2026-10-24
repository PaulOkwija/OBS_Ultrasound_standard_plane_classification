#!/usr/bin/env python3
"""
CLI training entry point.

Usage examples:
    python scripts/train.py --experiment baseline_fetal
    python scripts/train.py --experiment domain_shift_zero_shot --model resnet50
    python scripts/train.py --experiment transfer_learning --patients 6
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from pathlib import Path

from src.utils import set_seed, get_logger
from src.data import (
    FetalDataset, AfricanDataset, CombinedDataset,
    build_dataloaders, build_preprocessing_pipeline, build_transforms,
    CLASSES
)
from src.models import build_model
from src.training import Trainer

logger = get_logger("train")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_datasets_for_experiment(exp_cfg: dict, base_cfg: dict, args):
    """Build train/val/test datasets based on experiment config."""
    data_cfg = base_cfg["data"]

    # An experiment may define its own class list (e.g. the *_3c experiments);
    # otherwise drop "other" from the base list when include_other_class is false.
    if "classes" in exp_cfg:
        data_cfg["classes"] = list(exp_cfg["classes"])
    elif not data_cfg.get("include_other_class", True):
        data_cfg["classes"] = [c for c in data_cfg.get("classes", []) if c != "other"]
    data_cfg["num_classes"] = len(data_cfg["classes"])

    use_clahe        = exp_cfg.get("use_clahe", False)
    image_size       = data_cfg.get("image_size", 224)
    contrast_threshold = base_cfg["preprocessing"].get("contrast_threshold", 35)

    preprocessing = build_preprocessing_pipeline(
        use_clahe=use_clahe,
        clahe_clip_limit=base_cfg["preprocessing"].get("clahe_clip_limit", 2.0),
        clahe_tile_size=tuple(base_cfg["preprocessing"].get("clahe_tile_size", [8, 8])),
        contrast_threshold=contrast_threshold,   # conditional CLAHE
        remove_headers=base_cfg["preprocessing"].get("remove_headers", True),
    )

    train_tf = build_transforms(image_size, split="train")
    val_tf   = build_transforms(image_size, split="val")

    train_data_key = exp_cfg.get("train_data", "fetal")   # "fetal", "african", "combined"
    test_data_key  = exp_cfg.get("test_data",  "african_test")

    # Dataset factories

    def make_fetal_ds(split, transform):
        """Fetal DB - pre-split folders: train / val / test."""
        return FetalDataset(
            root_dir=str(Path(data_cfg["fetal_db_dir"]) / split),
            transform=transform,
            preprocessing=preprocessing,
            classes=data_cfg["classes"],
        )

    def make_african_ds(split, transform, max_patients=None):
        """African dataset - pre-split folders: train / val / test."""
        return AfricanDataset(
            root_dir=str(Path(data_cfg["african_dir"]) / split),
            max_patients=max_patients,
            transform=transform,
            preprocessing=preprocessing,
            classes=data_cfg["classes"],
        )

    # Train / val datasets

    if train_data_key == "fetal":
        train_ds = make_fetal_ds("train", train_tf)
        val_ds   = make_fetal_ds("val",   val_tf)

    elif train_data_key == "african":
        train_ds = make_african_ds("train", train_tf,
                                   max_patients=getattr(args, "patients", None))
        val_ds   = make_african_ds("val", val_tf)

    elif train_data_key == "combined":
        # Both datasets pooled for training - val from fetal only
        fetal_ds   = make_fetal_ds("train", train_tf)
        african_ds = make_african_ds("train", train_tf,
                                     max_patients=getattr(args, "patients", None))
        train_ds = CombinedDataset([fetal_ds, african_ds])
        val_ds   = make_fetal_ds("val", val_tf)

    else:
        raise ValueError(f"Unknown train_data key: {train_data_key}")

    # Test datasets
    # african_test -> held-out Malawi split
    # fetal_test   -> held-out Fetal DB split
    # both         -> evaluate on both separately

    if test_data_key == "african_test":
        test_ds      = make_african_ds("test", val_tf)
        test_ds_dict = {"african_test": test_ds}

    elif test_data_key == "fetal_test":
        test_ds      = make_fetal_ds("test", val_tf)
        test_ds_dict = {"fetal_test": test_ds}

    elif test_data_key == "both":
        african_test_ds = make_african_ds("test", val_tf)
        fetal_test_ds   = make_fetal_ds("test",   val_tf)
        test_ds         = None   # evaluated separately via test_ds_dict
        test_ds_dict    = {
            "african_test": african_test_ds,
            "fetal_test":   fetal_test_ds,
        }

    else:
        raise ValueError(f"Unknown test_data key: {test_data_key}")

    return train_ds, val_ds, test_ds, test_ds_dict


def main():
    parser = argparse.ArgumentParser(description="Fetal Ultrasound - Training")
    parser.add_argument("--experiment", type=str, required=True,  help="Experiment name from experiments.yaml")
    parser.add_argument("--model",      type=str, default=None,   help="Override model architecture")
    parser.add_argument("--patients",   type=int, default=None,   help="Max patients for African fine-tuning")
    parser.add_argument("--tracker",    type=str, default="wandb", choices=["wandb", "mlflow", "both", "none"])
    parser.add_argument("--config",     type=str, default="configs/base_config.yaml")
    parser.add_argument("--exp_config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for transfer learning")
    args = parser.parse_args()

    # Load configs
    base_cfg = load_config(args.config)
    exp_cfgs = load_config(args.exp_config)["experiments"]

    if args.experiment not in exp_cfgs:
        logger.error(f"Unknown experiment: {args.experiment}. Available: {list(exp_cfgs.keys())}")
        sys.exit(1)

    exp_cfg = exp_cfgs[args.experiment]
    logger.info(f"Running experiment  : {args.experiment}")
    logger.info(f"Description         : {exp_cfg.get('description', '')}")

    set_seed(args.seed)

    # Build datasets
    train_ds, val_ds, test_ds, test_ds_dict = build_datasets_for_experiment(
        exp_cfg, base_cfg, args
    )
    logger.info(f"Dataset sizes - train: {len(train_ds)}  val: {len(val_ds)}")

    # Build model
    arch             = args.model or exp_cfg.get("model", base_cfg["model"]["architecture"])
    fine_tune_layers = exp_cfg.get("fine_tune_layers", base_cfg["model"].get("fine_tune_layers", -1))
    num_classes      = base_cfg["data"]["num_classes"]   # set by build_datasets_for_experiment

    model = build_model(
        architecture=arch,
        num_classes=num_classes,
        pretrained=base_cfg["model"].get("pretrained", True),
        dropout=base_cfg["model"].get("dropout", 0.3),
        fine_tune_layers=fine_tune_layers,
    )

    # Load pretrained checkpoint (transfer learning)
    if args.pretrained_checkpoint and Path(args.pretrained_checkpoint).exists():
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded pretrained weights from {args.pretrained_checkpoint}")

        tl_cfg   = base_cfg.get("transfer_learning", {})
        tl_layers = exp_cfg.get("fine_tune_layers", tl_cfg.get("fine_tune_layers", 4))
        from src.models.model_zoo import _partial_unfreeze
        _partial_unfreeze(model, tl_layers)
        logger.info(f"Fine-tuning last {tl_layers} layers")

    from src.models import count_parameters
    param_info = count_parameters(model)
    logger.info(f"Parameters - total: {param_info['total']:,}  trainable: {param_info['trainable']:,}")

    # Build dataloaders
    loaders = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=base_cfg["training"]["batch_size"],
        num_workers=base_cfg["training"]["num_workers"],
    )

    # Training config
    train_cfg = {
        **base_cfg["training"],
        **base_cfg.get("logging", {}),
        "project_name": base_cfg["project_name"],
        "architecture": arch,
        "class_names": base_cfg["data"]["classes"],   # stored in every checkpoint
    }

    if args.pretrained_checkpoint:
        tl_cfg = base_cfg.get("transfer_learning", {})
        train_cfg["learning_rate"] = tl_cfg.get("fine_tune_lr", train_cfg["learning_rate"] / 2)
        train_cfg["num_epochs"]    = tl_cfg.get("fine_tune_epochs", 20)

    # Experiment name for tracking
    exp_name = args.experiment
    if args.model:
        exp_name += f"_{args.model}"
    if args.patients:
        exp_name += f"_p{args.patients}"

    # Train
    trainer = Trainer(
        model=model,
        loaders=loaders,
        config=train_cfg,
        experiment_name=exp_name,
        tracker=args.tracker,
    )

    results = trainer.train()
    logger.info(f"Training complete. Best val PR-AUC: {results['best_val_pr_auc']:.4f}")
    if results.get("test"):
        logger.info(f"Test results: {results['test']}")


if __name__ == "__main__":
    main()