"""
Leave-One-Country-Out Cross-Validation (LOCO-CV) module.

Implements the cross-validation logic for the African dataset, using the
`Center` column as the group variable.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut

from src.data import AfricanDatasetFromCSV, build_dataloaders
from src.models import build_model
from src.models.model_zoo import _partial_unfreeze
from src.training import Trainer

from src.utils import get_logger
logger = get_logger(__name__)

# Remap CSV plane names -> our class names
# CSV has "Fetal brain", "Fetal femur", "Fetal abdomen", "Fetal thorax"
# This is a different (exact-match) remapper from src/data/curate.py's
# remap_class() (last-word + catch-all-to-"other"). They agree on this dataset's
# actual strings today but are not interchangeable in general - don't unify them.
PLANE_REMAP = {
    "fetal brain"   : "brain",
    "fetal femur"   : "femur",
    "fetal abdomen" : "abdomen",
    "fetal thorax"  : "other",
}


def _prepare_df(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the African CSV:
      - Strip whitespace from Center
      - Remap 'Fetal brain' -> 'brain' etc. into a new 'label' column
      - Add .png extension to Filename -> 'Filename_png'
    """
    df = pd.read_csv(csv_path)
    df["Center"]      = df["Center"].astype(str).str.strip()
    df["label"]       = df["Plane"].str.lower().str.strip().map(PLANE_REMAP)
    df["Filename_png"] = df["Filename"].str.strip() + ".png"
    return df


def run_loco_cv(
    csv_path: str,
    img_dir: str,
    architecture: str,
    pretrained_checkpoint: str,
    train_tf,
    val_tf,
    preprocessing,
    train_cfg: dict,
    experiment_name: str,
    classes: list,
    fine_tune_layers: int = 4,
    device: str = "cuda",
    tracker: str = "wandb"
) -> dict:
    """
    Run Leave-One-Country-Out CV on the African dataset.
    Malawi is excluded from the CV loop and used as the final test set.

    Args:
        csv_path             : path to African_planes_database.csv
        img_dir              : root folder for african images (e.g. data/african/)
        architecture         : timm model architecture key
        pretrained_checkpoint: path to pre-trained .pt weights
        train_tf             : training transform pipeline
        val_tf               : validation/test transform pipeline
        preprocessing        : callable applied before transforms
        train_cfg            : Trainer config dictionary
        experiment_name      : prefix for checkpoints and W&B tracking
        classes              : list of class names (e.g. ["brain", "femur", "abdomen"])
        fine_tune_layers     : number of backbone layers to unfreeze
        device               : 'cuda' or 'cpu'
        tracker              : 'wandb' | 'none'

    Returns:
        dict with fold_results, mean/std CV metrics, and Malawi test metrics
    """
    df = _prepare_df(csv_path)

    # Filter to requested classes only
    df = df[df["label"].isin(classes)].copy()
    logger.info(f"After class filter: {len(df)} images  |  classes: {classes}")

    # Separate held-out test set (Malawi)
    test_country = "Malawi"
    df_test = df[df["Center"] == test_country].copy()
    df_cv   = df[df["Center"] != test_country].copy()

    logger.info(f"LOCO-CV pool : {len(df_cv)} images | "
                f"{df_cv['Center'].nunique()} countries")
    logger.info(f"Malawi test  : {len(df_test)} images")

    if len(df_test) == 0:
        raise ValueError("No Malawi images found - check Center column values in CSV.")
    if len(df_cv) == 0:
        raise ValueError("No non-Malawi images found for CV.")

    # Build Malawi test loader
    test_ds = AfricanDatasetFromCSV(
        df=df_test,
        img_dir=img_dir,
        label_col="label",           # use our remapped column
        filename_col="Filename_png", # use the .png extension column
        transform=val_tf,
        preprocessing=preprocessing,
        classes=classes,
    )
    test_loader = build_dataloaders(
        test_ds, test_ds,
        batch_size=train_cfg.get("batch_size", 32)
    )["val"]

    # LOCO-CV loop
    logo   = LeaveOneGroupOut()
    groups = df_cv["Center"].values

    fold_results          = {}
    best_models_per_fold  = {}

    for fold, (train_idx, val_idx) in enumerate(logo.split(df_cv, groups=groups)):
        df_train         = df_cv.iloc[train_idx]
        df_val           = df_cv.iloc[val_idx]
        held_out_country = df_val["Center"].iloc[0]

        logger.info(f"===== Fold {fold} | Val country: {held_out_country} =====")
        logger.info(f"  Train: {len(df_train)} | Val: {len(df_val)}")

        # Build datasets using remapped label and png filename columns
        train_ds = AfricanDatasetFromCSV(
            df=df_train, img_dir=img_dir,
            label_col="label", filename_col="Filename_png",
            transform=train_tf, preprocessing=preprocessing,
            classes=classes,
        )
        val_ds = AfricanDatasetFromCSV(
            df=df_val, img_dir=img_dir,
            label_col="label", filename_col="Filename_png",
            transform=val_tf, preprocessing=preprocessing,
            classes=classes,
        )
        loaders = build_dataloaders(
            train_ds, val_ds,
            batch_size=train_cfg.get("batch_size", 32)
        )

        # Build model and load pretrained weights
        model = build_model(
            architecture=architecture,
            num_classes=len(classes),
            pretrained=False,
        )

        ckpt       = torch.load(pretrained_checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys  : {missing}")
        logger.info(f"Unexpected keys: {unexpected}")

        _partial_unfreeze(model, fine_tune_layers)

        # Train this fold
        fold_exp_name = f"{experiment_name}_fold{fold}_{held_out_country.lower()}"
        trainer = Trainer(
            model=model,
            loaders=loaders,
            config=train_cfg,
            experiment_name=fold_exp_name,
            tracker=tracker,
        )
        res = trainer.train()

        fold_results[held_out_country] = {
            "val_loss"   : res["best_val_loss"],
            "val_pr_auc" : res["best_val_pr_auc"],
            "val_auc"    : max(res["history"].get("val_auc", [0])),
            "val_f1"     : max(res["history"].get("val_f1",  [0])),
            "val_acc"    : max(res["history"].get("val_acc", [0])),
        }

        best_models_per_fold[held_out_country] = trainer.ckpt_dir / "best_pr_auc.pt"
        logger.info(f"Fold {fold} done. Best PR-AUC: "
                    f"{fold_results[held_out_country]['val_pr_auc']:.4f}")

    # Evaluate each fold's best model on Malawi
    logger.info("===== Evaluating on Malawi (held-out test) =====")
    test_metrics_per_fold = {}

    for country, ckpt_path in best_models_per_fold.items():
        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint missing for {country}: {ckpt_path}")
            continue

        model = build_model(architecture=architecture,
                            num_classes=len(classes), pretrained=False)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Minimal loaders dict so Trainer initialises without error
        dummy_loaders = {"train": test_loader, "val": test_loader, "test": test_loader}
        eval_trainer  = Trainer(model=model, loaders=dummy_loaders,
                                config=train_cfg, tracker="none")
        metrics = eval_trainer._run_epoch(epoch=0, split="test")
        test_metrics_per_fold[country] = metrics
        logger.info(f"  [{country} model] Malawi PR-AUC: "
                    f"{metrics.get('pr_auc', 0):.4f}  "
                    f"F1: {metrics.get('f1', 0):.4f}  "
                    f"Acc: {metrics.get('accuracy', 0):.4f}")

    # Aggregate
    mean_cv  = {k: float(np.mean([r[k] for r in fold_results.values()]))
                for k in next(iter(fold_results.values())).keys()}
    std_cv   = {k: float(np.std([r[k] for r in fold_results.values()]))
                for k in next(iter(fold_results.values())).keys()}

    mean_test: dict = {}
    if test_metrics_per_fold:
        numeric_keys = [k for k, v in next(iter(test_metrics_per_fold.values())).items()
                        if isinstance(v, (int, float))]
        mean_test = {
            k: float(np.mean([r.get(k, 0) for r in test_metrics_per_fold.values()]))
            for k in numeric_keys
        }

    return {
        "fold_results"             : fold_results,
        "mean_cv_metrics"          : mean_cv,
        "std_cv_metrics"           : std_cv,
        "test_metrics_per_fold_model": {k: {kk: vv for kk, vv in v.items()
                                            if isinstance(vv, (int, float))}
                                        for k, v in test_metrics_per_fold.items()},
        "mean_test_metrics"        : mean_test,
    }
