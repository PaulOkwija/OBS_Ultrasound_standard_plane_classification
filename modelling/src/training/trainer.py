"""
Trainer class - encapsulates the full training loop with:
  - W&B and/or MLflow experiment tracking
  - Mixed precision (AMP)
  - Early stopping
  - LR scheduling
  - Checkpoint saving (best val loss + best val PR-AUC)
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = (
            (value < self.best - self.delta) if self.mode == "min"
            else (value > self.best + self.delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training loop for fetal US plane classification.

    Args:
        model:         nn.Module
        loaders:       dict with keys "train", "val", optional "test"
        config:        dict of hyperparameters (from YAML config)
        experiment_name: used for W&B run name and checkpoint directory
        tracker:       "wandb" | "mlflow" | "both" | "none"
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        config: dict,
        experiment_name: str = "experiment",
        tracker: str = "wandb",
    ):
        self.model = model
        self.loaders = loaders
        self.config = config
        import random
        # Append a random 4-digit number to ensure uniqueness
        suffix = random.randint(1000, 9999)
        self.experiment_name = f"{experiment_name}_{suffix}"
        self.tracker = tracker

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimiser
        self.optimizer = self._build_optimizer()

        # Loss
        self.criterion = self._build_criterion()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # AMP scaler
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            mode="min",
        )

        # Checkpoint dir
        ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints")) / self.experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        # Init trackers
        self._init_trackers()

    # Build helpers

    def _build_optimizer(self) -> optim.Optimizer:
        lr = self.config.get("learning_rate", 1e-4)
        wd = self.config.get("weight_decay", 0.01)
        opt_name = self.config.get("optimizer", "adam").lower()
        if opt_name == "adam":
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, weight_decay=wd,
            )
        elif opt_name == "sgd":
            return optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, momentum=0.9, weight_decay=wd,
            )
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_criterion(self) -> nn.Module:
        loss_name = self.config.get("loss", "cross_entropy")
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "focal":
            from .losses import FocalLoss
            return FocalLoss(gamma=2.0)
        elif loss_name == "weighted_ce":
            # weights are set per-dataset in train()
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unknown loss: {loss_name}")

    def _build_scheduler(self):
        sched_name = self.config.get("scheduler", "cosine")
        epochs = self.config.get("num_epochs", 50)
        if sched_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-7
            )
        elif sched_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif sched_name == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        return None

    def _init_trackers(self):
        if self.tracker in ("wandb", "both"):
            try:
                import wandb
                wandb.init(
                    project=self.config.get("project_name", "fetal-african-ai"),
                    name=self.experiment_name,
                    config=self.config,
                    reinit=True,
                )
                self._wandb = wandb
                logger.info("W&B tracker initialised")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")
                self._wandb = None
        else:
            self._wandb = None

        if self.tracker in ("mlflow", "both"):
            try:
                import mlflow
                mlflow.set_experiment(self.config.get("project_name", "fetal-african-ai"))
                mlflow.start_run(run_name=self.experiment_name)
                mlflow.log_params(self.config)
                self._mlflow = mlflow
                logger.info("MLflow tracker initialised")
            except Exception as e:
                logger.warning(f"MLflow init failed: {e}")
                self._mlflow = None
        else:
            self._mlflow = None

    def _log(self, metrics: dict, step: int):
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    # Training loop

    def train(self) -> dict:
        """
        Run full training and return best metrics.
        """
        num_epochs = self.config.get("num_epochs", 50)
        best_val_loss = float("inf")
        best_val_pr_auc = 0.0
        history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_f1": [], "val_auc": [], "val_acc": []}

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(epoch, split="train")
            val_metrics   = self._run_epoch(epoch, split="val")

            # LR scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            log_dict = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            self._log(log_dict, step=epoch)

            # History
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_pr_auc"].append(val_metrics.get("pr_auc", 0))
            history["val_f1"].append(val_metrics.get("f1", 0))
            history["val_auc"].append(val_metrics.get("auc", 0))
            history["val_acc"].append(val_metrics.get("accuracy", 0))

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1={val_metrics.get('f1', 0):.4f} | "
                f"val_pr_auc={val_metrics.get('pr_auc', 0):.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpointing
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best_loss.pt", val_metrics)

            if val_metrics.get("pr_auc", 0) > best_val_pr_auc:
                best_val_pr_auc = val_metrics.get("pr_auc", 0)
                self._save_checkpoint("best_pr_auc.pt", val_metrics)

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save training history
        history_path = self.ckpt_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Test evaluation if loader provided
        test_metrics = {}
        if "test" in self.loaders:
            logger.info("Running test evaluation with best_pr_auc checkpoint...")
            self._load_checkpoint("best_pr_auc.pt")
            test_metrics = self._run_epoch(0, split="test")
            logger.info(f"Test metrics: {test_metrics}")
            self._log({f"test/{k}": v for k, v in test_metrics.items()}, step=num_epochs)

        # Finish trackers
        if self._wandb:
            self._wandb.finish()
        if self._mlflow:
            self._mlflow.end_run()

        return {"history": history, "best_val_loss": best_val_loss,
                "best_val_pr_auc": best_val_pr_auc, "test": test_metrics}

    def _run_epoch(self, epoch: int, split: str) -> dict:
        loader = self.loaders[split]
        is_train = (split == "train")
        self.model.train(is_train)

        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.set_grad_enabled(is_train):
            pbar = tqdm(loader, desc=f"[{split}] epoch {epoch}", leave=False)
            for batch_idx, (imgs, labels) in enumerate(pbar):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self.model(imgs)
                    loss = self.criterion(logits, labels)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(logits.detach(), dim=1)
                preds = probs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )
        metrics["loss"] = avg_loss
        return metrics

    def _save_checkpoint(self, filename: str, metrics: dict):
        path = self.ckpt_dir / filename
        # Persist class_names so downstream evaluation never has to guess the
        # label-to-index mapping that was active at training time.
        from src.data import CLASSES
        class_names = self.config.get("class_names", CLASSES)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "class_names": class_names,
        }, path)

    def _load_checkpoint(self, filename: str):
        path = self.ckpt_dir / filename
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint: {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
