"""
Model Manager
-------------
Handles versioned model loading, in-memory caching, and lifecycle management.

Model registry layout:
    model_registry/
        efficientnet_b0/
            v1/
                model.pt          # TorchScript or state_dict
                config.json       # metadata (architecture, classes, etc.)
            v2/
                model.pt
                config.json

"latest" resolves to the highest version folder name in reverse string order,
so use zero-padded or single-digit names (v1, v2, ...) to keep it predictable.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

import torch

from app.core.config import settings
from app.models.loader import load_model_from_config

logger = logging.getLogger(__name__)


class LoadedModel:
    def __init__(self, name: str, version: str, model: torch.nn.Module, config: dict):
        self.name = name
        self.version = version
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def classes(self) -> list:
        return self.config.get("classes", [])

    @property
    def num_classes(self) -> int:
        return len(self.classes)


class ModelManager:
    """Thread-safe model registry with LRU-style management."""

    def __init__(self):
        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Registry discovery

    def list_available(self) -> list[dict]:
        """Return all model versions present in the registry directory."""
        registry = settings.registry_path
        result = []
        if not registry.exists():
            return result
        for model_dir in sorted(registry.iterdir()):
            if not model_dir.is_dir():
                continue
            for version_dir in sorted(model_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                cfg_path = version_dir / "config.json"
                config = {}
                if cfg_path.exists():
                    config = json.loads(cfg_path.read_text())
                result.append({
                    "name": model_dir.name,
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "loaded": f"{model_dir.name}:{version_dir.name}" in self._models,
                    "config": config,
                })
        return result

    def list_loaded(self) -> list[dict]:
        with self._lock:
            return [
                {"name": m.name, "version": m.version, "device": str(m.device)}
                for m in self._models.values()
            ]

    # Load / unload

    def load_model(self, name: str, version: str = "latest") -> LoadedModel:
        version = self._resolve_version(name, version)
        key = f"{name}:{version}"

        with self._lock:
            if key in self._models:
                logger.info(f"Model {key} already loaded.")
                return self._models[key]

        model_dir = settings.registry_path / name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        cfg_path = model_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json missing in {model_dir}")

        config = json.loads(cfg_path.read_text())
        model_path = model_dir / "model.pt"

        logger.info(f"Loading model {key} from {model_path} on {self._device}...")
        model = load_model_from_config(model_path, config, self._device)

        loaded = LoadedModel(name=name, version=version, model=model, config=config)
        with self._lock:
            self._models[key] = loaded

        logger.info(f"Model {key} loaded successfully.")
        return loaded

    def unload_model(self, name: str, version: str = "latest"):
        version = self._resolve_version(name, version)
        key = f"{name}:{version}"
        with self._lock:
            if key in self._models:
                del self._models[key]
                logger.info(f"Model {key} unloaded.")

    def unload_all(self):
        with self._lock:
            self._models.clear()
        logger.info("All models unloaded.")

    def get_model(self, name: str, version: str = "latest") -> LoadedModel:
        version = self._resolve_version(name, version)
        key = f"{name}:{version}"
        with self._lock:
            if key not in self._models:
                raise KeyError(f"Model {key} is not loaded. Call /models/load first.")
            return self._models[key]

    def get_or_load(self, name: str, version: str = "latest") -> LoadedModel:
        """Return a loaded model, loading it from the registry on first use.

        Raises FileNotFoundError if the model or version is not in the registry.
        """
        try:
            return self.get_model(name, version)
        except KeyError:
            return self.load_model(name, version)

    # Helpers

    def _resolve_version(self, name: str, version: str) -> str:
        if version != "latest":
            return version
        model_dir = settings.registry_path / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{name}' not in registry.")
        versions = sorted(
            [d.name for d in model_dir.iterdir() if d.is_dir()], reverse=True
        )
        if not versions:
            raise FileNotFoundError(f"No versions found for model '{name}'.")
        return versions[0]


# Singleton
model_manager = ModelManager()
