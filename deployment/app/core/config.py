"""
Application configuration - loaded from environment or .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model registry path (directory containing versioned model folders)
    MODEL_REGISTRY_DIR: str = "model_registry"

    # Model to load on startup (leave empty to load nothing automatically)
    DEFAULT_MODEL: str = ""

    # Inference defaults
    TOP_K: int = 5                  # Number of top predictions to return
    CONFIDENCE_THRESHOLD: float = 0.0  # Minimum confidence to include in results

    # Preprocessing
    IMAGE_SIZE: int = 224           # Resize shorter edge to this size
    NORMALIZE_MEAN: list = [0.485, 0.456, 0.406]
    NORMALIZE_STD: list = [0.229, 0.224, 0.225]

    # Batch inference
    MAX_BATCH_SIZE: int = 64
    BATCH_WORKERS: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def registry_path(self) -> Path:
        return Path(self.MODEL_REGISTRY_DIR)


settings = Settings()
