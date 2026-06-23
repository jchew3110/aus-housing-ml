"""
Model artifact registry: save and load trained models with metadata.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.config import MODELS_DIR
from src.models.base import BaseHousingModel
from src.models.evaluator import ModelMetrics

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str, version: str) -> Path:
        return self.models_dir / f"{model_name}_v{version}"

    def save(
        self,
        model: BaseHousingModel,
        metrics: dict[str, ModelMetrics],
        feature_cols: list[str],
    ) -> Path:
        """Save model.pkl + metadata.json to models/{name}_v{version}/."""
        out_dir = self._model_dir(model.name, model.version)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "name": model.name,
            "version": model.version,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_cols": feature_cols,
            "metrics": {split: m.to_dict() for split, m in metrics.items()},
        }
        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved %s to %s", model.name, out_dir)
        return out_dir

    def load(self, model_name: str, version: str = "1.0") -> tuple[BaseHousingModel, dict]:
        """Load model and metadata by name and version."""
        model_dir = self._model_dir(model_name, version)
        model_path = model_dir / "model.pkl"
        meta_path = model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(meta_path) as f:
            metadata = json.load(f)

        return model, metadata

    def list_models(self) -> list[dict]:
        """Return a list of saved model metadata dicts."""
        results = []
        for meta_path in self.models_dir.glob("*/metadata.json"):
            with open(meta_path) as f:
                meta = json.load(f)
            test_metrics = meta.get("metrics", {}).get("test", {})
            results.append(
                {
                    "name": meta["name"],
                    "version": meta["version"],
                    "training_date": meta["training_date"],
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "test_r2": test_metrics.get("r2"),
                }
            )
        return sorted(results, key=lambda x: x.get("test_rmse") or float("inf"))

    def load_best(self, metric: str = "test_rmse") -> tuple[BaseHousingModel, dict]:
        """Load the model with the best (lowest) value for the given metric."""
        models = self.list_models()
        if not models:
            raise FileNotFoundError("No trained models found in registry.")
        best = min(models, key=lambda x: x.get(metric) or float("inf"))
        logger.info("Loading best model: %s (v%s, %s=%.4f)", best["name"], best["version"], metric, best[metric])
        return self.load(best["name"], best["version"])
