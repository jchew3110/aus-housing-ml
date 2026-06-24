"""
Shared FastAPI dependencies: model loading and request-level access.
"""

import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request

from src.features.pipeline import FEATURE_COLS
from src.models.base import BaseHousingModel
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelState:
    """Holds the loaded model and its metadata for the duration of the process."""

    def __init__(self) -> None:
        self.model: BaseHousingModel | None = None
        self.metadata: dict = {}
        self.registry = ModelRegistry()

    def load_model(self, model_name: str | None = None) -> None:
        """Load the best model by test RMSE, or a specific named model."""
        try:
            if model_name:
                self.model, self.metadata = self.registry.load(model_name)
            else:
                self.model, self.metadata = self.registry.load_best()
            logger.info(
                "Loaded model: %s v%s (trained %s)",
                self.metadata.get("name"),
                self.metadata.get("version"),
                self.metadata.get("training_date"),
            )
            self._validate_model()
        except FileNotFoundError as exc:
            logger.error("Could not load model: %s", exc)
            self.model = None
            self.metadata = {}

    def _validate_model(self) -> None:
        """Smoke-test the loaded model and warn on feature mismatch."""
        if self.model is None:
            return

        # Warn if the model was trained with a different feature set
        model_cols = getattr(self.model, "_feature_cols", [])
        if model_cols and list(model_cols) != list(FEATURE_COLS):
            logger.warning(
                "Loaded model feature columns differ from current FEATURE_COLS. "
                "Model has %d features; pipeline expects %d. Re-train to sync.",
                len(model_cols),
                len(FEATURE_COLS),
            )

        # Dry-run predict on a zero-filled row to catch incompatible pkl artifacts
        try:
            dummy = pd.DataFrame(
                [np.zeros(len(FEATURE_COLS))], columns=FEATURE_COLS
            )
            self.model.predict(dummy)
            logger.info("Model startup validation passed.")
        except Exception as exc:
            logger.error(
                "Model startup validation FAILED — predictions will not work: %s", exc
            )
            self.model = None
            self.metadata = {}


def get_model_state(request: Request) -> ModelState:
    """FastAPI dependency that retrieves the loaded model state from app state."""
    state: ModelState = request.app.state.model_state
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return state
