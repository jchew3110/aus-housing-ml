"""
Shared FastAPI dependencies: model loading and request-level access.
"""

import logging

from fastapi import HTTPException, Request

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
        except FileNotFoundError as exc:
            logger.error("Could not load model: %s", exc)
            self.model = None
            self.metadata = {}


def get_model_state(request: Request) -> ModelState:
    """FastAPI dependency that retrieves the loaded model state from app state."""
    state: ModelState = request.app.state.model_state
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return state
