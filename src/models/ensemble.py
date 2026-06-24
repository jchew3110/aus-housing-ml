"""
Weighted ensemble of multiple housing models.

Weights are determined during fit() using inverse validation MAE —
better-performing models get a larger share of the blend.
The ensemble only exposes predict() and predict_with_interval(); it is
not retrained with Optuna (the constituent models already are).
"""

import logging

import numpy as np
import pandas as pd

from src.models.base import BaseHousingModel
from src.models.evaluator import evaluate

logger = logging.getLogger(__name__)


class EnsembleHousingModel(BaseHousingModel):
    name = "ensemble"

    def __init__(self, models: list[BaseHousingModel]) -> None:
        self.models = models
        self.weights: list[float] = []
        self._feature_cols: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        self._feature_cols = list(X_train.columns)

        # Compute validation MAE for each constituent model
        val_maes: list[float] = []
        for m in self.models:
            preds = m.predict(X_val)
            metrics = evaluate(y_val, preds)
            val_maes.append(metrics.mae)
            logger.info(
                "Ensemble constituent %s — val MAE=%.4f", m.name, metrics.mae
            )

        # Inverse-MAE weighting
        inv = [1.0 / mae for mae in val_maes]
        total = sum(inv)
        self.weights = [w / total for w in inv]
        logger.info(
            "Ensemble weights: %s",
            {m.name: f"{w:.3f}" for m, w in zip(self.models, self.weights)},
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        blended = sum(
            m.predict(X[self._feature_cols]) * w
            for m, w in zip(self.models, self.weights)
        )
        return np.asarray(blended)

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_points, all_lowers, all_uppers = [], [], []
        for m, w in zip(self.models, self.weights):
            p, lo, hi = m.predict_with_interval(X[self._feature_cols], confidence)
            all_points.append(p * w)
            all_lowers.append(lo * w)
            all_uppers.append(hi * w)
        return (
            np.asarray(sum(all_points)),
            np.asarray(sum(all_lowers)),
            np.asarray(sum(all_uppers)),
        )

    def get_feature_importance(self) -> pd.Series:
        importance = sum(
            m.get_feature_importance() * w
            for m, w in zip(self.models, self.weights)
        )
        return importance.sort_values(ascending=False)

    def compute_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        blended: pd.DataFrame | None = None
        for m, w in zip(self.models, self.weights):
            shap_df = m.compute_shap(X) * w
            blended = shap_df if blended is None else blended + shap_df
        return blended
