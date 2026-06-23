"""
Abstract base class for all housing price models.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.stats as stats


class BaseHousingModel(ABC):
    name: str
    version: str = "1.0"

    # Set by fit(); used to compute default prediction intervals
    _val_residual_std: float = 0.0

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (point_pred, lower_bound, upper_bound).

        Default implementation uses a Gaussian interval based on residual std
        from the validation set. Subclasses can override with quantile regression.
        """
        preds = self.predict(X)
        if self._val_residual_std > 0:
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * self._val_residual_std
            return preds, preds - margin, preds + margin
        return preds, preds, preds

    @abstractmethod
    def get_feature_importance(self) -> pd.Series: ...
