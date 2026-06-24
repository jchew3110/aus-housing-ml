"""
Ridge regression baseline.

Includes StandardScaler because Ridge is scale-sensitive.
The scaler is fit only on the training set and bundled into the model artifact.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseHousingModel


class RidgeHousingModel(BaseHousingModel):
    name = "ridge"

    ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=self.ALPHAS, cv=5)
        self._feature_cols: list[str] = []
        self._X_train_sample: np.ndarray | None = None
        self._explainer: shap.LinearExplainer | None = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self._feature_cols = list(X_train.columns)
        X_tr_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_tr_scaled, y_train)

        # Store up to 100 scaled training rows as background for LinearExplainer
        n = min(100, len(X_tr_scaled))
        self._X_train_sample = X_tr_scaled[:n]
        self._explainer = None  # invalidate on refit

        X_val_scaled = self.scaler.transform(X_val)
        val_preds = self.model.predict(X_val_scaled)
        self._val_residual_std = float(np.std(y_val.values - val_preds))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self._feature_cols])
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        coefs = np.abs(self.model.coef_)
        return pd.Series(coefs, index=self._feature_cols).sort_values(ascending=False)

    def compute_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._explainer is None:
            self._explainer = shap.LinearExplainer(
                self.model, self._X_train_sample, feature_perturbation="correlation_dependent"
            )
        X_scaled = self.scaler.transform(X[self._feature_cols])
        values = self._explainer.shap_values(X_scaled)
        return pd.DataFrame(values, columns=self._feature_cols, index=X.index)
