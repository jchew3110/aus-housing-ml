"""
LightGBM model with Optuna hyperparameter tuning and early stopping on val set.
"""

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap

from src.models.base import BaseHousingModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMHousingModel(BaseHousingModel):
    name = "lgbm"

    def __init__(self, n_trials: int = 50, timeout_seconds: int = 300) -> None:
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.best_params: dict = {}
        self.model: lgb.LGBMRegressor | None = None
        self._lower_model: lgb.LGBMRegressor | None = None
        self._upper_model: lgb.LGBMRegressor | None = None
        self._feature_cols: list[str] = []
        self._explainer: shap.TreeExplainer | None = None

    def _make_estimator(self, params: dict, objective: str = "regression") -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective=objective,
            random_state=42,
            verbose=-1,
            **params,
        )

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = self._make_estimator(params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val.values - preds) ** 2)))

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self._feature_cols = list(X_train.columns)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._optuna_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False,
        )
        self.best_params = study.best_params
        logger.info("LGBM best params: %s (val RMSE=%.4f)", self.best_params, study.best_value)

        self.model = self._make_estimator(self.best_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_preds = self.model.predict(X_val)
        self._val_residual_std = float(np.std(y_val.values - val_preds))

        # Quantile models for confidence intervals
        q_params = {k: v for k, v in self.best_params.items()}
        self._lower_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.05, random_state=42, verbose=-1, **q_params
        )
        self._upper_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.95, random_state=42, verbose=-1, **q_params
        )
        self._lower_model.fit(X_train, y_train)
        self._upper_model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._feature_cols])

    def predict_with_interval(
        self, X: pd.DataFrame, confidence: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = self.predict(X)
        X_feat = X[self._feature_cols]
        lower = self._lower_model.predict(X_feat)
        upper = self._upper_model.predict(X_feat)
        return preds, lower, upper

    def get_feature_importance(self) -> pd.Series:
        scores = self.model.feature_importances_
        return pd.Series(scores, index=self._feature_cols).sort_values(ascending=False)

    def compute_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        X_feat = X[self._feature_cols]
        values = self._explainer.shap_values(X_feat)
        return pd.DataFrame(values, columns=self._feature_cols, index=X_feat.index)
