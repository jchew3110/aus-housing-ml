"""
Model evaluation: computes MAE, RMSE, R², MAPE, and directional accuracy.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "n_samples": float(self.n_samples),
        }


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    split_name: str = "test",
) -> ModelMetrics:
    """Compute all evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    residuals = y_true - y_pred

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE: exclude samples where y_true == 0 to avoid division by zero
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    # Directional accuracy: fraction where sign(pred) == sign(true)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return ModelMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        directional_accuracy=dir_acc,
        n_samples=n,
    )


def compare_models(
    results: dict[str, dict[str, ModelMetrics]],
) -> pd.DataFrame:
    """
    Build a comparison table of test-split metrics across models.

    results: {model_name: {"train": ModelMetrics, "val": ModelMetrics, "test": ModelMetrics}}
    Returns a DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, splits in results.items():
        for split_name, metrics in splits.items():
            row = {"model": model_name, "split": split_name, **metrics.to_dict()}
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.pivot_table(index="model", columns="split", values=["mae", "rmse", "r2", "directional_accuracy"])


def calibration_coverage(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage of prediction intervals.

    Returns the fraction of y_true values that fall within [lower, upper].
    For a well-calibrated 90% interval this should be close to 0.90.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def walk_forward_cv(
    feature_df: pd.DataFrame,
    model_cls: type,
    n_splits: int = 5,
    min_train_quarters: int = 24,
    val_quarters: int = 8,
    model_kwargs: dict | None = None,
) -> list[ModelMetrics]:
    """
    Walk-forward (expanding window) cross-validation for time-series data.

    Each fold uses an expanding training window and a fixed-size validation
    window. No future information leaks into any fold's training set.

    Returns list of ModelMetrics, one per fold (use to compute mean/std of metrics).
    """
    from src.data.config import SplitConfig
    from src.features.pipeline import FeaturePipeline

    if model_kwargs is None:
        model_kwargs = {}

    pipeline = FeaturePipeline()
    periods = pd.PeriodIndex(feature_df["period"], freq="Q-DEC").unique()
    periods = periods.sort_values()

    total_quarters = len(periods)
    # First validation window starts after min_train_quarters
    first_val_start_idx = min_train_quarters
    # Compute fold starting points
    available = total_quarters - first_val_start_idx - val_quarters
    if available < 1:
        raise ValueError(
            f"Not enough data for walk-forward CV: need at least "
            f"{min_train_quarters + val_quarters} unique quarters, got {total_quarters}"
        )

    step = max(1, available // n_splits)
    fold_offsets = list(range(0, available, step))[:n_splits]

    fold_metrics: list[ModelMetrics] = []
    for fold_idx, offset in enumerate(fold_offsets):
        train_end_idx = first_val_start_idx + offset - 1
        val_end_idx = train_end_idx + val_quarters

        if val_end_idx >= total_quarters:
            break

        train_end = periods[train_end_idx]
        val_end = periods[min(val_end_idx, total_quarters - 1)]

        split_config = SplitConfig(
            train_end=str(train_end),
            val_end=str(val_end),
        )

        X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
        X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        metrics = evaluate(y_val, preds, split_name=f"fold_{fold_idx}")
        fold_metrics.append(metrics)

    return fold_metrics
