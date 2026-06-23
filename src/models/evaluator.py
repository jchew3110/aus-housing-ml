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
