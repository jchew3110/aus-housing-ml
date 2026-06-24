"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.models.evaluator import ModelMetrics, compare_models, evaluate


class TestEvaluate:
    def test_perfect_predictions_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = evaluate(y, y.copy())
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.r2 == pytest.approx(1.0)

    def test_directional_accuracy_perfect(self):
        y_true = np.array([1.0, -0.5, 2.0])
        y_pred = np.array([0.5, -0.1, 1.5])
        metrics = evaluate(y_true, y_pred)
        assert metrics.directional_accuracy == pytest.approx(1.0)

    def test_directional_accuracy_zero(self):
        y_true = np.array([1.0, -0.5, 2.0])
        y_pred = np.array([-0.5, 0.1, -1.5])
        metrics = evaluate(y_true, y_pred)
        assert metrics.directional_accuracy == pytest.approx(0.0)

    def test_mape_excludes_zero_actuals(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 1.1, 2.1])
        metrics = evaluate(y_true, y_pred)
        # Should not be NaN — zero actuals are excluded from MAPE
        assert not np.isnan(metrics.mape)

    def test_n_samples_correct(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate(y, y + 0.1)
        assert metrics.n_samples == 4

    def test_r2_negative_for_bad_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # inverted — worse than mean
        metrics = evaluate(y_true, y_pred)
        assert metrics.r2 < 0

    def test_rmse_greater_than_mae(self):
        y_true = np.array([1.0, 1.0, 1.0, 10.0])  # one outlier
        y_pred = np.zeros(4)
        metrics = evaluate(y_true, y_pred)
        assert metrics.rmse > metrics.mae


class TestCompareModels:
    def test_returns_dataframe(self):
        m = ModelMetrics(mae=0.5, rmse=0.6, r2=0.7, mape=5.0, directional_accuracy=0.8, n_samples=100)  # noqa: E501
        results = {"ridge": {"test": m}, "lgbm": {"test": m}}
        df = compare_models(results)
        assert df is not None
