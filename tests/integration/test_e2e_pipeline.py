"""
End-to-end integration tests: synthetic panel → feature build → train → save → load → predict.

These tests exercise the full stack without touching external data sources.
They are slower than unit tests (~5s) due to model fitting.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.config import CITIES, SplitConfig
from src.features.pipeline import FEATURE_COLS, FeaturePipeline
from src.models.evaluator import calibration_coverage, evaluate
from src.models.registry import ModelRegistry
from src.models.ridge import RidgeHousingModel


def _make_panel(n_quarters: int = 48) -> pd.DataFrame:
    """Multi-city synthetic panel large enough for train/val/test splits."""
    start = pd.Period("2010Q1", freq="Q-DEC")
    periods = pd.period_range(start=start, periods=n_quarters, freq="Q-DEC")
    rows = []
    for city in CITIES:
        rng = np.random.default_rng(CITIES.index(city))
        base = 100.0 + CITIES.index(city) * 5
        rate = 3.0
        cpi = 100.0
        unemp = 5.5
        for p in periods:
            base += rng.normal(0.5, 0.4)
            rate = max(0.1, rate + rng.normal(0, 0.15))
            cpi += rng.uniform(0.2, 0.5)
            unemp = max(3.0, unemp + rng.normal(0, 0.1))
            rows.append(
                {
                    "city": city,
                    "period": p,
                    "rppi_index": max(base, 50.0),
                    "cash_rate": rate,
                    "cpi": cpi,
                    "unemployment_rate": unemp,
                }
            )
    df = pd.DataFrame(rows).sort_values(["city", "period"]).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


@pytest.fixture(scope="module")
def feature_df_e2e(panel):
    return FeaturePipeline().build(panel)


@pytest.fixture(scope="module")
def split_cfg():
    return SplitConfig(train_end="2017Q4", val_end="2019Q4")


class TestFullPipeline:
    def test_feature_matrix_has_correct_columns(self, feature_df_e2e):
        for col in FEATURE_COLS:
            assert col in feature_df_e2e.columns, f"Missing feature: {col}"

    def test_no_nan_in_feature_matrix(self, feature_df_e2e):
        nan_counts = feature_df_e2e[FEATURE_COLS].isna().sum()
        assert nan_counts.sum() == 0, f"NaN features: {nan_counts[nan_counts > 0]}"

    def test_train_val_test_sizes_reasonable(self, feature_df_e2e, split_cfg):
        pipeline = FeaturePipeline()
        X_tr, y_tr = pipeline.get_X_y(feature_df_e2e, "train", split_cfg)
        X_v, y_v = pipeline.get_X_y(feature_df_e2e, "val", split_cfg)
        X_te, y_te = pipeline.get_X_y(feature_df_e2e, "test", split_cfg)
        assert len(X_tr) > 0
        assert len(X_v) > 0
        assert len(X_te) > 0

    def test_ridge_fit_predict(self, feature_df_e2e, split_cfg):
        pipeline = FeaturePipeline()
        X_tr, y_tr = pipeline.get_X_y(feature_df_e2e, "train", split_cfg)
        X_v, y_v = pipeline.get_X_y(feature_df_e2e, "val", split_cfg)
        X_te, y_te = pipeline.get_X_y(feature_df_e2e, "test", split_cfg)

        model = RidgeHousingModel()
        model.fit(X_tr, y_tr, X_v, y_v)
        preds = model.predict(X_te)

        assert len(preds) == len(y_te)
        assert not np.any(np.isnan(preds))

        metrics = evaluate(y_te, preds)
        assert np.isfinite(metrics.mae)
        assert metrics.mae > 0

    def test_calibration_coverage_function(self, feature_df_e2e, split_cfg):
        pipeline = FeaturePipeline()
        X_tr, y_tr = pipeline.get_X_y(feature_df_e2e, "train", split_cfg)
        X_v, y_v = pipeline.get_X_y(feature_df_e2e, "val", split_cfg)

        model = RidgeHousingModel()
        model.fit(X_tr, y_tr, X_v, y_v)
        _, lo, hi = model.predict_with_interval(X_v, confidence=0.90)
        coverage = calibration_coverage(y_v, lo, hi)

        assert 0.0 <= coverage <= 1.0


class TestModelRegistryRoundTrip:
    def test_save_load_predict_identical(self, feature_df_e2e, split_cfg):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            pipeline = FeaturePipeline()
            X_tr, y_tr = pipeline.get_X_y(feature_df_e2e, "train", split_cfg)
            X_v, y_v = pipeline.get_X_y(feature_df_e2e, "val", split_cfg)

            model = RidgeHousingModel()
            model.fit(X_tr, y_tr, X_v, y_v)
            preds_original = model.predict(X_v)

            metrics = {"val": evaluate(y_v, preds_original)}
            registry.save(model, metrics, pipeline.feature_cols, calibration_coverage=0.88)

            loaded_model, metadata = registry.load("ridge")
            preds_loaded = loaded_model.predict(X_v)

            np.testing.assert_array_almost_equal(preds_original, preds_loaded)
            assert metadata["calibration_coverage"] == pytest.approx(0.88)

    def test_list_models_includes_calibration(self, feature_df_e2e, split_cfg):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            pipeline = FeaturePipeline()
            X_tr, y_tr = pipeline.get_X_y(feature_df_e2e, "train", split_cfg)
            X_v, y_v = pipeline.get_X_y(feature_df_e2e, "val", split_cfg)

            model = RidgeHousingModel()
            model.fit(X_tr, y_tr, X_v, y_v)
            metrics = {"val": evaluate(y_v, model.predict(X_v))}
            registry.save(model, metrics, pipeline.feature_cols, calibration_coverage=0.75)

            listed = registry.list_models()
            assert len(listed) == 1
            assert "calibration_coverage" in listed[0]
            assert listed[0]["calibration_coverage"] == pytest.approx(0.75)
