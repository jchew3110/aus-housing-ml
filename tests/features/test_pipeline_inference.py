"""Tests for FeaturePipeline.build_inference_row() — the critical path for /predict/raw."""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLS, FeaturePipeline
from src.data.config import CITIES


def _make_panel(city: str = "Sydney", n_quarters: int = 8) -> pd.DataFrame:
    """Build a minimal synthetic single-city panel."""
    start = pd.Period("2018Q1", freq="Q-DEC")
    periods = pd.period_range(start=start, periods=n_quarters, freq="Q-DEC")
    rng = np.random.default_rng(42)
    rows = []
    base = 150.0
    for i, p in enumerate(periods):
        base += rng.normal(0.5, 0.3)
        rows.append(
            {
                "city": city,
                "period": p,
                "rppi_index": base,
                "cash_rate": max(0.1, 2.0 - i * 0.1),
                "cpi": 115 + i * 0.4,
                "unemployment_rate": max(3.5, 5.5 - i * 0.05),
            }
        )
    return pd.DataFrame(rows)


class TestBuildInferenceRow:
    def test_returns_one_row_for_6_quarters(self):
        panel = _make_panel(n_quarters=6)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        assert len(result) == 1

    def test_all_feature_cols_non_nan(self):
        panel = _make_panel(n_quarters=6)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        for col in FEATURE_COLS:
            assert not pd.isna(result[col].iloc[0]), f"Feature '{col}' is NaN"

    def test_last_row_period_is_last_provided_period(self):
        panel = _make_panel(n_quarters=6)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        last_period_in = panel["period"].iloc[-1]
        last_period_out = result["period"].iloc[-1]
        assert last_period_out == last_period_in

    def test_5_quarters_yields_empty_dataframe(self):
        """5 quarters is insufficient: cpi_yoy_pct_lag1 and rppi_yoy_pct_lag1 need 6."""
        panel = _make_panel(n_quarters=5)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        assert len(result) == 0

    def test_more_history_returns_more_rows(self):
        panel = _make_panel(n_quarters=10)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        # With 10 quarters we expect 5 valid rows (one per quarter from q6 onwards)
        assert len(result) > 1

    def test_city_dummy_correct(self):
        city = "Melbourne"
        panel = _make_panel(city=city, n_quarters=6)
        pipeline = FeaturePipeline()
        result = pipeline.build_inference_row(panel)
        assert result["city_Melbourne"].iloc[0] == 1
        assert result["city_Sydney"].iloc[0] == 0

    def test_does_not_require_non_nan_target(self):
        """build_inference_row() must work even though the last period has no target."""
        panel = _make_panel(n_quarters=6)
        pipeline = FeaturePipeline()
        result_infer = pipeline.build_inference_row(panel)
        result_train = pipeline.build(panel)
        # Training build drops NaN-target rows; inference build keeps the last period
        assert len(result_infer) >= len(result_train)
