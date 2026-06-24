"""Tests for momentum and acceleration features."""

import numpy as np
import pandas as pd

from src.features.momentum_features import _signed_streak, add_momentum_features


class TestSignedStreak:
    def test_consecutive_up(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = _signed_streak(s)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_consecutive_down(self):
        s = pd.Series([-1.0, -2.0, -3.0])
        result = _signed_streak(s)
        assert list(result) == [-1.0, -2.0, -3.0]

    def test_streak_resets_on_direction_change(self):
        s = pd.Series([1.0, 2.0, -1.0, 1.0])
        result = _signed_streak(s)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == -1.0
        assert result.iloc[3] == 1.0

    def test_nan_resets_streak(self):
        s = pd.Series([1.0, np.nan, 1.0])
        result = _signed_streak(s)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 1.0

    def test_zero_resets_streak(self):
        s = pd.Series([1.0, 2.0, 0.0, 1.0])
        result = _signed_streak(s)
        assert result.iloc[2] == 0.0
        assert result.iloc[3] == 1.0


class TestAddMomentumFeatures:
    def _make_df(self):
        periods = pd.period_range("2015Q1", periods=12, freq="Q-DEC")
        n = len(periods)
        df = pd.DataFrame(
            {
                "city": ["Sydney"] * n,
                "period": periods,
                "rppi_index": 100 + np.arange(n, dtype=float),
                "rppi_qoq_pct": [1.0, 1.5, 2.0, -1.0, -2.0, 1.0, 1.0, 1.0, 0.5, 0.5, -0.5, 1.0],
            }
        )
        return df

    def test_momentum_streak_lag1_created(self):
        df = add_momentum_features(self._make_df())
        assert "momentum_streak_lag1" in df.columns

    def test_price_acceleration_lag1_created(self):
        df = add_momentum_features(self._make_df())
        assert "price_acceleration_lag1" in df.columns

    def test_first_lag1_is_nan(self):
        df = add_momentum_features(self._make_df())
        assert np.isnan(df["momentum_streak_lag1"].iloc[0])

    def test_no_cross_city_contamination(self):
        periods = pd.period_range("2015Q1", periods=8, freq="Q-DEC")
        df = pd.DataFrame(
            {
                "city": ["Sydney"] * 4 + ["Melbourne"] * 4,
                "period": list(periods[:4]) + list(periods[:4]),
                "rppi_index": [100, 101, 102, 103, 200, 201, 202, 203],
                "rppi_qoq_pct": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
            }
        ).sort_values(["city", "period"]).reset_index(drop=True)

        result = add_momentum_features(df)
        # Melbourne's first row should not inherit Sydney's streak
        mel = result[result["city"] == "Melbourne"]
        assert np.isnan(mel["momentum_streak_lag1"].iloc[0])

    def test_original_not_modified(self):
        df = self._make_df()
        original_cols = set(df.columns)
        add_momentum_features(df)
        assert set(df.columns) == original_cols
