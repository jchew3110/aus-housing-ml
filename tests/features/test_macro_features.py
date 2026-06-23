"""Tests for macro feature transforms."""

import numpy as np
import pandas as pd
import pytest

from src.features.macro_features import (
    add_cash_rate_features,
    add_cpi_features,
    add_unemployment_features,
)


@pytest.fixture
def macro_df():
    n = 20
    return pd.DataFrame(
        {
            "cash_rate": np.linspace(4.5, 1.5, n),
            "cpi": np.linspace(100, 110, n),
            "unemployment_rate": np.linspace(5.5, 4.0, n),
        }
    )


class TestCashRateFeatures:
    def test_lag1_is_shifted_by_one(self, macro_df):
        df = add_cash_rate_features(macro_df)
        assert df["cash_rate_lag1"].iloc[1] == pytest.approx(macro_df["cash_rate"].iloc[0])

    def test_first_lag1_is_nan(self, macro_df):
        df = add_cash_rate_features(macro_df)
        assert pd.isna(df["cash_rate_lag1"].iloc[0])

    def test_delta_lag1_is_nan_at_start(self, macro_df):
        df = add_cash_rate_features(macro_df)
        # delta is NaN at index 0, delta_lag1 is NaN at indices 0 and 1
        assert pd.isna(df["cash_rate_delta_lag1"].iloc[0])
        assert pd.isna(df["cash_rate_delta_lag1"].iloc[1])


class TestCpiFeatures:
    def test_yoy_pct_correct(self, macro_df):
        df = add_cpi_features(macro_df)
        # yoy at index 4 = (cpi[4] / cpi[0] - 1) * 100
        expected = (macro_df["cpi"].iloc[4] / macro_df["cpi"].iloc[0] - 1) * 100
        assert df["cpi_yoy_pct"].iloc[4] == pytest.approx(expected, rel=1e-4)

    def test_cpi_lag1_shifted(self, macro_df):
        df = add_cpi_features(macro_df)
        assert df["cpi_lag1"].iloc[5] == pytest.approx(macro_df["cpi"].iloc[4])


class TestUnemploymentFeatures:
    def test_unemp_lag1_shifted(self, macro_df):
        df = add_unemployment_features(macro_df)
        assert df["unemp_lag1"].iloc[3] == pytest.approx(macro_df["unemployment_rate"].iloc[2])

    def test_delta_lag1_double_lagged(self, macro_df):
        df = add_unemployment_features(macro_df)
        # delta at index 1 = unemp[1] - unemp[0]; delta_lag1 at index 2 = delta[1]
        delta_at_1 = macro_df["unemployment_rate"].iloc[1] - macro_df["unemployment_rate"].iloc[0]
        assert df["unemp_delta_lag1"].iloc[2] == pytest.approx(delta_at_1, abs=1e-6)
