"""
Tests for lag feature engineering — the most leakage-sensitive module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.config import CITIES
from src.features.lag_features import (
    add_price_changes,
    add_price_lags,
    add_rolling_stats,
    create_target,
)


@pytest.fixture
def two_city_panel():
    """Two cities × 10 quarters, different price levels to make leakage obvious."""
    periods = pd.period_range("2015Q1", periods=10, freq="Q-DEC")
    rows = []
    for city, base in [("Sydney", 100.0), ("Melbourne", 200.0)]:
        for i, p in enumerate(periods):
            rows.append({"city": city, "period": p, "rppi_index": base + i})
    df = pd.DataFrame(rows).sort_values(["city", "period"]).reset_index(drop=True)
    return df


class TestAddPriceLags:
    def test_no_cross_city_contamination(self, two_city_panel):
        df = add_price_lags(two_city_panel, lags=[1])
        syd = df[df["city"] == "Sydney"].reset_index(drop=True)
        mel = df[df["city"] == "Melbourne"].reset_index(drop=True)

        # First row of Sydney should have NaN lag (not Melbourne's last value)
        assert pd.isna(syd.loc[0, "rppi_lag_1"])
        # First row of Melbourne should also have NaN lag
        assert pd.isna(mel.loc[0, "rppi_lag_1"])

    def test_lag_values_correct(self, two_city_panel):
        df = add_price_lags(two_city_panel, lags=[1, 2])
        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)

        # rppi_lag_1 at row 2 should equal rppi_index at row 1
        assert syd.loc[2, "rppi_lag_1"] == pytest.approx(syd.loc[1, "rppi_index"])
        # rppi_lag_2 at row 3 should equal rppi_index at row 1
        assert syd.loc[3, "rppi_lag_2"] == pytest.approx(syd.loc[1, "rppi_index"])

    def test_multiple_lags_created(self, two_city_panel):
        df = add_price_lags(two_city_panel, lags=[1, 2, 4])
        for lag in [1, 2, 4]:
            assert f"rppi_lag_{lag}" in df.columns

    def test_original_df_not_modified(self, two_city_panel):
        original_cols = list(two_city_panel.columns)
        add_price_lags(two_city_panel, lags=[1])
        assert list(two_city_panel.columns) == original_cols


class TestAddPriceChanges:
    def test_qoq_pct_correct(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)
        # QoQ change from 100 to 101 = 1%
        assert syd.loc[1, "rppi_qoq_pct"] == pytest.approx(1.0)

    def test_first_row_qoq_is_nan(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)
        assert pd.isna(syd.loc[0, "rppi_qoq_pct"])

    def test_no_cross_city_contamination_in_changes(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        mel = df[df["city"] == "Melbourne"].sort_values("period").reset_index(drop=True)
        # Melbourne first row should be NaN, not using Sydney's last price
        assert pd.isna(mel.loc[0, "rppi_qoq_pct"])


class TestAddRollingStats:
    def test_rolling_uses_shifted_window(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        df = add_rolling_stats(df, window=3, min_periods=2)

        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)

        # At index 3, rolling_mean_4q uses indices 0-2 of rppi_qoq_pct (shifted by 1),
        # which means the current period's QoQ change (index 3) is NOT included.
        # Manual verification: shift(1) at index 3 = rppi_qoq_pct[2], rolling(3) = [0,1,2]
        qoq = syd["rppi_qoq_pct"].values
        expected_at_3 = np.nanmean(qoq[:3])  # indices 0,1,2 (after shift)
        assert syd.loc[4, "rolling_mean_4q"] == pytest.approx(expected_at_3, abs=0.01)

    def test_rolling_std_non_negative(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        df = add_rolling_stats(df)
        std_col = df["rolling_std_4q"].dropna()
        assert (std_col >= 0).all()


class TestCreateTarget:
    def test_target_is_next_period_qoq(self, two_city_panel):
        df = add_price_changes(two_city_panel)
        df = create_target(two_city_panel)  # also calls add_price_changes internally

        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)
        # target[t] == rppi_qoq_pct[t+1]
        for i in range(len(syd) - 1):
            expected = syd.iloc[i + 1]["rppi_qoq_pct"]
            actual = syd.iloc[i]["target"]
            if not pd.isna(expected):
                assert actual == pytest.approx(expected, abs=1e-6)

    def test_last_row_target_is_nan(self, two_city_panel):
        df = create_target(two_city_panel)
        syd = df[df["city"] == "Sydney"].sort_values("period").reset_index(drop=True)
        assert pd.isna(syd.iloc[-1]["target"])

    def test_no_cross_city_target_contamination(self, two_city_panel):
        df = create_target(two_city_panel)
        mel = df[df["city"] == "Melbourne"].sort_values("period").reset_index(drop=True)
        # Melbourne's last row target should be NaN (not Sydney's first period)
        assert pd.isna(mel.iloc[-1]["target"])
