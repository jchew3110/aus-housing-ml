"""
Macroeconomic feature transforms: interest rates, CPI, unemployment.

All features are lagged by at least 1 period so no current-period macro
data leaks into the prediction of next-quarter prices.
"""

import numpy as np
import pandas as pd


def add_cash_rate_features(
    df: pd.DataFrame,
    rate_col: str = "cash_rate",
) -> pd.DataFrame:
    """
    Add lagged cash rate and its QoQ change.
    Features: cash_rate_lag1, cash_rate_delta (this quarter's change), cash_rate_delta_lag1
    """
    df = df.copy()
    df["cash_rate_delta"] = df[rate_col].diff()
    df["cash_rate_lag1"] = df[rate_col].shift(1)
    df["cash_rate_delta_lag1"] = df["cash_rate_delta"].shift(1)
    return df


def add_cpi_features(
    df: pd.DataFrame,
    cpi_col: str = "cpi",
) -> pd.DataFrame:
    """
    Add lagged CPI and year-over-year inflation rate.
    Features: cpi_lag1, cpi_yoy_pct (lagged by 1 period)
    """
    df = df.copy()
    df["cpi_yoy_pct"] = df[cpi_col].pct_change(periods=4) * 100
    df["cpi_lag1"] = df[cpi_col].shift(1)
    df["cpi_yoy_pct_lag1"] = df["cpi_yoy_pct"].shift(1)
    return df


def add_unemployment_features(
    df: pd.DataFrame,
    unemp_col: str = "unemployment_rate",
) -> pd.DataFrame:
    """
    Add lagged unemployment and its QoQ change.
    Features: unemp_lag1, unemp_delta (this quarter's change), unemp_delta_lag1
    """
    df = df.copy()
    df["unemp_delta"] = df[unemp_col].diff()
    df["unemp_lag1"] = df[unemp_col].shift(1)
    df["unemp_delta_lag1"] = df["unemp_delta"].shift(1)
    return df


def add_rate_regime_features(
    df: pd.DataFrame,
    rate_col: str = "cash_rate_lag1",
) -> pd.DataFrame:
    """
    Bucket the lagged cash rate into 4 ordinal regimes.

    0: very low  (< 1%)  — emergency/near-zero stimulus
    1: low       (1–3%)  — accommodative
    2: normal    (3–6%)  — historical mid-range
    3: high      (≥ 6%)  — restrictive

    Captures non-linear interest rate effects: the difference between
    moving from 5% → 6% is very different from moving from 0.1% → 1.1%.
    Requires cash_rate_lag1 to already be present (call add_cash_rate_features first).
    """
    df = df.copy()
    rate = df[rate_col].values
    # np.digitize([1.0, 3.0, 6.0]) bins: 0 if <1, 1 if 1-3, 2 if 3-6, 3 if ≥6
    bins = np.array([1.0, 3.0, 6.0])
    df["rate_regime"] = np.where(
        pd.isna(df[rate_col]),
        np.nan,
        np.digitize(rate, bins).astype(float),
    )
    return df
