"""
Macroeconomic feature transforms: interest rates, CPI, unemployment.

All features are lagged by at least 1 period so no current-period macro
data leaks into the prediction of next-quarter prices.
"""

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
