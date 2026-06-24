"""
Lag and rolling features for the RPPI price index.

All features use groupby(city).shift() — never raw shift() — to prevent
cross-city contamination (city A's last period must not become city B's lag).

The target is next-quarter QoQ % change, created by shift(-1) within each city.
"""

import pandas as pd


def add_price_lags(
    df: pd.DataFrame,
    price_col: str = "rppi_index",
    lags: list[int] = [1, 2, 4],
    group_col: str = "city",
) -> pd.DataFrame:
    """Add lagged price index values grouped by city."""
    df = df.copy()
    for lag in lags:
        df[f"rppi_lag_{lag}"] = df.groupby(group_col)[price_col].shift(lag)
    return df


def add_price_changes(
    df: pd.DataFrame,
    price_col: str = "rppi_index",
    group_col: str = "city",
) -> pd.DataFrame:
    """
    Add QoQ and YoY percentage changes grouped by city.
    rppi_qoq_pct = (rppi_t / rppi_{t-1} - 1) * 100
    rppi_yoy_pct = (rppi_t / rppi_{t-4} - 1) * 100
    """
    df = df.copy()
    grp = df.groupby(group_col)[price_col]
    df["rppi_qoq_pct"] = grp.pct_change(periods=1) * 100
    df["rppi_yoy_pct"] = grp.pct_change(periods=4) * 100
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    change_col: str = "rppi_qoq_pct",
    window: int = 4,
    group_col: str = "city",
    min_periods: int = 2,
) -> pd.DataFrame:
    """
    Rolling mean and std of quarterly price changes.

    Uses shift(1) before the rolling window so the current period's own
    QoQ change is never included in the rolling calculation.
    """
    df = df.copy()

    def _rolling_on_shifted(series: pd.Series) -> pd.DataFrame:
        shifted = series.shift(1)
        mean = shifted.rolling(window=window, min_periods=min_periods).mean()
        std = shifted.rolling(window=window, min_periods=min_periods).std()
        return pd.DataFrame({"rolling_mean_4q": mean, "rolling_std_4q": std})

    stats = df.groupby(group_col)[change_col].apply(
        lambda s: _rolling_on_shifted(s)
    )
    # Unstack the groupby result back onto df index
    stats = stats.reset_index(level=0, drop=True)
    df["rolling_mean_4q"] = stats["rolling_mean_4q"]
    df["rolling_std_4q"] = stats["rolling_std_4q"]
    return df


def create_target(
    df: pd.DataFrame,
    price_col: str = "rppi_index",
    group_col: str = "city",
) -> pd.DataFrame:
    """
    Create the prediction target: next-quarter QoQ price change (%).

    target[t] = rppi_qoq_pct[t+1]   (shift(-1) within each city group)

    Rows where target is NaN (last row per city) must be dropped before training.
    """
    df = df.copy()
    if "rppi_qoq_pct" not in df.columns:
        df = add_price_changes(df, price_col=price_col, group_col=group_col)
    df["target"] = df.groupby(group_col)["rppi_qoq_pct"].shift(-1)
    return df
