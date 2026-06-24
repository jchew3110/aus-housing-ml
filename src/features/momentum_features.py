"""
Momentum and acceleration features derived from RPPI price changes.

momentum_streak: signed count of consecutive up/down quarters (per city).
  +3 means three straight quarters of positive QoQ growth.
  -2 means two straight quarters of decline.

price_acceleration: second derivative of price — is the rate of change
  itself speeding up or slowing down? (QoQ % change minus prior QoQ % change)

Both are lagged by 1 period to prevent target leakage.
"""

import numpy as np
import pandas as pd


def _signed_streak(series: pd.Series) -> pd.Series:
    result = np.zeros(len(series), dtype=float)
    streak = 0.0
    for i, val in enumerate(series):
        if pd.isna(val):
            result[i] = np.nan
            streak = 0.0
            continue
        elif val > 0:
            streak = max(streak, 0) + 1
        elif val < 0:
            streak = min(streak, 0) - 1
        else:
            streak = 0.0
        result[i] = streak
    return pd.Series(result, index=series.index)


def add_momentum_features(
    df: pd.DataFrame,
    qoq_col: str = "rppi_qoq_pct",
    group_col: str = "city",
) -> pd.DataFrame:
    """
    Add momentum_streak and price_acceleration, both lagged by 1 period.

    Requires rppi_qoq_pct to already be present (call add_price_changes first).
    """
    df = df.copy()

    # Signed consecutive-quarter streak
    streak = df.groupby(group_col)[qoq_col].transform(
        lambda s: _signed_streak(s).values
    )
    df["momentum_streak"] = streak
    df["momentum_streak_lag1"] = df.groupby(group_col)["momentum_streak"].shift(1)

    # Second derivative: acceleration of QoQ growth rate
    df["price_accel"] = df.groupby(group_col)[qoq_col].diff()
    df["price_acceleration_lag1"] = df.groupby(group_col)["price_accel"].shift(1)

    return df
