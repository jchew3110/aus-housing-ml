"""
Seasonal features derived from the quarterly period index.
"""

import numpy as np
import pandas as pd


def add_seasonal_features(
    df: pd.DataFrame,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Add quarter number (1-4) and sin/cos encodings for seasonality.
    Sin/cos encoding preserves the cyclical nature of quarters:
    Q1 and Q4 are 'adjacent' in calendar terms.
    """
    df = df.copy()
    quarters = df[period_col].apply(lambda p: p.quarter)
    df["quarter"] = quarters
    df["quarter_sin"] = np.sin(2 * np.pi * quarters / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarters / 4)
    return df
