"""
FeaturePipeline: orchestrates all feature engineering and manages the
train/val/test split using strictly temporal boundaries.
"""

from __future__ import annotations

import pandas as pd

from src.data.config import CITIES, SplitConfig
from src.features.city_encoder import add_city_dummies
from src.features.lag_features import (
    add_price_changes,
    add_price_lags,
    add_rolling_stats,
    create_target,
)
from src.features.macro_features import (
    add_cash_rate_features,
    add_cpi_features,
    add_unemployment_features,
)
from src.features.seasonal_features import add_seasonal_features

FEATURE_COLS: list[str] = [
    # Lagged price index levels
    "rppi_lag_1",
    "rppi_lag_2",
    "rppi_lag_4",
    # Lagged price changes
    "rppi_qoq_pct_lag1",
    "rppi_yoy_pct_lag1",
    # Rolling statistics (window of 4 quarters, shifted by 1)
    "rolling_mean_4q",
    "rolling_std_4q",
    # Macro: interest rate (all lagged 1 period)
    "cash_rate_lag1",
    "cash_rate_delta_lag1",
    # Macro: inflation (lagged)
    "cpi_lag1",
    "cpi_yoy_pct_lag1",
    # Macro: labour market (lagged)
    "unemp_lag1",
    "unemp_delta_lag1",
    # Seasonal
    "quarter",
    "quarter_sin",
    "quarter_cos",
    # City one-hot dummies
    *[f"city_{city}" for city in CITIES],
]

TARGET_COL = "target"


class FeaturePipeline:
    """
    Applies all feature engineering transforms in a fixed order and provides
    time-based splits for train/val/test.
    """

    feature_cols: list[str] = FEATURE_COLS
    target_col: str = TARGET_COL

    def build(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transforms to the raw panel DataFrame.

        Input: raw merged panel (city, period, rppi_index, cash_rate, cpi, unemployment_rate)
        Output: feature matrix with FEATURE_COLS + target; NaN-target rows dropped.
        """
        df = panel_df.copy()
        df = df.sort_values(["city", "period"]).reset_index(drop=True)

        # Price-based features (within-city groupby to prevent leakage)
        df = add_price_lags(df)
        df = add_price_changes(df)
        df = add_rolling_stats(df)

        # Lagged price change features (lag the already-computed changes by 1 more)
        df["rppi_qoq_pct_lag1"] = df.groupby("city")["rppi_qoq_pct"].shift(1)
        df["rppi_yoy_pct_lag1"] = df.groupby("city")["rppi_yoy_pct"].shift(1)

        # Macro features (applied per-row, no groupby needed — macro is national)
        df = add_cash_rate_features(df)
        df = add_cpi_features(df)
        df = add_unemployment_features(df)

        # Seasonal features
        df = add_seasonal_features(df)

        # City one-hot encoding
        df = add_city_dummies(df)

        # Target: next-quarter QoQ % change
        df = create_target(df)

        # Drop rows with NaN target (last row per city) or NaN in any feature
        df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

        return df

    def get_X_y(
        self,
        df: pd.DataFrame,
        split: str,
        split_config: SplitConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return (X, y) for the requested split using time-based boundaries.

        split: 'train', 'val', or 'test'
        split_config: uses default SplitConfig if not provided
        """
        if split_config is None:
            split_config = SplitConfig()

        period = pd.PeriodIndex(df["period"], freq="Q-DEC")
        train_end = pd.Period(split_config.train_end, freq="Q-DEC")
        val_end = pd.Period(split_config.val_end, freq="Q-DEC")

        if split == "train":
            mask = period <= train_end
        elif split == "val":
            mask = (period > train_end) & (period <= val_end)
        elif split == "test":
            mask = period > val_end
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        split_df = df[mask]
        X = split_df[self.feature_cols].copy()
        y = split_df[self.target_col].copy()
        return X, y
