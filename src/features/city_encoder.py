"""
One-hot encode the city column.

Target encoding is explicitly avoided to prevent label leakage —
the encoding statistics would use target information from the full dataset.
"""

import pandas as pd

from src.data.config import CITIES


def add_city_dummies(
    df: pd.DataFrame,
    city_col: str = "city",
    cities: list[str] = CITIES,
) -> pd.DataFrame:
    """
    Add one-hot encoded city columns: city_Sydney, city_Melbourne, etc.
    Uses a fixed list of cities so the column set is always consistent
    regardless of which cities appear in the current data slice.
    """
    df = df.copy()
    for city in cities:
        df[f"city_{city}"] = (df[city_col] == city).astype(int)
    return df
