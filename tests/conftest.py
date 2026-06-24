"""
Shared test fixtures.
"""


import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.config import CITIES, SplitConfig

# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_panel_df() -> pd.DataFrame:
    """
    Synthetic panel DataFrame with 8 cities × 40 quarters.
    Mimics the output of merge_to_panel().
    """
    n_quarters = 40
    start = pd.Period("2010Q1", freq="Q-DEC")
    periods = pd.period_range(start=start, periods=n_quarters, freq="Q-DEC")

    rows = []
    for city in CITIES:
        base = 100.0 + CITIES.index(city) * 10
        for i, p in enumerate(periods):
            # Simulate a slowly rising index with noise
            rppi = base + i * 0.5 + np.random.default_rng(CITIES.index(city) + i).normal(0, 0.3)
            rows.append(
                {
                    "city": city,
                    "period": p,
                    "rppi_index": max(rppi, 50.0),
                    "cash_rate": max(0.1, 3.0 - i * 0.03 + np.random.default_rng(i).normal(0, 0.1)),
                    "cpi": 100 + i * 0.4,
                    "unemployment_rate": max(3.0, 5.5 - i * 0.02),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "period"]).reset_index(drop=True)
    return df


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig(train_end="2016Q4", val_end="2018Q4")


@pytest.fixture
def feature_df(sample_panel_df) -> pd.DataFrame:
    """Feature matrix built from synthetic panel."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    return pipeline.build(sample_panel_df)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_ridge(feature_df, split_config):
    """A Ridge model fit on synthetic training data."""
    from src.features.pipeline import FeaturePipeline
    from src.models.ridge import RidgeHousingModel

    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
    X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)
    model = RidgeHousingModel()
    model.fit(X_train, y_train, X_val, y_val)
    return model


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_model(trained_ridge):
    """FastAPI app with a pre-loaded Ridge model bypassing the lifespan."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from src.api.dependencies import ModelState
    from src.api.routers import health, predict

    state = ModelState()
    state.model = trained_ridge
    state.metadata = {
        "name": "ridge",
        "version": "1.0",
        "training_date": "2024-01-01T00:00:00+00:00",
        "feature_cols": trained_ridge._feature_cols,
        "metrics": {
            "train": {"mae": 0.3, "rmse": 0.4, "r2": 0.9, "mape": 5.0, "directional_accuracy": 0.8},
            "val": {"mae": 0.5, "rmse": 0.6, "r2": 0.7, "mape": 8.0, "directional_accuracy": 0.7},
            "test": {"mae": 0.6, "rmse": 0.7, "r2": 0.65, "mape": 9.0, "directional_accuracy": 0.65},  # noqa: E501
        },
    }

    @asynccontextmanager
    async def lifespan(app):
        app.state.model_state = state
        yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(predict.router, prefix="/api/v1")
    test_app.include_router(health.router)
    return test_app


@pytest.fixture
def client(app_with_model):
    with TestClient(app_with_model) as c:
        yield c


@pytest.fixture
def valid_predict_payload() -> dict:
    return {
        "city": "Sydney",
        "quarter": 2,
        "year": 2024,
        "rppi_current": 150.0,
        "rppi_lag1": 148.0,
        "rppi_lag2": 146.0,
        "rppi_lag4": 142.0,
        "rppi_qoq_pct_current": 1.35,
        "rppi_yoy_pct_current": 5.6,
        "rolling_mean_4q": 1.2,
        "rolling_std_4q": 0.5,
        "cash_rate": 4.35,
        "cash_rate_prev": 4.35,
        "cpi": 128.0,
        "cpi_prev_year": 122.0,
        "unemployment_rate": 4.1,
        "unemployment_rate_prev": 4.0,
    }
