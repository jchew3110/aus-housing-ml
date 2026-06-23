"""
POST /api/v1/predict — accepts feature inputs, returns price index prediction.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from src.api.dependencies import ModelState, get_model_state
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionInterval,
)
from src.data.config import CITIES
from src.features.pipeline import FEATURE_COLS

router = APIRouter()


def _request_to_features(req: PredictRequest) -> pd.DataFrame:
    """
    Map PredictRequest fields onto the model's FEATURE_COLS vector.

    The caller provides current-period values; the mapping applies the
    lags already baked into the field names (rppi_lag1 → rppi_lag_1, etc.).
    """
    city_dummies = {f"city_{city}": int(req.city.value == city) for city in CITIES}

    row = {
        "rppi_lag_1": req.rppi_lag1,
        "rppi_lag_2": req.rppi_lag2,
        "rppi_lag_4": req.rppi_lag4,
        "rppi_qoq_pct_lag1": req.rppi_qoq_pct_current,
        "rppi_yoy_pct_lag1": req.rppi_yoy_pct_current,
        "rolling_mean_4q": req.rolling_mean_4q,
        "rolling_std_4q": req.rolling_std_4q,
        "cash_rate_lag1": req.cash_rate_prev,
        "cash_rate_delta_lag1": req.cash_rate - req.cash_rate_prev,
        "cpi_lag1": req.cpi,
        "cpi_yoy_pct_lag1": (req.cpi / req.cpi_prev_year - 1) * 100,
        "unemp_lag1": req.unemployment_rate_prev,
        "unemp_delta_lag1": req.unemployment_rate - req.unemployment_rate_prev,
        "quarter": req.quarter,
        "quarter_sin": float(np.sin(2 * np.pi * req.quarter / 4)),
        "quarter_cos": float(np.cos(2 * np.pi * req.quarter / 4)),
        **city_dummies,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, state: ModelState = Depends(get_model_state)) -> PredictResponse:
    X = _request_to_features(req)
    point, lower, upper = state.model.predict_with_interval(X, confidence=0.90)

    pred = float(point[0])
    if pred > 0.05:
        direction = "up"
    elif pred < -0.05:
        direction = "down"
    else:
        direction = "flat"

    return PredictResponse(
        city=req.city.value,
        year=req.year,
        quarter=req.quarter,
        predicted_qoq_pct_change=round(pred, 4),
        direction=direction,
        confidence_interval=PredictionInterval(
            lower=round(float(lower[0]), 4),
            upper=round(float(upper[0]), 4),
            confidence=0.90,
        ),
        model_name=state.metadata.get("name", "unknown"),
        model_version=state.metadata.get("version", "1.0"),
    )
