"""
Prediction endpoints.

POST /api/v1/predict        — feature-vector input (advanced callers)
POST /api/v1/predict/raw    — raw time-series input (primary user-facing)
POST /api/v1/predict/batch  — batch of feature-vector requests
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from src.api.dependencies import ModelState, get_model_state
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    MacroObservation,
    PredictRawRequest,
    PredictRequest,
    PredictResponse,
    PredictionInterval,
    RppiObservation,
)
from src.data.config import CITIES
from src.features.pipeline import FEATURE_COLS

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_response(
    req_city: str,
    req_year: int,
    req_quarter: int,
    X: pd.DataFrame,
    state: ModelState,
) -> PredictResponse:
    point, lower, upper = state.model.predict_with_interval(X, confidence=0.90)
    pred = float(point[0])
    direction = "up" if pred > 0.0 else ("down" if pred < 0.0 else "flat")
    return PredictResponse(
        city=req_city,
        year=req_year,
        quarter=req_quarter,
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


def _request_to_features(req: PredictRequest) -> pd.DataFrame:
    """Map PredictRequest fields onto the model's FEATURE_COLS vector."""
    import numpy as _np

    city_dummies = {f"city_{city}": int(req.city.value == city) for city in CITIES}
    cash_rate_lag1 = req.cash_rate_prev
    rate_regime = float(_np.digitize(cash_rate_lag1, [1.0, 3.0, 6.0]))

    row = {
        "rppi_lag_1": req.rppi_lag1,
        "rppi_lag_2": req.rppi_lag2,
        "rppi_lag_4": req.rppi_lag4,
        "rppi_qoq_pct_lag1": req.rppi_qoq_pct_current,
        "rppi_yoy_pct_lag1": req.rppi_yoy_pct_current,
        "rolling_mean_4q": req.rolling_mean_4q,
        "rolling_std_4q": req.rolling_std_4q,
        "momentum_streak_lag1": req.momentum_streak,
        "price_acceleration_lag1": req.price_acceleration,
        "cash_rate_lag1": cash_rate_lag1,
        "cash_rate_delta_lag1": req.cash_rate - req.cash_rate_prev,
        "rate_regime": rate_regime,
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


def _raw_request_to_panel(
    city: str,
    rppi_history: list[RppiObservation],
    macro_history: list[MacroObservation],
) -> pd.DataFrame:
    """Build a single-city panel DataFrame from raw time-series observations."""
    rows = []
    for r, m in zip(rppi_history, macro_history):
        rows.append(
            {
                "city": city,
                "period": pd.Period(f"{r.year}Q{r.quarter}", freq="Q-DEC"),
                "rppi_index": r.rppi_index,
                "cash_rate": m.cash_rate,
                "cpi": m.cpi,
                "unemployment_rate": m.unemployment_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def _features_for_last_period(
    city: str,
    rppi_history: list[RppiObservation],
    macro_history: list[MacroObservation],
) -> tuple[pd.DataFrame, int, int]:
    """
    Build the feature vector for the last period in the history.
    Returns (X, prediction_year, prediction_quarter) where the year/quarter
    represent the NEXT period after the last observation.

    Uses build_inference_row() so the last provided period's features are
    available even though its target (next quarter) is unknown.
    """
    from src.features.pipeline import FeaturePipeline

    panel = _raw_request_to_panel(city, rppi_history, macro_history)
    pipeline = FeaturePipeline()
    feat_df = pipeline.build_inference_row(panel)

    if feat_df.empty:
        raise ValueError(
            "Not enough history to compute all features. Provide at least 6 consecutive "
            "quarters of RPPI and macro observations."
        )

    # Last row = last period with complete features. Predict the quarter after it.
    last_row = feat_df.iloc[[-1]]
    X = last_row[FEATURE_COLS].copy()

    last_period = pd.Period(last_row["period"].iloc[0], freq="Q-DEC")
    next_period = last_period + 1
    return X, next_period.year, next_period.quarter


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest, state: ModelState = Depends(get_model_state)
) -> PredictResponse:
    """
    Predict next-quarter QoQ price change from a pre-computed feature vector.

    All lag/rolling values must be provided by the caller. For a simpler
    interface that accepts raw RPPI + macro time series, use `/predict/raw`.
    """
    X = _request_to_features(req)
    next_period = pd.Period(f"{req.year}Q{req.quarter}", freq="Q-DEC") + 1
    return _make_response(req.city.value, next_period.year, next_period.quarter, X, state)


@router.post("/predict/raw", response_model=PredictResponse)
def predict_raw(
    req: PredictRawRequest, state: ModelState = Depends(get_model_state)
) -> PredictResponse:
    """
    Predict next-quarter QoQ price change from raw time-series data.

    Provide at least 5 consecutive quarters of RPPI index values and
    matching macro observations (cash rate, CPI, unemployment). The server
    computes all lag, rolling, momentum, and regime features automatically.

    The prediction is for the quarter immediately after the last provided period.
    """
    X, pred_year, pred_quarter = _features_for_last_period(
        req.city.value, req.rppi_history, req.macro_history
    )
    return _make_response(req.city.value, pred_year, pred_quarter, X, state)


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest, state: ModelState = Depends(get_model_state)
) -> BatchPredictResponse:
    """
    Predict for multiple cities/periods in a single request.

    Accepts up to 100 feature-vector requests. Responses preserve input order.
    """
    predictions = []
    for r in req.requests:
        X = _request_to_features(r)
        next_period = pd.Period(f"{r.year}Q{r.quarter}", freq="Q-DEC") + 1
        predictions.append(
            _make_response(r.city.value, next_period.year, next_period.quarter, X, state)
        )

    return BatchPredictResponse(
        predictions=predictions,
        model_name=state.metadata.get("name", "unknown"),
        model_version=state.metadata.get("version", "1.0"),
    )
