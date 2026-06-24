"""
Pydantic v2 request/response schemas for the prediction API.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class City(StrEnum):
    SYDNEY = "Sydney"
    MELBOURNE = "Melbourne"
    BRISBANE = "Brisbane"
    ADELAIDE = "Adelaide"
    PERTH = "Perth"
    HOBART = "Hobart"
    DARWIN = "Darwin"
    CANBERRA = "Canberra"


# ---------------------------------------------------------------------------
# /predict — feature-vector request (advanced / internal callers)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    city: City
    quarter: Literal[1, 2, 3, 4]
    year: int = Field(..., ge=2000, le=2050, description="Year of the prediction period")

    # The caller provides pre-computed lag/context values.
    rppi_current: float = Field(..., gt=0, description="Current period RPPI index value")
    rppi_lag1: float = Field(..., gt=0, description="RPPI from 1 quarter ago")
    rppi_lag2: float = Field(..., gt=0, description="RPPI from 2 quarters ago")
    rppi_lag4: float = Field(..., gt=0, description="RPPI from 4 quarters ago")
    rppi_qoq_pct_current: float = Field(
        ..., description="Current quarter QoQ % price change"
    )
    rppi_yoy_pct_current: float = Field(
        ..., description="Current YoY % price change (4-quarter)"
    )
    rolling_mean_4q: float = Field(
        ..., description="Rolling 4-quarter mean of QoQ % changes (lagged)"
    )
    rolling_std_4q: float = Field(
        ..., ge=0, description="Rolling 4-quarter std of QoQ % changes (lagged)"
    )
    momentum_streak: float = Field(
        default=0.0,
        description="Signed count of consecutive up/down quarters (lagged). "
        "Positive = consecutive increases, negative = consecutive decreases.",
    )
    price_acceleration: float = Field(
        default=0.0,
        description="Change in QoQ growth rate from one quarter to the next (lagged).",
    )
    cash_rate: float = Field(..., ge=0, le=30, description="Current quarter-end cash rate (%)")
    cash_rate_prev: float = Field(..., ge=0, le=30, description="Previous quarter cash rate (%)")
    cpi: float = Field(..., gt=0, description="Current CPI index value")
    cpi_prev_year: float = Field(..., gt=0, description="CPI from 4 quarters ago")
    unemployment_rate: float = Field(
        ..., ge=0, le=100, description="Current quarter unemployment rate (%)"
    )
    unemployment_rate_prev: float = Field(
        ..., ge=0, le=100, description="Previous quarter unemployment rate (%)"
    )

    @model_validator(mode="after")
    def check_rppi_lags_reasonable(self) -> PredictRequest:
        ratio = self.rppi_current / self.rppi_lag4
        if not (0.5 < ratio < 2.0):
            raise ValueError(
                f"rppi_current / rppi_lag4 = {ratio:.2f} is outside the plausible range (0.5–2.0). "
                "Check your lag values."
            )
        return self


# ---------------------------------------------------------------------------
# /predict/raw — time-series request (primary user-facing endpoint)
# ---------------------------------------------------------------------------

class RppiObservation(BaseModel):
    year: int = Field(..., ge=2000, le=2050)
    quarter: Literal[1, 2, 3, 4]
    rppi_index: float = Field(..., gt=0, description="RPPI index value for this quarter")


class MacroObservation(BaseModel):
    year: int = Field(..., ge=2000, le=2050)
    quarter: Literal[1, 2, 3, 4]
    cash_rate: float = Field(..., ge=0, le=30)
    cpi: float = Field(..., gt=0)
    unemployment_rate: float = Field(..., ge=0, le=100)


class PredictRawRequest(BaseModel):
    """
    User-friendly prediction request that accepts raw time-series data.

    Provide at least 5 consecutive quarters of RPPI and macro observations
    in chronological order. The server computes all lag/rolling features
    internally and returns a prediction for the quarter immediately following
    the last observation provided.
    """

    city: City
    rppi_history: list[RppiObservation] = Field(
        ...,
        min_length=6,
        max_length=40,
        description="Chronological quarterly RPPI observations (oldest first). Minimum 6.",
    )
    macro_history: list[MacroObservation] = Field(
        ...,
        min_length=6,
        max_length=40,
        description="Chronological quarterly macro observations (oldest first). "
        "Must have the same length and periods as rppi_history.",
    )

    @model_validator(mode="after")
    def validate_histories(self) -> PredictRawRequest:
        if len(self.rppi_history) != len(self.macro_history):
            raise ValueError(
                f"rppi_history ({len(self.rppi_history)} items) and macro_history "
                f"({len(self.macro_history)} items) must have the same length."
            )
        for r, m in zip(self.rppi_history, self.macro_history):
            if r.year != m.year or r.quarter != m.quarter:
                raise ValueError(
                    f"Period mismatch: rppi_history has {r.year}Q{r.quarter} "
                    f"but macro_history has {m.year}Q{m.quarter}."
                )
        return self


# ---------------------------------------------------------------------------
# /predict/batch — multiple feature-vector predictions in one call
# ---------------------------------------------------------------------------

class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Shared response models
# ---------------------------------------------------------------------------

class PredictionInterval(BaseModel):
    lower: float
    upper: float
    confidence: float = 0.90


class PredictResponse(BaseModel):
    city: str
    year: int
    quarter: int
    predicted_qoq_pct_change: float = Field(
        description="Predicted next-quarter QoQ property price % change"
    )
    direction: Literal["up", "down", "flat"]
    confidence_interval: PredictionInterval
    model_name: str
    model_version: str


class ExplainResponse(PredictResponse):
    """Extends PredictResponse with per-feature SHAP contributions."""

    shap_values: dict[str, float] = Field(
        description="Feature name → SHAP contribution (additive, sums to prediction − base_value)"
    )
    base_value: float = Field(
        description="Model's expected output (SHAP intercept / mean training prediction)"
    )


class BatchPredictError(BaseModel):
    index: int = Field(description="Zero-based index of the failed request in the batch")
    city: str
    detail: str = Field(description="Human-readable error description")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    errors: list[BatchPredictError] = []
    success_count: int
    error_count: int
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    training_date: str
    feature_count: int
    metrics: dict[str, dict[str, float]]
    cities_supported: list[str]
    calibration_coverage: float | None = None
