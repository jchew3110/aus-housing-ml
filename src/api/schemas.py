"""
Pydantic v2 request/response schemas for the prediction API.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class City(str, Enum):
    SYDNEY = "Sydney"
    MELBOURNE = "Melbourne"
    BRISBANE = "Brisbane"
    ADELAIDE = "Adelaide"
    PERTH = "Perth"
    HOBART = "Hobart"
    DARWIN = "Darwin"
    CANBERRA = "Canberra"


class PredictRequest(BaseModel):
    city: City
    quarter: Literal[1, 2, 3, 4]
    year: int = Field(..., ge=2000, le=2050, description="Year of the prediction period")

    # The caller provides pre-computed lag/context values.
    # These map directly onto the model's feature columns.
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
