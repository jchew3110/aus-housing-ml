"""
GET /health and GET /model-info endpoints.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse, ModelInfoResponse
from src.data.config import CITIES

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state.model_state
    model_loaded = state.model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    state = request.app.state.model_state
    meta = state.metadata

    feature_cols = meta.get("feature_cols", [])
    raw_metrics = meta.get("metrics", {})

    # Convert nested metric dicts to float-only (remove n_samples key for clean display)
    clean_metrics: dict[str, dict[str, float]] = {}
    for split, m in raw_metrics.items():
        clean_metrics[split] = {k: v for k, v in m.items() if k != "n_samples"}

    calibration = meta.get("calibration_coverage")

    return ModelInfoResponse(
        model_name=meta.get("name", "unknown"),
        model_version=meta.get("version", "1.0"),
        training_date=meta.get("training_date", ""),
        feature_count=len(feature_cols),
        metrics=clean_metrics,
        cities_supported=CITIES,
        calibration_coverage=round(calibration, 4) if calibration is not None else None,
    )
