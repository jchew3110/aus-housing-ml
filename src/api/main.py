"""
FastAPI application factory.

The model is loaded once at startup via the lifespan context manager and
stored in app.state.model_state for reuse across requests.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.dependencies import ModelState
from src.api.middleware import PredictionMetricsMiddleware, set_model_info
from src.api.routers import health, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ModelState()
    model_name = os.getenv("MODEL_NAME")  # optionally pin a specific model
    state.load_model(model_name)
    app.state.model_state = state

    # Populate Prometheus model_info gauge
    meta = state.metadata
    set_model_info(
        model_name=meta.get("name", "unknown"),
        model_version=meta.get("version", "1.0"),
        training_date=meta.get("training_date", ""),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AUS Housing Price Index Predictor",
        description=(
            "Predicts next-quarter residential property price index change (QoQ %) "
            "by Australian capital city using macroeconomic features."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(PredictionMetricsMiddleware)

    app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
    app.include_router(health.router, tags=["health"])

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(
            generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()
