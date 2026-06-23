"""
FastAPI application factory.

The model is loaded once at startup via the lifespan context manager and
stored in app.state.model_state for reuse across requests.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import ModelState
from src.api.routers import health, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ModelState()
    model_name = os.getenv("MODEL_NAME")  # optionally pin a specific model
    state.load_model(model_name)
    app.state.model_state = state
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

    app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
    app.include_router(health.router, tags=["health"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()
