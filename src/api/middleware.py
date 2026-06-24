"""
Prometheus metrics + structured request logging middleware.

Metrics exposed:
  prediction_requests_total{endpoint, model, status_code}  — Counter
  prediction_latency_seconds{endpoint}                     — Histogram
  model_info{model_name, model_version, training_date}     — Info gauge (set at startup)
"""

import logging
import time
import uuid

from prometheus_client import Counter, Histogram, Info
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction API requests",
    ["endpoint", "model", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction API request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

MODEL_INFO = Info("model", "Loaded model metadata")


def set_model_info(model_name: str, model_version: str, training_date: str) -> None:
    MODEL_INFO.info(
        {
            "model_name": model_name,
            "model_version": model_version,
            "training_date": training_date,
        }
    )


class PredictionMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start

        endpoint = request.url.path
        status_code = str(response.status_code)

        # Resolve model name from app state if available
        try:
            meta = request.app.state.model_state.metadata
            model_name = meta.get("name", "unknown")
        except AttributeError:
            model_name = "unknown"

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            model=model_name,
            status_code=status_code,
        ).inc()

        if endpoint.startswith("/api/"):
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        logger.info(
            "%s | %s %s | %s | %.1fms",
            request_id,
            request.method,
            endpoint,
            status_code,
            latency * 1000,
        )

        return response
