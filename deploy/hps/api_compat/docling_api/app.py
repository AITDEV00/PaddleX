"""Docling-compatible FastAPI application — thin wiring layer.

This module creates the FastAPI app, wires up the lifespan (which starts
the inference thread), and mounts routers from the Docling-specific slice
and the shared _core health slice.

All business logic lives in:
  - .._core/     — shared infrastructure (config, inference, image, health)
  - converter/   — PaddleX → DoclingDocument conversion
  - routes.py    — /v1/convert/* endpoints
  - service.py   — conversion pipeline orchestration
  - schema.py    — Docling request/response models
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use orjson for response serialization — 3-10x faster than stdlib json.
# Falls back to FastAPI's default JSONResponse if orjson is unavailable.
try:
    from fastapi.responses import ORJSONResponse as _DefaultResponse
except ImportError:  # pragma: no cover
    from fastapi.responses import JSONResponse as _DefaultResponse  # type: ignore[assignment]

from .._core.config import STARTUP_TIMEOUT, setup_logging
from .._core.health.routes import router as health_router
from .._core.inference import state
from .routes import router as convert_router

logger = logging.getLogger("hps_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the inference backend at startup, signal shutdown on exit.

    For the Triton backend, checks that the Triton server is reachable
    and the model is loaded. For the direct backend, starts the
    inference thread and waits for model loading.
    """
    setup_logging()
    state.start_inference()

    from .._core.config import INFERENCE_BACKEND

    if INFERENCE_BACKEND == "triton":
        # Check Triton server readiness (non-blocking — model may still
        # be loading, so we poll with a timeout)
        from .._core import triton_client
        import asyncio
        import time as _time

        deadline = _time.monotonic() + STARTUP_TIMEOUT
        while _time.monotonic() < deadline:
            if await triton_client.is_server_ready():
                logger.info("Triton server and model ready")
                break
            await asyncio.sleep(2)
        else:
            state.shutdown()
            raise RuntimeError(
                f"Triton server not ready within {STARTUP_TIMEOUT}s"
            )
    else:
        # Direct backend: wait for model to load in inference thread
        if not state.wait_ready(STARTUP_TIMEOUT):
            state.shutdown()
            raise RuntimeError(f"Model failed to load within {STARTUP_TIMEOUT}s")

    logger.info("Application ready")
    yield

    # Shutdown: stop backend cleanly
    state.shutdown()
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Factory function — creates the FastAPI application."""
    app = FastAPI(
        title="PaddleX HPS Docling API",
        description=(
            "Docling-compatible document conversion API backed by "
            "PaddleX PP-DocLayoutV3 TensorRT. "
            "Currently provides layout detection only."
        ),
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=_DefaultResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(convert_router)

    return app


# Module-level app instance for Granian
app = create_app()
