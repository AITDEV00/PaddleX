"""Mistral OCR-compatible FastAPI application — thin wiring layer.

Creates the FastAPI app, wires the lifespan (starts the shared inference
backend), and mounts the health router plus the ``/v1/ocr`` router.

All business logic lives in:
  - service.py  — OCR pipeline orchestration
  - converter/  — PaddleX → Mistral OCR conversion
  - schema.py   — official Mistral DTOs

The lifespan is imported from ``docling_api.app`` so both layers share the
same single inference backend.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use orjson for JSON response serialization when available.
try:
    from fastapi.responses import ORJSONResponse as _DefaultResponse
except ImportError:  # pragma: no cover
    from fastapi.responses import JSONResponse as _DefaultResponse  # type: ignore[assignment]

from .._core.config import setup_logging
from .._core.health.routes import router as health_router
from ..docling_api.app import lifespan
from .routes import router as ocr_router


def create_app() -> FastAPI:
    """Factory function — creates the Mistral OCR FastAPI app."""
    setup_logging()

    app = FastAPI(
        title="PaddleX HPS Mistral OCR API",
        description=(
            "Mistral OCR-compatible endpoint (/v1/ocr) backed by "
            "PaddleX layout detection (PP-DocLayoutV3 TensorRT)."
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
    app.include_router(ocr_router)

    return app


# Module-level app instance for Granian
app = create_app()