"""Master FastAPI app — mounts enabled API layers.

This is the default entry point (``api_compat.web_app.app:app``) for Granian.
It mounts the shared ``_core.health`` router plus the routers of every API
layer listed in ``HPS_API_LAYERS`` (see ``_core/config.py``).

Per-layer apps (e.g. ``api_compat.docling_api.app:app``) remain available for
deployments that want a dedicated process/port per protocol.

The runtime is shared: one ``_core.inference`` backend drives every layer, so
enabling more layers does not add a second model or GPU context.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use orjson for response serialization when available.
try:
    from fastapi.responses import ORJSONResponse as _DefaultResponse
except ImportError:  # pragma: no cover
    from fastapi.responses import JSONResponse as _DefaultResponse  # type: ignore[assignment]

from ._core.config import API_LAYERS, setup_logging
from ._core.health.routes import router as health_router
from .docling_api.app import lifespan

logger = logging.getLogger("hps_api")


def _create_router(layer: str):
    """Import and return the router for *layer* (deferred import).

    Deferring the import means enabling a layer is the only thing that pulls
    in its (optional) dependencies.  Raises a clear ImportError if a requested
    layer's dependency is missing.
    """
    if layer == "docling":
        from .docling_api.routes import router

        return router
    if layer == "mistral":
        from .mistral_ocr_api.routes import router

        return router
    raise ValueError(f"unknown API layer {layer!r}")


def create_app() -> FastAPI:
    """Factory function — creates the FastAPI app with enabled layers."""
    setup_logging()

    app = FastAPI(
        title="PaddleX HPS API",
        description=(
            "PaddleX document/layout inference exposed through one or more "
            "API protocols (Docling, Mistral OCR). Enabled layers: "
            + ", ".join(API_LAYERS)
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
    for layer in API_LAYERS:
        try:
            app.include_router(_create_router(layer))
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"HPS_API_LAYERS includes {layer!r} but its dependencies are "
                f"not installed. For 'mistral': install the 'mistralai' "
                f"package."
            ) from exc
        except ValueError:
            logger.warning("unknown API layer %r in HPS_API_LAYERS, ignoring", layer)
            continue
        logger.info("enabled API layer: %s", layer)

    return app


# Module-level app instance for Granian
app = create_app()