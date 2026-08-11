"""Route handlers for health, readiness, and version endpoints.

Mirrors upstream docling-serve health endpoints:
  /health      — liveness probe (always ok)
  /health-check — liveness alias
  /ready       — readiness probe (model loaded)
  /readyz      — readiness alias (k8s convention)
  /livez       — liveness alias (k8s convention)
  /version     — version information
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from fastapi import APIRouter, HTTPException

from ..inference import state
from .schema import (
    HealthCheckResponse,
    ModelInventoryResponse,
    ModelMetadataResponse,
    ModelEntry,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health():
    """Liveness probe."""
    return HealthCheckResponse(status="ok")


@router.get("/health-check", response_model=HealthCheckResponse)
async def health_check():
    """Liveness probe alias (for platforms that expect /health-check)."""
    return HealthCheckResponse(status="ok")


@router.get("/livez", response_model=HealthCheckResponse)
async def livez():
    """Liveness probe — k8s convention."""
    return HealthCheckResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready():
    """Readiness probe — model must be loaded."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ReadinessResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz():
    """Readiness probe — k8s convention."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ReadinessResponse(status="ok")


@router.get("/version")
async def version():
    """Version information."""
    try:
        docling_slim_ver = pkg_version("docling-slim")
    except PackageNotFoundError:
        docling_slim_ver = "unknown"
    try:
        docling_core_ver = pkg_version("docling-core")
    except PackageNotFoundError:
        docling_core_ver = "unknown"

    return {
        "version": "0.1.0",
        "docling_slim": docling_slim_ver,
        "docling_core": docling_core_ver,
    }


# ─── Triton-compatible /v1/models ─────────────────────────────────────────────
#
# Triton exposes a model inventory at /v1/models.  These routes implement a
# compatible subset that works regardless of backend:
#   - direct backend → reports the in-process PaddleX model(s) from
#                      AppState.list_models()
#   - triton backend → reports the models loaded in the Triton server
#
# This lets a client written against Triton's inventory API discover the
# PP-DocLayoutV3 model id without caring which backend is active.

@router.get("/v1/models", response_model=ModelInventoryResponse)
async def list_models():
    """List all models (Triton-compatible)."""
    models = state.list_models()
    return ModelInventoryResponse(
        models=[ModelEntry(**m) for m in models]
    )


@router.get("/v1/models/{name}", response_model=ModelMetadataResponse)
async def get_model_metadata(name: str):
    """Get metadata for a single model (Triton-compatible)."""
    status = state.get_model_status(name)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return ModelMetadataResponse(
        name=name,
        versions=[status.get("version", "1")],
        platform=status.get("platform"),
        backend=status.get("backend"),
        inputs=status.get("inputs", []),
        outputs=status.get("outputs", []),
    )


@router.get("/v1/models/{name}/ready")
async def get_model_ready(name: str):
    """Readiness of a single model (Triton-compatible)."""
    status = state.get_model_status(name)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    ready = status.get("ready", status.get("state") == "READY")
    if not ready:
        raise HTTPException(status_code=503, detail=f"Model '{name}' not ready")
    return {"status": "ok"}
