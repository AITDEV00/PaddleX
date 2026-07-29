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
from .schema import HealthCheckResponse, ReadinessResponse

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
