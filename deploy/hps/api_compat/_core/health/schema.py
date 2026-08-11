"""Response models for health and readiness endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str = "ok"


# ─── Triton-compatible /v1/models ─────────────────────────────────────────────
#
# Triton's HTTP model-inventory endpoints:
#   GET /v1/models                → list all models
#   GET /v1/models/{name}        → get a model's metadata
#   GET /v1/models/{name}/ready  → readiness of one model
#
# We implement a compatible subset so clients written against Triton's
# inventory API work against the direct path too.  The wire shape mirrors
# Triton's repository index entries.


class ModelEntry(BaseModel):
    """A single model in the inventory (mirrors Triton's index entry)."""

    name: str
    version: str = "1"
    state: str = "READY"  # READY | UNAVAILABLE | LOADING | UNLOADED
    reason: str | None = None
    ready: bool = True
    active: bool = True


class ModelInventoryResponse(BaseModel):
    """Body for GET /v1/models."""

    models: list[ModelEntry]


class ModelMetadataResponse(BaseModel):
    """Body for GET /v1/models/{name}."""

    name: str
    versions: list[str]
    platform: str | None = None
    backend: str | None = None
    inputs: list[dict] = []
    outputs: list[dict] = []
