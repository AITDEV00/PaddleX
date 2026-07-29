"""Response models for health and readiness endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str = "ok"
