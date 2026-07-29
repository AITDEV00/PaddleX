"""Health slice — liveness and readiness endpoints.

Owned by _core because every API compatibility layer needs the same
health/readiness probes.
"""

from __future__ import annotations

from .routes import router

__all__ = ["router"]
