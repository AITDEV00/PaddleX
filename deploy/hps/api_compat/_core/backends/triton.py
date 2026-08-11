"""Triton backend — client to a Triton Inference Server via gRPC.

This backend does NOT load the model in-process.  It sends requests to a
Triton Inference Server, which owns the model and provides continuous
batching via its dynamic batcher.

Select via ``HPS_API_BACKEND=triton``.

The actual gRPC transport lives in ``..triton_client``; this class adapts
it to the :class:`InferenceBackend` interface and adds model inventory
reporting from Triton's own model repository.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ..config import PIPELINE_DEPTH, TRITON_MODEL_NAME, TRITON_URL
from .base import InferenceBackend

logger = logging.getLogger("hps_api")


class TritonBackend(InferenceBackend):
    """Triton Inference Server gRPC backend."""

    name = "triton"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(PIPELINE_DEPTH)
        # A module-level shared client is used by triton_client; we don't
        # pre-create it (server may not be up yet).
        self._url = kwargs.get("url") or TRITON_URL
        self._model_name = kwargs.get("model_name") or TRITON_MODEL_NAME

    # ── InferenceBackend lifecycle ───────────────────────────────────────────
    def start(self) -> None:
        """Triton manages the model.  Mark ready (readiness is polled later)."""
        self._ready = True
        logger.info(
            "Triton backend active (url=%s, model=%s) — managed by Triton server",
            self._url, self._model_name,
        )

    def wait_ready(self, timeout: float) -> bool:
        return self._ready

    def shutdown(self) -> None:
        """Close the gRPC client (best-effort, fire-and-forget)."""
        try:
            from .. import triton_client

            loop = asyncio.new_event_loop()
            loop.run_until_complete(triton_client.close_client())
            loop.close()
        except Exception:
            pass

    # ── InferenceBackend interface ───────────────────────────────────────────
    async def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Send image to Triton via gRPC; Triton handles batching."""
        from .. import triton_client

        async with self.semaphore:
            try:
                return await triton_client.detect_layout(image)
            except Exception as e:
                raise RuntimeError(f"Triton inference failed: {e}") from e

    async def _list_models_remote(self) -> list[dict[str, Any]]:
        """Query Triton's model inventory (best-effort)."""
        try:
            from .. import triton_client

            client = triton_client._ensure_client()
            repo = await client.get_model_repository_index()
            models = []
            for m in repo.models:
                ready = await client.is_model_ready(m.name)
                models.append({
                    "name": m.name,
                    "version": "1",
                    "ready": bool(ready),
                    "active": bool(ready),
                })
            return models
        except Exception as e:
            logger.warning("Failed to list Triton models: %s", e)
            return []

    def list_models(self) -> list[dict[str, Any]]:
        """Return the Triton model inventory (sync wrapper around async query).

        Returns the static declaration if the server is unreachable.
        """
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._list_models_remote())
            finally:
                loop.close()
        except Exception:
            return [{
                "name": self._model_name,
                "version": "1",
                "ready": False,
                "active": False,
            }]

    def get_model_status(self, name: str) -> dict[str, Any] | None:
        """Return readiness for a single model, or None if unknown."""
        try:
            from .. import triton_client

            loop = asyncio.new_event_loop()
            try:
                ready = loop.run_until_complete(
                    triton_client.is_model_ready(name)
                )
            finally:
                loop.close()
            return {
                "name": name,
                "version": "1",
                "state": "READY" if ready else "UNAVAILABLE",
                "reason": None if ready else "model not ready",
            }
        except Exception:
            return None


__all__ = ["TritonBackend"]