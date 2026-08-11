"""Backend abstraction — the interface every inference backend implements.

This is the seam that lets the API layer stay backend-agnostic.  Instead of
hard-coded ``if backend == "triton"`` / ``elif backend == "direct"`` branches
scattered through the codebase, each backend is a self-contained class that
implements this interface.  Adding a future backend (e.g. "custom", "grpc",
"http", "vllm", …) is a matter of adding one class — nothing in the API layer
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceBackend(ABC):
    """Abstract interface shared by all inference backends.

    Lifecycle:
      1. ``start()``   — acquire resources, load the model (called once at app
                         startup from the app lifespan).
      2. ``detect(image)`` — run layout detection on a single image. May be
                         called concurrently; backends are responsible for
                         their own batching/concurrency.
      3. ``wait_ready(timeout)`` — block until the model is ready to serve.
      4. ``stop()`` — release resources on shutdown.

    Optional introspection for the Triton-compatible ``/v1/models`` endpoint:
      - ``list_models()`` — return the model inventory.
      - ``get_model_status(name)`` — readiness/state of one model.
      Backends that don't host named models (e.g. a remote client) can leave
      the defaults; the platform falls back to a statically-declared inventory.
    """

    #: Short identifier for this backend (also the value of HPS_API_BACKEND).
    name: str

    def __init__(self) -> None:
        self._ready = False

    # ── Lifecycle ───────────────────────────────────────────────────────────
    @abstractmethod
    def start(self) -> None:
        """Acquire resources and make the backend ready to serve."""

    def wait_ready(self, timeout: float) -> bool:
        """Block until ready (or raise on fatal load error).

        Default implementation: backend marks itself ready during ``start()``.
        Subclasses that load asynchronously (e.g. a worker thread) override.
        """
        return self._ready

    def shutdown(self) -> None:
        """Release resources. Default: no-op."""

    # ── Inference ───────────────────────────────────────────────────────────
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run layout detection on one image.

        Args:
            image: (H, W, 3) numpy array, uint8 (BGR).

        Returns:
            List of box dicts: {label, score, coordinate, order, cls_id,
            polygon_points}.
        """

    # ── Optional: model inventory (for Triton-compatible /v1/models) ───────
    def list_models(self) -> list[dict[str, Any]]:
        """Return the model inventory for ``/v1/models``.

        Each entry: {"name": str, "version": str, "ready": bool,
        "active": bool}.  Backends that don't own named models should return
        an empty list — the platform layer falls back to a static inventory
        declared in config.
        """
        return []

    def get_model_status(self, name: str) -> dict[str, Any] | None:
        """Return readiness/state for a single model name, or None if unknown."""
        return None