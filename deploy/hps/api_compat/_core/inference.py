"""Layout detection inference — pluggable backend facade.

This module is the single entry point the API layer uses to run layout
detection.  The actual work is delegated to an :class:`InferenceBackend`
selected via ``HPS_API_BACKEND``:

  - ``direct`` (default for lean images): PaddleX/TensorRT loaded in-process
    (see ``backends/direct.py``).
  - ``triton``: client to a Triton Inference Server via gRPC (see
    ``backends/triton.py``).
  - future: add a new module under ``backends/`` and register it in
    :func:`create_backend` — the API layer does not change.

This module exposes a module-level singleton ``state`` (an :class:`AppState`
holding the active backend) and the ``run_layout_detection()`` coroutine.

The same :class:`AppState` is also the seam for the Triton-compatible
``/v1/models`` endpoint: :meth:`AppState.list_models` and
:meth:`AppState.get_model_status` delegate to the active backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .backends import create_backend
from .config import INFERENCE_BACKEND
from .latency import is_latency_logging_enabled

logger = logging.getLogger("hps_api")


class AppState:
    """Holds the active inference backend and its concurrency state.

    The backend (Triton vs direct) is chosen once at startup from
    ``HPS_API_BACKEND``.  All lifecycle (``start_inference``,
    ``wait_ready``, ``shutdown``) and inference (``run_layout_detection``)
    are delegated to it.
    """

    def __init__(self) -> None:
        self._backend = create_backend()
        self._ready = False

    @property
    def backend(self):
        """The active :class:`InferenceBackend` instance."""
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def ready(self) -> bool:
        return self._ready or self._backend._ready

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def start_inference(self) -> None:
        """Start the active backend."""
        self._backend.start()
        self._ready = True

    def wait_ready(self, timeout: float) -> bool:
        """Wait for the backend model to be ready."""
        return self._backend.wait_ready(timeout)

    def shutdown(self) -> None:
        """Shut down the active backend."""
        self._backend.shutdown()

    # ── Model inventory (Triton-compatible /v1/models) ──────────────────────
    def list_models(self) -> list[dict[str, Any]]:
        """Model inventory from the active backend."""
        models = self._backend.list_models()
        if models:
            return models
        # Fallback: static declaration if the backend doesn't own named models.
        from .config import MODEL_NAME

        return [{
            "name": MODEL_NAME,
            "version": "1",
            "ready": self.ready,
            "active": self.ready,
        }]

    def get_model_status(self, name: str) -> dict[str, Any] | None:
        """Readiness/state for a single model from the active backend."""
        status = self._backend.get_model_status(name)
        if status is not None:
            return status
        from .config import MODEL_NAME

        if name == MODEL_NAME:
            return {
                "name": name,
                "version": "1",
                "state": "READY" if self.ready else "LOADING",
                "reason": None if self.ready else "model loading",
            }
        return None


# Singleton — one backend per process
state = AppState()


async def run_layout_detection(image: np.ndarray) -> list[dict[str, Any]]:
    """Run layout detection on a single image via the active backend.

    Delegates to the selected backend's :meth:`detect`.  The backend is
    responsible for its own batching/concurrency.

    Raises:
        RuntimeError: If the backend is unreachable or inference fails.
    """
    _log = is_latency_logging_enabled()
    t0 = time.perf_counter() if _log else 0.0
    boxes = await state.backend.detect(image)
    if _log:
        elapsed = time.perf_counter() - t0
        logger.info(
            '{"event":"latency","stage":"backend_detect","backend":"%s",'
            '"wait_s":%.6f,"n_boxes":%d}',
            state.backend_name, elapsed, len(boxes),
        )
    return boxes


__all__ = ["AppState", "state", "run_layout_detection"]