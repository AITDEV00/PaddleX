"""Inference backends — pluggable layout-detection backends.

Each backend implements :class:`InferenceBackend` and is selected via the
``HPS_API_BACKEND`` env var:

  - ``direct``  — load PaddleX in-process (TensorRT), micro-batched.
                  Reuses the existing in-process pipeline. (default)
  - ``triton``  — send requests to a Triton Inference Server via gRPC.
  - future      — add a new module under ``backends/`` and register it in
                  :func:`create_backend`.

The platform layer (``inference.py``, routes) only talks to the
``InferenceBackend`` interface, so adding a new backend never touches the
API/HTTP layer.
"""

from __future__ import annotations

from typing import Any

from .base import InferenceBackend


def create_backend(name: str | None = None, **kwargs: Any) -> InferenceBackend:
    """Instantiate the requested inference backend.

    Args:
        name: Backend id — "direct" or "triton". If None, reads
            ``HPS_API_BACKEND``. Unknown names raise a clear error.
        **kwargs: Optional backend-specific overrides.

    The default is ``direct`` (matches the lean images). Previously the
    default was ``triton`` (the full/Triton-based images); that is now
    explicit via ``HPS_API_BACKEND=triton``.
    """
    from . import direct, triton  # deferred to avoid heavy imports

    backend_id = (name or "").strip().lower()
    if not backend_id:
        from ..config import INFERENCE_BACKEND
        backend_id = INFERENCE_BACKEND.strip().lower()

    if backend_id == "direct":
        return direct.DirectBackend(**kwargs)
    if backend_id == "triton":
        return triton.TritonBackend(**kwargs)
    raise ValueError(
        f"Unknown HPS_API_BACKEND '{backend_id}'. "
        f"Supported: direct, triton"
    )


__all__ = ["InferenceBackend", "create_backend"]