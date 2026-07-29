"""Latency tracing — per-stage timing instrumentation for the conversion pipeline.

Controlled by a single environment variable for easy enable/disable:

    HPS_LATENCY_LOG=1   →  enable structured per-request timing logs
    HPS_LATENCY_LOG=0   →  disable (default — zero overhead)

When enabled, each request logs a JSON line with the full stage breakdown:

    {"event":"latency_trace","request_id":"abc123","filename":"doc.png",
     "stages":{"image_load":0.003,"layout_detection":0.045,
               "convert_to_docling":0.002,"export_formats":0.001},
     "total":0.051,"to_formats":["md","json"]}

When disabled, :class:`LatencyTracer` is a thin no-op that does not call
``time.perf_counter`` or allocate dicts, so production throughput is
unaffected.

Usage in pipeline code::

    from .._core.latency import LatencyTracer

    tracer = LatencyTracer("doc.png", ["md"])
    with tracer.stage("image_load"):
        image = load_image_from_bytes(data)
    with tracer.stage("layout_detection"):
        boxes = await run_layout_detection(image)
    tracer.finish()  # logs if enabled
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("hps_api.latency")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Single env-var gate — checked once at import, then reused as a module-level
# boolean for zero-overhead fast-path when disabled.
_LATENCY_ENABLED: bool = os.environ.get("HPS_LATENCY_LOG", "0") in ("1", "true", "True", "yes")


def is_latency_logging_enabled() -> bool:
    """Return whether latency tracing is currently enabled.

    Re-reads the env var so tests can toggle it at runtime via
    ``monkeypatch`` or direct ``os.environ`` manipulation.
    """
    return os.environ.get("HPS_LATENCY_LOG", "0") in ("1", "true", "True", "yes")


def set_latency_logging(enabled: bool) -> None:
    """Enable or disable latency logging at runtime (for tests / debugging).

    Sets the ``HPS_LATENCY_LOG`` environment variable.
    """
    os.environ["HPS_LATENCY_LOG"] = "1" if enabled else "0"


class LatencyTracer:
    """Per-request latency tracer with stage-level granularity.

    When latency logging is disabled, methods are no-ops with minimal
    overhead (a single boolean check per call, no dict allocation).

    When enabled, accumulates per-stage timings and logs a structured
    JSON summary on :meth:`finish`.
    """

    __slots__ = (
        "_enabled", "_request_id", "_filename", "_to_formats",
        "_stages", "_t0",
    )

    def __init__(
        self,
        filename: str = "",
        to_formats: list[str] | None = None,
    ) -> None:
        self._enabled = is_latency_logging_enabled()
        if not self._enabled:
            return
        self._request_id: str = uuid.uuid4().hex[:12]
        self._filename: str = filename
        self._to_formats: list[str] = [str(f) for f in (to_formats or [])]
        self._stages: dict[str, float] = {}
        self._t0: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> "Iterator[None]":
        """Context manager that times a pipeline stage.

        Usage::

            with tracer.stage("image_load"):
                image = load_image_from_bytes(data)
        """
        if not self._enabled:
            yield
            return
        t_start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = time.perf_counter() - t_start

    def record(self, name: str, elapsed: float) -> None:
        """Manually record a stage timing (instead of using the context manager)."""
        if not self._enabled:
            return
        self._stages[name] = elapsed

    def finish(self) -> dict[str, float] | None:
        """Log the accumulated timings and return the stages dict.

        Returns ``None`` when disabled (no overhead).
        """
        if not self._enabled:
            return None
        total = time.perf_counter() - self._t0
        trace = {
            "event": "latency_trace",
            "request_id": self._request_id,
            "filename": self._filename,
            "to_formats": self._to_formats,
            "stages": self._stages,
            "total": round(total, 6),
        }
        logger.info(json.dumps(trace))
        return self._stages

    @property
    def stages(self) -> dict[str, float]:
        """Return accumulated stage timings (empty if disabled)."""
        if not self._enabled:
            return {}
        return self._stages
