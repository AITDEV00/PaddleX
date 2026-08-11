"""Debug visualization for the Docling API.

This package implements per-model *vertical slices* of debug functionality.
Each model (e.g. PP-DocLayoutV3) gets its own debug renderer that knows how
to visualize that model's specific output — a layout model draws bounding
boxes, a table model might draw grid lines, etc.

The seam is :class:`DebugRenderer` (an ABC) + a registry.  A new model's
debug slice is added by:

  1. Subclassing :class:`DebugRenderer` in a new module (e.g.
     ``tabledet.py``).
  2. Registering it in :func:`get_debugger`.

Nothing else changes — the HTTP layer calls :func:`render_debug` with a
model id and the model's raw boxes, and the registry routes to the right
vertical slice.

HTTP contract
-------------
When a request to ``/v1/convert/*`` passes ``debug=1`` (or ``debug=true``),
the convert endpoint returns the **image with the model's output drawn on
it** (content-type ``image/png``) instead of the usual conversion response.
This is a debugging convenience and is intentionally separate from the
normal conversion pipeline.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("hps_api")


class DebugRenderer(ABC):
    """A per-model debug slice: how to draw a model's output on an image.

    Each subclass is a *vertical slice* for exactly one model family.  It
    receives the model's raw output (a list of box dicts for layout) and
    the original image, and returns the annotated image bytes (PNG).
    """

    #: Model id(s) this debugger serves.  Used by the registry.
    models: tuple[str, ...] = ()

    @abstractmethod
    def render(self, image: np.ndarray, output: Any) -> bytes:
        """Draw *output* on *image* and return PNG bytes."""

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_box(
        canvas: np.ndarray,
        coord: list[float] | np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one axis-aligned box (xmin,ymin,xmax,ymax) + label."""
        xmin, ymin, xmax, ymax = (int(v) for v in coord[:4])
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                canvas,
                (xmin, max(0, ymin - th - 6)),
                (min(canvas.shape[1], xmin + tw + 6), ymin),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (xmin + 3, max(th + 2, ymin - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def to_png(canvas: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("Failed to encode debug image as PNG")
        return io.BytesIO(buf).getvalue()


class LayoutDebugRenderer(DebugRenderer):
    """Debug slice for layout models (PP-DocLayoutV3)."""

    models = ("PP-DocLayoutV3", "PP-DocLayoutV2", "PP-DocLayoutV1")

    #: Color per label (BGR). Falls back to a fixed palette when unknown.
    _PALETTE: tuple[tuple[int, int, int], ...] = (
        (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 255, 0),
    )

    def _color_for(self, label: str) -> tuple[int, int, int]:
        # Stable color per label (hash-based), fallback to palette
        idx = abs(hash(label)) % len(self._PALETTE)
        return self._PALETTE[idx]

    def render(self, image: np.ndarray, output: Any) -> bytes:
        canvas = image.copy()
        boxes = output if isinstance(output, list) else getattr(output, "boxes", [])
        for box in boxes:
            if not isinstance(box, dict):
                continue
            label = box.get("label", "")
            score = box.get("score")
            coord = box.get("coordinate") or box.get("coords") or []
            if len(coord) < 4:
                continue
            text = f"{label}:{score:.2f}" if score is not None else label
            self._draw_box(canvas, coord, text, self._color_for(label))
        return self.to_png(canvas)


# ─── Registry ─────────────────────────────────────────────────────────────
_REGISTRY: dict[str, DebugRenderer] = {}


def _build_registry() -> dict[str, DebugRenderer]:
    reg: dict[str, DebugRenderer] = {}
    for cls in DebugRenderer.__subclasses__():
        for m in cls.models:
            reg[m] = cls()
    return reg


def get_debugger(model_id: str) -> DebugRenderer | None:
    """Return the debug renderer for *model_id*, or None if none registered.

    Falls back to the first renderer if model_id is unknown (so a generic
    box-drawer still works), but prefers the exact model slice.
    """
    global _REGISTRY
    if not _REGISTRY:
        _REGISTRY = _build_registry()
    renderer = _REGISTRY.get(model_id)
    if renderer is not None:
        return renderer
    # Fallback: first registered renderer that accepts any output
    return _REGISTRY.get("PP-DocLayoutV3")


def render_debug(
    model_id: str, image: np.ndarray, output: Any
) -> bytes | None:
    """Render *output* for *model_id* as PNG bytes, or None if unsupported."""
    renderer = get_debugger(model_id)
    if renderer is None:
        return None
    try:
        return renderer.render(image, output)
    except Exception as e:
        logger.exception("Debug render failed for %s: %s", model_id, e)
        return None


__all__ = [
    "DebugRenderer",
    "LayoutDebugRenderer",
    "get_debugger",
    "render_debug",
]