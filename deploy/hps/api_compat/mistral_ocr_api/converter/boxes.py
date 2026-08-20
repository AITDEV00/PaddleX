"""Canonical layout-box model and per-model box adapters.

This is the extension point that keeps the Mistral emitter decoupled from any
single PaddleX model's native output structure.

A backend model emits boxes with **model-specific** structure — dicts or
objects, and different key names / coordinate layouts.  PP-DocLayoutV3 uses
``{label, score, coordinate, order, cls_id, polygon_points}``.  A future model
(e.g. a multi-column layout net, an OCR-VL pipeline, or a different label
taxonomy) may emit something entirely different.

Rather than special-casing every shape inside the emitter, each model
registers a :class:`BoxAdapter` subclass that normalizes its native boxes
into the canonical :class:`LayoutBox`.  The emitter only ever sees
``LayoutBox`` instances, so adding a new model is a *pure additive* change:
write a new adapter, register it, done.  No edits to the emitter.

    adapter = box_adapter_registry.get("PP-DocLayoutV3")
    canonical = [adapter.to_box(raw) for raw in raw_boxes]

Every accessor is defensive: it accepts either a ``dict`` or an attribute-
bearing object, tolerates ``None``/missing values, and normalizes to a valid
canonical field rather than raising. Subclasses override the accessor(s) that
differ, not the whole pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

logger = logging.getLogger("hps_api")

#: A native model box is an unstructured dict-or-object. Adapters are
#: responsible for the exact key layout; the emitter never touches raw boxes.
RawBox: TypeAlias = Any


@dataclass(frozen=True)
class LayoutBox:
    """Model-agnostic layout box consumed by the Mistral emitter.

    All coordinates are pixel values in the original image space. ``label``
    is the PaddleX label string (``"text"``, ``"table"``, ...). ``cls_id``
    and ``order`` are optional model bookkeeping that some models emit and
    others do not; when absent they default to ``-1`` / ``None``.

    ``coordinate`` is always exactly ``[xmin, ymin, xmax, ymax]`` (4 ints);
    if the source model only provides a polygon, the adapter derives this
    bounding box from it.
    """

    label: str
    score: float
    coordinate: list[int]  # exactly [xmin, ymin, xmax, ymax]
    cls_id: int = -1
    order: int | None = None
    polygon_points: list[list[float]] | None = None


class BoxAdapter:
    """Base class for a model-specific box normalizer.

    Subclasses declare ``model_id`` and may override any field accessor
    (``_label``, ``_score``, ``_coordinate``, ``_cls_id``, ``_order``,
    ``_polygon``) or the whole ``to_box``. All accessors are defensive and
    return a valid canonical value even for malformed input.

    Tunable class attributes (override in a subclass):

    * ``default_label`` / ``default_score`` / ``default_cls_id`` — fallbacks
      used when a field is missing or unparseable.
    * ``clamp_scores`` / ``min_score`` / ``max_score`` — whether and how to
      clamp confidence scores to a sane range.
    * ``min_polygon_points`` — discard polygons shorter than this.
    """

    #: Stable model id this adapter handles (used as the registry key).
    model_id: ClassVar[str] = ""

    # Fallback values for missing / unparseable fields.
    default_label: ClassVar[str] = "text"
    default_score: ClassVar[float] = 0.0
    default_cls_id: ClassVar[int] = -1

    # Score sanitization.
    clamp_scores: ClassVar[bool] = True
    min_score: ClassVar[float] = 0.0
    max_score: ClassVar[float] = 1.0

    # Minimum polygon points to keep; shorter shapes are treated as "no
    # polygon" (degenerate boxes are already covered by the coordinate).
    min_polygon_points: ClassVar[int] = 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_box(self, raw: RawBox) -> LayoutBox:
        """Normalize one native box into a canonical :class:`LayoutBox`."""
        return LayoutBox(
            label=self._label(raw),
            score=self._score(raw),
            coordinate=self._coordinate(raw),
            cls_id=self._cls_id(raw),
            order=self._order(raw),
            polygon_points=self._polygon(raw),
        )

    def to_boxes(self, raws: list[RawBox]) -> list[LayoutBox]:
        """Normalize a list of native boxes, dropping none (each maps to a
        valid :class:`LayoutBox` even on malformed input)."""
        return [self.to_box(raw) for raw in raws]

    # ------------------------------------------------------------------
    # Field accessors. Override in a subclass to handle a different shape
    # without rewriting the whole pipeline.
    # ------------------------------------------------------------------
    def _label(self, raw: RawBox) -> str:
        label = self._read(raw, "label", self.default_label)
        if label is None:
            return self.default_label
        return str(label)

    def _score(self, raw: RawBox) -> float:
        try:
            score = float(self._read(raw, "score", self.default_score))
        except (TypeError, ValueError):
            return self.default_score
        if not math.isfinite(score):
            return self.default_score
        if self.clamp_scores:
            score = min(self.max_score, max(self.min_score, score))
        return score

    def _coordinate(self, raw: RawBox) -> list[int]:
        # Prefer an explicit coordinate; fall back to a bounding box derived
        # from the polygon so a model that only emits points still yields a
        # well-formed canonical box.
        coord = self._read(raw, "coordinate")
        if coord is None:
            coord = self._read(raw, "polygon_points")
        return self._normalize_coordinate(coord)

    def _cls_id(self, raw: RawBox) -> int:
        try:
            return int(self._read(raw, "cls_id", self.default_cls_id))
        except (TypeError, ValueError):
            return self.default_cls_id

    def _order(self, raw: RawBox) -> int | None:
        order = self._read(raw, "order")
        if order is None:
            return None
        try:
            return int(order)
        except (TypeError, ValueError):
            return None

    def _polygon(self, raw: RawBox) -> list[list[float]] | None:
        return self._normalize_polygon(self._read(raw, "polygon_points"))

    # ------------------------------------------------------------------
    # Shared robust helpers (overridable per subclass if a model needs to
    # diverge, but usually not).
    # ------------------------------------------------------------------
    @staticmethod
    def _read(raw: RawBox, key: str, default: Any = None) -> Any:
        """Read ``key`` from a dict or an attribute-bearing object."""
        if raw is None:
            return default
        if isinstance(raw, dict):
            return raw.get(key, default)
        return getattr(raw, key, default)

    @staticmethod
    def _to_floats(value: Any) -> list[float]:
        """Flatten any numeric container (list, tuple, numpy array) to
        ``list[float]``. Returns ``[]`` on unparseable input."""
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return []

    @classmethod
    def _normalize_coordinate(cls, value: Any) -> list[int]:
        """Return a 4-int ``[xmin, ymin, xmax, ymax]`` from any of:

        * ``[xmin, ymin, xmax, ymax]`` (4 values) — used verbatim.
        * a flat polygon ``[x0, y0, x1, y1, ...]`` (>=4 values) — bbox derived.
        * a short/empty/missing value — ``[0, 0, 0, 0]``.

        Out-of-order min/max are swapped so the box is always well-formed.
        """
        nums = cls._to_floats(value)
        if len(nums) < 4:
            return [0, 0, 0, 0]
        if len(nums) == 4:
            xmin, ymin, xmax, ymax = nums
        else:
            # Flat polygon → bounding box from the interleaved points.
            xs = nums[0::2]
            ys = nums[1::2]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
        return [
            int(round(min(xmin, xmax))),
            int(round(min(ymin, ymax))),
            int(round(max(xmin, xmax))),
            int(round(max(ymin, ymax))),
        ]

    @classmethod
    def _normalize_polygon(cls, value: Any) -> list[list[float]] | None:
        """Return a list of ``[x, y]`` pairs from a flat ``[x0,y0,x1,y1,...]``
        or a nested ``[[x,y], ...]`` container. Returns ``None`` for
        malformed / too-small shapes. A trailing closing point that duplicates
        the first is dropped."""
        pairs: list[list[float]] | None = None

        # Case 1: flat [x0, y0, x1, y1, ...]
        flat = cls._to_floats(value)
        if len(flat) >= 4 and len(flat) % 2 == 0:
            pairs = [
                [flat[i], flat[i + 1]] for i in range(0, len(flat), 2)
            ]

        # Case 2: nested [[x, y], ...] (flat parse of the container failed
        # because each element is itself a sequence).
        if pairs is None and value is not None and not isinstance(
            value, (int, float)
        ):
            try:
                pairs = [[float(p[0]), float(p[1])] for p in value]
            except (TypeError, ValueError, IndexError):
                pairs = None

        if pairs is None or len(pairs) < cls.min_polygon_points:
            return None

        # Drop a trailing duplicate of the first point (closed loop).
        if len(pairs) > 1 and pairs[-1] == pairs[0]:
            pairs = pairs[:-1]
        return pairs or None


class PaddleXDocLayoutV3Adapter(BoxAdapter):
    """Default adapter for PP-DocLayoutV3's native box dicts.

    The native dict shape matches the canonical fields, so the base accessors
    already apply. This subclass exists to give the primary model a named,
    documented home and a clear template for future adapters.
    """

    model_id = "PP-DocLayoutV3"


class BoxAdapterRegistry:
    """Registry mapping a model id -> :class:`BoxAdapter`.

    Adapters are looked up by the effective model id reported for the
    request. Unknown models fall back to :class:`PaddleXDocLayoutV3Adapter`
    (with a warning) so the wire contract stays stable while future models
    register their own adapters.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BoxAdapter] = {}
        self._default_model: str | None = None

    def register(
        self,
        adapter: type[BoxAdapter],
        *,
        default: bool = False,
        replace: bool = False,
    ) -> None:
        """Register an adapter class under its ``model_id``.

        Args:
            adapter: A :class:`BoxAdapter` subclass with a non-empty
                ``model_id``.
            default: If True, this becomes the fallback adapter for unknown
                model ids.
            replace: Allow re-registering an already-registered id (otherwise
                a ``ValueError`` guards against accidental shadowing).

        Raises:
            TypeError: If ``adapter`` is not a ``BoxAdapter`` subclass.
            ValueError: If ``adapter`` has no ``model_id``, or its id is
                already registered and ``replace=False``.
        """
        if not issubclass(adapter, BoxAdapter):
            raise TypeError(f"{adapter!r} is not a BoxAdapter subclass")
        model_id = getattr(adapter, "model_id", "") or ""
        if not model_id:
            raise ValueError(
                f"{adapter.__name__} must declare a non-empty `model_id`"
            )
        if model_id in self._adapters and not replace:
            raise ValueError(
                f"adapter for model_id={model_id!r} already registered; "
                "pass replace=True to override"
            )
        self._adapters[model_id] = adapter()
        if default or self._default_model is None:
            self._default_model = model_id

    def get(self, model_id: str | None) -> BoxAdapter:
        """Return the adapter for ``model_id``.

        Unknown/None model ids resolve to the default adapter (with a debug
        log) so callers always get a working adapter.
        """
        if model_id and model_id in self._adapters:
            return self._adapters[model_id]
        if model_id:
            logger.debug(
                "No BoxAdapter registered for %r; using default %r",
                model_id, self._default_model,
            )
        assert self._default_model is not None, "registry has no default adapter"
        return self._adapters[self._default_model]

    def normalize(
        self, raws: list[RawBox], model_id: str | None = None
    ) -> list[LayoutBox]:
        """Convenience: resolve the adapter and normalize a whole list."""
        return self.get(model_id).to_boxes(raws)

    @property
    def available(self) -> list[str]:
        """Ids of all registered adapters."""
        return list(self._adapters.keys())

    @property
    def default_model(self) -> str | None:
        """Id of the fallback adapter for unknown models."""
        return self._default_model


# Global registry, populated with the built-in adapter at import time.
# Future models call ``box_adapter_registry.register(MyAdapter, default=...)``.
box_adapter_registry = BoxAdapterRegistry()
box_adapter_registry.register(PaddleXDocLayoutV3Adapter, default=True)