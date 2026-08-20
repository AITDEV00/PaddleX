"""Request/response models for the Mistral OCR layer.

Re-exports the official ``mistralai`` generated DTOs so we never drift from
the wire contract that Mistral's own client produces.  Do NOT hand-write these
classes.

The import is gated: if ``mistralai`` is not installed, importing this module
raises a clear ``ImportError``.  The master app turns that into a startup
error only when the ``mistral`` layer is explicitly enabled.
"""

from __future__ import annotations

import functools
from typing import Annotated, Any, List, Optional, Union

try:  # pragma: no cover - depends on optional dependency
    from mistralai.client import models
    from pydantic import (
        BaseModel,
        BeforeValidator,
        ConfigDict,
        Field,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mistral' API layer requires the 'mistralai' package. Install it "
        "with `pip install mistralai` (see deploy/hps/requirements.api_compat.txt)."
    ) from exc


def as_optional(value: Any) -> Any:
    """Normalize a pydantic ``Unset()`` sentinel to ``None``.

    The generated ``mistralai`` DTOs use an ``Unset()`` sentinel (not ``None``)
    for optional fields the client did not send. Passing ``Unset()`` downstream
    (e.g. into a ``PaddleXMetadata`` field, or ``bool()``-checking it) is a bug
    source: ``bool(Unset())`` is truthy and ``Unset`` fails pydantic validation.

    The sentinel is detected by its class name rather than importing the
    internal ``mistralai.client.types.basemodel`` module, which is not a
    stable public import surface.
    """
    if value is not None and type(value).__name__ == "Unset":
        return None
    return value

# Top-level request/response
OCRRequest = models.OCRRequest
OCRResponse = models.OCRResponse

# Document union + concrete chunks
DocumentUnion = models.DocumentUnion
DocumentURLChunk = models.DocumentURLChunk
ImageURLChunk = models.ImageURLChunk
FileChunk = models.FileChunk

# Page object
OCRPageObject = models.OCRPageObject
OCRPageDimensions = models.OCRPageDimensions
OCRUsageInfo = models.OCRUsageInfo

# Confidence score DTOs (native Mistral — round-trip through the stock client)
OCRBlockConfidenceScores = models.OCRBlockConfidenceScores
OCRPageConfidenceScores = models.OCRPageConfidenceScores


class PaddleXOCRRequest(OCRRequest):
    """Server-side extension of the official ``OCRRequest``.

    Adds PaddleX-native layout detection knobs that the stock Mistral schema
    has no field for.  These are optional and ignored by clients that only
    send stock fields; when present they are threaded through to the backend.

    .. note::
        The official ``mistralai`` client cannot *send* these (its ``process``
        has a fixed signature). They are intended for the LiteLLM/raw-HTTP
        path, which forwards unknown top-level fields. The stock client
        simply omits them — behaviour is unchanged.
    """

    # Per-request detection threshold (float or per-class dict). Falls back
    # to the deployment default (predictor.threshold = 0.5) when unset.
    threshold: Optional[Any] = Field(default=None)
    # Layout post-processing knobs (mirror PaddleX predictor kwargs).
    layout_nms: Optional[bool] = Field(default=None)
    layout_unclip_ratio: Optional[Any] = Field(default=None)
    layout_merge_bboxes_mode: Optional[Any] = Field(default=None)
    layout_shape_mode: Optional[str] = Field(default=None)
    filter_overlap_boxes: Optional[bool] = Field(default=None)
    # Include the full native PaddleX metadata (label, cls_id, score, order,
    # polygon) on the response under the top-level ``paddlex`` field.
    include_paddlex_metadata: Optional[bool] = Field(default=False)
    # Emit the native PaddleX ``label`` on each block, alongside the stock
    # Mistral ``type``.  ``type`` stays as the Mistral enum (protocol
    # compatible); ``label`` carries the lossy-collapsed native label
    # (e.g. ``chart``/``number``) so clients can route on it without the
    # ``paddlex`` container.
    include_native_labels: Optional[bool] = Field(default=False)


class PaddleXBox(BaseModel):
    """Native PaddleX layout-box metadata, surfaced under ``paddlex``.

    One entry per block, in the same reading order as ``page.blocks``. The
    ``block_index`` links it back to the corresponding Mistral block so
    consumers can align boxes and blocks without re-sorting.
    """

    block_index: int
    label: str
    cls_id: int
    score: float
    order: Optional[int] = None
    coordinate: list[int] = Field(default_factory=list)
    polygon_points: Optional[list[list[float]]] = None


class PaddleXPageMetadata(BaseModel):
    """Per-page native metadata (surfaced under ``paddlex.pages``)."""

    page_index: int
    threshold_used: Optional[float] = None
    boxes: list[PaddleXBox] = Field(default_factory=list)


class PaddleXMetadata(BaseModel):
    """Top-level ``paddlex`` metadata container added to the response.

    Holds data that the stock Mistral block union cannot carry without
    breaking serialization (see feasibility study): the native ``label``,
    ``cls_id``, per-box ``score``, reading ``order`` and ``polygon_points``.
    The native per-box ``score`` is also mirrored onto each block's
    ``confidence_scores.block_type_confidence_score`` so stock clients see it.
    """

    model: str = "PP-DocLayoutV3"
    layout_nms: Optional[bool] = None
    threshold: Optional[float] = None
    layout_shape_mode: Optional[str] = None
    confidence_scores_granularity: Optional[str] = None
    pages: list[PaddleXPageMetadata] = Field(default_factory=list)


class PaddleXOCRResponse(OCRResponse):
    """Extended ``OCRResponse`` that carries a top-level ``paddlex`` container.

    Only present when ``include_paddlex_metadata`` was requested; the stock
    ``OCRResponse`` parses it as an unknown field (ignored) so default
    Mistral clients are unaffected.

    ``pages`` is widened to a union so it can carry either a stock
    ``OCRPageObject`` (default) or a ``PaddleXOCRPageObject`` whose blocks
    additionally carry the native ``label`` (when ``include_native_labels`` is
    requested).  The stock class stays first so strict parsing is unchanged.
    """

    pages: List[Union[OCRPageObject, PaddleXOCRPageObject]] = Field(
        default_factory=list
    )
    paddlex: Optional[PaddleXMetadata] = None


# Whether to include PaddleX-native metadata by default.
DEFAULT_INCLUDE_PADDLEX_METADATA = False

# Concrete block classes (never the `Block` union — build these directly)
OCRTextBlock = models.OCRTextBlock
OCRTitleBlock = models.OCRTitleBlock
OCRTableBlock = models.OCRTableBlock
OCRImageBlock = models.OCRImageBlock
OCRListBlock = models.OCRListBlock
OCREquationBlock = models.OCREquationBlock
OCRCaptionBlock = models.OCRCaptionBlock
OCRCodeBlock = models.OCRCodeBlock
OCRHeaderBlock = models.OCRHeaderBlock
OCRFooterBlock = models.OCRFooterBlock
OCRSignatureBlock = models.OCRSignatureBlock
OCRReferencesBlock = models.OCRReferencesBlock
OCRAsideTextBlock = models.OCRAsideTextBlock


# ═══════════════════════════════════════════════════════════════════════════════
# Native-label block variants (Option A: emit ``label`` alongside ``type``)
#
# The stock Mistral block DTOs cannot carry the native PaddleX label: their
# ``type`` is a fixed ``Literal`` (locked to the Mistral enum), ``model_config``
# has no ``extra``, so an extra ``label`` field is silently dropped.  To expose
# the lossy-collapsed label (``chart``→``image``, ``number``→``text``, …) we
# subclass each block, add a ``label`` field, and re-type ``blocks`` with a
# discriminated union over these labeled variants.
#
# ``type`` stays as the stock Mistral enum value (protocol compatibility);
# ``label`` carries the native PaddleX label.  Stock clients that parse a strict
# ``OCRResponse`` ignore the extra field; clients that request
# ``include_native_labels`` get the native label on every block.
# ═══════════════════════════════════════════════════════════════════════════════

# Block type string → stock Mistral block class.
_NATIVE_LABEL_BLOCK_BASE: dict[str, type[BaseModel]] = {
    "text": OCRTextBlock,
    "title": OCRTitleBlock,
    "table": OCRTableBlock,
    "image": OCRImageBlock,
    "list": OCRListBlock,
    "equation": OCREquationBlock,
    "caption": OCRCaptionBlock,
    "code": OCRCodeBlock,
    "header": OCRHeaderBlock,
    "footer": OCRFooterBlock,
    "signature": OCRSignatureBlock,
    "references": OCRReferencesBlock,
    "aside_text": OCRAsideTextBlock,
}


def _make_labeled_block(base: type[BaseModel]) -> type[BaseModel]:
    """Return a subclass of a Mistral block DTO that adds a ``label`` field."""
    return type(
        f"Labeled{base.__name__}",
        (base,),
        {
            "__annotations__": {"label": Optional[str]},
            "label": Field(default=None, description="Native PaddleX label"),
            "model_config": ConfigDict(extra="allow"),
        },
    )


# Labeled variants keyed by Mistral block type.
LABELED_BLOCK_CLASS: dict[str, type[BaseModel]] = {
    mtype: _make_labeled_block(base)
    for mtype, base in _NATIVE_LABEL_BLOCK_BASE.items()
}

# Discriminated union over the labeled variants (same forward-compatible
# mechanism the stock ``OCRPageObject.blocks`` uses; unknown ``type`` values
# fall back to ``UnknownBlock`` and preserve the raw payload).
try:  # pragma: no cover - depends on mistralai internals
    from mistralai.client.models.ocrpageobject import (  # noqa: PLC0415
        UnknownBlock as _LabeledUnknownBlock,
    )
    from mistralai.client.models.ocrpageobject import (  # noqa: PLC0415
        parse_open_union as _parse_open_union,
    )
except ImportError:  # pragma: no cover - unexpected structure
    _LabeledUnknownBlock = None
    _parse_open_union = None


def _identity_validator(v: Any) -> Any:
    """Pass-through validator for the labeled union.

    Used only if ``mistralai``'s internal ``parse_open_union`` helper cannot be
    imported (an unexpected structure change upstream). In that case we degrade
    to pydantic's default union resolution instead of crashing at import time.
    """
    return v


_LABELED_BLOCK_MEMBERS: list[type] = list(LABELED_BLOCK_CLASS.values())
if _LabeledUnknownBlock is not None:
    _LABELED_BLOCK_MEMBERS.append(_LabeledUnknownBlock)

# Build the union programmatically (PEP 604/star-unpacking in a Union is 3.11+;
# this layer must support Python 3.10).
_LABELED_BLOCK_UNION = Union[tuple(_LABELED_BLOCK_MEMBERS)]  # type: ignore[valid-type]

# If the internal ``parse_open_union`` helper is unavailable, fall back to a
# no-op validator so ``functools.partial(None, ...)`` never raises TypeError at
# import time (that would defeat the guard above).
_union_validator = (
    functools.partial(
        _parse_open_union,
        disc_key="type",
        variants=LABELED_BLOCK_CLASS,
        unknown_cls=_LabeledUnknownBlock,
        union_name="LabeledBlock",
    )
    if _parse_open_union is not None
    else _identity_validator
)

LabeledBlock = Annotated[
    _LABELED_BLOCK_UNION,
    BeforeValidator(_union_validator),
]


class PaddleXOCRPageObject(models.OCRPageObject):
    """``OCRPageObject`` whose ``blocks`` carry the native PaddleX ``label``.

    Used only when ``include_native_labels`` is requested.  ``type`` retains
    the stock Mistral enum value; ``label`` carries the lossy-prevented native
    label.  Default responses use the stock ``OCRPageObject`` unchanged.
    """

    blocks: Optional[List[LabeledBlock]] = Field(default=None)

__all__ = [
    "OCRRequest",
    "OCRResponse",
    "DocumentUnion",
    "DocumentURLChunk",
    "ImageURLChunk",
    "FileChunk",
    "OCRPageObject",
    "OCRPageDimensions",
    "OCRUsageInfo",
    "OCRBlockConfidenceScores",
    "OCRPageConfidenceScores",
    "OCRTextBlock",
    "OCRTitleBlock",
    "OCRTableBlock",
    "OCRImageBlock",
    "OCRListBlock",
    "OCREquationBlock",
    "OCRCaptionBlock",
    "OCRCodeBlock",
    "OCRHeaderBlock",
    "OCRFooterBlock",
    "OCRSignatureBlock",
    "OCRReferencesBlock",
    "OCRAsideTextBlock",
    "PaddleXOCRRequest",
    "PaddleXOCRResponse",
    "PaddleXMetadata",
    "PaddleXPageMetadata",
    "PaddleXBox",
    "LABELED_BLOCK_CLASS",
    "LabeledBlock",
    "PaddleXOCRPageObject",
    "DEFAULT_INCLUDE_PADDLEX_METADATA",
    "as_optional",
]