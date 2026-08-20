"""Business logic for the Mistral OCR layer.

Orchestrates: resolve the ``document`` → bytes → decode image → layout
detection (via the shared ``_core.inference`` backend) → convert to a
``Mistral OCRResponse``.

CPU-bound stages (image decode) are offloaded to a thread pool so they never
block the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .._core.config import CPU_POOL_SIZE, MODEL_NAME, MODEL_THRESHOLD
from .._core.image import fetch_image_from_url, load_image_from_bytes
from .._core.inference import run_layout_detection
from .converter import ConvertOptions, box_adapter_registry
from .converter.service import converter
from .schema import (
    DEFAULT_INCLUDE_PADDLEX_METADATA,
    OCRRequest,
    OCRResponse,
    OCRUsageInfo,
    PaddleXMetadata,
    PaddleXOCRResponse,
    as_optional,
)

logger = logging.getLogger("hps_api")

# Dedicated pool for CPU-bound image decode (mirrors docling_api/service.py).
_cpu_pool = ThreadPoolExecutor(max_workers=CPU_POOL_SIZE, thread_name_prefix="hps-ocr")


def resolve_model(model: str | None) -> str:
    """Return the effective backend model for a request.

    Today there is a single configured model, so any non-empty model id is
    accepted and echoed back (never rejected).  When multiple models/backends
    exist this becomes a registry lookup and ``run_layout_detection`` gains a
    backend selector.
    """
    configured = MODEL_NAME
    if model and model != configured:
        logger.warning("model=%r requested, serving %r", model, configured)
    return configured


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Pydantic object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _document_url(document: Any) -> str | None:
    """Extract the URL from a ``document_url``/``image_url`` chunk."""
    dtype = _field(document, "type")
    if dtype == "document_url":
        return _field(document, "document_url")
    if dtype == "image_url":
        raw = _field(document, "image_url")
        if isinstance(raw, dict):
            return raw.get("url")
        return raw
    return None


async def _resolve_image_bytes(document: Any) -> bytes:
    """Return raw image bytes for a Mistral document chunk (obj or dict)."""
    dtype = _field(document, "type")

    # URL-based chunks → async HTTP fetch.
    url = _document_url(document)
    if url is not None:
        try:
            return await fetch_image_from_url(url, headers={})
        except Exception as exc:  # noqa: BLE001 - map any fetch failure to 400
            logger.exception("Failed to fetch image from URL %r", url)
            raise ValueError(f"failed to fetch image from URL: {exc}") from exc

    # The official FileChunk DTO has a ``file_id`` that references a file
    # previously uploaded via the /v1/files store — this server has NO file
    # store, so a file reference cannot be resolved here.
    if dtype == "file":
        raise ValueError(
            "type=file is not supported (no file upload store); "
            "use a document_url or image_url chunk"
        )

    raise ValueError(f"unsupported document type: {dtype!r}")


async def process_ocr(request: OCRRequest) -> OCRResponse:
    """Run the full OCR pipeline and return a Mistral-compatible response."""
    model = resolve_model(request.model)
    image_bytes = await _resolve_image_bytes(request.document)

    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(_cpu_pool, load_image_from_bytes, image_bytes)

    # Resolve the effective per-request threshold (falls back to the
    # deployment default) and the PaddleX-native layout knobs. The stock
    # mistralai DTOs use the ``Unset()`` sentinel for unset optional fields,
    # so every read goes through ``as_optional`` to normalise it to ``None``.
    threshold = as_optional(getattr(request, "threshold", None))
    if threshold is None:
        threshold = MODEL_THRESHOLD
    native = bool(
        as_optional(
            getattr(request, "include_paddlex_metadata", DEFAULT_INCLUDE_PADDLEX_METADATA)
        )
    )
    include_scores = native or bool(
        as_optional(getattr(request, "confidence_scores_granularity", None))
    )

    # Thread layout post-processing knobs through to the backend.
    layout_kwargs = {
        "layout_nms": as_optional(getattr(request, "layout_nms", None)),
        "layout_unclip_ratio": as_optional(
            getattr(request, "layout_unclip_ratio", None)
        ),
        "layout_merge_bboxes_mode": as_optional(
            getattr(request, "layout_merge_bboxes_mode", None)
        ),
        "layout_shape_mode": as_optional(getattr(request, "layout_shape_mode", None)),
        "filter_overlap_boxes": as_optional(
            getattr(request, "filter_overlap_boxes", None)
        ),
    }
    layout_kwargs = {k: v for k, v in layout_kwargs.items() if v is not None}

    boxes = await run_layout_detection(image, threshold=threshold, **layout_kwargs)

    # Normalize the backend's raw box dicts into canonical LayoutBox objects
    # via the per-model adapter. The emitter never sees model-specific shapes,
    # so a future model only needs to register its own BoxAdapter.
    adapter = box_adapter_registry.get(model)
    canonical_boxes = adapter.to_boxes(boxes)

    include_blocks = bool(request.include_blocks)
    native_labels = bool(
        as_optional(getattr(request, "include_native_labels", None))
    )
    options = ConvertOptions(
        page_no=0,
        include_blocks=include_blocks,
        include_scores=include_scores,
        include_native_metadata=native,
        include_native_labels=native_labels,
        threshold=threshold if native else None,
    )
    # Offload the box→Mistral-block conversion to the CPU pool (mirrors the
    # docling layer). Keeps the asyncio event loop free to accept/fetch other
    # requests while this request builds its block objects, which matters
    # under single-worker concurrency.
    page = await loop.run_in_executor(
        _cpu_pool,
        functools.partial(
            converter.convert,
            canonical_boxes,
            image,
            options,
        ),
    )

    usage_info = OCRUsageInfo(
        pages_processed=1,
        doc_size_bytes=len(image_bytes),
    )

    # If native metadata was requested, attach the top-level ``paddlex``
    # container. The extended response class adds it without breaking the
    # stock Mistral serialization; stock clients ignore it.
    if native:
        page_metadata = (
            [page._paddlex_page_metadata]
            if hasattr(page, "_paddlex_page_metadata")
            else []
        )
        paddlex = PaddleXMetadata(
            model=model,
            threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
            layout_nms=layout_kwargs.get("layout_nms"),
            layout_shape_mode=layout_kwargs.get("layout_shape_mode"),
            confidence_scores_granularity=as_optional(
                getattr(request, "confidence_scores_granularity", None)
            ),
            pages=page_metadata,
        )
        return PaddleXOCRResponse(
            model=model,
            pages=[page],
            usage_info=usage_info,
            paddlex=paddlex,
        )

    return OCRResponse(
        model=model,
        pages=[page],
        usage_info=usage_info,
    )