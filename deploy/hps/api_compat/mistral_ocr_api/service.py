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

from .._core.config import CPU_POOL_SIZE, MODEL_NAME
from .._core.image import fetch_image_from_url, load_image_from_bytes
from .._core.inference import run_layout_detection
from .converter.service import converter
from .schema import OCRRequest, OCRResponse, OCRUsageInfo

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


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key from a Pydantic object or plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _document_url(document: Any) -> str | None:
    """Extract the URL from a ``document_url``/``image_url`` chunk."""
    dtype = _get_attr(document, "type")
    if dtype == "document_url":
        return _get_attr(document, "document_url")
    if dtype == "image_url":
        raw = _get_attr(document, "image_url")
        if isinstance(raw, dict):
            return raw.get("url")
        return raw
    return None


async def _resolve_image_bytes(document: Any) -> bytes:
    """Return raw image bytes for a Mistral document chunk (obj or dict)."""
    dtype = _get_attr(document, "type")

    # URL-based chunks → async HTTP fetch.
    url = _document_url(document)
    if url is not None:
        return await fetch_image_from_url(url, headers={})

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

    boxes = await run_layout_detection(image)

    include_blocks = bool(request.include_blocks)
    # Offload the box→Mistral-block conversion to the CPU pool (mirrors the
    # docling layer). Keeps the asyncio event loop free to accept/fetch other
    # requests while this request builds its block objects, which matters
    # under single-worker concurrency.
    page = await loop.run_in_executor(
        _cpu_pool,
        functools.partial(
            converter.convert,
            boxes,
            image,
            page_no=0,
            include_blocks=include_blocks,
        ),
    )

    return OCRResponse(
        model=model,
        pages=[page],
        usage_info=OCRUsageInfo(
            pages_processed=1,
            doc_size_bytes=len(image_bytes),
        ),
    )