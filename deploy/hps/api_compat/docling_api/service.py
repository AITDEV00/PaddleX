"""Conversion pipeline — orchestrates layout detection → DoclingDocument → export.

This is the core business logic for the Docling convert endpoints. It ties together:
  - Image loading (_core.image)
  - Layout inference (_core.inference)
  - DoclingDocument conversion (converter.service)
  - Multi-format export
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from docling_core.types.doc import DoclingDocument

from .._core.config import CPU_POOL_SIZE
from .._core.image import load_image_from_bytes, to_rgb
from .._core.inference import run_layout_detection
from .._core.latency import LatencyTracer
from .converter.service import PaddleXToDoclingConverter
from .schema import (
    ConversionStatus,
    ConvertDocumentResponse,
    DoclingComponentType,
    ErrorItem,
    ExportDocumentResponse,
    FailureCategory,
    OutputFormat,
    ProfilingItem,
    ProfilingScope,
)

logger = logging.getLogger("hps_api")

# Dedicated thread pool for CPU-bound work (image decode, DoclingDocument
# conversion, format export).  This prevents one request's CPU work from
# blocking the asyncio event loop and stalling all other requests.
_cpu_pool = ThreadPoolExecutor(max_workers=CPU_POOL_SIZE, thread_name_prefix="hps-cpu")

# ─── Error category constants ──────────────────────────────────────────────────
# Centralized so error categories are never typo'd across files.
# Maps our error scenarios to docling's FailureCategory enum values.
ERR_IMAGE_LOAD = FailureCategory.INFERENCE_FAILURE
ERR_LAYOUT_DETECTION = FailureCategory.INFERENCE_FAILURE
ERR_FETCH = FailureCategory.SOURCE_UNAVAILABLE
ERR_DECODE = FailureCategory.INTERNAL

# Stateless converter — safe to share across requests
_converter = PaddleXToDoclingConverter()


@dataclass
class ExportResult:
    """Container for multi-format export output."""
    md_content: str | None = None
    json_content: DoclingDocument | None = None
    html_content: str | None = None
    text_content: str | None = None
    doctags_content: str | None = None
    doclang_content: str | None = None
    errors: list[ErrorItem] = field(default_factory=list)


def error_response(
    error_category: FailureCategory,
    error_message: str,
    filename: str = "unknown",
    processing_time: float = 0.0,
) -> ConvertDocumentResponse:
    """Build a failure response with a single error item."""
    return ConvertDocumentResponse(
        document=ExportDocumentResponse(filename=filename),
        status=ConversionStatus.FAILURE,
        errors=[ErrorItem(
            component_type=DoclingComponentType.MODEL,
            module_name="layout",
            error_message=error_message,
            category=error_category,
        )],
        processing_time=processing_time,
    )


async def convert_image(
    image_data: bytes,
    filename: str,
    to_formats: list[OutputFormat],
) -> ConvertDocumentResponse:
    """Full conversion pipeline: load image → layout detection → build document.

    CPU-bound stages (image decode, DoclingDocument conversion, format
    export) are offloaded to a dedicated thread pool so they never block
    the asyncio event loop.  This allows other requests to proceed
    concurrently even while this request does heavy CPU work.

    The export and confidence-computation stages run in parallel since
    they are independent of each other.

    Args:
        image_data: Raw image bytes (PNG, JPEG, etc.)
        filename: Original filename for the response.
        to_formats: List of desired output formats.

    Returns:
        A ConvertDocumentResponse with the converted content and metadata.
    """
    t0 = time.perf_counter()
    fmt_strs = [str(f) for f in to_formats]
    tracer = LatencyTracer(filename=filename, to_formats=fmt_strs)
    loop = asyncio.get_running_loop()

    try:
        with tracer.stage("image_load"):
            # cv2.imdecode + resize is fast (~2ms for typical images).
            # run_in_executor overhead (thread scheduling + context switch)
            # adds ~0.2-0.5ms — not worth it for such a quick operation.
            # Run inline on the event loop.
            image = load_image_from_bytes(image_data)
    except Exception as e:
        logger.exception("Failed to load image")
        elapsed = time.perf_counter() - t0
        tracer.finish()
        return error_response(ERR_IMAGE_LOAD, str(e), filename, elapsed)

    try:
        with tracer.stage("layout_detection"):
            boxes = await run_layout_detection(image)
    except Exception as e:
        logger.exception("Layout detection failed")
        elapsed = time.perf_counter() - t0
        tracer.finish()
        return error_response(ERR_LAYOUT_DETECTION, str(e), filename, elapsed)

    layout_time = time.perf_counter() - t0
    tracer.record("layout_detection_total", layout_time)

    if not boxes:
        logger.warning("Layout detection returned 0 boxes for %s", filename)

    with tracer.stage("convert_to_docling"):
        # Offload DoclingDocument construction to thread pool (CPU-bound)
        # Convert BGR→RGB only here — the converter needs RGB for page
        # metadata, but layout detection works on BGR (PaddleX handles it).
        image_rgb = to_rgb(image)
        doc = await loop.run_in_executor(
            _cpu_pool,
            functools.partial(
                _converter.convert,
                boxes=boxes, image=image_rgb, filename=filename, page_no=1,
            ),
        )

    # Export and confidence are independent — run them concurrently
    with tracer.stage("export_formats"):
        export_task = loop.run_in_executor(
            _cpu_pool, functools.partial(_export_to_formats, doc, to_formats),
        )
        confidence_task = loop.run_in_executor(
            _cpu_pool, functools.partial(_converter.compute_confidence, boxes),
        )
        result, confidence = await asyncio.gather(export_task, confidence_task)

    tracer.finish()

    if not result.errors:
        status = ConversionStatus.SUCCESS
    else:
        status = ConversionStatus.PARTIAL_SUCCESS

    return ConvertDocumentResponse(
        document=ExportDocumentResponse(
            filename=filename,
            md_content=result.md_content,
            json_content=result.json_content,
            html_content=result.html_content,
            text_content=result.text_content,
            doctags_content=result.doctags_content,
            doclang_content=result.doclang_content,
        ),
        status=status,
        errors=result.errors,
        processing_time=time.perf_counter() - t0,
        timings={"layout": ProfilingItem(
            scope=ProfilingScope.DOCUMENT,
            count=1,
            times=[layout_time],
        )},
        confidence=confidence,
    )


def _call_optional_export(doc: DoclingDocument, method_name: str) -> str | None:
    """Call a DoclingDocument export method if it exists, else return None.

    Logs a warning if the method is missing — this usually indicates a
    version mismatch between the code and the installed docling-core.
    """
    method = getattr(doc, method_name, None)
    if method is None:
        logger.warning(
            "DoclingDocument has no method '%s' — returning None for this format. "
            "Check docling-core version compatibility.",
            method_name,
        )
        return None
    return method()


# Dispatch table: OutputFormat → exporter function.
# Each exporter takes (doc, cached_md) and returns the exported content.
_EXPORTERS: dict[OutputFormat, Any] = {
    OutputFormat.MARKDOWN: lambda doc, md: md,
    OutputFormat.JSON: lambda doc, md: doc,
    OutputFormat.TEXT: lambda doc, md: _call_optional_export(doc, "export_to_text"),
    OutputFormat.HTML: lambda doc, md: _call_optional_export(doc, "export_to_html"),
    OutputFormat.DOCTAGS: lambda doc, md: _call_optional_export(
        doc, "export_to_doctags",
    ),
    OutputFormat.DOCLANG: lambda doc, md: _call_optional_export(
        doc, "export_to_doclang",
    ),
}


# Maps OutputFormat → ExportResult attribute name for setattr
_FIELD_MAP: dict[OutputFormat, str] = {
    OutputFormat.MARKDOWN: "md_content",
    OutputFormat.JSON: "json_content",
    OutputFormat.TEXT: "text_content",
    OutputFormat.HTML: "html_content",
    OutputFormat.DOCTAGS: "doctags_content",
    OutputFormat.DOCLANG: "doclang_content",
}


def _export_to_formats(
    doc: DoclingDocument,
    to_formats: list[OutputFormat],
) -> ExportResult:
    """Export a DoclingDocument to each requested format."""
    result = ExportResult()

    # Pre-compute markdown once — only the MARKDOWN exporter reuses it
    need_md = OutputFormat.MARKDOWN in to_formats
    cached_md = doc.export_to_markdown() if need_md else None

    for fmt in to_formats:
        exporter = _EXPORTERS.get(fmt)
        if exporter is None:
            result.errors.append(ErrorItem(
                component_type=DoclingComponentType.MODEL,
                module_name="layout",
                error_message=f"Unsupported output format: {fmt}",
                category=FailureCategory.INTERNAL,
            ))
            continue
        try:
            content = exporter(doc, cached_md)
            setattr(result, _FIELD_MAP[fmt], content)
        except Exception as e:
            logger.exception("Failed to export %s", fmt)
            result.errors.append(ErrorItem(
                component_type=DoclingComponentType.MODEL,
                module_name="layout",
                error_message=f"Failed to export {fmt}: {e}",
                category=FailureCategory.INTERNAL,
            ))

    return result
