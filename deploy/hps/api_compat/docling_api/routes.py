"""HTTP route handlers for the Docling convert endpoints.

Uses APIRouter so routes are self-contained and testable without creating
the full FastAPI app.
"""

from __future__ import annotations

import base64
import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from .._core.image import fetch_image_from_url
from .schema import (
    BatchConvertSourcesRequest,
    ConvertDocumentResponse,
    ConvertSourcesRequest,
    OutputFormat,
)
from .service import (
    ERR_DECODE,
    ERR_FETCH,
    convert_image,
    error_response,
)

logger = logging.getLogger("hps_api")

router = APIRouter(tags=["convert"])


def _parse_to_formats(to_formats: list[str] | None) -> list[OutputFormat]:
    """Parse ``to_formats`` form values into a list of OutputFormat enums.

    Upstream docling-serve's ``FormDepends(ConvertDocumentsOptions)`` keeps
    the ``to_formats`` field (``list[OutputFormat]``) as a **list type** — it
    is NOT JSON-serialized (``is_json_field`` only handles ``dict`` origin,
    not ``list``).  Clients like httpx therefore send repeated form fields:

        to_formats=md&to_formats=json

    FastAPI receives these as ``list[str]`` when the parameter is declared
    ``list[str] | None = Form(default=None)``.  Each element may itself be a
    comma-separated string (defensive parsing for simple clients that send a
    single ``to_formats=md,json`` field).

    Raises HTTPException(400) on invalid format value.
    Defaults to ``[MARKDOWN]`` when *to_formats* is None or empty.
    """
    if not to_formats:
        return [OutputFormat.MARKDOWN]

    # Flatten: each element may contain comma-separated values
    raw_values: list[str] = []
    for item in to_formats:
        if not item:
            continue
        raw_values.extend(
            part.strip() for part in item.split(",") if part.strip()
        )

    if not raw_values:
        return [OutputFormat.MARKDOWN]

    try:
        return [OutputFormat(v) for v in raw_values]
    except ValueError:
        valid = ", ".join(f.value for f in OutputFormat)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid to_formats. Valid options: {valid}",
        ) from None


@router.post(
    "/v1/convert/source",
    response_model=ConvertDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def convert_source(request: ConvertSourcesRequest):
    """Convert documents from HTTP URLs or base64-encoded file sources.

    Note: Only the first source is processed. The API accepts a list for
    forward-compatibility with batch processing.

    S3/Azure/GCS/GoogleDrive source kinds are accepted in the schema for
    compatibility but return 501 Not Implemented.
    """
    to_formats = request.options.to_formats
    source = request.sources[0]

    if source.kind == "http":
        filename = source.url.path.rsplit("/", 1)[-1] or "document"
        try:
            image_data = await fetch_image_from_url(str(source.url), source.headers)
        except Exception as e:
            logger.exception("URL fetch failed")
            return error_response(ERR_FETCH, str(e), filename)
    elif source.kind == "file":
        try:
            image_data = base64.b64decode(source.base64_string)
        except Exception as e:
            logger.exception("Base64 decode failed")
            return error_response(ERR_DECODE, str(e), source.filename)
        filename = source.filename
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Source kind '{source.kind}' is not supported by PaddleX HPS. "
            f"Only 'http' and 'file' sources are accepted.",
        )

    return await convert_image(image_data, filename, to_formats)


@router.post(
    "/v1/convert/file",
    response_model=ConvertDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def convert_file(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI requires File() as default
    to_formats: list[str] | None = Form(default=None),  # noqa: B008
):
    """Convert an uploaded file (multipart/form-data).

    Accepts ``to_formats`` as repeated multipart form fields — matching
    upstream docling-serve's ``FormDepends(ConvertDocumentsOptions)``
    pattern.  Upstream keeps ``list[OutputFormat]`` as a list type (not
    JSON-serialized), so clients send repeated form fields:

        to_formats=md&to_formats=json

    FastAPI collects these into a ``list[str]``.  Defaults to
    ``[MARKDOWN]`` when omitted.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    formats = _parse_to_formats(to_formats)
    image_data = await file.read()
    return await convert_image(image_data, file.filename, formats)


@router.post(
    "/v1/convert/source/async",
    status_code=status.HTTP_202_ACCEPTED,
)
async def convert_source_async(request: ConvertSourcesRequest):  # noqa: ARG001
    """Async conversion — accepted for API compatibility, returns 501.

    PaddleX HPS currently only supports synchronous conversion.
    """
    raise HTTPException(
        status_code=501,
        detail="Async conversion is not supported by PaddleX HPS. "
        "Use /v1/convert/source for synchronous conversion.",
    )


@router.post(
    "/v1/convert/file/async",
    status_code=status.HTTP_202_ACCEPTED,
)
async def convert_file_async(
    file: UploadFile = File(...),  # noqa: B008, ARG001 - FastAPI requires File() as default
    to_formats: list[str] | None = None,  # noqa: ARG001
):
    """Async file conversion — accepted for API compatibility, returns 501."""
    raise HTTPException(
        status_code=501,
        detail="Async conversion is not supported by PaddleX HPS. "
        "Use /v1/convert/file for synchronous conversion.",
    )


@router.post(
    "/v1/convert/source/batch",
    status_code=status.HTTP_200_OK,
)
async def convert_source_batch(request: BatchConvertSourcesRequest):  # noqa: ARG001
    """Batch conversion — accepted for API compatibility, returns 501.

    PaddleX HPS currently only supports single-document synchronous conversion.
    """
    raise HTTPException(
        status_code=501,
        detail="Batch conversion is not supported by PaddleX HPS. "
        "Use /v1/convert/source for single-document conversion.",
    )
