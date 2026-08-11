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
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response

from .._core.image import fetch_image_from_url
from .schema import (
    BatchConvertSourcesRequest,
    ConversionStatus,
    ConvertDocumentResponse,
    ConvertSourcesRequest,
    InBodyTarget,
    OutputFormat,
)
from .service import (
    ERR_DECODE,
    ERR_FETCH,
    convert_image,
    error_response,
    render_debug_image,
)
from .task_store import task_store

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


# Target types that are NOT supported by PaddleX HPS.
# The official ``DoclingServiceClient`` first tries ``presigned_url`` and
# falls back to ``inbody`` only when the server rejects with 400/422 whose
# detail mentions ``presigned_url`` and a validation-style phrase.
# See ``DoclingServiceClient._should_fallback_from_presigned_target``.
_UNSUPPORTED_TARGET_KINDS = {"presigned_url", "zip", "s3", "azure_blob", "gcs", "gdrive"}


def _reject_unsupported_target(target_kind: str | None) -> None:
    """Raise 422 if *target_kind* is not ``inbody``.

    The error detail includes ``presigned_url`` and ``validation error`` so
    the official SDK's fallback heuristic triggers and it retries with
    ``target_type=inbody``.
    """
    if target_kind is None or target_kind == "inbody":
        return
    raise HTTPException(
        status_code=422,
        detail=f"Validation error: target_type '{target_kind}' is not supported. "
        f"presigned_url and other storage targets require artifact storage to be "
        f"configured. Use 'inbody' instead.",
    )


@router.post(
    "/v1/convert/source",
    response_model=ConvertDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def convert_source(
    request: ConvertSourcesRequest,
    debug: bool = Query(
        default=False,
        description=(
            "Debug mode: return the image with detected bounding boxes "
            "drawn (PNG) instead of a conversion response."
        ),
    ),
):
    """Convert documents from HTTP URLs or base64-encoded file sources.

    Note: Only the first source is processed. The API accepts a list for
    forward-compatibility with batch processing.

    S3/Azure/GCS/GoogleDrive source kinds are accepted in the schema for
    compatibility but return 501 Not Implemented.

    When ``debug=true``, runs layout detection and returns the annotated
    image (PNG) instead of the Docling conversion.
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

    if debug:
        png = await render_debug_image(image_data, filename)
        return Response(
            content=png,
            media_type="image/png",
            headers={"X-Debug-Model": "PP-DocLayoutV3"},
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
    debug: bool = Form(default=False),  # noqa: B008 - debug: return annotated PNG
):
    """Convert an uploaded file (multipart/form-data).

    Accepts ``to_formats`` as repeated multipart form fields — matching
    upstream docling-serve's ``FormDepends(ConvertDocumentsOptions)``
    pattern.  Upstream keeps ``list[OutputFormat]`` as a list type (not
    JSON-serialized), so clients send repeated form fields:

        to_formats=md&to_formats=json

    FastAPI collects these into a ``list[str]``.  Defaults to
    ``[MARKDOWN]`` when omitted.  When ``debug=1``, returns the annotated
    image (PNG) instead of the conversion response.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    if debug:
        image_data = await file.read()
        png = await render_debug_image(image_data, file.filename)
        return Response(
            content=png,
            media_type="image/png",
            headers={"X-Debug-Filename": "true"},
        )

    formats = _parse_to_formats(to_formats)
    image_data = await file.read()
    return await convert_image(image_data, file.filename, formats)


@router.post(
    "/v1/convert/file/async",
    status_code=status.HTTP_200_OK,
)
async def convert_file_async(
    files: UploadFile = File(..., alias="files"),  # noqa: B008 - client SDK sends 'files' (plural)
    to_formats: list[str] | None = Form(default=None),  # noqa: B008
    target_type: str | None = Form(default=None),  # noqa: ARG001, B008 - ignored, client sends it
):
    """Async file conversion — submit a task, return task ID for polling.

    The official ``DoclingServiceClient`` always uses this async flow:
    submit → poll → result.  PaddleX HPS performs the conversion
    synchronously during submission and caches the result for retrieval
    via ``/v1/result/{task_id}``.

    The client SDK sends the file under the ``files`` field name (plural),
    not ``file`` — we use ``alias="files"`` to accept it.
    """
    _reject_unsupported_target(target_type)

    if not files.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    formats = _parse_to_formats(to_formats)
    image_data = await files.read()
    response = await convert_image(image_data, files.filename, formats)

    task_id = await task_store.create(
        status=response.status,
        response=response,
        error_message=response.errors[0].error_message if response.errors else None,
    )
    status_response = await task_store.get_status(task_id)
    logger.info(
        "Async file conversion submitted: task_id=%s, status=%s",
        task_id,
        status_response.task_status,
    )
    return status_response


@router.post(
    "/v1/convert/source/async",
    status_code=status.HTTP_200_OK,
)
async def convert_source_async(request: ConvertSourcesRequest):
    """Async source conversion — submit a task, return task ID for polling.

    Same as ``/v1/convert/file/async`` but accepts a JSON body with
    HTTP or base64-encoded file sources (same schema as ``/v1/convert/source``).
    """
    _reject_unsupported_target(request.target.kind if request.target else None)

    to_formats = request.options.to_formats
    source = request.sources[0]

    if source.kind == "http":
        filename = source.url.path.rsplit("/", 1)[-1] or "document"
        try:
            image_data = await fetch_image_from_url(str(source.url), source.headers)
        except Exception as e:
            logger.exception("URL fetch failed")
            response = error_response(ERR_FETCH, str(e), filename)
            task_id = await task_store.create(
                status=ConversionStatus.FAILURE,
                response=response,
                error_message=str(e),
            )
            return await task_store.get_status(task_id)
    elif source.kind == "file":
        try:
            image_data = base64.b64decode(source.base64_string)
        except Exception as e:
            logger.exception("Base64 decode failed")
            response = error_response(ERR_DECODE, str(e), source.filename)
            task_id = await task_store.create(
                status=ConversionStatus.FAILURE,
                response=response,
                error_message=str(e),
            )
            return await task_store.get_status(task_id)
        filename = source.filename
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Source kind '{source.kind}' is not supported by PaddleX HPS. "
            f"Only 'http' and 'file' sources are accepted.",
        )

    response = await convert_image(image_data, filename, to_formats)
    task_id = await task_store.create(
        status=response.status,
        response=response,
        error_message=response.errors[0].error_message if response.errors else None,
    )
    status_response = await task_store.get_status(task_id)
    logger.info(
        "Async source conversion submitted: task_id=%s, status=%s",
        task_id,
        status_response.task_status,
    )
    return status_response


@router.get(
    "/v1/status/poll/{task_id}",
    status_code=status.HTTP_200_OK,
)
async def poll_task_status(
    task_id: str,
    wait: float = Query(default=0.0, ge=0.0),  # noqa: ARG001 - server-side wait ignored (task already complete)
):
    """Poll the status of an async conversion task.

    The official client calls this with ``wait=N`` for server-side
    long-polling.  Since PaddleX HPS completes conversion synchronously
    during submission, the task is already in a terminal state by the
    time this endpoint is called — the ``wait`` parameter is accepted
    but ignored.
    """
    try:
        return await task_store.get_status(task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found.",
        ) from None


@router.get(
    "/v1/result/{task_id}",
    response_model=ConvertDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_task_result(task_id: str):
    """Retrieve the conversion result for a completed task."""
    try:
        return await task_store.get_result(task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found.",
        ) from None


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
