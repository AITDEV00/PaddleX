"""HTTP route handlers for the Mistral OCR layer.

Thin router: parse the request, call ``service.process_ocr``, return the
official ``OCRResponse``.  No business logic here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from .._core.inference import state
from .schema import OCRRequest, OCRResponse
from .service import process_ocr

logger = logging.getLogger("hps_api")

router = APIRouter(tags=["ocr"])


@router.post("/v1/ocr", response_model=OCRResponse)
async def ocr(request: OCRRequest) -> OCRResponse:
    """Run OCR/layout detection on a document and return a Mistral response."""
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not ready",
        )
    try:
        return await process_ocr(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc