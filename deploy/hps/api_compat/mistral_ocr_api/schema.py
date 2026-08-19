"""Request/response models for the Mistral OCR layer.

Re-exports the official ``mistralai`` generated DTOs so we never drift from
the wire contract that Mistral's own client produces.  Do NOT hand-write these
classes.

The import is gated: if ``mistralai`` is not installed, importing this module
raises a clear ``ImportError``.  The master app turns that into a startup
error only when the ``mistral`` layer is explicitly enabled.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on optional dependency
    from mistralai.client import models
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mistral' API layer requires the 'mistralai' package. Install it "
        "with `pip install mistralai` (see deploy/hps/requirements.api_compat.txt)."
    ) from exc

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
]