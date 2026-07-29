"""Docling-serve API schema — thin re-export layer.

All request/response models, enums, source/target models, and discriminated
unions are imported directly from the ``docling`` package (docling-slim 2.114.0).
This guarantees 1:1 compatibility with upstream docling-serve by construction.

The only custom symbols are:
  - ``ImageRefMode``       — re-exported from ``docling_core.types.doc``
  - ``SourceRequestItem``  — alias matching the local-only source subset
  - ``ExportDocumentResponse`` — re-exported from docling.service.responses

If upstream adds/removes/renames a field, we get it automatically on the
next ``pip install --upgrade docling-slim`` — no manual sync required.
"""

from __future__ import annotations

# ─── Enums (base_models) ──────────────────────────────────────────────────────
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    InputFormat,
    OutputFormat,
    QualityGrade,
)

# ─── Enums (pipeline_options) ─────────────────────────────────────────────────
from docling.datamodel.pipeline_options import (
    PdfBackend,
    ProcessingPipeline,
    TableFormerMode,
)

# ─── Service: callbacks ───────────────────────────────────────────────────────
from docling.datamodel.service.callbacks import CallbackSpec

# ─── Service: options ─────────────────────────────────────────────────────────
from docling.datamodel.service.options import ConvertDocumentsOptions

# ─── Service: source request models + discriminated unions ────────────────────
# BatchTargetRequest lives in requests too (not targets)
# ─── Local alias ──────────────────────────────────────────────────────────────
#
# Upstream's ``SourceRequestItem`` union only includes ``FileSourceRequest``
# and ``HttpSourceRequest`` — which is exactly the subset PaddleX HPS supports.
# We re-export it under the same name so downstream code doesn't change.
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    AzureBlobSourceRequest,
    BatchConvertSourcesRequest,
    BatchSourceRequestItem,
    BatchTargetRequest,
    ConvertSourcesRequest,
    FileSourceRequest,
    GoogleCloudStorageSourceRequest,
    GoogleDriveSourceRequest,
    HttpSourceRequest,
    S3SourceRequest,
    SourceRequestItem,
    TargetName,
    TargetRequest,
)

# ─── Service: responses ───────────────────────────────────────────────────────
from docling.datamodel.service.responses import (
    ConfidenceScores,
    ConvertDocumentErrorResponse,
    ConvertDocumentResponse,
    ExportDocumentResponse,
)

# ─── Service: sources (coordinate models) ─────────────────────────────────────
from docling.datamodel.service.sources import (
    AzureBlobCoordinates,
    GoogleCloudStorageCoordinates,
    GoogleDriveCoordinates,
    S3Coordinates,
)

# ─── Service: targets ─────────────────────────────────────────────────────────
from docling.datamodel.service.targets import (
    AzureBlobTarget,
    GoogleCloudStorageTarget,
    GoogleDriveTarget,
    InBodyTarget,
    PresignedUrlTarget,
    PutTarget,
    S3Target,
    ZipTarget,
)

# ─── Settings ─────────────────────────────────────────────────────────────────
from docling.datamodel.settings import (
    DEFAULT_PAGE_RANGE,
    DocumentLimits,
    PageRange,
)

# ─── Profiling ────────────────────────────────────────────────────────────────
from docling.utils.profiling import (
    ProfilingItem,
    ProfilingScope,
)

# ─── Re-export from docling_core ──────────────────────────────────────────────
from docling_core.types.doc import ImageRefMode

__all__ = [
    # base_models enums
    "ConversionStatus",
    "DoclingComponentType",
    "ErrorItem",
    "FailureCategory",
    "InputFormat",
    "OutputFormat",
    "QualityGrade",
    # pipeline_options enums
    "PdfBackend",
    "ProcessingPipeline",
    "TableFormerMode",
    # settings
    "DEFAULT_PAGE_RANGE",
    "DocumentLimits",
    "PageRange",
    # profiling
    "ProfilingItem",
    "ProfilingScope",
    # service options
    "ConvertDocumentsOptions",
    # service sources
    "AzureBlobCoordinates",
    "GoogleCloudStorageCoordinates",
    "GoogleDriveCoordinates",
    "S3Coordinates",
    # service source requests + unions
    "AnyHttpSourceRequest",
    "AzureBlobSourceRequest",
    "BatchConvertSourcesRequest",
    "BatchSourceRequestItem",
    "ConvertSourcesRequest",
    "FileSourceRequest",
    "GoogleCloudStorageSourceRequest",
    "GoogleDriveSourceRequest",
    "HttpSourceRequest",
    "S3SourceRequest",
    "SourceRequestItem",
    "TargetName",
    # service targets
    "AzureBlobTarget",
    "BatchTargetRequest",
    "GoogleCloudStorageTarget",
    "GoogleDriveTarget",
    "InBodyTarget",
    "PresignedUrlTarget",
    "PutTarget",
    "S3Target",
    "TargetRequest",
    "ZipTarget",
    # service callbacks
    "CallbackSpec",
    # service responses
    "ConfidenceScores",
    "ConvertDocumentErrorResponse",
    "ConvertDocumentResponse",
    "ExportDocumentResponse",
    # docling_core
    "ImageRefMode",
]
