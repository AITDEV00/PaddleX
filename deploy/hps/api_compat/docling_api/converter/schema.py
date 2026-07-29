"""Converter schema — thin re-export of upstream confidence models.

``QualityGrade`` and ``ConfidenceScores`` are defined in
``docling.datamodel.base_models`` and ``docling.datamodel.service.responses``
respectively. We re-export them here so converter code imports from a
single local module without coupling to upstream import paths.

Note: upstream ``QualityGrade`` has 5 members (poor, fair, good, excellent,
unspecified) — NOT the high/medium/low we had in the manual copy.
``ConfidenceScores`` has 8 fields, all defaulting to None / UNSPECIFIED.
"""

from __future__ import annotations

from docling.datamodel.base_models import QualityGrade
from docling.datamodel.service.responses import ConfidenceScores

__all__ = [
    "ConfidenceScores",
    "QualityGrade",
]
