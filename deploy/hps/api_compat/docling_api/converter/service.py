"""PaddleX layout detection → DoclingDocument converter.

Takes PP-DocLayoutV3 output (boxes with labels, scores, coordinates, polygon
points, reading order) and builds a DoclingDocument with proper provenance,
page metadata, and a document tree respecting reading order.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel

from .labels import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    PADDLEX_TO_DOCLING,
)
from .schema import ConfidenceScores, QualityGrade

logger = logging.getLogger("hps_api")


def _add_special_item(
    doc: DoclingDocument, label: DocItemLabel, prov: ProvenanceItem
) -> None:
    """Add items that use dedicated add_* methods (no text placeholder)."""
    if label in (DocItemLabel.PICTURE, DocItemLabel.CHART):
        doc.add_picture(prov=prov)
    elif label == DocItemLabel.TABLE:
        doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=prov)
    elif label == DocItemLabel.FORMULA:
        doc.add_text(label=label, text="", prov=prov)


def _add_text_item(
    doc: DoclingDocument, label: DocItemLabel, placeholder: str, prov: ProvenanceItem
) -> None:
    """Add text-bearing items, dispatching title/heading vs generic text."""
    if label == DocItemLabel.TITLE:
        doc.add_title(text=placeholder, prov=prov)
    elif label == DocItemLabel.SECTION_HEADER:
        doc.add_heading(text=placeholder, level=1, prov=prov)
    elif label in _TEXT_VALID_LABELS:
        doc.add_text(label=label, text=placeholder, prov=prov)
    else:
        # Label not valid for add_text (e.g. CHART, KEY_VALUE_REGION,
        # DOCUMENT_INDEX) — fall back to picture to avoid a crash.
        logger.warning(
            "Label %s is not valid for add_text; adding as picture", label
        )
        doc.add_picture(prov=prov)


# Labels that use special add_* methods instead of text placeholders
_SPECIAL_LABELS = frozenset({
    DocItemLabel.PICTURE,
    DocItemLabel.TABLE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHART,
})

# Labels accepted by DoclingDocument.add_text() — used to guard against
# labels that are only valid for visual items (CHART, PICTURE, TABLE, etc.).
# NOTE: FORMULA is excluded here because it's handled by _SPECIAL_LABELS
# (add_text with empty text), not the generic text-item path.
_TEXT_VALID_LABELS = frozenset({
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.TEXT,
    DocItemLabel.TITLE,
    DocItemLabel.CODE,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.HANDWRITTEN_TEXT,
    DocItemLabel.EMPTY_VALUE,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.REFERENCE,
    DocItemLabel.FIELD_HEADING,
    DocItemLabel.FIELD_KEY,
    DocItemLabel.FIELD_VALUE,
    DocItemLabel.FIELD_HINT,
    DocItemLabel.MARKER,
})


class PaddleXToDoclingConverter:
    """Converts PaddleX layout detection results to a DoclingDocument."""

    def convert(
        self,
        boxes: list[dict[str, Any]],
        image: np.ndarray,
        filename: str = "document",
        page_no: int = 1,
    ) -> DoclingDocument:
        """Convert PaddleX layout boxes to a DoclingDocument.

        Args:
            boxes: List of box dicts from PaddleX LayoutAnalysisResult.
                   Each has: label, score, coordinate [xmin,ymin,xmax,ymax],
                   order (reading order), cls_id, polygon_points.
            image: Original image as numpy array (H, W, 3) BGR or RGB uint8.
                   Only used for dimensions (shape[:2]), color order irrelevant.
            filename: Base filename for the document.
            page_no: 1-based page number.

        Returns:
            A populated DoclingDocument with layout items, provenance,
            and page metadata.
        """
        h, w = image.shape[:2]
        page_size = Size(width=float(w), height=float(h))

        doc = DoclingDocument(name=filename)
        doc.add_page(page_no=page_no, size=page_size)

        if not boxes:
            logger.warning(
                "No layout boxes detected for %s — returning empty document",
                filename,
            )

        sorted_boxes = self._sort_by_reading_order(boxes)

        for box in sorted_boxes:
            label_str = box.get("label", "text")
            coord = box.get("coordinate", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = (float(c) for c in coord)

            docling_label = PADDLEX_TO_DOCLING.get(label_str, DocItemLabel.TEXT)

            bbox = BoundingBox(
                l=xmin, t=ymin, r=xmax, b=ymax,
                coord_origin="TOPLEFT",
            )
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=bbox,
                charspan=(0, 0),
            )

            if docling_label in _SPECIAL_LABELS:
                _add_special_item(doc, docling_label, prov)
            else:
                placeholder = f"[{label_str}]"
                _add_text_item(doc, docling_label, placeholder, prov)

        logger.debug(
            "Converted %d boxes to DoclingDocument for %s", len(sorted_boxes), filename
        )
        return doc

    def compute_confidence(
        self, boxes: list[dict[str, Any]]
    ) -> ConfidenceScores:
        """Compute layout confidence from box scores.

        Boxes without a 'score' key are excluded from the mean (rather than
        treated as 0.0, which would skew the average downward).
        """
        scores = [float(b["score"]) for b in boxes if "score" in b]
        if not scores:
            return ConfidenceScores(
                layout_score=0.0,
                mean_score=0.0,
                mean_grade=QualityGrade.POOR,
                low_score=0.0,
                low_grade=QualityGrade.POOR,
            )
        mean_score = sum(scores) / len(scores)
        low_score = min(scores)
        grade = self._grade_confidence(mean_score)
        low_grade = self._grade_confidence(low_score)
        return ConfidenceScores(
            layout_score=mean_score,
            mean_score=mean_score,
            mean_grade=grade,
            low_score=low_score,
            low_grade=low_grade,
        )

    @staticmethod
    def _grade_confidence(mean_score: float) -> QualityGrade:
        """Map a mean confidence score to a quality grade.

        Uses upstream QualityGrade members: excellent / good / poor.
        Thresholds: >=0.85 → excellent, >=0.6 → good, else → poor.
        """
        if mean_score >= CONFIDENCE_HIGH_THRESHOLD:
            return QualityGrade.EXCELLENT
        if mean_score >= CONFIDENCE_MEDIUM_THRESHOLD:
            return QualityGrade.GOOD
        return QualityGrade.POOR

    @staticmethod
    def _sort_by_reading_order(
        boxes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort boxes by reading order field, falling back to top-to-bottom."""
        def sort_key(b: dict) -> tuple[int, float]:
            order = b.get("order")
            if order is not None:
                return (0, float(order))
            coord = b.get("coordinate", [0, 0, 0, 0])
            return (1, float(coord[1]))

        return sorted(boxes, key=sort_key)
