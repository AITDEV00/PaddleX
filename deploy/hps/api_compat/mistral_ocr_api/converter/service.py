"""Converter — canonical layout boxes → Mistral ``OCRPageObject``.

Stateless and safe to share across requests (mirrors
``docling_api/converter/service.py``).  Since the layout model produces no
OCR text, ``content`` is a placeholder like the Docling converter uses
(``[text]``, ``[title]``, ...).  When a PaddleOCR-VL backend is wired up,
real text can be filled in here without changing the route or schema.

The emitter consumes **canonical** :class:`LayoutBox` objects (see
``boxes.py``).  Raw, model-specific box dicts are normalized by a
:class:`BoxAdapter` before reaching this module, so adding a new model never
requires editing the emitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import (
    LABELED_BLOCK_CLASS,
    OCRAsideTextBlock,
    OCRBlockConfidenceScores,
    OCRCaptionBlock,
    OCRCodeBlock,
    OCREquationBlock,
    OCRFooterBlock,
    OCRHeaderBlock,
    OCRImageBlock,
    OCRListBlock,
    OCRPageDimensions,
    OCRPageObject,
    OCRReferencesBlock,
    OCRSignatureBlock,
    OCRTableBlock,
    OCRTextBlock,
    OCRTitleBlock,
    PaddleXBox,
    PaddleXOCRPageObject,
    PaddleXPageMetadata,
)
from .boxes import LayoutBox
from .labels import to_mistral_type

logger = logging.getLogger("hps_api")

# Layout-only placeholder content per Mistral block type.
_CONTENT_PLACEHOLDER = {
    "text": "[text]",
    "title": "[title]",
    "table": "[table]",
    "image": "[image]",
    "list": "[list]",
    "equation": "[formula]",
    "caption": "[caption]",
    "code": "[code]",
    "header": "[header]",
    "footer": "[footer]",
    "signature": "[signature]",
    "references": "[references]",
    "aside_text": "[aside]",
}

# Mistral concrete block class per type.  Image blocks additionally carry an
# ``image_id`` reference; table blocks carry a ``table_id``.
#
# Every key in ``_CONTENT_PLACEHOLDER`` (and every value ``to_mistral_type`` can
# return) MUST be present here.  If a type is missing it silently collapses to
# ``OCRTextBlock`` and the structural type is lost on the wire.
_BLOCK_CLASS = {
    "text": OCRTextBlock,
    "title": OCRTitleBlock,
    "table": OCRTableBlock,
    "image": OCRImageBlock,
    "list": OCRListBlock,
    "equation": OCREquationBlock,
    "caption": OCRCaptionBlock,
    "code": OCRCodeBlock,
    "header": OCRHeaderBlock,
    "footer": OCRFooterBlock,
    "signature": OCRSignatureBlock,
    "references": OCRReferencesBlock,
    "aside_text": OCRAsideTextBlock,
}


@dataclass(frozen=True)
class ConvertOptions:
    """Knobs controlling how canonical boxes are turned into a page.

    Grouped into one object so ``convert`` keeps a small signature and future
    options (e.g. text extraction toggles) can be added without churn.
    """

    page_no: int = 0
    include_blocks: bool = True
    include_scores: bool = False
    include_native_metadata: bool = False
    include_native_labels: bool = False
    threshold: float | None = None


class PaddleXToMistralConverter:
    """Converts canonical layout boxes to a Mistral OCRPageObject."""

    def convert(
        self,
        boxes: list[LayoutBox],
        image: np.ndarray,
        options: ConvertOptions | None = None,
    ) -> OCRPageObject:
        """Build an OCRPageObject from canonical layout boxes.

        Args:
            boxes: Canonical :class:`LayoutBox` list (already normalized from
                a model's native output via :class:`BoxAdapter`).
            image: Original decoded image (H, W, 3).
            options: Conversion knobs (:class:`ConvertOptions`); defaults
                produce a minimal page.
        """
        opts = options or ConvertOptions()
        h, w = image.shape[:2]

        blocks = None
        if opts.include_blocks and boxes:
            blocks = self._to_blocks(
                boxes,
                include_scores=opts.include_scores,
                include_native_labels=opts.include_native_labels,
            )

        page_cls = PaddleXOCRPageObject if opts.include_native_labels else OCRPageObject
        page = page_cls(
            index=opts.page_no,
            markdown="",  # layout-only: no text pipeline yet
            images=[],
            dimensions=OCRPageDimensions(dpi=300, height=int(h), width=int(w)),
            blocks=blocks,
        )

        # Attach native metadata as a private attribute so the service can
        # lift it into the top-level ``paddlex`` container. The block/page
        # DTOs cannot carry it in the wire schema (see feasibility study).
        if opts.include_native_metadata:
            page._paddlex_page_metadata = self._native_metadata(
                boxes, opts.page_no, opts.threshold
            )

        return page

    @staticmethod
    def _sort_by_reading_order(boxes: list[LayoutBox]) -> list[LayoutBox]:
        """Sort boxes by reading order, falling back to top-to-bottom."""
        def sort_key(b: LayoutBox) -> tuple[int, float]:
            if b.order is not None:
                return (0, float(b.order))
            return (1, float(b.coordinate[1]))

        return sorted(boxes, key=sort_key)

    def _to_blocks(
        self,
        boxes: list[LayoutBox],
        include_scores: bool = False,
        include_native_labels: bool = False,
    ) -> list[Any]:
        """Build concrete Mistral block instances from layout boxes.

        When ``include_native_labels`` is set, the block instances are the
        labeled subclasses so each block carries the native PaddleX ``label``
        alongside the stock Mistral ``type``.
        """
        blocks: list[Any] = []
        image_counter = 0
        table_counter = 0

        # Choose the block class pool: stock DTOs, or the labeled subclasses
        # that add a ``label`` field (see schema.LABELED_BLOCK_CLASS). The
        # fallback also switches to the labeled text block so a label is never
        # silently dropped when native labels are requested.
        block_cls = LABELED_BLOCK_CLASS if include_native_labels else _BLOCK_CLASS
        fallback_cls = block_cls.get("text", OCRTextBlock)

        for box in self._sort_by_reading_order(boxes):
            mtype = to_mistral_type(box.label)
            xmin, ymin, xmax, ymax = box.coordinate

            content = _CONTENT_PLACEHOLDER.get(mtype, f"[{box.label}]")

            cls = block_cls.get(mtype, fallback_cls)

            # Native per-box confidence, mirrored onto the Mistral block's
            # native confidence_scores field so stock clients see it.
            confidence = None
            if include_scores:
                try:
                    confidence = OCRBlockConfidenceScores(
                        block_type_confidence_score=float(box.score),
                        average_content_confidence_score=float(box.score),
                    )
                except (TypeError, ValueError):
                    confidence = None

            common = dict(
                top_left_x=xmin,
                top_left_y=ymin,
                bottom_right_x=xmax,
                bottom_right_y=ymax,
                content=content,
            )
            if confidence is not None:
                common["confidence_scores"] = confidence
            if include_native_labels:
                common["label"] = box.label

            if mtype == "image":
                image_id = f"image_{image_counter}"
                image_counter += 1
                blocks.append(cls(image_id=image_id, **common))
            elif mtype == "table":
                table_id = f"table_{table_counter}"
                table_counter += 1
                blocks.append(cls(table_id=table_id, **common))
            else:
                blocks.append(cls(**common))

        logger.debug(
            "Converted %d boxes to %d Mistral blocks", len(boxes), len(blocks)
        )
        return blocks

    def _native_metadata(
        self,
        boxes: list[LayoutBox],
        page_no: int,
        threshold: float | None,
    ) -> PaddleXPageMetadata:
        """Build per-page native PaddleX metadata aligned with ``blocks``.

        ``boxes`` must already be in reading order (as emitted by
        ``_to_blocks``) so ``block_index`` aligns with ``page.blocks``.
        """
        native_boxes: list[PaddleXBox] = []
        for i, box in enumerate(self._sort_by_reading_order(boxes)):
            native_boxes.append(
                PaddleXBox(
                    block_index=i,
                    label=box.label,
                    cls_id=box.cls_id,
                    score=float(box.score),
                    order=box.order,
                    coordinate=list(box.coordinate),
                    polygon_points=box.polygon_points,
                )
            )
        return PaddleXPageMetadata(
            page_index=page_no,
            threshold_used=threshold,
            boxes=native_boxes,
        )


# Stateless converter — safe to share
converter = PaddleXToMistralConverter()