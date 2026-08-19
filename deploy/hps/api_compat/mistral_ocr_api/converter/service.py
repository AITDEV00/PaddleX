"""Converter — PaddleX layout boxes → Mistral ``OCRPageObject``.

Stateless and safe to share across requests (mirrors
``docling_api/converter/service.py``).  Since the layout model produces no
OCR text, ``content`` is a placeholder like the Docling converter uses
(``[text]``, ``[title]``, ...).  When a PaddleOCR-VL backend is wired up,
real text can be filled in here without changing the route or schema.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..schema import (
    OCRAsideTextBlock,
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
)
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


class PaddleXToMistralConverter:
    """Converts PaddleX layout boxes to a Mistral OCRPageObject."""

    def convert(
        self,
        boxes: list[dict[str, Any]],
        image: np.ndarray,
        page_no: int = 0,
        include_blocks: bool = True,
    ) -> OCRPageObject:
        """Build an OCRPageObject from layout boxes.

        Args:
            boxes: List of box dicts from ``run_layout_detection``. Each has
                ``label``, ``score``, ``coordinate:[xmin,ymin,xmax,ymax]``,
                ``order``.
            image: Original decoded image (H, W, 3).
            page_no: 0-based page index.
            include_blocks: If False, blocks are omitted from the page.
        """
        h, w = image.shape[:2]

        blocks = None
        if include_blocks and boxes:
            blocks = self._to_blocks(boxes)

        return OCRPageObject(
            index=page_no,
            markdown="",  # layout-only: no text pipeline yet
            images=[],
            dimensions=OCRPageDimensions(dpi=300, height=int(h), width=int(w)),
            blocks=blocks,
        )

    @staticmethod
    def _sort_by_reading_order(
        boxes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort boxes by reading order, falling back to top-to-bottom."""
        def sort_key(b: dict) -> tuple[int, float]:
            order = b.get("order")
            if order is not None:
                return (0, float(order))
            coord = b.get("coordinate", [0, 0, 0, 0])
            return (1, float(coord[1]))

        return sorted(boxes, key=sort_key)

    def _to_blocks(self, boxes: list[dict[str, Any]]) -> list[Any]:
        """Build concrete Mistral block instances from layout boxes."""
        blocks: list[Any] = []
        image_counter = 0
        table_counter = 0

        for box in self._sort_by_reading_order(boxes):
            label = str(box.get("label", "text"))
            mtype = to_mistral_type(label)
            coord = box.get("coordinate", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = (int(round(float(c))) for c in coord)

            content = _CONTENT_PLACEHOLDER.get(mtype, f"[{label}]")

            cls = _BLOCK_CLASS.get(mtype, OCRTextBlock)

            if mtype == "image":
                image_id = f"image_{image_counter}"
                image_counter += 1
                blocks.append(
                    cls(
                        top_left_x=xmin,
                        top_left_y=ymin,
                        bottom_right_x=xmax,
                        bottom_right_y=ymax,
                        content=content,
                        image_id=image_id,
                    )
                )
            elif mtype == "table":
                table_id = f"table_{table_counter}"
                table_counter += 1
                blocks.append(
                    cls(
                        top_left_x=xmin,
                        top_left_y=ymin,
                        bottom_right_x=xmax,
                        bottom_right_y=ymax,
                        content=content,
                        table_id=table_id,
                    )
                )
            else:
                blocks.append(
                    cls(
                        top_left_x=xmin,
                        top_left_y=ymin,
                        bottom_right_x=xmax,
                        bottom_right_y=ymax,
                        content=content,
                    )
                )

        logger.debug("Converted %d boxes to %d Mistral blocks", len(boxes), len(blocks))
        return blocks


# Stateless converter — safe to share
converter = PaddleXToMistralConverter()