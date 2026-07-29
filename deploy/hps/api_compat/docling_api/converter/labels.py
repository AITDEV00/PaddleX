"""Label mapping and confidence thresholds for PaddleX → Docling conversion."""

from __future__ import annotations

from docling_core.types.doc.labels import DocItemLabel

# PP-DocLayoutV3 25 categories → Docling DocItemLabel
PADDLEX_TO_DOCLING: dict[str, DocItemLabel] = {
    "abstract": DocItemLabel.TEXT,
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "chart": DocItemLabel.CHART,
    "content": DocItemLabel.TEXT,
    "display_formula": DocItemLabel.FORMULA,
    "doc_title": DocItemLabel.TITLE,
    "figure_title": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    "footer_image": DocItemLabel.PICTURE,
    "footnote": DocItemLabel.FOOTNOTE,
    "formula_number": DocItemLabel.TEXT,
    "header": DocItemLabel.PAGE_HEADER,
    "header_image": DocItemLabel.PICTURE,
    "image": DocItemLabel.PICTURE,
    "inline_formula": DocItemLabel.FORMULA,
    "number": DocItemLabel.TEXT,
    "paragraph_title": DocItemLabel.SECTION_HEADER,
    "reference": DocItemLabel.TEXT,
    "reference_content": DocItemLabel.TEXT,
    "seal": DocItemLabel.PICTURE,
    "table": DocItemLabel.TABLE,
    "text": DocItemLabel.TEXT,
    "vertical_text": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.CAPTION,
}

# Confidence thresholds for quality grading
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.6
