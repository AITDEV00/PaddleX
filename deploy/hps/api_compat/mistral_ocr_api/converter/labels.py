"""PaddleX → Mistral OCR block-type mapping.

Mirrors ``docling_api/converter/labels.py``: the mapping is data, not logic,
so it lives in one auditable place.

PP-DocLayoutV3 emits 25 layout categories.  Mistral's OCR response uses
paragraph-level blocks, each with a structural ``type``.  We map each PaddleX
label to a Mistral block *type string*.  The converter maps the type string to
the concrete Mistral DTO class (``OCRTextBlock``, ``OCRTitleBlock``, ...).
"""

from __future__ import annotations

# Mistral block type strings → which PaddleX labels produce them.
PADDLEX_TO_MISTRAL_TYPE: dict[str, str] = {
    # Text-like
    "text": "text",
    "content": "text",
    "abstract": "text",
    "number": "text",
    "reference": "text",
    "reference_content": "text",
    "formula_number": "text",
    "aside_text": "aside_text",
    "vertical_text": "text",
    # Titles / headers
    "doc_title": "title",
    "paragraph_title": "title",
    "figure_title": "title",
    "header": "header",
    "footer": "footer",
    # Tables / structure
    "table": "table",
    "algorithm": "code",
    # Formulas
    "display_formula": "equation",
    "inline_formula": "equation",
    # Images
    "image": "image",
    "header_image": "image",
    "footer_image": "image",
    "seal": "image",
    "chart": "image",
    # Footnotes / captions
    "footnote": "caption",
    "vision_footnote": "caption",
}


def to_mistral_type(label: str) -> str:
    """Return the Mistral block type for a PaddleX label, defaulting to text."""
    return PADDLEX_TO_MISTRAL_TYPE.get(label, "text")