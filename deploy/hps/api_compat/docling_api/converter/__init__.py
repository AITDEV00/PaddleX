"""Converter slice — PaddleX layout → DoclingDocument conversion.

Owns the mapping from PaddleX detection labels to Docling labels, confidence
scoring, and the document-tree builder.
"""

from __future__ import annotations

from .service import PaddleXToDoclingConverter

__all__ = ["PaddleXToDoclingConverter"]
