"""Converter package — maps model layout boxes to Mistral OCR blocks.

Exposes the canonical :class:`LayoutBox` model, the per-model
:class:`BoxAdapter` registry (the extension point for future models), and the
:class:`PaddleXToMistralConverter` emitter that consumes canonical boxes.
"""

from .boxes import (
    BoxAdapter,
    BoxAdapterRegistry,
    LayoutBox,
    PaddleXDocLayoutV3Adapter,
    box_adapter_registry,
)
from .service import ConvertOptions, converter

__all__ = [
    "BoxAdapter",
    "BoxAdapterRegistry",
    "LayoutBox",
    "PaddleXDocLayoutV3Adapter",
    "box_adapter_registry",
    "ConvertOptions",
    "converter",
]