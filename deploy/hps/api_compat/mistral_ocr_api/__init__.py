"""Mistral OCR-compatible API layer for PaddleX HPS.

Serves ``POST /v1/ocr`` using the official Mistral DTOs
(``mistralai.client.models.*``) so the wire contract matches what Mistral's
own generated client produces.  Reuses the shared ``_core`` infrastructure
(config, engine, image, inference, health).

This layer is opt-in: enable it by including ``mistral`` in ``HPS_API_LAYERS``
(see ``.._core.config``).  It requires the pinned ``mistralai`` package.
"""