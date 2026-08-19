"""API compatibility layers for PaddleX HPS.

This package contains thin FastAPI serving layers that expose PaddleX
inference through various API protocols (Docling, Mistral, etc.).

Structure:
  - _core/        — shared infrastructure (config, inference, image, health)
  - docling_api/  — Docling-compatible API (/v1/convert/*)
  - mistral_ocr_api/ — Mistral OCR-compatible API (/v1/ocr)
  - web_app.py    — master app that mounts enabled layers per HPS_API_LAYERS
  - unstructured_api/ — (future) Unstructured-compatible API
"""
