"""API compatibility layers for PaddleX HPS.

This package contains thin FastAPI serving layers that expose PaddleX
inference through various API protocols (Docling, Unstructured, etc.).

Structure:
  - _core/        — shared infrastructure (config, inference, image, health)
  - docling_api/  — Docling-compatible API
  - unstructured_api/ — (future) Unstructured-compatible API
"""
