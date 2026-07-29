"""Shared serving infrastructure — reusable across all API compatibility layers.

Contains:
  - config.py:    Environment-driven configuration (single source of truth)
  - engine.py:    TensorRT engine caching
  - inference.py: PaddleX model loading + dedicated inference thread
  - image.py:     Image loading from bytes and URLs
  - health/       Liveness and readiness endpoints
"""
