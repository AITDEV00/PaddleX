"""TensorRT engine caching — avoid 3-minute rebuilds on every restart.

TRT engines are GPU-specific (built for the exact CUDA/TRT/GPU combo), so
they can't be downloaded from HuggingFace — they must be built once per
environment. This module finds a pre-built engine and copies it to /tmp
(writable) so PaddleX loads it directly with no rebuild.
"""

from __future__ import annotations

import logging
import os
import shutil

from .config import (
    BUILTIN_ENGINE_DIRS,
    ENGINE_DIR,
    MODEL_PRECISION,
)

logger = logging.getLogger("hps_api")


def prepare_engine() -> str:
    """Ensure a TRT engine is available in /tmp.

    Search order:
      1. /tmp/paddlex-engines/  (previous copy or persisted volume — fastest)
      2. Pre-built engine in known locations (Dockerfile, container, custom)

    Returns the path to the engine file. If no pre-built engine exists
    anywhere, returns the expected /tmp path — PaddleX will auto-build
    from ONNX (slow, ~3 min).
    """
    engine_name = f"inference_{MODEL_PRECISION}.trt"
    cache_name = "timing.cache"
    tmp_engine = os.path.join(ENGINE_DIR, engine_name)
    tmp_cache = os.path.join(ENGINE_DIR, cache_name)

    # Fast path: engine already in /tmp
    if os.path.exists(tmp_engine):
        logger.info("Found cached engine at %s — instant load", tmp_engine)
        return tmp_engine

    # Search for a pre-built engine in known locations
    for src_dir in BUILTIN_ENGINE_DIRS:
        src_engine = os.path.join(src_dir, engine_name)
        if os.path.exists(src_engine):
            logger.info("Copying pre-built engine %s → %s", src_engine, tmp_engine)
            shutil.copy2(src_engine, tmp_engine)
            src_cache = os.path.join(src_dir, cache_name)
            if os.path.exists(src_cache) and not os.path.exists(tmp_cache):
                shutil.copy2(src_cache, tmp_cache)
            return tmp_engine

    # No pre-built engine — PaddleX will auto-build (slow)
    logger.warning(
        "No pre-built engine found in %s — will auto-build from ONNX (~3 min)",
        BUILTIN_ENGINE_DIRS,
    )
    return tmp_engine
