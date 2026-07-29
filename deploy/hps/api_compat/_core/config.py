"""Environment-driven configuration for the HPS API compatibility layers.

Single source of truth for all env-var reads. No other module should call
``os.environ`` directly — import from here instead.
"""

from __future__ import annotations

import logging
import os

# ─── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("HPS_API_LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure root logging — call once at startup (not at import time).

    Idempotent: safe to call multiple times.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ─── Model / Engine ────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("HPS_API_MODEL", "PP-DocLayoutV3")
MODEL_PRECISION = os.environ.get("HPS_API_PRECISION", "fp8")
MODEL_DEVICE_ID = int(os.environ.get("HPS_API_DEVICE_ID", "0"))

# ─── Engine cache directories (all writable paths in /tmp) ─────────────────────

ENGINE_DIR = os.environ.get("HPS_API_ENGINE_DIR", "/tmp/paddlex-engines")
os.makedirs(ENGINE_DIR, exist_ok=True)

# PADDLEX_HOME overrides where PaddleX stores/looks for model files.
# We don't set it ourselves — just read it so BUILTIN_ENGINE_DIRS can include it.
PADDLEX_HOME = os.environ.get("PADDLEX_HOME")

# Search order for pre-built TRT engines (first match wins)
BUILTIN_ENGINE_DIRS: list[str] = [
    "/opt/models/PP-DocLayoutV3",
    "/root/.paddlex/official_models/PP-DocLayoutV3",
    os.path.expanduser("~/.paddlex/official_models/PP-DocLayoutV3"),
]
if PADDLEX_HOME:
    BUILTIN_ENGINE_DIRS.append(
        os.path.join(PADDLEX_HOME, ".paddlex/official_models/PP-DocLayoutV3")
    )

# ─── Limits / Timeouts ─────────────────────────────────────────────────────────

MAX_IMAGE_DIM = int(os.environ.get("HPS_API_MAX_IMAGE_DIM", "4096"))
DEFAULT_TIMEOUT = float(os.environ.get("HPS_API_TIMEOUT", "30"))
STARTUP_TIMEOUT = int(os.environ.get("HPS_API_STARTUP_TIMEOUT", "300"))

# ─── Concurrency Pipeline ──────────────────────────────────────────────────────
# How many GPU inference tasks can be in-flight simultaneously.
# Depth=1 → fully serial (original behavior, safest for single GPU).
# Depth=3 → allows overlap: while request A waits for its GPU result,
#           request B can submit work and request C can load its image.
# The inference thread still processes one at a time (CUDA context is
# single-threaded), but the asyncio layer pipelines submission and
# result-wait so GPU idle time between requests is minimized.
#
# Throughput tuning: higher depth increases GPU utilization under load but
# also increases p50 latency due to queuing.  For best throughput on a
# single GPU, depth=4-6 is a good starting point (allows 4-6 requests
# overlapping image-load + GPU + post-processing).
PIPELINE_DEPTH = int(os.environ.get("HPS_API_PIPELINE_DEPTH", "4"))

# Dedicated thread pool for CPU-bound pipeline stages (image decode,
# DoclingDocument conversion, format export).  Keeping these off the
# event loop prevents one request's CPU work from blocking all others.
# Set to >= PIPELINE_DEPTH so CPU stages never block GPU submission.
CPU_POOL_SIZE = int(os.environ.get("HPS_API_CPU_POOL_SIZE", "8"))

# ─── GPU Micro-Batching (legacy: used only by DirectBackend) ──────────────────
# When using the direct in-process backend (HPS_API_BACKEND=direct), the
# inference thread collects up to BATCH_SIZE images before issuing a
# single predict() call.  When using the Triton backend (default), Triton's
# own dynamic_batching handles this — these settings are ignored.
#
# For direct backend throughput: BATCH_SIZE=2-4 with short timeout
# amortizes kernel launch overhead and improves GPU utilization without
# adding much latency under load.
BATCH_SIZE = int(os.environ.get("HPS_API_BATCH_SIZE", "2"))
BATCH_TIMEOUT_MS = float(os.environ.get("HPS_API_BATCH_TIMEOUT_MS", "3"))

# ─── Inference Backend ────────────────────────────────────────────────────────
# "triton" (default) — Connect to a Triton Inference Server via gRPC.
#                       Triton's dynamic_batching provides continuous batching,
#                       non-blocking request queuing, and GPU saturation.
# "direct"            — Load the PaddleX model in-process (no Triton).
#                       Uses our custom micro-batching.  Simpler but less
#                       performant under concurrent load.
INFERENCE_BACKEND = os.environ.get("HPS_API_BACKEND", "triton")

# ─── Triton Server Configuration ──────────────────────────────────────────────
TRITON_URL = os.environ.get("HPS_TRITON_URL", "localhost:8001")
TRITON_MODEL_NAME = os.environ.get("HPS_TRITON_MODEL_NAME", "doclayout-v3")
TRITON_REQUEST_TIMEOUT = float(os.environ.get("HPS_TRITON_TIMEOUT", "30"))
