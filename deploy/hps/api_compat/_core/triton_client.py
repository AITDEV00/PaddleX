"""Async Triton gRPC client for layout detection.

This module replaces the direct in-process ``paddlex.create_model()`` call
with a gRPC request to a Triton Inference Server.  Triton's dynamic batcher
provides continuous batching: incoming requests are queued non-blockingly,
collected into batches of up to ``max_batch_size``, and dispatched to the
GPU.  While one batch executes, the next is already being collected.

Wire protocol (matches ``triton/model_repo/doclayout-v3/1/model.py``):
  Request:  Raw binary — 12-byte header (H,W,C as int32 LE) + raw image bytes
  Response: JSON {"boxes": [...], "error": null | "..."}

  (v1 protocol used base64-encoded JPEG — eliminated for latency: saves
   JPEG re-encode + base64 encode + base64 decode + JPEG decode per request)

Architecture:

  ┌──────────────────────┐     gRPC      ┌───────────────────────────┐
  │  api_compat (FastAPI) │──────────────▶│  Triton Inference Server   │
  │  triton_client.py     │              │  doclayout-v3 model         │
  │                       │◀──────────────│  dynamic_batching { }       │
  │  await detect(image)  │   response   │  model.predict(batch)       │
  └──────────────────────┘              └───────────────────────────┘

The client uses ``tritonclient.grpc.aio`` (async) so requests don't block
the asyncio event loop.  Multiple concurrent ``detect()`` calls are
automatically batched by Triton — no client-side batching logic needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import TRITON_MODEL_NAME, TRITON_REQUEST_TIMEOUT, TRITON_URL
from .latency import is_latency_logging_enabled

# orjson is 3-10x faster than stdlib json for both serialization and
# deserialization.  It handles numpy types natively when we pass
# default=_numpy_default.  Falls back to stdlib json if orjson is missing.
try:
    import orjson

    def _json_loads(data: bytes):
        return orjson.loads(data)

    def _json_dumps(obj) -> bytes:
        return orjson.dumps(obj)

    _HAS_ORJSON = True
except ImportError:
    import json as _stdjson

    def _json_loads(data: bytes):
        return _stdjson.loads(data)

    def _json_dumps(obj) -> bytes:
        return _stdjson.dumps(obj).encode("utf-8")

    _HAS_ORJSON = False

logger = logging.getLogger("hps_api")

# Triton client is optional at import time (tests stub it)
_triton_client = None
_triton_client_ready = False


def _ensure_client():
    """Lazily create the async Triton gRPC client.

    We can't create it at import time because the Triton server may not
    be running yet during tests.  The client is created on first use.
    """
    global _triton_client, _triton_client_ready
    if _triton_client is not None:
        return _triton_client

    try:
        from tritonclient.grpc.aio import InferenceServerClient
    except ImportError as e:
        raise RuntimeError(
            "tritonclient[grpc] is required for the Triton backend. "
            "Install it with: pip install tritonclient[grpc]"
        ) from e

    _triton_client = InferenceServerClient(
        url=TRITON_URL,
        verbose=False,
    )
    _triton_client_ready = True
    logger.info("Triton gRPC client created (url=%s)", TRITON_URL)
    return _triton_client


async def is_server_ready() -> bool:
    """Check if the Triton server and model are ready."""
    try:
        client = _ensure_client()
        server_ready = await client.is_server_ready()
        if not server_ready:
            return False
        model_ready = await client.is_model_ready(TRITON_MODEL_NAME)
        return bool(model_ready)
    except Exception as e:
        logger.warning("Triton readiness check failed: %s", e)
        return False


async def detect_layout(image: np.ndarray) -> list[dict[str, Any]]:
    """Send an image to Triton for layout detection.

    This is an async call — multiple concurrent invocations are
    automatically batched by Triton's dynamic batcher.  No client-side
    batching, no queue management, no inference thread.

    Args:
        image: RGB or BGR numpy array (H, W, 3) uint8.

    Returns:
        List of box dicts, each with keys: label, score, coordinate,
        order, cls_id, polygon_points.

    Raises:
        RuntimeError: If Triton returns an error or is unreachable.
    """
    import struct
    import time

    from tritonclient.grpc.aio import InferInput

    _log = is_latency_logging_enabled()
    t0 = time.perf_counter() if _log else 0.0

    # ── Raw-bytes wire protocol (v2) ─────────────────────────────────
    # Send raw numpy bytes with a compact 12-byte header:
    #   [H: int32 LE][W: int32 LE][C: int32 LE][raw pixel data]
    # This eliminates JPEG re-encode + base64 encode/decode (saves
    # ~5-15ms CPU per request depending on image size).
    h, w = image.shape[0], image.shape[1]
    c = image.shape[2] if image.ndim == 3 else 1
    header = struct.pack("<III", h, w, c)
    # Use bytes() (immutable) — numpy treats bytes as a scalar object,
    # but would iterate over bytearray, creating a 3D array.
    payload_bytes = header + image.tobytes()

    # Create Triton input tensor: shape [1, 1], dtype BYTES
    input_tensor = InferInput("input", [1, 1], "BYTES")
    input_data = np.array([[payload_bytes]], dtype=np.object_)
    input_tensor.set_data_from_numpy(input_data)

    client = _ensure_client()

    # Send the inference request
    # tritonclient expects an integer timeout (seconds), not float
    t_send = time.perf_counter() if _log else 0.0
    response = await client.infer(
        TRITON_MODEL_NAME,
        inputs=[input_tensor],
        timeout=int(TRITON_REQUEST_TIMEOUT),
    )
    t_recv = time.perf_counter() if _log else 0.0

    # Parse the response — use orjson for 3-10x faster deserialization
    output = response.as_numpy("output")
    if output is None:
        raise RuntimeError("Triton returned no output tensor")

    result = _json_loads(output[0, 0])

    if result.get("error"):
        raise RuntimeError(f"Triton inference error: {result['error']}")

    boxes = result.get("boxes", [])

    if _log:
        t_done = time.perf_counter()
        logger.info(
            '{"event":"latency","stage":"triton_round_trip",'
            '"encode_s":%.6f,"server_s":%.6f,"decode_s":%.6f,'
            '"total_s":%.6f,"n_boxes":%d}',
            t_send - t0, t_recv - t_send, t_done - t_recv,
            t_done - t0, len(boxes),
        )

    return boxes


async def close_client() -> None:
    """Close the gRPC client on shutdown."""
    global _triton_client, _triton_client_ready
    if _triton_client is not None:
        await _triton_client.close()
        _triton_client = None
        _triton_client_ready = False
        logger.info("Triton gRPC client closed")
