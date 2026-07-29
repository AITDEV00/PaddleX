"""Triton Python backend model for PP-DocLayoutV3 layout detection.

This model runs inside the Triton Inference Server process.  Triton's
dynamic batcher collects concurrent requests and calls ``execute()``
with a batch of them.  We decode each image, call
``model.predict(images, batch_size=N)`` in a single GPU forward pass,
then split results back to individual responses.

This is the core of the continuous-batching architecture:
  - Triton queues incoming requests (non-blocking)
  - Triton's dynamic_batcher collects up to ``max_batch_size`` requests
    (or flushes after ``max_queue_delay_microseconds``)
  - ``execute()`` receives the batch and does ONE ``predict()`` call
  - While the GPU is busy, Triton is already collecting the NEXT batch
  - Results are split by position and returned to each request's response

Wire format (shared with the api_compat gRPC client):
  Request:  Raw bytes — 12-byte header (H,W,C as int32 LE) + raw pixel data
  Response: JSON string {"boxes": [...], "error": null | "..."}

  (v1 used base64-encoded JPEG — eliminated for latency.)
"""

from __future__ import annotations

import logging
import os
import struct
import time
from typing import Any

import numpy as np

# orjson is 3-10x faster than stdlib json for serialization.  It handles
# numpy types via the default= callback.  Falls back to stdlib json if
# orjson is not available.
try:
    import orjson

    # orjson.OPT_SERIALIZE_NUMPY handles numpy arrays and scalars natively
    # without any Python-level callback — much faster than default= approach.
    _ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY

    def _json_dumps_bytes(obj: Any) -> bytes:
        """Serialize to bytes using orjson with native numpy support."""
        return orjson.dumps(obj, option=_ORJSON_OPTS)

    def _json_loads_config(data: str | bytes) -> Any:
        """Deserialize config using orjson (fast path)."""
        return orjson.loads(data)

    _HAS_ORJSON = True
except ImportError:
    import json as _stdjson

    def _json_dumps_bytes(obj: Any) -> bytes:
        """Serialize to bytes using stdlib json (fallback)."""
        return _stdjson.dumps(
            obj, separators=(",", ":"), default=_numpy_default,
        ).encode("utf-8")

    def _json_loads_config(data: str | bytes) -> Any:
        """Deserialize config using stdlib json (fallback)."""
        return _stdjson.loads(data)

    _HAS_ORJSON = False

# triton_python_backend_utils is injected by Triton at runtime
import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]

logger = logging.getLogger("doclayout_v3")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

INPUT_NAME = "input"
OUTPUT_NAME = "output"

MODEL_NAME = os.environ.get("HPS_API_MODEL", "PP-DocLayoutV3")
MODEL_PRECISION = os.environ.get("HPS_API_PRECISION", "fp8")
DEVICE_ID = int(os.environ.get("HPS_API_DEVICE_ID", "0"))


def _decode_image(payload: bytes) -> np.ndarray:
    """Decode raw-bytes wire format to a BGR numpy array.

    v2 protocol: 12-byte header (H, W, C as int32 LE) + raw pixel data.
    The image arrives as-is from the client — no JPEG/base64 overhead.
    """
    if len(payload) < 12:
        raise ValueError(f"Payload too short: {len(payload)} bytes")
    h, w, c = struct.unpack_from("<III", payload, 0)
    expected = h * w * c
    raw = payload[12:]
    if len(raw) != expected:
        raise ValueError(
            f"Payload size mismatch: expected {expected}, got {len(raw)}"
        )
    img = np.frombuffer(raw, np.uint8)
    if c == 1:
        img = img.reshape(h, w)
    else:
        img = img.reshape(h, w, c)
    return img


def _extract_boxes(result: Any) -> list[dict[str, Any]]:
    """Extract layout boxes from a PaddleX prediction result."""
    if isinstance(result, dict):
        boxes = result.get("boxes", [])
    else:
        boxes = getattr(result, "boxes", [])
    # numpy types are converted at JSON serialization time via _numpy_default
    return boxes


def _numpy_default(obj: Any) -> Any:
    """JSON default function — converts numpy types to native Python.

    Replaces the recursive _to_native() traversal: json.dumps calls this
    only on objects it can't serialize, which is much faster than
    pre-traversing the entire box tree.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class TritonPythonModel:
    """Triton Python backend model for PP-DocLayoutV3.

    Lifecycle:
      - ``initialize()``: Loads the PaddleX model (creates CUDA context)
      - ``execute()``:    Called by Triton with a batch of requests
      - ``finalize()``:   Cleanup on shutdown
    """

    def initialize(self, args: dict) -> None:
        """Load the PaddleX model at startup.

        The CUDA context is created in this thread and stays alive for
        the lifetime of the model instance.
        """
        self.model_config = _json_loads_config(args["model_config"])
        logger.info(
            "Initializing PP-DocLayoutV3 model (precision=%s, device=%d)",
            MODEL_PRECISION, DEVICE_ID,
        )

        from paddlex import create_model

        engine_path = os.environ.get("HPS_API_ENGINE_PATH", "")
        engine_config: dict[str, Any] = {
            "precision": MODEL_PRECISION,
            "device_id": DEVICE_ID,
        }
        if engine_path and os.path.exists(engine_path):
            engine_config["engine_path"] = engine_path

        self.model = create_model(
            MODEL_NAME,
            engine="tensorrt",
            engine_config=engine_config,
        )
        logger.info("PP-DocLayoutV3 model loaded successfully")

    def execute(self, requests: list) -> list:
        """Process a batch of inference requests.

        Called by Triton's dynamic batcher with 1–8 requests.  We:
          1. Decode each request's image from base64
          2. Call ``model.predict(images, batch_size=N)`` once
          3. Split results back to individual responses

        If image decoding fails for a request, that request gets an error
        response but the rest of the batch still processes.

        Latency instrumentation: when HPS_LATENCY_LOG=1, each phase is
        timed separately and logged as a structured JSON line. This lets
        us identify whether the bottleneck is decode, preprocess, GPU
        inference, postprocess, or serialize.
        """
        _latency = os.environ.get("HPS_LATENCY_LOG", "0") in ("1", "true", "True")
        batch_size = len(requests)
        t0 = time.perf_counter()

        # Phase 1: Decode all images, collect errors
        images: list[np.ndarray | None] = []
        errors: list[str | None] = []
        valid_indices: list[int] = []

        for i, request in enumerate(requests):
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(
                    request, INPUT_NAME
                )
                if input_tensor is None:
                    errors.append("Missing input tensor")
                    images.append(None)
                    continue

                input_data = input_tensor.as_numpy()
                # Shape is [1, 1] containing raw bytes payload.
                # Triton BYTES tensors return Python bytes objects directly.
                raw_payload = input_data[0, 0]
                if isinstance(raw_payload, np.ndarray):
                    payload = raw_payload.tobytes()
                elif isinstance(raw_payload, bytes):
                    payload = raw_payload
                else:
                    payload = bytes(raw_payload)

                img = _decode_image(payload)
                images.append(img)
                errors.append(None)
                valid_indices.append(i)
            except Exception as e:
                logger.exception("Failed to decode image %d", i)
                errors.append(str(e))
                images.append(None)

        t_decode = time.perf_counter()

        # Phase 2: Batch predict on valid images
        all_boxes: dict[int, list[dict[str, Any]]] = {}

        if valid_indices:
            valid_images = [images[i] for i in valid_indices]
            n_valid = len(valid_images)
            try:
                gen = self.model.predict(valid_images, batch_size=n_valid)
                results = list(gen)

                if len(results) != n_valid:
                    raise RuntimeError(
                        f"Batch predict returned {len(results)} results "
                        f"for {n_valid} images"
                    )

                for j, idx in enumerate(valid_indices):
                    all_boxes[idx] = _extract_boxes(results[j])
            except Exception as e:
                logger.exception(
                    "Batch predict failed (%d images)", n_valid
                )
                for idx in valid_indices:
                    errors[idx] = f"Inference error: {e}"

        t_predict = time.perf_counter()

        # Phase 3: Build responses
        responses = []
        for i in range(batch_size):
            if errors[i] is not None:
                payload = {"boxes": [], "error": errors[i]}
            else:
                payload = {
                    "boxes": all_boxes.get(i, []),
                    "error": None,
                }
            # _numpy_default handles numpy→native conversion at serialization
            # orjson is 3-10x faster than stdlib json
            data = _json_dumps_bytes(payload)
            data = np.array([[data]], dtype=np.object_)
            output_tensor = pb_utils.Tensor(OUTPUT_NAME, data)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )

        t_serialize = time.perf_counter()
        elapsed_ms = (t_serialize - t0) * 1000

        if _latency:
            decode_ms = (t_decode - t0) * 1000
            predict_ms = (t_predict - t_decode) * 1000
            serialize_ms = (t_serialize - t_predict) * 1000
            logger.info(
                '{"event":"latency","stage":"triton_execute",'
                '"batch_size":%d,"valid":%d,'
                '"decode_ms":%.3f,"predict_ms":%.3f,"serialize_ms":%.3f,'
                '"total_ms":%.3f}',
                batch_size, len(valid_indices),
                decode_ms, predict_ms, serialize_ms,
                elapsed_ms,
            )
        else:
            logger.info(
                "Batch processed: %d requests (%d valid) in %.1fms",
                batch_size, len(valid_indices), elapsed_ms,
            )

        return responses

    def finalize(self) -> None:
        """Cleanup on shutdown."""
        logger.info("Finalizing PP-DocLayoutV3 model")
