"""Layout detection inference — Triton (default) or direct in-process backend.

This module provides ``run_layout_detection(image)`` which dispatches to
one of two backends based on ``HPS_API_BACKEND``:

**Triton backend** (default, ``HPS_API_BACKEND=triton``):

  ┌────────────────────────┐       gRPC       ┌──────────────────────────┐
  │  api_compat (FastAPI)   │──────────────────▶│  Triton Inference Server │
  │  triton_client.py       │                   │  doclayout-v3 model      │
  │                         │◀──────────────────│  dynamic_batching { }    │
  │  await detect(image)    │     response     │  model.predict(batch)    │
  └────────────────────────┘                   └──────────────────────────┘

  Triton's dynamic batcher collects concurrent requests into batches of
  up to 8, dispatches them in one GPU forward pass, and queues the next
  batch while the current one executes.  This is continuous batching —
  the GPU never idles between batches.  No inference thread, no custom
  batching logic, no in-process model loading.

**Direct backend** (``HPS_API_BACKEND=direct``):

  Pycuda/TensorRT CUDA contexts are thread-local. We use a dedicated
  inference thread that loads the model and processes tasks from a queue,
  returning results via a future-based mechanism.

  When BATCH_SIZE > 1, the inference thread collects up to BATCH_SIZE
  images (or waits BATCH_TIMEOUT_MS) before issuing a single predict()
  call.  This is static micro-batching — simpler but less performant than
  Triton's continuous batching.

Both backends use the same semaphore (PIPELINE_DEPTH) to limit concurrent
in-flight requests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Optional

import numpy as np

from .config import (
    BATCH_SIZE,
    BATCH_TIMEOUT_MS,
    INFERENCE_BACKEND,
    MODEL_DEVICE_ID,
    MODEL_NAME,
    MODEL_PRECISION,
    PIPELINE_DEPTH,
)
from .latency import is_latency_logging_enabled

logger = logging.getLogger("hps_api")

# Conditionally import paddlex (only needed for the "direct" backend).
# The "triton" backend (default) never loads a model in-process.
if INFERENCE_BACKEND == "direct":
    from paddlex import create_model
    from .engine import prepare_engine


def _extract_boxes(result: Any) -> list[dict[str, Any]]:
    """Extract layout boxes from a PaddleX prediction result.

    Handles both dict-style and attribute-style results from PaddleX.
    """
    if isinstance(result, dict):
        return result.get("boxes", [])
    return getattr(result, "boxes", [])


class AppState:
    """Holds the loaded model and concurrency primitives.

    Backend selection (HPS_API_BACKEND):
      - "triton" (default): No model loaded in-process. Requests go to a
        Triton Inference Server via gRPC. Triton's dynamic_batching handles
        continuous batching, GPU scheduling, and request queuing.
      - "direct": Loads the PaddleX model in a dedicated thread. Uses our
        custom micro-batching. Simpler but less performant under load.

    The semaphore depth (PIPELINE_DEPTH) controls how many requests can
    be in-flight simultaneously. With the Triton backend, this limits
    concurrent gRPC calls; with the direct backend, it limits concurrent
    queued tasks.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(PIPELINE_DEPTH)
        self._ready = False
        self._load_error: Optional[Exception] = None
        # Dedicated inference thread (direct backend only)
        self._infer_thread: Optional[threading.Thread] = None
        self._task_queue: queue.Queue = queue.Queue()
        # Pending futures keyed by task_id — the inference thread resolves
        # the matching future when it finishes each task.
        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self._load_done = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready

    def start_inference(self) -> None:
        """Start the inference backend.

        For the "direct" backend, starts the dedicated inference thread
        (which loads the model). For "triton", just marks as ready — the
        Triton server manages its own model lifecycle.
        """
        if INFERENCE_BACKEND == "triton":
            # Triton manages the model. We just need the gRPC client.
            # Readiness is checked on first request.
            self._ready = True
            self._load_done.set()
            logger.info(
                "Triton backend active — model is managed by Triton server"
            )
        else:
            self._infer_thread = threading.Thread(
                target=_inference_worker, daemon=True, name="inference-worker"
            )
            self._infer_thread.start()

    def wait_ready(self, timeout: float) -> bool:
        """Block until the model is loaded or *timeout* seconds elapse.

        Returns True if the model loaded successfully.  If the inference
        thread crashed during ``load_model()``, re-raises the captured
        exception so the caller can fail fast instead of waiting for the
        full timeout.
        """
        self._load_done.wait(timeout=timeout)
        if self._load_error is not None:
            raise self._load_error
        return self._ready

    def shutdown(self) -> None:
        """Signal the inference backend to stop.

        For the "direct" backend, signals the inference thread to stop.
        For "triton", closes the gRPC client.
        """
        if INFERENCE_BACKEND == "triton":
            # Close the gRPC client asynchronously — fire and forget
            try:
                from . import triton_client
                loop = asyncio.new_event_loop()
                loop.run_until_complete(triton_client.close_client())
                loop.close()
            except Exception:
                pass  # best-effort cleanup
        else:
            self._task_queue.put((None, None, None))
            if self._infer_thread is not None:
                self._infer_thread.join(timeout=5)


# Singleton — one model instance per process
state = AppState()


def load_model() -> None:
    """Load the PP-DocLayoutV3 TensorRT model at startup."""
    engine_path = prepare_engine()
    logger.info(
        "Loading model %s (precision=%s, device=%d, engine=%s)",
        MODEL_NAME, MODEL_PRECISION, MODEL_DEVICE_ID, engine_path,
    )
    engine_config: dict[str, Any] = {
        "precision": MODEL_PRECISION,
        "device_id": MODEL_DEVICE_ID,
    }
    if os.path.exists(engine_path):
        engine_config["engine_path"] = engine_path

    state.model = create_model(
        MODEL_NAME,
        engine="tensorrt",
        engine_config=engine_config,
    )
    state._ready = True
    logger.info("Model loaded successfully")


def _inference_worker() -> None:
    """Dedicated thread: loads the model, then processes inference requests.

    The CUDA context stays in the thread that created it.  Tasks are
    processed one at a time (GPU is serial), but the asyncio layer can
    have multiple tasks queued simultaneously via the semaphore.

    When BATCH_SIZE > 1, the worker collects up to BATCH_SIZE images
    (waiting at most BATCH_TIMEOUT_MS for the batch to fill) and issues
    a single predict() call.  This amortizes kernel-launch overhead and
    lets the GPU process multiple images in one forward pass.

    Each result is delivered by resolving the matching Future, so the
    asyncio side can ``await`` without head-of-line blocking.
    """
    try:
        load_model()
    except Exception as e:
        state._load_error = e
        state._load_done.set()
        logger.exception("Model failed to load in inference thread")
        return
    state._load_done.set()
    logger.info(
        "Inference thread ready (pipeline_depth=%d, batch_size=%d, "
        "batch_timeout_ms=%.0f), waiting for tasks...",
        PIPELINE_DEPTH, BATCH_SIZE, BATCH_TIMEOUT_MS,
    )

    while True:
        # Block on the first task — no work to do until one arrives
        task = state._task_queue.get()
        if task[0] is None:  # shutdown sentinel
            break

        if BATCH_SIZE <= 1:
            _process_single(task)
        else:
            # Collect additional tasks up to BATCH_SIZE with a timeout
            batch = [task]
            deadline = None
            while len(batch) < BATCH_SIZE:
                if deadline is None:
                    deadline = time.monotonic() + BATCH_TIMEOUT_MS / 1000.0
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    extra = state._task_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if extra[0] is None:  # shutdown sentinel during batch wait
                    # Process what we have, then shut down
                    _process_batch(batch)
                    return
                batch.append(extra)
            _process_batch(batch)


def _process_single(task: tuple) -> None:
    """Process a single inference task (BATCH_SIZE=1 path)."""
    task_id, image, future = task
    try:
        gen = state.model.predict(image)
        results = list(gen)
        boxes = _extract_boxes(results[0]) if results else []
        future.set_result(boxes)
    except Exception as e:
        logger.exception("Inference failed for task %s", task_id)
        future.set_exception(e)
    finally:
        with state._pending_lock:
            state._pending.pop(task_id, None)


def _process_batch(batch: list[tuple]) -> None:
    """Process a batch of tasks in a single predict() call.

    Issues one predict() with all images, then splits results back to
    each task's future by position.  If the batch call fails, each
    future receives the exception.
    """
    n = len(batch)
    if n == 0:
        return

    _log = is_latency_logging_enabled()
    t0 = time.perf_counter() if _log else 0.0

    images = [t[1] for t in batch]

    try:
        gen = state.model.predict(images, batch_size=n)
        results = list(gen)

        if len(results) != n:
            raise RuntimeError(
                f"Batch predict returned {len(results)} results for "
                f"{n} images"
            )

        for i, (task_id, _img, future) in enumerate(batch):
            boxes = _extract_boxes(results[i])
            future.set_result(boxes)
    except Exception as e:
        logger.exception("Batch inference failed (%d images)", n)
        for task_id, _img, future in batch:
            if not future.done():
                future.set_exception(e)
    finally:
        if _log:
            elapsed = time.perf_counter() - t0
            logger.info(
                '{"event":"latency","stage":"batch_inference",'
                '"batch_size":%d,"elapsed_s":%.6f}',
                n, elapsed,
            )
        with state._pending_lock:
            for task_id, _img, _fut in batch:
                state._pending.pop(task_id, None)


async def run_layout_detection(image: np.ndarray) -> list[dict[str, Any]]:
    """Run layout detection on a single image.

    Dispatches to the configured inference backend:
      - "triton" (default): Sends image via gRPC to Triton Inference Server.
        Triton's dynamic_batching provides continuous batching — multiple
        concurrent calls to this function are automatically batched by
        Triton into a single GPU forward pass.
      - "direct": Submits work to the dedicated inference thread. Uses
        our custom micro-batching (BATCH_SIZE, BATCH_TIMEOUT_MS).

    The semaphore limits concurrent in-flight requests to PIPELINE_DEPTH.
    With the Triton backend, this controls how many gRPC calls are
    simultaneously awaiting response.

    Raises:
        RuntimeError: If the backend is unreachable or inference fails.
    """
    if INFERENCE_BACKEND == "triton":
        return await _run_triton(image)
    else:
        return await _run_direct(image)


async def _run_triton(image: np.ndarray) -> list[dict[str, Any]]:
    """Triton backend: send image via gRPC, let Triton handle batching."""
    from . import triton_client

    _log = is_latency_logging_enabled()
    t_sem_acquire = time.perf_counter() if _log else 0.0

    async with state.semaphore:
        if _log:
            sem_wait = time.perf_counter() - t_sem_acquire
            logger.info(
                '{"event":"latency","stage":"semaphore_wait",'
                '"wait_s":%.6f}',
                sem_wait,
            )

        t_infer_start = time.perf_counter() if _log else 0.0
        try:
            boxes = await triton_client.detect_layout(image)
        except Exception as e:
            raise RuntimeError(f"Triton inference failed: {e}") from e

        if _log:
            infer_time = time.perf_counter() - t_infer_start
            logger.info(
                '{"event":"latency","stage":"gpu_inference",'
                '"wait_s":%.6f}',
                infer_time,
            )

    return boxes


async def _run_direct(image: np.ndarray) -> list[dict[str, Any]]:
    """Direct backend: submit to the dedicated inference thread.

    The semaphore limits concurrent in-flight tasks to PIPELINE_DEPTH,
    allowing GPU work to be pipelined while keeping the event loop free.

    With ``PIPELINE_DEPTH=1`` this is fully serial (original behavior).
    With ``PIPELINE_DEPTH>1`` multiple requests can be queued, so the
    inference thread always has work ready — minimizing GPU idle gaps.

    Raises:
        RuntimeError: If the inference thread is no longer alive (crashed
            after startup).
    """
    INFERENCE_QUEUE_TIMEOUT = 120  # seconds

    _log = is_latency_logging_enabled()
    t_sem_acquire = time.perf_counter() if _log else 0.0

    async with state.semaphore:
        if _log:
            sem_wait = time.perf_counter() - t_sem_acquire
            logger.info(
                '{"event":"latency","stage":"semaphore_wait",'
                '"wait_s":%.6f}',
                sem_wait,
            )

        task_id = uuid.uuid4().hex
        future: Future = Future()

        with state._pending_lock:
            state._pending[task_id] = future

        t_infer_start = time.perf_counter() if _log else 0.0
        state._task_queue.put((task_id, image, future))

        # Await the future directly via asyncio.wrap_future — this avoids
        # the extra thread hop of run_in_executor(None, future.result),
        # which wastes a thread pool slot just to block on a Future.
        # wrap_future creates an asyncio Future backed by the concurrent
        # Future, so the event loop is notified when it completes.
        try:
            boxes = await asyncio.wait_for(
                asyncio.wrap_future(future, loop=asyncio.get_running_loop()),
                timeout=INFERENCE_QUEUE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            with state._pending_lock:
                state._pending.pop(task_id, None)
            raise RuntimeError(
                f"Inference timed out after {INFERENCE_QUEUE_TIMEOUT}s "
                f"for task {task_id}"
            ) from None

        if _log:
            infer_time = time.perf_counter() - t_infer_start
            logger.info(
                '{"event":"latency","stage":"gpu_inference",'
                '"wait_s":%.6f}',
                infer_time,
            )

    return boxes
