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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from .config import (
    BATCH_SIZE,
    BATCH_TIMEOUT_MS,
    CPU_POOL_SIZE,
    INFERENCE_BACKEND,
    MODEL_DEVICE_ID,
    MODEL_NAME,
    MODEL_PRECISION,
    PIPELINE_DEPTH,
    PRE_POST_POOL_SIZE,
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
        self._load_error: Exception | None = None
        # Dedicated inference thread (direct backend only)
        self._infer_thread: threading.Thread | None = None
        self._task_queue: queue.Queue = queue.Queue()
        # Monotonic task counter — replaces uuid.uuid4() which calls
        # os.urandom (syscall) per request.  task_id is only used for
        # logging, never for lookup.
        self._task_counter = 0
        self._task_counter_lock = threading.Lock()
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

    def _next_task_id(self) -> int:
        """Generate a monotonic task ID (replaces uuid.uuid4().hex)."""
        with self._task_counter_lock:
            tid = self._task_counter
            self._task_counter += 1
        return tid

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
            self._task_queue.put((None, None, None, 0.0))
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


# ─── Pipelined inference: split pre / GPU / post ─────────────────────────────
#
# The original code calls state.model.predict(images), which runs ALL of
# pre-processing → GPU inference → post-processing sequentially on the
# inference thread.  This means the GPU sits idle during CPU-bound pre/post
# work (~5-8 ms per batch).
#
# By splitting into three stages we can overlap post-processing with the
# next batch's pre+GPU:
#
#   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
#   │ Pre batch N │───▶│ GPU batch N │───▶│ Post batch N│
#   │ (CPU pool)  │    │ (infer thd) │    │ (CPU pool)  │
#   └─────────────┘    └─────────────┘    └─────────────┘
#          ↑                   ↑                   ↑
#          │   serial (await)  │   non-blocking    │
#          │                   │   overlaps N+1    │
#
# Pre-processing (ReadImage, Resize, Normalize, ToCHW, ToBatch) runs on
# the _pre_post_pool thread pool, producing batch_inputs ready for the GPU.
# The inference thread only does: runner(batch_inputs) + _format_output().
# Post-processing (LayoutAnalysisProcess) also runs on _pre_post_pool.
#
# IMPORTANT — GIL constraint (measured 2026-07-30):
# Pre-processing is submitted to the pool but AWAITED serially before GPU.
# A double-buffered approach (submitting pre(N+1) before GPU(N) to overlap
# them) was tested and REVERTED — it causes 2-3× H2D copy inflation:
#   - pycuda's memcpy_htod_async() Python wrapper needs the GIL
#   - numpy/cv2 pre-processing on the pool thread holds the GIL
#   - Result: H2D copy went from 9 ms → 25 ms, negating the overlap gain
# Post-processing is GIL-safe to overlap: it only does CPU-side box
# decoding (no CUDA calls), so it doesn't contend with H2D/D2H copies.

# Dedicated pool for pre/post-processing (separate from CPU_POOL_SIZE which
# is used by the API layer for image decode / DoclingDocument conversion).
_pre_post_pool: ThreadPoolExecutor | None = None


def _get_pre_post_pool() -> ThreadPoolExecutor:
    """Lazily create the pre/post-processing thread pool."""
    global _pre_post_pool
    if _pre_post_pool is None:
        _pre_post_pool = ThreadPoolExecutor(
            max_workers=PRE_POST_POOL_SIZE,
            thread_name_prefix="pre-post",
        )
    return _pre_post_pool


# ─── Fused pre-processing (Fix 7) ─────────────────────────────────────────────
#
# Per-op profiling revealed Normalize (10ms) + ToBatch (5ms) = 15ms per
# batch of 4, dominating pre-processing (23.8ms total).  The current flow:
#   Normalize: img.astype(float32) * alpha  → 2 memory passes per image
#   ToCHW:     img.transpose(2,0,1)         → strided view (free)
#   ToBatch:   np.stack(imgs)               → 1 copy per image
# Total: ~3 passes over 7.68 MB/image × 4 = ~46 MB memory traffic.
#
# The fused approach replaces Normalize+ToCHW+ToBatch with a single
# pre-allocated (N,3,H,W) float32 buffer:
#   1. buf[i] = img.transpose(2,0,1)  → uint8→float32 + HWC→CHW in 1 pass
#   2. buf *= alpha                    → in-place normalize (contiguous, cache-friendly)
# Total: ~2 passes over 7.68 MB/image × 4 = ~23 MB memory traffic.
#
# Benchmark (host, batch=4, 800×800):
#   Current (Normalize+ToCHW+ToBatch): 17.0 ms
#   Fused:                              2.9 ms  (5.9× faster)
#
# ReadImage (BGR→RGB, 4ms) and Resize (cv2.resize, 5ms) remain as-is —
# they are needed for correctness and post-processing metadata.

def _fused_normalize_to_batch(
    datas: list[dict],
    alpha: float,
) -> list[np.ndarray]:
    """Fuse Normalize + ToCHW + ToBatch into a single pre-allocated buffer.

    Returns a list of batched arrays matching ToBatch's output format:
    [img, img_size, scale_factors] for PP-DocLayoutV3 (sorted by input name).

    Thread-safe: allocates per call (np.empty is ~0.01ms for 29MB, negligible
    vs the 3ms copy).  A module-level reused buffer would risk data races
    when pre(N+1) overlaps with pre(N) on the pool.
    """
    n = len(datas)

    # Get target shape from first image (already resized)
    first_img = datas[0]["img"]
    c, h, w = first_img.shape[2], first_img.shape[0], first_img.shape[1]

    # Single allocation for the full batch (N, C, H, W) float32
    buf = np.empty((n, c, h, w), dtype=np.float32)

    # Single-pass: uint8→float32 + HWC→CHW for each image
    for i, data in enumerate(datas):
        img = data["img"]  # (H, W, C) uint8, already RGB (ReadImage converted)
        buf[i] = img.transpose(2, 0, 1)

    # In-place normalize (contiguous buffer → cache-friendly)
    buf *= alpha

    # Batch metadata (img_size, scale_factors) — same as ToBatch.apply
    img_sizes = np.stack(
        [data["img_size"][::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)
    scale_factors = np.stack(
        [data.get("scale_factors", [1.0, 1.0])[::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)

    # Return in the same order as ToBatch.__call__:
    # ordered_required_keys = ("img_size", "img", "scale_factors")
    # The runner re-sorts by model input name, so we must match ToBatch's
    # output order exactly.
    result = [img_sizes, buf, scale_factors]
    return result


# ─── Pinned-memory pool for async H2D uploads ─────────────────────────────────
#
# torch.from_numpy(pageable).to("cuda", non_blocking=True) is effectively
# SYNCHRONOUS because non_blocking only works with pinned (page-locked)
# memory.  Under load this turns the GPU pre path from 0.5ms (standalone)
# into 8ms (C=20), defeating the purpose.
#
# Solution: use a pinned-memory staging buffer.  Copy each image into a
# pre-allocated pinned buffer, then launch a truly async H2D from there.
# The CPU is free to continue while the DMA runs on the GPU.

_PINNED_BUF = None  # torch.Tensor (pinned uint8 staging), lazy-allocated


def _get_pinned_staging(h: int, w: int, n: int):
    """Get or grow a pinned-memory buffer for staging (N, H, W, C) uint8.

    Pinned memory is expensive to allocate, so we reuse a single buffer
    and grow it only when the batch size or image dimensions change.
    """
    import torch

    global _PINNED_BUF
    needed = n * h * w * 3  # 3 channels, uint8
    if _PINNED_BUF is None or _PINNED_BUF.numel() < needed:
        # Allocate with extra headroom to avoid frequent reallocation
        alloc_n = max(n, 8)
        _PINNED_BUF = torch.empty(
            (alloc_n, h, w, 3), dtype=torch.uint8, pin_memory=True
        )
    return _PINNED_BUF[:n, :h, :w, :]


def _fused_normalize_to_batch_gpu(
    datas: list[dict],
    alpha: float,
) -> list:
    """GPU-accelerated fused Normalize + ToCHW + ToBatch using torch.

    Uses pinned-memory staging for truly async H2D uploads, then
    performs permute + normalize + stack on-device.  Returns a
    GPU-resident torch tensor for the "image" input.  The runner detects
    the GPU tensor and sets the TensorRT tensor address directly —
    eliminating the separate H2D copy in the runner entirely.

    Returns [img_sizes(np), img_gpu(torch), scale_factors(np)] matching
    ToBatch's output order.  The metadata arrays remain on CPU (small,
    and the runner needs to set_input_shape from them).
    """
    import torch

    n = len(datas)
    first_img = datas[0]["img"]
    h, w, c = first_img.shape[0], first_img.shape[1], first_img.shape[2]

    # Stage all images into a pinned buffer, then launch ONE async H2D
    # for the entire batch.  This is faster than per-image uploads and
    # the pinned source enables true non_blocking DMA.
    staging = _get_pinned_staging(h, w, n)
    for i, data in enumerate(datas):
        # numpy → pinned torch (CPU memcpy, fast)
        staging[i] = torch.from_numpy(data["img"])

    # Single async H2D: pinned → GPU (truly non-blocking)
    gpu_staging = staging.to("cuda", non_blocking=True)

    # On-GPU: permute HWC→CHW, float, normalize, stack — all device kernels
    batch = gpu_staging.permute(0, 3, 1, 2).float() * alpha
    batch = batch.contiguous()  # ensure contiguous for TRT

    # Metadata stays on CPU (small: 2 floats per image)
    img_sizes = np.stack(
        [data["img_size"][::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)
    scale_factors = np.stack(
        [data.get("scale_factors", [1.0, 1.0])[::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)

    return [img_sizes, batch, scale_factors]


def _run_preprocessing(predictor: Any, images: list[np.ndarray]) -> tuple:
    """Run pre-processing (CPU) to produce batch_inputs ready for GPU.

    Returns (batch_inputs, datas) where:
      - batch_inputs: list of numpy arrays for the GPU runner
      - datas: list of per-image dicts (needed by post-processing)
    """
    # Build batch_data the same way ImageBatchSampler would
    from paddlex.inference.common.batch_sampler.image_batch_sampler import (
        ImgBatch,
    )

    batch_data = ImgBatch()
    for img in images:
        batch_data.append(img, None, None, None)

    datas = batch_data.instances

    _log = is_latency_logging_enabled()
    _t = [time.perf_counter()] if _log else None

    # Apply ReadImage + Resize only (pre_ops[:-3] skips Normalize, ToCHW, ToBatch)
    # pre_ops layout: [ReadImage, Resize, Normalize, ToCHWImage, ToBatch]
    pre_ops_head = predictor.pre_ops[:-3]
    for pre_op in pre_ops_head:
        datas = pre_op(datas)
        if _log:
            _t.append(time.perf_counter())

    # Fused Normalize + ToCHW + ToBatch
    # Extract alpha from the Normalize op (pre_ops[-3])
    norm_op = predictor.pre_ops[-3]
    alpha = (
        norm_op._alpha_scalar
        if getattr(norm_op, "_alpha_scalar", None) is not None
        else float(norm_op.alpha[0])
    )

    # GPU pre-processing: upload + normalize on GPU, return GPU-resident
    # tensor. The runner detects it and skips H2D copy entirely.
    # Controlled by HPS_GPU_PRE=1 (default off — safe fallback).
    _use_gpu_pre = os.environ.get("HPS_GPU_PRE", "0") in ("1", "true", "True")
    if _use_gpu_pre:
        batch_inputs = _fused_normalize_to_batch_gpu(datas, alpha)
    else:
        batch_inputs = _fused_normalize_to_batch(datas, alpha)
    if _log:
        _t.append(time.perf_counter())
        fused_name = "FusedNormToBatchGPU" if _use_gpu_pre else "FusedNormToBatch"
        op_names = [type(op).__name__ for op in pre_ops_head] + [fused_name]
        parts = ",".join(
            f'"{op_names[i]}":{( _t[i+1] - _t[i]) * 1000:.3f}'
            for i in range(len(op_names))
        )
        total = (_t[-1] - _t[0]) * 1000
        logger.info(
            '{"event":"latency","stage":"pre_ops_detail",'
            '"total_ms":%.3f,"ops":[%s]}',
            total, parts,
        )

    return batch_inputs, datas


def _run_gpu_inference(
    predictor: Any, batch_inputs: tuple
) -> list[dict]:
    """Run GPU inference + output formatting (inference thread only).

    Returns preds_list — a list of per-image prediction dicts.
    """
    batch_preds = predictor.runner(batch_inputs)
    preds_list = predictor._format_output(batch_preds)
    return preds_list


def _run_postprocessing(
    predictor: Any, preds_list: list[dict], datas: list[dict]
) -> list[list[dict]]:
    """Run post-processing (CPU) to produce layout boxes.

    Returns a list of box-lists, one per image.
    """
    boxes = predictor.post_op(
        preds_list,
        datas,
        threshold=predictor.threshold,
        layout_nms=predictor.layout_nms,
        layout_unclip_ratio=predictor.layout_unclip_ratio,
        layout_merge_bboxes_mode=predictor.layout_merge_bboxes_mode,
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
        skip_order_labels=None,
    )
    return boxes if isinstance(boxes, list) else [boxes]


# ─── Pipeline state: overlap pre of batch N+1 with GPU of batch N ────────────
#
# The inference worker loop implements double-buffered pipelining inline
# (see _inference_worker).  No module-level globals needed — the pipeline
# state is local to the worker loop.


def _process_batch_pipelined(batch: list[tuple]) -> None:
    """Process a batch with pre on CPU pool, GPU on inference thread, post on CPU pool.

    This is the non-pipelined fallback (no double-buffer overlap).
    Used for shutdown sentinel flush and first batch when the main
    pipeline loop isn't active.
    """
    n = len(batch)
    if n == 0:
        return

    _log = is_latency_logging_enabled()
    t0 = time.perf_counter() if _log else 0.0
    t_enqueue_min = min(t[3] for t in batch)
    images = [t[1] for t in batch]

    pool = _get_pre_post_pool()
    predictor = state.model._predictor

    try:
        pre_future = pool.submit(_run_preprocessing, predictor, images)
        batch_inputs, datas = pre_future.result()
        preds_list = _run_gpu_inference(predictor, batch_inputs)
        boxes_list = _run_postprocessing(predictor, preds_list, datas)

        if len(boxes_list) != n:
            raise RuntimeError(
                f"Post-processing returned {len(boxes_list)} results "
                f"for {n} images"
            )

        for i, (_task_id, _img, future, _t) in enumerate(batch):
            future.set_result(boxes_list[i])
    except Exception as e:
        logger.exception("Batch inference failed (%d images)", n)
        for _task_id, _img, future, _t in batch:
            if not future.done():
                future.set_exception(e)
    finally:
        if _log:
            elapsed = time.perf_counter() - t0
            queue_wait = (t0 - t_enqueue_min) * 1000
            logger.info(
                '{"event":"latency","stage":"batch_inference",'
                '"batch_size":%d,"queue_wait_ms":%.3f,'
                '"predict_ms":%.3f,"elapsed_s":%.6f}',
                n, queue_wait, elapsed * 1000, elapsed,
            )


def _resolve_batch(
    batch: list[tuple],
    preds_list: list[dict],
    datas: list[dict],
) -> None:
    """Run post-processing for a batch and resolve its futures.

    NON-BLOCKING: submits post-processing to the CPU pool and returns
    immediately.  Futures are resolved via add_done_callback when the
    pool task completes.  This lets the inference thread proceed to
    collect the next batch without waiting for post to finish.
    """
    n = len(batch)
    if n == 0:
        return

    _log = is_latency_logging_enabled()
    t0 = time.perf_counter() if _log else 0.0
    predictor = state.model._predictor
    pool = _get_pre_post_pool()

    def _on_post_done(post_future: Future) -> None:
        """Resolve the batch futures when post-processing completes."""
        try:
            boxes_list = post_future.result()
            if len(boxes_list) != n:
                raise RuntimeError(
                    f"Post-processing returned {len(boxes_list)} results "
                    f"for {n} images"
                )
            for i, (_task_id, _img, future, _t) in enumerate(batch):
                future.set_result(boxes_list[i])
        except Exception as e:
            logger.exception("Post-processing failed (%d images)", n)
            for _task_id, _img, future, _t in batch:
                if not future.done():
                    future.set_exception(e)
        finally:
            if _log:
                elapsed = time.perf_counter() - t0
                logger.info(
                    '{"event":"latency","stage":"post_resolve",'
                    '"batch_size":%d,"post_ms":%.3f,"elapsed_s":%.6f}',
                    n, elapsed * 1000, elapsed,
                )

    # Submit post to pool — non-blocking
    pool.submit(
        _run_postprocessing, predictor, preds_list, datas
    ).add_done_callback(_on_post_done)


def _inference_worker() -> None:
    """Dedicated thread: loads the model, then processes inference requests.

    The CUDA context stays in the thread that created it.  Tasks are
    processed one at a time (GPU is serial), but the asyncio layer can
    have multiple tasks queued simultaneously via the semaphore.

    **Double-buffer pre/GPU overlap with non-blocking post**:

        Time ──▶
        Pre  N:   ████
        GPU  N:        ████
        Post N:             ████  (non-blocking → pool)
        Pre  N+1:     ████       (submitted BEFORE GPU N, overlaps!)
        GPU  N+1:          ████
        Post N+1:               ████

    Pre(N+1) is submitted to the pool BEFORE GPU(N) starts, so CPU
    pre-processing overlaps with GPU execution.  After GPU(N) completes,
    we await pre(N+1) (which is likely already done) then run GPU(N+1).

    GIL safety: pycuda's ``memcpy_htod_async`` and ``stream.synchronize``
    BOTH release the GIL (verified empirically).  Concurrent numpy
    pre-processing on the pool thread does not block CUDA calls.
    Memory-bandwidth contention is minimal (1.1x measured).

    Post-processing is non-blocking (submitted to pool with callback)
    so it overlaps with the next batch's pre+GPU without blocking.
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
        "batch_timeout_ms=%.0f, pre_post_pool=%d), waiting for tasks...",
        PIPELINE_DEPTH, BATCH_SIZE, BATCH_TIMEOUT_MS, PRE_POST_POOL_SIZE,
    )

    predictor = state.model._predictor
    pool = _get_pre_post_pool()

    # prev_post: (batch, preds, datas) awaiting post-processing, to be
    # resolved (non-blocking) at the start of the next iteration so it
    # overlaps with this batch's pre + GPU.
    prev_post: tuple | None = None  # (batch, preds, datas)

    # next_pre: future for pre-processing of the NEXT batch, submitted
    # before GPU(N) so it overlaps.  Awaited after GPU(N) completes.
    next_pre: Future | None = None
    next_batch: list[tuple] | None = None

    def _collect_batch_blocking() -> list[tuple] | None:
        """Block on the first task, then fill non-blocking.  Two-phase
        with BATCH_TIMEOUT_MS wait for the batch to fill."""
        task = state._task_queue.get()
        if task[0] is None:
            return [task]  # sentinel
        batch = [task]
        # Phase 1: non-blocking fill
        while len(batch) < BATCH_SIZE:
            try:
                extra = state._task_queue.get_nowait()
            except queue.Empty:
                break
            if extra[0] is None:
                batch.append(extra)
                return batch
            batch.append(extra)
        # Phase 2: timeout wait for more
        while len(batch) < BATCH_SIZE:
            try:
                extra = state._task_queue.get(
                    timeout=BATCH_TIMEOUT_MS / 1000.0
                )
            except queue.Empty:
                break
            if extra[0] is None:
                batch.append(extra)
                return batch
            batch.append(extra)
        return batch

    def _has_sentinel(batch: list[tuple]) -> bool:
        return any(t[0] is None for t in batch)

    def _try_collect_nonblocking() -> list[tuple] | None:
        """Non-blocking attempt to collect a full batch.  Returns None
        if queue is empty.  Used for pre-fetching the next batch."""
        try:
            task = state._task_queue.get_nowait()
        except queue.Empty:
            return None
        if task[0] is None:
            return [task]  # sentinel
        batch = [task]
        while len(batch) < BATCH_SIZE:
            try:
                extra = state._task_queue.get_nowait()
            except queue.Empty:
                break
            if extra[0] is None:
                batch.append(extra)
                return batch
            batch.append(extra)
        return batch

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        # 0a. If we have a pre-fetched next_batch from the previous
        #     iteration, use it.  Otherwise collect a new batch (blocking).
        if next_batch is not None:
            batch = next_batch
            next_batch = None
        else:
            batch = _collect_batch_blocking()

        if batch is None or _has_sentinel(batch):
            if prev_post is not None:
                _resolve_batch(*prev_post)
                prev_post = None
            if next_pre is not None:
                # Drain the pre-fetched pre-processing result
                try:
                    next_pre.result(timeout=1.0)
                except Exception:
                    pass
                next_pre = None
            if batch is not None:
                real = [t for t in batch if t[0] is not None]
                if real:
                    _process_batch_pipelined(real)
            break

        _log = is_latency_logging_enabled()
        t0 = time.perf_counter() if _log else 0.0
        t_enqueue_min = min(t[3] for t in batch)

        # 1. Resolve post of PREVIOUS batch (non-blocking — submits to pool)
        if prev_post is not None:
            _resolve_batch(*prev_post)
            prev_post = None

        # 2. Get pre-processing result.
        #    If next_pre exists (submitted in previous iteration), await it.
        #    Otherwise, submit and await (first iteration or low concurrency).
        t_pre_wait = time.perf_counter() if _log else 0.0
        if next_pre is not None:
            batch_inputs, datas = next_pre.result()
            next_pre = None
        else:
            images = [t[1] for t in batch]
            pre_future = pool.submit(_run_preprocessing, predictor, images)
            batch_inputs, datas = pre_future.result()
        t_pre_done = time.perf_counter() if _log else 0.0

        # 3. Pre-fetch next batch and submit its pre-processing BEFORE
        #    GPU(N).  This overlaps CPU pre(N+1) with GPU(N).
        #    Only do this if the queue has items (high concurrency).
        if state._task_queue.qsize() > 0:
            next_batch = _try_collect_nonblocking()
            if next_batch is not None and not _has_sentinel(next_batch):
                next_images = [t[1] for t in next_batch]
                next_pre = pool.submit(
                    _run_preprocessing, predictor, next_images
                )
            else:
                # Got sentinel or None — don't pre-fetch
                if next_batch is not None and _has_sentinel(next_batch):
                    # Put sentinel back — handle next iteration
                    pass
                next_batch = None

        # 4. Run GPU inference — overlaps with pre(N+1) on the pool thread
        t_gpu_start = time.perf_counter() if _log else 0.0
        preds_list = _run_gpu_inference(predictor, batch_inputs)

        if _log:
            gpu_ms = (time.perf_counter() - t_gpu_start) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            queue_wait = (t0 - t_enqueue_min) * 1000
            pre_wait_ms = (t_pre_done - t_pre_wait) * 1000
            overlapped = next_pre is not None
            logger.info(
                '{"event":"latency","stage":"pipeline_iter",'
                '"batch_size":%d,"queue_wait_ms":%.3f,'
                '"pre_wait_ms":%.3f,"gpu_ms":%.3f,"total_ms":%.3f,'
                '"overlapped":%s}',
                len(batch), queue_wait, pre_wait_ms, gpu_ms, total_ms,
                "true" if overlapped else "false",
            )

        # 5. Save for next iteration's non-blocking post-processing
        prev_post = (batch, preds_list, datas)

        # 6. If no next batch was pre-fetched (low concurrency), flush
        #    post immediately for prompt future resolution.
        if next_batch is None and prev_post is not None:
            _resolve_batch(*prev_post)
            prev_post = None

    # End of loop — flush any remaining post
    if prev_post is not None:
        _resolve_batch(*prev_post)


def _process_single(task: tuple) -> None:
    """Process a single inference task (BATCH_SIZE=1 path).

    Delegates to _process_batch_pipelined which splits pre/GPU/post
    across threads for better overlap.
    """
    _process_batch_pipelined([task])


def _process_batch(batch: list[tuple]) -> None:
    """Process a batch of tasks — delegates to the pipelined version.

    The pipelined version splits pre-processing and post-processing onto
    a CPU thread pool, so they overlap with GPU work from the previous
    batch instead of blocking the inference thread.
    """
    _process_batch_pipelined(batch)


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

        task_id = state._next_task_id()
        future: Future = Future()

        t_enqueue = time.perf_counter()
        t_infer_start = t_enqueue if _log else 0.0
        state._task_queue.put((task_id, image, future, t_enqueue))

        # Await the future directly via asyncio.wrap_future — this avoids
        # the extra thread hop of run_in_executor(None, future.result),
        # which wastes a thread pool slot just to block on a Future.
        # wrap_future creates an asyncio Future backed by the concurrent
        # Future, so the event loop is notified when it completes.
        #
        # NOTE: asyncio.wait_for removed — it creates a TimerHandle per
        # request that's never triggered (timeout is 120s).  Direct await
        # is simpler and saves ~0.05 ms of event-loop overhead.
        try:
            boxes = await asyncio.wrap_future(
                future, loop=asyncio.get_running_loop()
            )
        except Exception:
            raise RuntimeError(
                f"Inference failed for task {task_id}"
            ) from None

        if _log:
            infer_time = time.perf_counter() - t_infer_start
            logger.info(
                '{"event":"latency","stage":"gpu_inference",'
                '"wait_s":%.6f}',
                infer_time,
            )

    return boxes
