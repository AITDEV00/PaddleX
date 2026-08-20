"""Direct in-process backend — PaddleX/TensorRT loaded in the API process.

This backend runs the PP-DocLayoutV3 model in-process (no Triton server).
It is the original lean-image inference path, extracted here so the API
layer only talks to the :class:`InferenceBackend` interface.

Highlights
----------
- A dedicated inference thread owns the CUDA context and loads the model.
- Requests are submitted to a queue and resolved via ``concurrent.futures.Future``.
- Pre-processing and post-processing run on a CPU thread pool and overlap
  with GPU inference (double-buffered pipeline).
- Micro-batching: ``BATCH_SIZE`` images are collected before a single
  ``predict()`` call (or flushed after ``BATCH_TIMEOUT_MS``).

Select via ``HPS_API_BACKEND=direct`` (the default for lean images).
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

from ..config import (
    BATCH_SIZE,
    BATCH_TIMEOUT_MS,
    MODEL_DEVICE_ID,
    MODEL_NAME,
    MODEL_PRECISION,
    MODEL_THRESHOLD,
    PIPELINE_DEPTH,
    PRE_POST_POOL_SIZE,
)
from ..latency import is_latency_logging_enabled
from .base import InferenceBackend

logger = logging.getLogger("hps_api")


def _extract_boxes(result: Any) -> list[dict[str, Any]]:
    """Extract layout boxes from a PaddleX prediction result."""
    if isinstance(result, dict):
        return result.get("boxes", [])
    return getattr(result, "boxes", [])


# ─── Fused pre-processing ─────────────────────────────────────────────────────
#
# Replace Normalize + ToCHW + ToBatch with a single pre-allocated
# (N,3,H,W) float32 buffer.  Benchmark (host, batch=4, 800x800):
#   Current (Normalize+ToCHW+ToBatch): 17.0 ms
#   Fused:                              2.9 ms  (5.9x faster)


def _fused_normalize_to_batch(
    datas: list[dict],
    alpha: float,
) -> list[np.ndarray]:
    """Fuse Normalize + ToCHW + ToBatch into a single buffer (CPU)."""
    n = len(datas)
    first_img = datas[0]["img"]
    c, h, w = first_img.shape[2], first_img.shape[0], first_img.shape[1]

    buf = np.empty((n, c, h, w), dtype=np.float32)
    for i, data in enumerate(datas):
        img = data["img"]  # (H, W, C) uint8, already RGB
        buf[i] = img.transpose(2, 0, 1)
    buf *= alpha

    img_sizes = np.stack(
        [data["img_size"][::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)
    scale_factors = np.stack(
        [data.get("scale_factors", [1.0, 1.0])[::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)

    # Match ToBatch.__call__ output order: (img_size, img, scale_factors)
    return [img_sizes, buf, scale_factors]


# Pinned-memory pool for async H2D uploads (GPU pre path)
_PINNED_BUF = None  # torch.Tensor, lazy-allocated


def _get_pinned_staging(h: int, w: int, n: int):
    import torch

    global _PINNED_BUF
    needed = n * h * w * 3
    if _PINNED_BUF is None or _PINNED_BUF.numel() < needed:
        alloc_n = max(n, 8)
        _PINNED_BUF = torch.empty(
            (alloc_n, h, w, 3), dtype=torch.uint8, pin_memory=True
        )
    return _PINNED_BUF[:n, :h, :w, :]


def _fused_normalize_to_batch_gpu(
    datas: list[dict],
    alpha: float,
) -> list:
    """GPU-accelerated fused Normalize + ToCHW + ToBatch (torch)."""
    import torch

    n = len(datas)
    first_img = datas[0]["img"]
    h, w, c = first_img.shape[0], first_img.shape[1], first_img.shape[2]

    staging = _get_pinned_staging(h, w, n)
    for i, data in enumerate(datas):
        staging[i] = torch.from_numpy(data["img"])

    gpu_staging = staging.to("cuda", non_blocking=True)
    batch = gpu_staging.permute(0, 3, 1, 2).float() * alpha
    batch = batch.contiguous()

    img_sizes = np.stack(
        [data["img_size"][::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)
    scale_factors = np.stack(
        [data.get("scale_factors", [1.0, 1.0])[::-1] for data in datas], axis=0
    ).astype(dtype=np.float32, copy=False)

    return [img_sizes, batch, scale_factors]


class DirectBackend(InferenceBackend):
    """In-process PaddleX/TensorRT layout-detection backend."""

    name = "direct"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model: Any = None
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(PIPELINE_DEPTH)
        self._infer_thread: threading.Thread | None = None
        self._task_queue: queue.Queue = queue.Queue()
        self._load_error: Exception | None = None
        self._load_done = threading.Event()
        self._task_counter = 0
        self._task_counter_lock = threading.Lock()
        self._pre_post_pool: ThreadPoolExecutor | None = None
        self._default_threshold: float | dict | None = MODEL_THRESHOLD

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get_pre_post_pool(self) -> ThreadPoolExecutor:
        if self._pre_post_pool is None:
            self._pre_post_pool = ThreadPoolExecutor(
                max_workers=PRE_POST_POOL_SIZE,
                thread_name_prefix="pre-post",
            )
        return self._pre_post_pool

    def _next_task_id(self) -> int:
        with self._task_counter_lock:
            tid = self._task_counter
            self._task_counter += 1
        return tid

    # ── InferenceBackend lifecycle ───────────────────────────────────────────
    def start(self) -> None:
        """Start the dedicated inference thread (which loads the model)."""
        self._infer_thread = threading.Thread(
            target=self._infer_worker,
            daemon=True,
            name="inference-worker",
        )
        self._infer_thread.start()

    def wait_ready(self, timeout: float) -> bool:
        """Block until the model is loaded (or raise on fatal load error)."""
        self._load_done.wait(timeout=timeout)
        if self._load_error is not None:
            raise self._load_error
        return self._ready

    def shutdown(self) -> None:
        """Signal the inference thread to stop (best-effort)."""
        self._task_queue.put((None, None, None, 0.0, None, None))
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=5)
        if self._pre_post_pool is not None:
            self._pre_post_pool.shutdown(wait=False)

    # ── model loading ────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Load the PP-DocLayoutV3 TensorRT model in the current thread."""
        from paddlex import create_model
        from ..engine import prepare_engine

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

        self.model = create_model(
            MODEL_NAME,
            engine="tensorrt",
            engine_config=engine_config,
        )
        self._ready = True
        logger.info("Model loaded successfully")

    # ── pipeline stages ──────────────────────────────────────────────────────
    def _run_preprocessing(self, predictor: Any, images: list[np.ndarray]) -> tuple:
        """CPU pre-processing → batch_inputs ready for GPU."""
        from paddlex.inference.common.batch_sampler.image_batch_sampler import (
            ImgBatch,
        )

        batch_data = ImgBatch()
        for img in images:
            batch_data.append(img, None, None, None)
        datas = batch_data.instances

        _log = is_latency_logging_enabled()
        _t = [time.perf_counter()] if _log else None

        pre_ops_head = predictor.pre_ops[:-3]  # skip Normalize, ToCHW, ToBatch
        for pre_op in pre_ops_head:
            datas = pre_op(datas)
            if _log:
                _t.append(time.perf_counter())

        norm_op = predictor.pre_ops[-3]
        alpha = (
            norm_op._alpha_scalar
            if getattr(norm_op, "_alpha_scalar", None) is not None
            else float(norm_op.alpha[0])
        )

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
                f'"{op_names[i]}":{(_t[i + 1] - _t[i]) * 1000:.3f}'
                for i in range(len(op_names))
            )
            logger.info(
                '{"event":"latency","stage":"pre_ops_detail","total_ms":%.3f,'
                '"ops":[%s]}',
                (_t[-1] - _t[0]) * 1000, parts,
            )

        return batch_inputs, datas

    @staticmethod
    def _run_gpu_inference(predictor: Any, batch_inputs: tuple) -> list[dict]:
        """Run GPU inference + output formatting (inference thread only)."""
        batch_preds = predictor.runner(batch_inputs)
        return predictor._format_output(batch_preds)

    @staticmethod
    def _run_postprocessing(
        predictor: Any,
        preds_list: list[dict],
        datas: list[dict],
        threshold: float | dict | None = None,
        layout_kwargs: dict[str, Any] | None = None,
    ) -> list[list[dict]]:
        """Run post-processing (CPU) to produce layout boxes.

        ``threshold`` and ``layout_kwargs`` are per-request overrides; any
        ``None`` value falls back to the deployment default on the predictor.
        """
        threshold = threshold if threshold is not None else predictor.threshold
        layout_kwargs = layout_kwargs or {}
        boxes = predictor.post_op(
            preds_list,
            datas,
            threshold=threshold,
            layout_nms=layout_kwargs.get(
                "layout_nms", predictor.layout_nms
            ),
            layout_unclip_ratio=layout_kwargs.get(
                "layout_unclip_ratio", predictor.layout_unclip_ratio
            ),
            layout_merge_bboxes_mode=layout_kwargs.get(
                "layout_merge_bboxes_mode",
                predictor.layout_merge_bboxes_mode,
            ),
            layout_shape_mode=layout_kwargs.get("layout_shape_mode", "auto"),
            filter_overlap_boxes=layout_kwargs.get(
                "filter_overlap_boxes", True
            ),
            skip_order_labels=None,
        )
        return boxes if isinstance(boxes, list) else [boxes]

    # ── batch processing ─────────────────────────────────────────────────────
    def _process_batch_pipelined(self, batch: list[tuple]) -> None:
        """Process a batch with pre/post on CPU pool, GPU on inference thread.

        Non-pipelined fallback (no double-buffer overlap).
        """
        n = len(batch)
        if n == 0:
            return

        _log = is_latency_logging_enabled()
        t0 = time.perf_counter() if _log else 0.0
        t_enqueue_min = min(t[3] for t in batch)
        images = [t[1] for t in batch]

        pool = self._get_pre_post_pool()
        predictor = self.model._predictor

        try:
            pre_future = pool.submit(self._run_preprocessing, predictor, images)
            batch_inputs, datas = pre_future.result()
            preds_list = self._run_gpu_inference(predictor, batch_inputs)
            # Each task carries its own threshold/layout overrides; use the
            # first task's (batching is only meaningful when they agree).
            _thr = batch[0][4]
            _lw = batch[0][5]
            boxes_list = self._run_postprocessing(
                predictor, preds_list, datas, _thr, _lw
            )

            if len(boxes_list) != n:
                raise RuntimeError(
                    f"Post-processing returned {len(boxes_list)} results "
                    f"for {n} images"
                )

            for i, (_task_id, _img, future, _t, _thr2, _lw2) in enumerate(batch):
                future.set_result(boxes_list[i])
        except Exception as e:
            logger.exception("Batch inference failed (%d images)", n)
            for _task_id, _img, future, _t, _thr2, _lw2 in batch:
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
        self,
        batch: list[tuple],
        preds_list: list[dict],
        datas: list[dict],
    ) -> None:
        """Submit post-processing to the CPU pool, resolve futures via callback."""
        n = len(batch)
        if n == 0:
            return

        _log = is_latency_logging_enabled()
        t0 = time.perf_counter() if _log else 0.0
        predictor = self.model._predictor
        pool = self._get_pre_post_pool()

        def _on_post_done(post_future: Future) -> None:
            try:
                boxes_list = post_future.result()
                if len(boxes_list) != n:
                    raise RuntimeError(
                        f"Post-processing returned {len(boxes_list)} results "
                        f"for {n} images"
                    )
                for i, (_task_id, _img, future, _t, _thr, _lw) in enumerate(batch):
                    future.set_result(boxes_list[i])
            except Exception as e:
                logger.exception("Post-processing failed (%d images)", n)
                for _task_id, _img, future, _t, _thr, _lw in batch:
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

        _thr = batch[0][4] if batch else None
        _lw = batch[0][5] if batch else None
        pool.submit(
            self._run_postprocessing, predictor, preds_list, datas, _thr, _lw
        ).add_done_callback(_on_post_done)

    # ── inference worker loop ────────────────────────────────────────────────
    def _infer_worker(self) -> None:
        """Dedicated thread: load model, then process queued requests."""
        try:
            self._load_model()
        except Exception as e:
            self._load_error = e
            self._load_done.set()
            logger.exception("Model failed to load in inference thread")
            return
        self._load_done.set()
        logger.info(
            "Inference thread ready (pipeline_depth=%d, batch_size=%d, "
            "batch_timeout_ms=%.0f, pre_post_pool=%d), waiting for tasks...",
            PIPELINE_DEPTH, BATCH_SIZE, BATCH_TIMEOUT_MS, PRE_POST_POOL_SIZE,
        )

        predictor = self.model._predictor
        pool = self._get_pre_post_pool()

        prev_post: tuple | None = None  # (batch, preds, datas)
        next_pre: Future | None = None
        next_batch: list[tuple] | None = None

        def _collect_batch_blocking() -> list[tuple] | None:
            task = self._task_queue.get()
            if task[0] is None:
                return [task]  # sentinel
            batch = [task]
            while len(batch) < BATCH_SIZE:
                try:
                    extra = self._task_queue.get_nowait()
                except queue.Empty:
                    break
                if extra[0] is None:
                    batch.append(extra)
                    return batch
                batch.append(extra)
            while len(batch) < BATCH_SIZE:
                try:
                    extra = self._task_queue.get(timeout=BATCH_TIMEOUT_MS / 1000.0)
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
            try:
                task = self._task_queue.get_nowait()
            except queue.Empty:
                return None
            if task[0] is None:
                return [task]
            batch = [task]
            while len(batch) < BATCH_SIZE:
                try:
                    extra = self._task_queue.get_nowait()
                except queue.Empty:
                    break
                if extra[0] is None:
                    batch.append(extra)
                    return batch
                batch.append(extra)
            return batch

        while True:
            if next_batch is not None:
                batch = next_batch
                next_batch = None
            else:
                batch = _collect_batch_blocking()

            if batch is None or _has_sentinel(batch):
                if prev_post is not None:
                    self._resolve_batch(*prev_post)
                    prev_post = None
                if next_pre is not None:
                    try:
                        next_pre.result(timeout=1.0)
                    except Exception:
                        pass
                    next_pre = None
                if batch is not None:
                    real = [t for t in batch if t[0] is not None]
                    if real:
                        self._process_batch_pipelined(real)
                break

            _log = is_latency_logging_enabled()
            t0 = time.perf_counter() if _log else 0.0
            t_enqueue_min = min(t[3] for t in batch)

            if prev_post is not None:
                self._resolve_batch(*prev_post)
                prev_post = None

            t_pre_wait = time.perf_counter() if _log else 0.0
            if next_pre is not None:
                batch_inputs, datas = next_pre.result()
                next_pre = None
            else:
                images = [t[1] for t in batch]
                pre_future = pool.submit(self._run_preprocessing, predictor, images)
                batch_inputs, datas = pre_future.result()
            t_pre_done = time.perf_counter() if _log else 0.0

            if self._task_queue.qsize() > 0:
                next_batch = _try_collect_nonblocking()
                if next_batch is not None and not _has_sentinel(next_batch):
                    next_images = [t[1] for t in next_batch]
                    next_pre = pool.submit(
                        self._run_preprocessing, predictor, next_images
                    )
                else:
                    next_batch = None

            t_gpu_start = time.perf_counter() if _log else 0.0
            preds_list = self._run_gpu_inference(predictor, batch_inputs)

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

            prev_post = (batch, preds_list, datas)

            if next_batch is None and prev_post is not None:
                self._resolve_batch(*prev_post)
                prev_post = None

        if prev_post is not None:
            self._resolve_batch(*prev_post)

    # ── InferenceBackend interface ───────────────────────────────────────────
    async def detect(
        self,
        image: np.ndarray,
        *,
        threshold: float | dict | None = None,
        **layout_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Submit an image to the inference thread and await the result.

        ``threshold`` and ``layout_kwargs`` are per-request overrides for the
        layout post-processing stage; ``None`` values fall back to the
        deployment defaults baked into the predictor config.
        """
        _log = is_latency_logging_enabled()
        t_sem_acquire = time.perf_counter() if _log else 0.0

        async with self.semaphore:
            if _log:
                sem_wait = time.perf_counter() - t_sem_acquire
                logger.info(
                    '{"event":"latency","stage":"semaphore_wait","wait_s":%.6f}',
                    sem_wait,
                )
            if threshold is None:
                threshold = self._default_threshold

            task_id = self._next_task_id()
            future: Future = Future()
            t_enqueue = time.perf_counter() if _log else 0.0
            t_infer_start = t_enqueue if _log else 0.0
            self._task_queue.put(
                (task_id, image, future, t_enqueue, threshold, layout_kwargs)
            )

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
                    '{"event":"latency","stage":"gpu_inference","wait_s":%.6f}',
                    infer_time,
                )

        return boxes

    def list_models(self) -> list[dict[str, Any]]:
        """Report the single in-process model."""
        return [{
            "name": MODEL_NAME,
            "version": "1",
            "ready": bool(self._ready),
            "active": bool(self._ready),
        }]

    def get_model_status(self, name: str) -> dict[str, Any] | None:
        if name == MODEL_NAME:
            return {
                "name": name,
                "version": "1",
                "state": "READY" if self._ready else "LOADING",
                "reason": None if self._ready else "model loading",
            }
        return None


__all__ = ["DirectBackend"]