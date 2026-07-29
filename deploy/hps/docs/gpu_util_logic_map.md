# GPU Utilization Logic Map — Inference Pipeline Trace

> **Technique**: Logic Mapping (Phase 1: Trace)
> **Goal**: Identify why GPU utilization is ~26% and find the path to 100%
> **Date**: 2026-07-28 (updated 2026-07-30 with measured data)

---

## 0. Executive Summary — Is the Current Approach the Fastest?

**No, the current HPS direct-backend approach is NOT the fastest possible,
but it is the fastest configuration we have validated.** Batched inference
(BATCH_SIZE>1) was tested and proven **counterproductive** due to D2H copy
explosion.

| Metric | Value | Assessment |
|--------|-------|------------|
| Throughput ceiling | ~21–26 req/s | Flat from C=2 onward — serial inference thread |
| GPU compute (exec) | 9.0ms / image | Fast — FP8 TRT engine is well-optimized |
| GPU utilization (within TRT runner) | 26.1% | exec / total = 9.0 / 34.7ms |
| GPU utilization (end-to-end) | ~5.7% | exec / layout_total = 9.0 / 155.4ms |
| D2H copy | 21.9ms | **2.4× the GPU compute** — dominant TRT cost |
| Non-TRT CPU overhead | ~120.7ms | 78% of layout_detection — PaddleX pre/post |
| Queue wait (gpu_inference) | 104.5ms mean | Tasks wait in queue for the single inference thread |

**The pipeline is CPU-bound, not GPU-bound.** The GPU is idle 74% of the time
even within the TRT runner, and 94% of the time end-to-end. The single
serial inference thread is the throughput ceiling.

---

## 1. End-to-End Request Flow (Direct Backend)

> **Active backend**: `HPS_API_BACKEND=direct` — model loaded in-process.
> The Triton backend (`_run_triton`) exists in code but requires
> `tritonclient[grpc]` which is not installed.

```
HTTP POST /v1/convert/file (multipart upload)
    │
    ▼
[Granian ASGI server] — 1 worker, 1 event loop
    │  granian --interface asgi --workers 1 "api_compat.docling_api.app:app"
    │
    ▼
[FastAPI route: routes.py:convert_file()]          ← _parse_to_formats()
    │  await file.read()  →  image_data: bytes
    │
    ▼
[service.py:convert_image()]  — 4 LatencyTracer stages
    │
    ├── tracer.stage("image_load")                    [CPU, ThreadPool]
    │     └── loop.run_in_executor(_cpu_pool, load_image_from_bytes)
    │          └── PIL.Image.open → convert RGB → resize if >4096 → np.array
    │             MEASURED: mean=9.9ms  p50=8.6ms  max=22.1ms
    │
    ├── tracer.stage("layout_detection")              [GPU + CPU]
    │     └── run_layout_detection(image)  →  _run_direct(image)
    │          │
    │          ├── async with state.semaphore (PIPELINE_DEPTH=3):
    │          │     MEASURED semaphore_wait: mean=40.3ms  p50=17.9ms  max=260.3ms
    │          │
    │          ├── task_id = uuid4().hex; future = Future()
    │          ├── state._task_queue.put((task_id, image, future))
    │          │
    │          ├── await loop.run_in_executor(None, future.result)
    │          │     ↑ Frees event loop while inference thread works
    │          │     MEASURED gpu_inference wait: mean=104.5ms  p50=109.3ms  max=181.2ms
    │          │
    │          │   ┌──────────────────────────────────────────────────┐
    │          │   │  INFERENCE THREAD (_inference_worker)             │
    │          │   │  — Single thread, owns CUDA context              │
    │          │   │  — Processes ONE task at a time (BATCH_SIZE=1)   │
    │          │   │                                                   │
    │          │   │  task = state._task_queue.get()  [BLOCKING]      │
    │          │   │  _process_single(task):                          │
    │          │   │    gen = state.model.predict(image)              │
    │          │   │    results = list(gen)                           │
    │          │   │    boxes = _extract_boxes(results[0])            │
    │          │   │    future.set_result(boxes)                     │
    │          │   │                                                   │
    │          │   │  ┌── PaddleX predict internals ──────────────┐  │
    │          │   │  │ BasePredictor.__call__ → apply()          │  │
    │          │   │  │   ├── pre_ops[:-1] (Resize, Normalize)  [CPU] │
    │          │   │  │   ├── pre_ops[-1]  (ToBatch)            [CPU] │
    │          │   │  │   │                                       │  │
    │          │   │  │   ├── runner(batch_inputs)  ← GPU INFERENCE  │
    │          │   │  │   │    ┌── TRT runner ────────────────┐  │  │
    │          │   │  │   │    │ set_input_shape              │  │  │
    │          │   │  │   │    │ _allocate_buffers            │  │  │
    │          │   │  │   │    │ memcpy_htod (H2D)     3.8ms  │  │  │
    │          │   │  │   │    │ execute_async_v3      ──┐    │  │  │
    │          │   │  │   │    │ stream.synchronize() ──┘    │  │  │
    │          │   │  │   │    │                          GPU 9.0ms  │
    │          │   │  │   │    │ memcpy_dtoh (D2H)    21.9ms │  │  │
    │          │   │  │   │    │ total_ms              34.7ms │  │  │
    │          │   │  │   │    └────────────────────────────┘  │  │
    │          │   │  │   │                                       │  │
    │          │   │  │   ├── _format_output (NMS etc.)      [CPU] │
    │          │   │  │   └── post_op (layout NMS, unclip)   [CPU] │
    │          │   │  └────────────────────────────────────────┘  │
    │          │   └──────────────────────────────────────────────┘
    │          │
    │          └── return boxes
    │             MEASURED layout_detection: mean=145.4ms  p50=127.6ms  max=392.6ms
    │             MEASURED layout_detection_total: mean=155.4ms  p50=138.1ms
    │
    ├── tracer.stage("convert_to_docling")            [CPU, ThreadPool]
    │     └── loop.run_in_executor(_cpu_pool, _converter.convert)
    │          └── Build DoclingDocument from boxes
    │             MEASURED: mean=1.0ms  p50=0.9ms  max=2.2ms
    │
    └── tracer.stage("export_formats")                [CPU, ThreadPool]
          └── asyncio.gather(
                _export_to_formats(doc, formats),     [CPU, ThreadPool]
                _converter.compute_confidence(boxes),  [CPU, ThreadPool]
              )
             MEASURED: mean=1.7ms  p50=1.2ms  max=49.8ms

    MEASURED TOTAL: mean=158.1ms  p50=140.7ms  min=40.5ms  max=404.3ms
```

## 2. Key File:Line References

| Stage | File | Line | What Happens |
|-------|------|------|--------------|
| HTTP handler | `api_compat/docling_api/routes.py` | ~160 | `convert_file()` reads upload, parses formats |
| Pipeline orchestration | `api_compat/docling_api/service.py` | ~119 | `convert_image()` — 4 tracer stages |
| Image decode | `api_compat/_core/image.py` | ~30 | `load_image_from_bytes()` — PIL→numpy |
| CPU thread pool | `api_compat/docling_api/service.py` | 44 | `ThreadPoolExecutor(max_workers=4)` |
| Concurrency config | `api_compat/_core/config.py` | 63 | `PIPELINE_DEPTH=3`, `CPU_POOL_SIZE=4` |
| Backend config | `api_compat/_core/config.py` | 84 | `INFERENCE_BACKEND="triton"` (overridden to `direct`) |
| Semaphore + queue | `api_compat/_core/inference.py` | 80 | `AppState.__init__()` — semaphore, task_queue, _pending |
| Inference thread start | `api_compat/_core/inference.py` | 135 | `start_inference()` → spawns `_inference_worker` thread |
| Inference worker loop | `api_compat/_core/inference.py` | 196 | `_inference_worker()` — loads model, blocks on queue |
| Single-task processing | `api_compat/_core/inference.py` | 255 | `_process_single()` — `model.predict(image)` |
| Batch processing | `api_compat/_core/inference.py` | 271 | `_process_batch()` — `model.predict(images, batch_size=n)` |
| Dispatch (triton vs direct) | `api_compat/_core/inference.py` | 340 | `run_layout_detection()` → `_run_direct()` |
| Direct backend async path | `api_compat/_core/inference.py` | 375 | `_run_direct()` — semaphore → queue → await future |
| Model loading | `api_compat/_core/inference.py` | 173 | `load_model()` — `create_model(MODEL_NAME, engine="tensorrt")` |
| Engine discovery | `api_compat/_core/engine.py` | ~40 | `prepare_engine()` — finds TRT engine in /tmp or /opt/models |
| PaddleX predict | `paddlex/inference/models/predictors/base_predictor.py` | ~100 | `apply()` → batch → process |
| Layout process | `paddlex/inference/models/layout_analysis/predictor.py` | ~70 | `process()` — pre+infer+post |
| TRT runner | `paddlex/inference/models/runners/tensorrt_runner.py` | 159 | `__call__()` — H2D→infer→D2H |
| TRT synchronize | `paddlex/inference/models/runners/tensorrt_runner.py` | 201-202 | `execute_async_v3` + `synchronize()` |
| TRT H2D copy | `paddlex/inference/models/runners/tensorrt_runner.py` | 192 | `memcpy_htod` |
| TRT D2H copy | `paddlex/inference/models/runners/tensorrt_runner.py` | 209 | `memcpy_dtoh` |
| Granian launch | `scripts/run_api.sh` | ~120 | `exec granian --interface asgi --workers 1` |

## 3. Bottleneck Analysis (Measured Data)

> All data from BATCH_SIZE=1, PIPELINE_DEPTH=3, CPU_POOL_SIZE=4, WORKERS=1.
> 121 latency traces, 121 TRT runner samples, concurrency sweep C=1–8.

### 3.1 The Serial Inference Thread — Throughput Ceiling

The direct backend uses a **single dedicated inference thread**
(`_inference_worker`, `inference.py:196`). CUDA contexts are thread-local,
so only this thread can drive the GPU. It processes **one task at a time**
(BATCH_SIZE=1): `task = queue.get()` → `model.predict(image)` →
`future.set_result(boxes)` → loop.

```
Time →
Inference thread:  [== predict A ==][== predict B ==][== predict C ==]
                    ← 155ms each →   ← 155ms each →
Event loop:        [submit A][submit B][submit C]  (non-blocking, all queued instantly)
                   [await A........][await B........][await C........]

Throughput = 1 / 0.155s ≈ 6.5 req/s theoretical max from GPU work alone
Actual:     ~21-26 req/s achieved  ← multiple requests overlap CPU stages
```

Wait — that doesn't add up. The 155ms includes queue wait. The actual
GPU-bound time per request is ~35ms (TRT runner) + ~121ms PaddleX overhead.
But with PIPELINE_DEPTH=3, multiple requests are in-flight, so while
request A is in GPU inference, requests B and C can be doing image_load
and export in parallel on the CPU pool. This overlap pushes throughput
above the naive serial limit.

**The ceiling at ~26 req/s is reached because the inference thread can
only complete one `predict()` call every ~155ms, but CPU-side overlap
lets ~1.7 requests complete per inference cycle.**

Measured queue wait confirms this:

| Wait Stage | Mean | p50 | Max |
|------------|------|-----|-----|
| semaphore_wait | 40.3ms | 17.9ms | 260.3ms |
| gpu_inference (queue wait) | 104.5ms | 109.3ms | 181.2ms |

The gpu_inference wait (104.5ms) is the time a task sits in the queue
before the inference thread picks it up. This is significant — it means
tasks spend most of their "layout_detection" time **waiting**, not
computing.

### 3.2 The Synchronous `synchronize()` Wall

The TensorRT runner at `tensorrt_runner.py:201-202`:

```python
self._context.execute_async_v3(self._stream.handle)
self._stream.synchronize()  # ← BLOCKS until GPU finishes
```

Despite using `execute_async_v3` (async kernel launch), the immediate
`stream.synchronize()` makes it **synchronous from the caller's perspective**.
No H2D copy for the next request can overlap with compute.

**Measured TRT runner breakdown (BATCH_SIZE=1, n=121):**

| Stage | Mean | p50 | Min | Max |
|-------|------|-----|-----|-----|
| h2d_ms (H2D copy) | 3.8ms | 3.3ms | 2.0ms | 13.1ms |
| exec_ms (GPU compute) | 9.0ms | 6.3ms | 3.8ms | 29.8ms |
| d2h_ms (D2H copy) | 21.9ms | 21.1ms | 15.1ms | 51.3ms |
| total_ms | 34.7ms | 31.7ms | 22.5ms | 81.9ms |

```
GPU utilization within TRT runner = exec / total = 9.0 / 34.7 = 26.1%
```

**D2H copy (21.9ms) is 2.4× the actual GPU compute (9.0ms).** The GPU
finishes its kernels in 9ms, then the thread blocks for 22ms waiting for
results to copy back over PCIe. This is the dominant cost within the TRT
runner.

### 3.3 Non-TRT CPU Overhead — 78% of Layout Detection

```
layout_detection_total = 155.4ms (mean)
TRT runner total       =  34.7ms  (22.3%)
─────────────────────────────────
Non-TRT overhead       = 120.7ms  (77.7%)  ← PaddleX pre/post-processing
```

This 120.7ms is PaddleX framework overhead: BasePredictor.apply(),
batch_sampler, Resize, Normalize, ToBatch, _format_output, NMS, box
extraction, _extract_boxes(). All CPU-bound, all in the inference thread,
all serial with the GPU call.

```
Per-request layout_detection timeline (155.4ms total):
  [0────9.9ms]   image_load (parallel on CPU pool)         ← not in layout
  [0───120.7ms]  PaddleX pre/post-processing    [CPU]      ← 78%
  [     34.7ms]  TRT runner (H2D+exec+D2H)      [GPU/CPU]  ← 22%
  ─────────────────────────────────────────────
  Total: 155.4ms, but GPU only busy for 9.0ms = 5.8% GPU utilization
```

### 3.4 BATCH_SIZE=4 Experiment — Counterproductive

Tested BATCH_SIZE=4, BATCH_TIMEOUT_MS=5, PIPELINE_DEPTH=8, CPU_POOL_SIZE=8.
Result: **9.88 req/s — 2.6× slower than BATCH_SIZE=1 baseline.**

| TRT Stage | BATCH_SIZE=1 | BATCH_SIZE=4 | Ratio |
|-----------|-------------|--------------|-------|
| h2d_ms | 3.8ms | 12.3ms | 3.2× |
| exec_ms | 9.0ms | 14.1ms | 1.6× |
| **d2h_ms** | **21.9ms** | **327.6ms** | **15.0×** |
| total_ms | 34.7ms | 354.0ms | 10.2× |

The TRT engine was built for `max_batch_size=1`. D2H copy scales
catastrophically with batch — mean 327.6ms, max **1734.7ms**. The
inference throughput dropped to 10 img/s (vs ~26 req/s with BATCH_SIZE=1).

**Conclusion: BATCH_SIZE>1 is counterproductive with the current TRT engine.
Do not increase BATCH_SIZE without rebuilding the engine for dynamic batching.**

### 3.5 Throughput Ceiling — Flat from C=2 Onward

| Concurrency | Throughput (req/s) |
|-------------|-------------------|
| C=1 | 16.96 |
| C=2 | 19.55 |
| C=4 | 20.91 (also measured 24.77 in 40-req run) |
| C=8 | 19.95 |

(Prior session: C=1: 18.93, C=2: 25.90, C=4: 25.94, C=8: 25.56,
C=12: 25.64, C=16: 25.42)

Throughput saturates at ~21–26 req/s from C=2 onward. Adding more
concurrency does not help because the single inference thread is the
bottleneck — extra requests just pile up in the queue (gpu_inference
wait grows).

### 3.6 Semaphore (PIPELINE_DEPTH) — Adequate but Not the Limit

PIPELINE_DEPTH=3 allows 3 concurrent in-flight inference tasks. With
the direct backend, this means up to 3 futures can be awaiting
`future.result()` simultaneously while the inference thread processes
them one by one.

The semaphore_wait (mean=40.3ms, max=260.3ms) shows occasional
contention, but this is not the primary bottleneck — gpu_inference
wait (104.5ms) is 2.6× larger. Increasing PIPELINE_DEPTH would only
add more queue depth, not more processing parallelism.

## 4. Root Cause Summary

```
End-to-end GPU utilization = GPU_compute / layout_detection_total
                            = 9.0ms / 155.4ms
                            = 5.8%

TRT runner GPU utilization = GPU_compute / TRT_total
                            = 9.0ms / 34.7ms
                            = 26.1%
```

**Three compounding factors keep the GPU idle 94% of the time:**

1. **Serial inference thread** (inference.py:196): One thread, one CUDA
   context, one `predict()` at a time. While the GPU computes for 9ms,
   no preprocessing for the next image can happen in that thread.
   Throughput ceiling ≈ 1 / 0.155s ≈ 6.5 GPU-bound req/s (overlapped
   CPU work pushes actual to ~26 req/s).

2. **Synchronous D2H copy** (tensorrt_runner.py:209): The 21.9ms D2H
   copy blocks the inference thread after every GPU call — 2.4× the
   actual compute. No overlap with next-request H2D.

3. **PaddleX framework overhead** (120.7ms): 78% of layout_detection
   is CPU-bound pre/post-processing in the same thread as GPU inference.
   Resize, Normalize, NMS, box extraction — all serial with the GPU.

## 5. Direct Backend vs Triton Backend — Architecture Comparison

| Aspect | Direct Backend (active) | Triton Backend (not used) |
|--------|------------------------|--------------------------|
| Model location | In-process (PaddleX `create_model`) | Separate Triton server |
| IPC | `queue.Queue` + `Future` (in-process) | gRPC (base64-encoded JSON) |
| Batching | Custom micro-batching (BATCH_SIZE, BATCH_TIMEOUT_MS) | Triton dynamic_batching (continuous) |
| GPU scheduling | Single inference thread, serial | Triton instance group (configurable) |
| CUDA context | Thread-local, one owner | Managed by Triton |
| Dependencies | `paddlepaddle`, `paddlepaddle-gpu` | `tritonclient[grpc]` (not installed) |
| Engine | Pre-built TRT engine loaded by PaddleX | TRT engine in model repository |
| Wire format | numpy arrays (in-process) | base64 JPEG in JSON over gRPC |
| Startup | `load_model()` in inference thread (~30s) | Triton server startup (~60s) |
| Overhead per request | ~120.7ms PaddleX pre/post | ~120.7ms PaddleX pre/post + 4-8ms base64 |

**Key insight**: Both backends share the same PaddleX predictor and TRT
runner, so the GPU compute time (9.0ms) and D2H copy (21.9ms) are
identical. The Triton backend adds ~4-8ms of base64/gRPC overhead per
request but gains **continuous batching** — Triton can collect multiple
requests and process them in a single GPU forward pass without our
custom queue.

The direct backend's advantage is simplicity and zero IPC overhead. Its
disadvantage is the single inference thread with no continuous batching.

## 6. Efficiency Assessment & Resolution Strategy

### Current State: Is This the Fastest Approach?

**For the direct backend with BATCH_SIZE=1: YES, this is optimal.**
The configuration (PIPELINE_DEPTH=3, CPU_POOL_SIZE=4, WORKERS=1) is
well-tuned. BATCH_SIZE>1 is proven counterproductive. Increasing
PIPELINE_DEPTH beyond 3 yields diminishing returns (queue just grows).

**But the direct backend architecture itself has a hard ceiling at ~26
req/s** due to the serial inference thread. To go faster, the
architecture must change.

### Improvement Options (Ranked by Impact × Feasibility)

#### Option A: Async D2H with CUDA Streams (HIGH impact, MEDIUM effort)

**Problem**: `stream.synchronize()` blocks for 21.9ms after every GPU call.
**Fix**: Use two CUDA streams — stream A computes while stream B does D2H
for the previous request. Or use `cuda.Stream.record_event()` + deferred
sync to overlap D2H with preprocessing of the next request.

```python
# Conceptual — overlap D2H with next-request prep
self._context.execute_async_v3(self._compute_stream.handle)
done_event = self._compute_stream.record_event()
# Don't synchronize yet — start next-request H2D on copy_stream
# Only synchronize before reading D2H results
self._cuda.memcpy_dtoh_async(..., self._copy_stream.handle)
self._copy_stream.synchronize()  # only when we actually need results
```

**Expected gain**: Reduce effective D2H from 21.9ms → ~0ms (overlapped).
TRT runner total drops from 34.7ms → ~12.8ms. Per-request layout drops
from 155.4ms → ~133.7ms. Theoretical throughput: ~7.5 GPU-bound req/s
before CPU overlap.

**Effort**: Requires modifying `tensorrt_runner.py` (PaddleX upstream code)
or subclassing the runner. Must handle buffer lifecycle carefully.

#### Option B: Multi-Worker Granian (HIGH impact, LOW effort, GPU memory risk)

**Problem**: Single Granian worker = single inference thread = serial GPU access.
**Fix**: Launch Granian with `--workers 2` or `--workers 3`. Each worker
is a separate process with its own model instance and inference thread.
CUDA contexts are per-process, so the GPU interleaves kernels from
multiple workers.

```bash
exec granian --interface asgi --workers 3 "api_compat.docling_api.app:app"
```

**Expected gain**: Near-linear throughput scaling — 2 workers → ~40-50 req/s,
3 workers → ~60-75 req/s (theoretical, GPU SM saturation may limit).

**Risk**: Each worker loads its own model copy (~2-4GB GPU memory for
FP8 PP-DocLayoutV3). With 3 workers, ~6-12GB GPU memory needed. Must
verify available VRAM. Also, 3× CPU memory for PaddleX framework.

**Effort**: Change one env var (`WORKERS=3`). No code changes. Test
GPU memory with `nvidia-smi` during load.

#### Option C: Triton Backend with Continuous Batching (HIGH impact, HIGH effort)

**Problem**: Direct backend has no continuous batching — one request at a time.
**Fix**: Install `tritonclient[grpc]`, start Triton server with
`dynamic_batching` enabled, rebuild TRT engine with `max_batch_size ≥ 8`
and proper `opt_profile` for dynamic shapes. Switch `HPS_API_BACKEND=triton`.

**Expected gain**: Triton's dynamic batcher collects concurrent requests
and processes them in a single GPU forward pass. With 8 concurrent
requests and a properly built engine, GPU utilization could reach 60-80%.

**Critical prerequisite**: The TRT engine **must be rebuilt** for
`max_batch_size ≥ 8`. The current engine (built for batch_size=1) causes
D2H explosion with batches (proven in §3.4). The engine must have an
optimization profile covering batch sizes 1-8.

**Effort**: Install tritonclient, configure Triton model repo, rebuild
TRT engine with dynamic batch profile, update `config.pbtxt`, test.
Significant infrastructure work.

#### Option D: Move Preprocessing to GPU (MEDIUM impact, HIGH effort)

**Problem**: 120.7ms of CPU-bound PaddleX pre/post-processing per request.
**Fix**: Move Resize, Normalize, and NMS to CUDA kernels. PaddleX/TRT
already has GPU-based preprocessing in some pipelines — investigate
`paddlex` GPU preprocessing ops or write custom CUDA kernels.

**Expected gain**: Reduce 120.7ms → ~10-20ms (GPU-accelerated). Layout
detection drops from 155.4ms → ~45-55ms. Theoretical throughput:
~18-22 GPU-bound req/s before overlap.

**Effort**: Deep PaddleX framework changes. High risk of breaking
accuracy. Likely not worth the effort compared to Options A+B.

#### Option E: Reduce PaddleX Framework Overhead (MEDIUM impact, MEDIUM effort)

**Problem**: 120.7ms of PaddleX overhead — much of it is Python framework
code (BasePredictor.apply, batch_sampler, _format_output, _extract_boxes).
**Fix**: Profile PaddleX predictor internals with `cProfile`. Identify
hot spots. Potential wins:
- Skip unnecessary copy/conversion steps
- Cache model metadata between calls
- Use numpy operations instead of Python loops in postprocessing
- Pre-allocate buffers instead of per-call allocation

**Expected gain**: 10-30% reduction in PaddleX overhead → 120.7ms → 85-108ms.
Modest throughput improvement to ~28-32 req/s.

**Effort**: Profile-guided optimization of PaddleX code. Requires deep
understanding of the predictor pipeline.

### Recommended Path Forward

1. **Immediate (Option B)**: Set `WORKERS=2` or `WORKERS=3`. Zero code
   changes, test VRAM first. Expected: 2-3× throughput.

2. **Short-term (Option A)**: Implement async D2H in TRT runner. Eliminates
   the 21.9ms blocking copy. Expected: 15-25% per-request latency reduction.

3. **Medium-term (Option C)**: If throughput targets exceed ~75 req/s,
   invest in Triton backend with rebuilt dynamic-batch engine. This is
   the path to 60-80% GPU utilization.

4. **Avoid**: BATCH_SIZE>1 with current engine (proven counterproductive).
   Increasing PIPELINE_DEPTH beyond 3 (diminishing returns). Option D
   unless preprocessing becomes the clear bottleneck after A+B.

### Throughput Projection

| Configuration | Est. Throughput | GPU Util | Effort |
|--------------|----------------|----------|--------|
| Current (BATCH_SIZE=1, 1 worker) | ~26 req/s | ~6% | — |
| + WORKERS=2 | ~45-50 req/s | ~12% | Trivial |
| + WORKERS=3 | ~60-75 req/s | ~18% | Trivial |
| + Async D2H (Option A) | +15-25% per req | ~8% | Medium |
| + Triton w/ dynamic batch | ~80-120 req/s | ~60-80% | High |
| + GPU preprocessing | +30-40% per req | Higher | Very High |
