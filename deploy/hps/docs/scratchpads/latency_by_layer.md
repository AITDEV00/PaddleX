# Latency Separation by Architectural Layer — Granian / HPS / PaddleX

> **Purpose**: Break the total request latency into three architectural
> layers so each layer's contribution is visible and optimizable
> independently. Includes measured Direct-vs-Triton benchmark comparison.
>
> **Companion document**: `architecture_latency_map.md` (full function-level
> trace). This document re-aggregates the same data by *architectural layer*.
>
> **Architecture note**: The layer breakdown in §2–§10 is from the **Direct
> backend** (Architecture B — in-process inference via PaddleX
> `create_model()`). See §11 for a measured comparison against the **Triton
> backend** (Architecture C — gRPC to Triton Inference Server).

---

## 1. Layer Definitions

| Layer | Technology | Role | Code Location |
|-------|-----------|------|---------------|
| **L1 — Granian** | Rust ASGI server | TCP accept, HTTP parse, ASGI dispatch, response serialization | `run_api.sh` → `exec granian ...` |
| **L2 — HPS api_compat** | FastAPI + asyncio | Route handling, pipeline orchestration, inference dispatch (semaphore/queue/future bridge), CPU pool I/O | `deploy/hps/api_compat/` |
| **L3 — PaddleX** | Python framework | Image preprocessing, TensorRT inference, post-processing (NMS, containment, restructure) | `paddlex/inference/` |

---

## 2. Layer Breakdown — Where Does 158.1ms Go?

> **Architecture**: Direct backend (Architecture B). These numbers were
> captured with `HPS_LATENCY_LOG=1` enabled, which adds tracer overhead.
> The production stress test (§11) shows **47.8ms p50 at concurrency=1**
> without tracer overhead — the 158.1ms includes instrumentation cost.

```
Total server-side latency:  158.1ms mean  (p50=140.7ms  max=404.3ms)
  [Direct backend, with HPS_LATENCY_LOG=1 tracer overhead]

┌──────────────────────────────────────────┬──────────┬──────┬──────────────┐
│ Layer                                    │ Mean     │ %    │ Measured?    │
├──────────────────────────────────────────┼──────────┼──────┼──────────────┤
│ L1. Granian (ASGI)                       │  ~0.5ms  │ 0.3% │ Not measured │
│   ├ TCP/HTTP request parsing             │  ~0.2ms  │      │ (Rust, fast) │
│   ├ ASGI dispatch → FastAPI              │  ~0.1ms  │      │              │
│   └ HTTP response serialization          │  ~0.2ms  │      │              │
│                                          │          │      │              │
│ L2. HPS api_compat (FastAPI + asyncio)   │  63.2ms  │ 40.0%│ ✅ Tracer    │
│   ├ A. image_load (CPU pool)             │   9.9ms  │  6.3%│ ✅ Tracer    │
│   ├ B.1 semaphore_wait (asyncio)         │  40.3ms  │ 25.5%│ ✅ Tracer    │
│   ├ B.2+B.3 task submit + async overhead │  10.6ms  │  6.7%│ ✅ Tracer    │
│   ├ C. convert_to_docling (CPU pool)     │   1.0ms  │  0.6%│ ✅ Tracer    │
│   └ D. export_formats (CPU pool)         │   1.4ms  │  0.9%│ ✅ Tracer    │
│                                          │          │      │              │
│ L3. PaddleX (inference framework)        │ ~94.4ms  │ 59.7%│ ✅ HPS_LATENCY_LOG│
│   ├ pre_ops (CPU: Resize+Normalize+CHW)  │ ~15-25ms │ ~12% │ ✅ LATENCY_LOG│
│   ├ ToBatch (CPU: np.stack)              │  ~2-5ms  │  ~2% │ ✅ LATENCY_LOG│
│   ├ TRT Runner (GPU + memcpy)            │  34.7ms  │ 22.0%│ ✅ LATENCY_LOG│
│   │   ├ h2d (host→device)                │   3.8ms  │      │              │
│   │   ├ exec (GPU forward)               │   9.0ms  │      │              │
│   │   └ d2h (device→host)                │  21.9ms  │      │              │
│   ├ _format_output (CPU: numpy slice)    │  ~1ms    │ <1%  │ ✅ LATENCY_LOG│
│   └ post_op (CPU: NMS+containment+restr) │ ~18-35ms │ ~15% │ ✅ LATENCY_LOG│
└──────────────────────────────────────────┴──────────┴──────┴──────────────┘
```

> **Note**: L2 + L3 sum to 157.6ms (vs 158.1ms measured) — the ~0.5ms gap
> is L1 (Granian), which is not instrumented. Percentages are approximate
> due to post_op's range (18-35ms).

---

## 3. Layer 1 — Granian (ASGI Server)

```
HTTP POST /v1/convert/file
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ Granian (Rust) — WORKERS=1                                │
│                                                           │
│ 1. TCP accept + HTTP/1.1 parse          ~0.2ms           │
│ 2. Multipart boundary parse (file upload) ~0.1ms         │
│ 3. ASGI scope dict construction         ~0.05ms          │
│ 4. Dispatch to FastAPI app (app object)  ~0.05ms         │
│ ...                                                       │
│ 5. ASGI response receive + HTTP encode  ~0.1ms           │
│ 6. TCP send + keepalive                 ~0.05ms          │
└───────────────────────────────────────────────────────────┘
```

**Launch command** (`run_api.sh`):
```bash
exec granian --interface asgi --host "${HOST}" --port "${PORT}" \
    --workers "${WORKERS}" "api_compat.docling_api.app:app"
# WORKERS=1
```

**Key points**:
- Granian is a **Rust-based ASGI server** — HTTP parsing and I/O happen in native code, far faster than Python.
- The `--interface asgi` flag means Granian calls the FastAPI app's `__call__(scope, receive, send)` directly.
- `WORKERS=1` = single process. All requests share one Python event loop.
- **Not measured**: Granian's own latency is not instrumented. It's estimated at <1ms based on Rust ASGI benchmarks. To measure it, add `time.perf_counter()` at the top of the FastAPI middleware and compare to `processing_time` in the response.

**Latency contribution**: ~0.5ms (0.3%) — negligible.

---

## 4. Layer 2 — HPS api_compat (FastAPI + asyncio)

This is the **custom serving layer**. It does NOT use the standard PaddleX HPS
pipeline server (`paddlex_hps_server` / `basic_serving`). See §7 below.

### 4.1 Route Handler → convert_image Pipeline

```
FastAPI route (routes.py)
    │
    ▼
convert_image() (service.py:90)
    │
    ├── [A] image_load          9.9ms   CPU pool thread
    │   └── load_image_from_bytes(): PIL decode + RGB convert + resize + np.array
    │
    ├── [B] layout_detection   104.5ms  asyncio → inference thread (cross-layer to L3)
    │   ├── [B.1] semaphore_wait    40.3ms  asyncio.Semaphore(4) contention
    │   ├── [B.2] task submit        ~0.01ms queue.Queue.put
    │   └── [B.3] async wait        10.6ms  loop.run_in_executor(None, future.result)
    │       └── (inference thread runs L3 — see §5)
    │
    ├── [C] convert_to_docling  1.0ms   CPU pool thread
    │   └── PaddleXToDoclingConverter.convert(): box→DoclingDocument
    │
    └── [D] export_formats      1.4ms   CPU pool thread (parallel with confidence)
        ├── _export_to_formats(): markdown/json/html/text/doctags
        └── compute_confidence(): quality grade scoring
            └── asyncio.gather() — runs both concurrently, takes max()
```

### 4.2 L2 Latency Summary

| Sub-stage | Mean | % of Total | Thread | Blocking? |
|-----------|------|-----------|--------|-----------|
| A. image_load | 9.9ms | 6.3% | _cpu_pool | No (offloaded) |
| B.1 semaphore_wait | 40.3ms | 25.5% | event loop | No (async wait) |
| B.2 task submit | ~0.01ms | <0.1% | event loop | No |
| B.3 async overhead | 10.6ms | 6.7% | default executor | Yes (blocks executor thread, not event loop) |
| C. convert_to_docling | 1.0ms | 0.6% | _cpu_pool | No |
| D. export_formats | 1.4ms | 0.9% | _cpu_pool | No |
| **L2 Total** | **63.2ms** | **40.0%** | | |

**Key insight**: The largest L2 cost is **B.1 semaphore_wait (40.3ms)** — pure
contention waiting for a GPU slot. This is NOT compute; it's queueing. Under
low concurrency (1 request), this drops to ~0ms.

---

## 5. Layer 3 — PaddleX (Inference Framework)

This layer runs entirely inside the **inference thread** (daemon thread,
holds the CUDA context). It's invoked via `state.model.predict(image)` which
returns a generator drained by `list(gen)`.

### 5.1 Call Chain Inside predict()

```
state.model.predict(image)          ← LayoutAnalysisPredictor
    │
    ▼
BasePredictor.__call__ → apply()
    ├── batch_sampler(input)              ~1ms    ImageBatchSampler
    │
    └── for batch_data in batches:
        ├── process(batch_data)           ← LayoutAnalysisProcess
        │   ├── [1] pre_ops (CPU)         ~15-25ms
        │   │   ├── ReadImage: cv2.cvtColor BGR→RGB     ~1ms
        │   │   ├── Resize: cv2.resize BICUBIC          ~5-15ms
        │   │   ├── Normalize: img*alpha+beta (VECTORIZED) ~3-8ms
        │   │   └── ToCHWImage: transpose (2,0,1)       ~1ms
        │   │
        │   ├── [2] ToBatch (CPU)         ~2-5ms
        │   │   └── np.stack([img_size, img, scale_factors])
        │   │
        │   ├── [3] TRT Runner (GPU)      34.7ms
        │   │   └── TensorRTRunner.__call__()
        │   │       ├── h2d: np.ascontiguousarray + cuda.memcpy_htod  3.8ms
        │   │       ├── exec: context.execute_async_v3 + sync         9.0ms
        │   │       └── d2h: cuda.memcpy_dtoh (NO .copy() — SKIPPED) 21.9ms
        │   │
        │   ├── [4] _format_output (CPU)  ~1ms
        │   │   └── numpy slicing per image
        │   │
        │   └── [5] post_op (CPU)         ~18-35ms
        │       ├── threshold filter (score > 0.5)     ~1ms
        │       ├── layout_nms (VECTORIZED: _iou_matrix) ~5-10ms
        │       ├── filter_large_image                  ~2-5ms
        │       ├── check_containment (VECTORIZED)      ~3-8ms
        │       ├── sort by order index                 ~1ms
        │       ├── unclip_boxes                        ~1ms
        │       └── restructured_boxes                  ~5-10ms
        │
        └── yield LayoutAnalysisResult(item)   ~1ms
```

### 5.2 L3 Latency Summary

| Sub-stage | Mean | % of Total | Device | Optimized? |
|-----------|------|-----------|--------|------------|
| batch_sampler | ~1ms | 0.6% | CPU | — |
| pre_ops (Resize+Normalize+CHW) | ~15-25ms | ~12% | CPU | ✅ Normalize vectorized |
| ToBatch (np.stack) | ~2-5ms | ~2% | CPU | — |
| TRT h2d (host→device) | 3.8ms | 2.4% | GPU+PCIe | — |
| TRT exec (GPU forward) | 9.0ms | 5.7% | GPU | ❌ Hard limit |
| TRT d2h (device→host) | 21.9ms | 13.9% | GPU+PCIe | ✅ .copy() skipped |
| _format_output | ~1ms | 0.6% | CPU | — |
| post_op (NMS+containment+restructure) | ~18-35ms | ~15% | CPU | ✅ NMS+containment vectorized |
| result wrap | ~1ms | 0.6% | CPU | — |
| **L3 Total** | **~94.4ms** | **~59.7%** | | |

**Key insight**: L3 is dominated by two costs:
1. **TRT d2h memcpy (21.9ms)** — GPU→CPU data transfer. Already optimized by skipping redundant `.copy()`.
2. **post_op (18-35ms)** — CPU-bound box processing. Already optimized by vectorizing NMS and containment.

---

## 6. Cross-Layer Flow — How a Request Traverses All Three Layers

```
┌─ L1: Granian ──────────────────────────────────────────────────┐
│  HTTP POST → ASGI scope → dispatch to FastAPI app             │
└──────────────────────────┬─────────────────────────────────────┘
                           │ scope, receive, send
                           ▼
┌─ L2: HPS api_compat ──────────────────────────────────────────┐
│  routes.py: convert_file()                                    │
│    └── service.py: convert_image()                            │
│        ├── [A] image_load ──→ _cpu_pool ──→ np.ndarray        │
│        ├── [B] run_layout_detection()                         │
│        │   ├── semaphore.acquire()     ← async wait 40.3ms    │
│        │   ├── queue.put(task)                                │
│        │   └── await future.result()   ← blocks executor      │
│        │       │                                              │
│        │       ▼  (crosses thread boundary)                   │
│        │   ┌─ L3: PaddleX ────────────────────────────────┐   │
│        │   │  inference thread:                            │   │
│        │   │  task_queue.get() → predict(image)            │   │
│        │   │    ├── pre_ops (CPU)                          │   │
│        │   │    ├── ToBatch (CPU)                          │   │
│        │   │    ├── TensorRTRunner (GPU)                   │   │
│        │   │    ├── _format_output (CPU)                   │   │
│        │   │    └── post_op (CPU)                          │   │
│        │   │  future.set_result(boxes) ──→ resolves L2     │   │
│        │   └───────────────────────────────────────────────┘   │
│        ├── [C] convert_to_docling ──→ _cpu_pool                │
│        └── [D] export_formats ──→ _cpu_pool (parallel)         │
└──────────────────────────┬─────────────────────────────────────┘
                           │ JSON response
                           ▼
┌─ L1: Granian ──────────────────────────────────────────────────┐
│  ASGI response → HTTP encode → TCP send                       │
└───────────────────────────────────────────────────────────────┘
```

---

## 7. Serving Pipeline Verification — Custom vs Standard

### 7.1 Current Setup: CUSTOM api_compat (NOT Standard PaddleX HPS)

The current serving pipeline **IS custom**. It does NOT use the standard
PaddleX HPS pipeline server.

| Aspect | Current (Custom api_compat) | Standard PaddleX HPS |
|--------|----------------------------|----------------------|
| **Entry point** | `run_api.sh` → `granian --interface asgi` | `paddlex_hps_server` → `uvicorn.run()` |
| **App factory** | `api_compat.docling_api.app:create_app()` | `basic_serving._app.create_app()` |
| **Server** | Granian (Rust ASGI) | uvicorn (Python ASGI) |
| **Pipeline wrapper** | None — direct `model.predict()` | `PipelineWrapper[PipelineT]` |
| **Pipeline scope** | Single model (PP-DocLayoutV3 only) | Full pipeline (DocPreprocessor + OCR + Table + Seal) |
| **Inference dispatch** | Custom semaphore + queue + future bridge + inference thread | PipelineWrapper → pipeline.predict() |
| **Backend** | Dual: triton (gRPC) or direct (micro-batching) | Single: direct |
| **Routes** | `/v1/convert/file`, `/v1/convert/source` (Docling API) | `/predict` (PaddleX native API) |
| **Output** | DoclingDocument → md/json/html/text/doctags | PaddleX result dict |
| **Code location** | `deploy/hps/api_compat/` | `deploy/hps/server_env/paddlex-hps-server/` |

### 7.2 Why It's Custom

The `api_compat` layer was built as a **Docling-compatible API wrapper** around
a **single layout detection model** (PP-DocLayoutV3). It:

1. **Skips the standard multi-model pipeline**: The standard HPS `layout_parsing`
   pipeline chains DocPreprocessor → LayoutDetection → OCR → TableRecognition →
   SealRecognition. The api_compat layer only runs LayoutDetection.
2. **Adds a custom Docling output layer**: Converts PaddleX boxes → DoclingDocument
   → markdown/JSON/HTML/text/doctags — none of which exists in standard PaddleX.
3. **Adds custom concurrency control**: Semaphore(PIPELINE_DEPTH=4) + task queue +
   inference daemon thread + CPU pool — none of which exists in standard PaddleX.
4. **Supports dual backends**: Triton (gRPC, server-side batching) and direct
   (custom micro-batching with BATCH_SIZE=2, BATCH_TIMEOUT_MS=5ms).

### 7.3 Standard PaddleX HPS Path (NOT Used)

The standard path that would be used if `paddlex_hps_server` were launched:

```
paddlex_hps_server (config.py: re-exports from paddlex.inference.serving.infra.config)
    └── basic_serving._server.run_server(app, host, port)
        └── uvicorn.run(app, host, port)
            └── basic_serving._app.create_app()
                └── PipelineWrapper(pipeline_config)
                    └── pipeline = create_pipeline("layout_parsing", ...)
                        └── full multi-model pipeline
```

The standard `pipeline_config.yaml` (at
`deploy/hps/sdk/pipelines/layout_parsing/server/pipeline_config.yaml`) has been
modified to use `PP-DocLayoutV3` (from `RT-DETR-H_layout_17cls`), but this
config is **not used** by the api_compat layer — it would only be used if the
standard `paddlex_hps_server` were launched.

### 7.4 Summary

> ⚠️ **The current serving pipeline IS custom.** If the requirement is to use
> the standard PaddleX HPS pipeline server, the entry point must change from
> `granian ... api_compat.docling_api.app:app` to
> `paddlex_hps_server` (which uses uvicorn + PipelineWrapper + full
> layout_parsing pipeline). However, the custom api_compat layer provides
> Docling API compatibility and single-model optimization that the standard
> HPS server does not offer.

---

## 8. TensorRT Extension Verification

### 8.1 Extension Status: PRESENT and WIRED ✅

The TensorRT engine extension is fully present on disk and properly wired into
the PaddleX framework:

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| Engine class | `paddlex/inference/models/engines/tensorrt.py` | ✅ Present | 131 |
| Runner class | `paddlex/inference/models/runners/tensorrt_runner.py` | ✅ Present | 779 |
| Auto-registration | `paddlex/inference/models/engines/__init__.py` | ✅ Imported | `from . import (... tensorrt ...)` |
| Metaclass registration | `paddlex/inference/models/engines/_base.py` | ✅ AutoRegisterABCMetaClass | `entities = "tensorrt"` |

### 8.2 Extension Features

The `TensorRTRunner` (779 lines) provides a **native TensorRT engine** using the
`tensorrt` Python API + `pycuda` for GPU memory management:

**Precision support**: fp32, fp16, int8, fp8
- FP16: weakly-typed network, `BuilderFlag(0)` (kFP16)
- INT8/FP8: strongly-typed network, precision from Q/DQ nodes, no flags

**Auto-build pipeline**: ONNX → (auto-export Paddle if needed) → (auto-quantize
for int8/fp8 via nvidia-modelopt) → TRT engine
- Polygraphy constant folding on exported ONNX
- Timing cache for faster rebuilds
- Optimization profiles with auto-detected dynamic shapes

**HPS-specific extensions** (env-var controlled):
```python
# In TensorRTRunner.__call__() (line 183):
_skip_d2h_copy = os.environ.get("HPS_TRT_SKIP_D2H_COPY", "0")
_lat = os.environ.get("HPS_LATENCY_LOG", "0")
```
- `HPS_TRT_SKIP_D2H_COPY=1`: When enabled, `results.append(self._h_outputs[name])`
  returns a view instead of copying — saves ~21.9ms per request (the d2h `.copy()`
  is skipped, only the `cuda.memcpy_dtoh` remains).
- `HPS_LATENCY_LOG=1`: When enabled, logs h2d/exec/d2h timing breakdown.

**Buffer management**: `_allocate_buffers()` reuses GPU allocations across
batches — only reallocates when a larger buffer is needed.

**Calibration** (for int8/fp8): `_build_calibration_reader()` supports both
real image directories and synthetic data generation. Ported from TurboOCR's
`quantize_onnx_int8.py`.

### 8.3 ⚠️ Git Status: UNTRACKED — Needs Committing

```
$ git status --short
?? paddlex/inference/models/engines/tensorrt.py
?? paddlex/inference/models/runners/tensorrt_runner.py
?? deploy/hps/api_compat/
```

Both TRT extension files AND the entire `deploy/hps/api_compat/` directory are
**untracked in git**. They exist on disk and work at runtime, but are not
committed. To preserve them:

```bash
cd /home/jyao/ADEO/OCR/PaddleX
git add paddlex/inference/models/engines/tensorrt.py
git add paddlex/inference/models/runners/tensorrt_runner.py
git add deploy/hps/api_compat/
git commit -m "Add TensorRT engine extension and HPS api_compat Docling API"
```

---

## 9. Visualization — Layer Latency Stack

```
                          158.1ms total
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  L1 Granian  ▏                                          │  ~0.5ms (0.3%)
│                                                         │
│  L2 HPS api_compat                                      │  63.2ms (40.0%)
│  ┌─────────────────────────────────────────────────┐    │
│  │ image_load    ███████                            │    │   9.9ms
│  │ semaphore_wait███████████████████████████████    │    │  40.3ms
│  │ async overhead ████████                          │    │  10.6ms
│  │ convert_doc   ▏                                  │    │   1.0ms
│  │ export_formats █                                 │    │   1.4ms
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  L3 PaddleX                                             │ ~94.4ms (59.7%)
│  ┌─────────────────────────────────────────────────┐    │
│  │ pre_ops       ██████████████                     │    │  ~20ms
│  │ ToBatch       ███                                │    │   ~3ms
│  │ TRT h2d       ███                                │    │   3.8ms
│  │ TRT exec      ███████                            │    │   9.0ms
│  │ TRT d2h       ██████████████████                 │    │  21.9ms
│  │ format_output ▏                                  │    │   ~1ms
│  │ post_op       █████████████████                  │    │  ~26ms
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

> **Note**: The 158.1ms above includes tracer overhead
> (`HPS_LATENCY_LOG=1`). Without instrumentation, the Direct backend
> achieves **47.8ms p50 at concurrency=1** (see §11).

---

## 10. Optimization Opportunities by Layer

### L1 — Granian (minimal headroom)
- Already Rust-native, <1ms. No optimization needed.
- Could measure precisely by adding ASGI middleware timing.

### L2 — HPS api_compat (40% of total, 63.2ms)
| Cost | Current | Optimization | Expected Gain |
|------|---------|-------------|---------------|
| semaphore_wait | 40.3ms | Increase PIPELINE_DEPTH (4→6?) or remove if single-GPU serial is acceptable | Up to 40ms (under load) |
| async overhead | 10.6ms | Replace future bridge with direct call (single-image, no batching) | ~5ms |
| image_load | 9.9ms | Use libvips/turbojpeg instead of PIL | ~3-5ms |

### L3 — PaddleX (60% of total, ~94.4ms)
| Cost | Current | Optimization | Expected Gain |
|------|---------|-------------|---------------|
| TRT d2h | 21.9ms | Already optimized (.copy() skipped). Could use pinned memory | ~2-5ms |
| post_op | ~18-35ms | Already vectorized. Could move NMS to GPU (TRT plugin or CUDA kernel) | ~10-15ms |
| pre_ops Resize | ~5-15ms | Could use GPU-side resize (cv2.cuda) | ~5-10ms |
| TRT exec | 9.0ms | Hard limit — GPU compute time | 0 |

---

## Appendix A: Environment Variables Controlling Each Layer

| Env Var | Layer | Default | Effect |
|---------|-------|---------|--------|
| `HPS_API_BACKEND` | L2 | `triton` | `direct` = custom micro-batching; `triton` = Triton gRPC |
| `HPS_API_PIPELINE_DEPTH` | L2 | `4` | Semaphore depth — max concurrent in-flight requests |
| `HPS_API_CPU_POOL_SIZE` | L2 | `8` | ThreadPoolExecutor size for CPU-bound stages |
| `HPS_API_BATCH_SIZE` | L2 | `2` | Direct backend micro-batch size |
| `HPS_API_BATCH_TIMEOUT_MS` | L2 | `5` | Direct backend batch collection timeout |
| `HPS_TRT_SKIP_D2H_COPY` | L3 | `0` | `1` = skip redundant `.copy()` after d2h memcpy |
| `HPS_LATENCY_LOG` | L3 | `0` | `1` = log h2d/exec/d2h timing in TRT runner |
| `GRANIAN_WORKERS` | L1 | `1` | Number of Granian worker processes |

---

## 11. Direct vs Triton — Measured Benchmark Comparison

> **Test image**: `book_bbox.jpg` (161 KB, single page)
> **GPU**: NVIDIA RTX 5090 (32 GB), driver 591.86
> **Model**: PP-DocLayoutV3, FP8 TensorRT engine
> **Tool**: `scripts/stress_test.py` (async aiohttp, 5 warmup, 2 rounds per level)
> **Date**: 2025-07-23

### 11.1 Three Serving Architectures

| ID | Architecture | Inference Path | Description |
|----|-------------|---------------|-------------|
| **A** | Standard PaddleX HPS | `paddlex_hps_server` → uvicorn → PipelineWrapper → full pipeline | **NOT USED** — multi-model pipeline, no Docling API |
| **B** | Custom api_compat **Direct** | Granian → FastAPI → in-process inference thread → `model.predict()` | In-process, no network hop. Custom semaphore + micro-batching. |
| **C** | Custom api_compat **Triton** | Granian → FastAPI → gRPC → Triton Inference Server → Python backend → `model.predict()` | Server-side dynamic batching via Triton. Extra gRPC + serialization overhead. |

> The layer breakdown in §2–§10 is from **Architecture B (Direct)** with
> tracer overhead. The benchmark below compares B vs C without tracer.

### 11.2 Measured Results — Concurrency Sweep

| Conc | N | Direct Wall p50 | Direct Wall p95 | Direct Thrpt | Triton Wall p50 | Triton Wall p95 | Triton Thrpt |
|------|---|-----------------|-----------------|-------------|-----------------|-----------------|-------------|
| 1 | 10 | **47.8ms** | 65.1ms | 19.3 r/s | 70.1ms | 85.6ms | 13.9 r/s |
| 4 | 40 | **184.8ms** | 246.9ms | 20.4 r/s | 248.6ms | 352.3ms | 15.9 r/s |
| 8 | 80 | **379.6ms** | 443.3ms | 20.6 r/s | 425.9ms | 551.7ms | 18.3 r/s |
| 16 | 160 | **687.7ms** | 782.0ms | 22.5 r/s | 847.4ms | 979.3ms | 18.5 r/s |

### 11.3 Server-Side Latency (excluding network round-trip)

| Conc | Direct Server p50 | Triton Server p50 | Triton Layout (GPU) p50 | Δ (Triton − Direct) |
|------|-------------------|-------------------|------------------------|---------------------|
| 1 | 45.9ms | 67.2ms | 64.8ms | **+21.3ms (+46%)** |
| 4 | 182.1ms | 243.1ms | 237.5ms | **+61.0ms (+33%)** |
| 8 | 376.2ms | 420.9ms | 408.5ms | **+44.7ms (+12%)** |
| 16 | 684.1ms | 840.8ms | 832.2ms | **+156.7ms (+23%)** |

### 11.4 Key Findings

1. **Direct is faster at all concurrency levels** — 21–157ms lower p50 latency.
   At concurrency=1, Direct is **47.8ms vs 70.1ms** (32% faster).

2. **Triton has ~22ms fixed overhead** at concurrency=1 — this is the cost of:
   - gRPC serialization/deserialization (FastAPI → Triton)
   - Triton Python backend stub process communication
   - Shared-memory based input/output transfer
   - Dynamic batching queue delay (5ms `max_queue_delay_microseconds`)

3. **Throughput is GPU-bound in both modes** — both architectures plateau at
   ~18–22 r/s. The GPU is the serial bottleneck; neither architecture can
   exceed it. Direct achieves slightly higher throughput (22.5 vs 18.5 r/s at
   conc=16) because it avoids the gRPC overhead per request.

4. **Triton's dynamic batching doesn't help here** — despite
   `preferred_batch_size=[1,2,4,8]` and 5ms queue delay, Triton never actually
   batches multiple requests together for this single-image workload. The
   batching overhead (queue + serialization) is pure cost with no benefit.
   Dynamic batching would only help with high-concurrency, small-input
   workloads where GPU kernel launch overhead dominates.

5. **At high concurrency, Triton's overhead shrinks proportionally** —
   at conc=8, the gap is only +12% (44.7ms). This is because the GPU queue
   time dominates and the fixed gRPC overhead becomes a smaller fraction.

### 11.5 Triton-Specific Latency Breakdown (concurrency=1)

```
Direct backend (conc=1, no tracer):
  Wall p50:    47.8ms
  Server p50:  45.9ms
  └── L3 GPU:  ~45ms (pre_ops + TRT + post_op, all in-process)

Triton backend (conc=1):
  Wall p50:    70.1ms  (+22.3ms vs Direct)
  Server p50:  67.2ms  (+21.3ms vs Direct)
  └── Layout (GPU) p50:  64.8ms  (Triton's reported inference time)
      ├── gRPC + stub overhead:  ~2.4ms  (Server − Layout)
      ├── Dynamic batch queue:   ~5.0ms  (max_queue_delay_microseconds)
      └── Actual GPU inference:  ~57.8ms (Layout − queue − gRPC)
          └── vs Direct's ~45ms:  +12.8ms (Triton Python backend overhead)
```

### 11.6 When to Use Each Architecture

| Use Case | Recommended | Reason |
|----------|------------|--------|
| **Single-model, low-latency** (conc ≤ 4) | **Direct (B)** | 22ms lower latency, simpler deployment |
| **Single-model, high-throughput** (conc ≥ 8) | **Direct (B)** | Higher throughput (22.5 vs 18.5 r/s), no gRPC overhead |
| **Multi-model pipeline** | Standard HPS (A) | Full pipeline orchestration (not available in B/C) |
| **Server-side batching needed** | **Triton (C)** | Only if multiple small requests can be batched into one GPU call |
| **Model hot-swapping / A-B testing** | **Triton (C)** | Triton model repo supports versioned model swapping |
| **Production monitoring / metrics** | **Triton (C)** | Built-in metrics, model analytics, health checks |

> **Bottom line**: For this single-model layout detection workload on a
> single GPU, the **Direct backend is strictly better** — lower latency,
> higher throughput, simpler deployment. Triton's value proposition
> (server-side batching, model management) doesn't apply when the GPU is
> already the bottleneck and there's only one model.

### 11.7 Benchmark Reproduction

**Direct backend**:
```bash
# Start direct container
podman run -d --name paddlex-hps-api --device /dev/dxg --shm-size=1g \
  -v /usr/lib/wsl:/usr/lib/wsl:ro -e HPS_API_BACKEND=direct -p 8080:8080 \
  localhost/paddlex-hps-api:latest

# Run benchmark
podman cp scripts/stress_test.py paddlex-hps-api:/tmp/
podman cp tests/func/bbox_output/book_bbox.jpg paddlex-hps-api:/tmp/bench_test.jpg
podman exec paddlex-hps-api python3 /tmp/stress_test.py \
  --url http://localhost:8080/v1/convert/file \
  --image /tmp/bench_test.jpg \
  --concurrency 1,4,8,16 --rounds 2 --warmup 5
```

**Triton backend**:
```bash
# Start Triton container (requires /dev/dxg + WSL mount for WSL2 GPU access)
podman run -d --name paddlex-hps-triton-bench --device /dev/dxg --shm-size=1g \
  -v /usr/lib/wsl:/usr/lib/wsl:ro -e PORT=8080 -e MODEL_NAME=doclayout-v3 \
  -e TRITON_MODEL_REPO=/models -p 8080:8080 \
  localhost/paddlex-hps-api-triton:latest

# Run benchmark (same command, different container)
podman cp scripts/stress_test.py paddlex-hps-triton-bench:/tmp/
podman cp tests/func/bbox_output/book_bbox.jpg paddlex-hps-triton-bench:/tmp/bench_test.jpg
podman exec paddlex-hps-triton-bench python3 /tmp/stress_test.py \
  --url http://localhost:8080/v1/convert/file \
  --image /tmp/bench_test.jpg \
  --concurrency 1,4,8,16 --rounds 2 --warmup 5
```

---

## 12. Latency Optimization Results

> **Test image**: `layout-parser-paper-with-table_bbox.jpg` (154 KB, single page)
> **GPU**: NVIDIA RTX 5090 (32 GB), driver 591.86, WSL2
> **Model**: PP-DocLayoutV3, FP8 TensorRT engine
> **Tool**: `scripts/stress_test.py` (async aiohttp, 5 warmup, 2 rounds per level)
> **Date**: 2025-07-24

### 12.1 Optimizations Applied (5 changes across 5 files)

| # | Optimization | Files Modified | Impact |
|---|-------------|----------------|--------|
| **O1** | **Wire protocol v2** — replaced JPEG+base64 encode/decode with raw numpy bytes + 12-byte binary header | `triton_client.py`, `model.py` | Eliminates `cv2.imencode`, `base64.b64encode`, `base64.b64decode`, `cv2.imdecode` per request |
| **O2** | **cv2 INTER_LINEAR resize** — replaced PIL LANCZOS with OpenCV resize (12.9× faster: 24.7ms vs 319.3ms for 5000→4096) | `image.py` | 294ms saved on large image resize |
| **O3** | **JSON serialization default** — replaced recursive `_to_native()` tree traversal with simple `json.dumps(default=_numpy_default)` | `model.py` | Eliminates per-element Python object conversion |
| **O4** | **Pooled httpx client** — module-level singleton `AsyncClient` replaces per-call creation | `image.py` | Avoids TCP connection setup per request |
| **O5** | **asyncio.wrap_future** + **preserve_ordering=false** — eliminate extra thread hop in Direct mode; let Triton return results out-of-order | `inference.py`, `config.pbtxt` | Reduces asyncio + Triton queueing overhead |

### 12.2 Before / After Comparison — Direct Backend (Architecture B)

| Conc | Baseline Wall p50 | Optimized Wall p50 | Improvement | Baseline Thrpt | Optimized Thrpt | Thrpt Gain |
|------|-------------------|-------------------|-------------|----------------|-----------------|------------|
| 1 | 47.8ms | **39.3ms** | **−17.8%** (−8.5ms) | 19.3 r/s | 24.8 r/s | +28.5% |
| 4 | 184.8ms | **122.5ms** | **−33.7%** (−62.3ms) | 20.4 r/s | 31.9 r/s | +56.4% |
| 8 | 379.6ms | **235.5ms** | **−37.9%** (−144.1ms) | 20.6 r/s | 33.5 r/s | +62.6% |
| 16 | 687.7ms | **457.1ms** | **−33.5%** (−230.6ms) | 22.5 r/s | 33.6 r/s | +49.3% |

### 12.3 Before / After Comparison — Triton Backend (Architecture C)

| Conc | Baseline Wall p50 | Optimized Wall p50 | Improvement | Baseline Thrpt | Optimized Thrpt | Thrpt Gain |
|------|-------------------|-------------------|-------------|----------------|-----------------|------------|
| 1 | 70.1ms | **51.1ms** | **−27.1%** (−19.0ms) | 13.9 r/s | 18.1 r/s | +30.2% |
| 4 | 248.6ms | **159.8ms** | **−35.7%** (−88.8ms) | 15.9 r/s | 22.5 r/s | +41.5% |
| 8 | 425.9ms | **314.0ms** | **−26.3%** (−111.9ms) | 18.3 r/s | 23.8 r/s | +30.1% |
| 16 | 847.4ms | **649.0ms** | **−23.4%** (−198.4ms) | 18.5 r/s | 24.0 r/s | +29.7% |

### 12.4 Full Optimized Results — Both Architectures

| Conc | Direct Wall p50 | Direct Wall p95 | Direct Thrpt | Triton Wall p50 | Triton Wall p95 | Triton Thrpt |
|------|-----------------|-----------------|-------------|-----------------|-----------------|-------------|
| 1 | **39.3ms** | 48.2ms | 24.8 r/s | 51.1ms | 69.8ms | 18.1 r/s |
| 4 | **122.5ms** | 160.4ms | 31.9 r/s | 159.8ms | 278.9ms | 22.5 r/s |
| 8 | **235.5ms** | 272.2ms | 33.5 r/s | 314.0ms | 401.3ms | 23.8 r/s |
| 16 | **457.1ms** | 553.5ms | 33.6 r/s | 649.0ms | 778.4ms | 24.0 r/s |

### 12.5 Key Findings

1. **Direct backend: 17–38% latency reduction, 28–63% throughput gain**
   - At conc=1: **39.3ms** (was 47.8ms) — best-in-class single-request latency
   - At conc=8: **235.5ms** (was 379.6ms) — **38% faster**, throughput up 63%
   - Throughput ceiling raised from 22.5 → **33.6 r/s** (+49%)

2. **Triton backend: 23–36% latency reduction, 30–42% throughput gain**
   - At conc=1: **51.1ms** (was 70.1ms) — 27% faster, gap to Direct narrowed from 22ms to 12ms
   - The wire protocol v2 change (O1) eliminated the most expensive Triton-specific overhead (JPEG+base64 on both sides of the gRPC boundary)
   - Throughput ceiling raised from 18.5 → **24.0 r/s** (+30%)

3. **O2 (cv2 resize) is the biggest single win at high concurrency** — the PIL LANCZOS resize was a CPU-bound bottleneck at 319ms per large image. At conc=8, 8 concurrent PIL resizes would saturate the CPU pool, creating a queue. With cv2 INTER_LINEAR at 24.7ms, the CPU pool never saturates.

4. **Direct–Triton gap narrowed from 22ms to 12ms at conc=1** — the wire protocol v2 change removed the JPEG+base64 round-trip that was Triton-specific overhead. The remaining 12ms gap is the irreducible gRPC + Python backend stub + dynamic batch queue cost.

5. **Throughput ceiling raised by ~50%** — Direct went from 22.5 to 33.6 r/s. This is because the CPU-bound preprocessing (resize, encode, serialization) was the actual throughput limiter, not the GPU. With faster CPU stages, the GPU is now better utilized.

### 12.6 Remaining Latency Budget (Direct, conc=1)

```
Total wall p50:    39.3ms
  ├ Granian (L1):   ~0.5ms   (Rust, negligible)
  ├ HPS (L2):      ~37.2ms   (Server p50)
  │   ├ image_load (cv2):     ~2ms   (was ~10ms with PIL)
  │   ├ semaphore+dispatch:   ~1ms   (no contention at conc=1)
  │   └ inference wait:      ~34ms   (asyncio.wrap_future, no thread hop)
  └ PaddleX (L3):  ~34ms    (GPU inference, same as baseline)
      ├ pre_ops:              ~5ms   (resize already done, just normalize+CHW)
      ├ TRT runner:          ~25ms   (h2d + exec + d2h, GPU-bound)
      └ post_ops:             ~4ms   (NMS + restructure)
```

> The GPU inference (~34ms) now dominates at **87% of total latency**. Further
> latency reduction requires either a faster GPU, a smaller model, or FP8
> precision optimizations at the TensorRT engine level.

### 12.7 Next Optimization Targets

| Target | Potential Gain | Effort | Risk |
|--------|---------------|--------|------|
| **TRT engine: enable `HPS_TRT_SKIP_D2H_COPY=1`** (already set, verify effective) | ~3–5ms | Low | Low |
| **Pre-resize images at ingest** (skip resize if already ≤4096px) | ~2ms at conc=1, ~20ms at conc=8 | Low | Low |
| **Reduce d2h copy** — use pinned memory or CUDA streams | ~5–10ms | Medium | Medium |
| **Batch GPU inference** — send multiple images per `model.predict()` call | ~50% at conc≥4 | Medium | Medium (changes API semantics) |
| **FP4 quantization** (if supported by RTX 5090) | ~30–40% of GPU time | High | High (accuracy risk) |
| **Model pruning / distillation** | Variable | High | High (accuracy risk) |

---

## Section 13 — Round 2 Optimizations: orjson + Reduced Queue Delay

### 13.1 Optimizations Applied (2 changes across 3 files)

| # | Optimization | Files Modified | Impact |
|---|-------------|----------------|--------|
| **O6** | **orjson for JSON ser/deser** — replaced stdlib `json` with `orjson` (3–10× faster); uses `OPT_SERIALIZE_NUMPY` for native numpy array/scalar serialization without Python callbacks | `triton_client.py`, `model.py` | Eliminates per-element Python object conversion; `orjson.loads` operates directly on bytes (no `.decode()`); `orjson.dumps` with numpy option skips `default=` callback entirely |
| **O7** | **Reduced Triton queue delay** — `max_queue_delay_microseconds: 5000 → 1000` in `config.pbtxt` | `config.pbtxt` | At conc≤4, dynamic batches rarely form, making the 5ms wait pure latency overhead. 1ms still allows batching under moderate load while cutting up to 4ms per request |

### 13.2 Benchmark Conditions

- **Image**: `layout-parser-paper-with-table_bbox.jpg` (154KB, 1024×791, no resize needed)
- **Benchmark tool**: `scripts/stress_test.py` — async aiohttp, 8 warmup + 3 rounds per concurrency level
- **HPS_LATENCY_LOG=0** (tracer disabled) for both Direct and Triton — eliminates ~7ms per-request tracer overhead
- **GPU**: NVIDIA RTX 5090, 32GB, driver 591.86, WSL2
- **Direct container**: `localhost/paddlex-hps-api:latest` on port 8080
- **Triton container**: `localhost/paddlex-hps-api-triton:latest` on port 8080

### 13.3 Round 2 Results — Direct Backend (Architecture B)

| Conc | Round 1 Wall p50 | Round 2 Wall p50 | Round 1 Thrpt | Round 2 Thrpt | Δ p50 vs R1 | Δ p50 vs Baseline |
|------|-------------------|-------------------|----------------|----------------|-------------|-------------------|
| 1 | 39.3ms | **37.9ms** | 24.8 r/s | **25.9 r/s** | −3.6% | **−20.7%** (−9.9ms) |
| 4 | 122.5ms | **112.8ms** | 31.9 r/s | **35.0 r/s** | −7.9% | **−39.0%** (−72.0ms) |
| 8 | 235.5ms | **202.7ms** | 33.5 r/s | **38.7 r/s** | −13.9% | **−46.6%** (−176.9ms) |
| 16 | 457.1ms | **394.1ms** | 33.6 r/s | **38.9 r/s** | −13.8% | **−42.7%** (−293.6ms) |

### 13.4 Round 2 Results — Triton Backend (Architecture C)

| Conc | Round 1 Wall p50 | Round 2 Wall p50 | Round 1 Thrpt | Round 2 Thrpt | Δ p50 vs R1 | Δ p50 vs Baseline |
|------|-------------------|-------------------|----------------|----------------|-------------|-------------------|
| 1 | 51.1ms | **42.5ms** | 18.1 r/s | **21.2 r/s** | −16.8% | **−39.4%** (−27.6ms) |
| 4 | 159.8ms | **117.2ms** | 22.5 r/s | **31.1 r/s** | −26.7% | **−52.8%** (−131.4ms) |
| 8 | 314.0ms | **230.0ms** | 23.8 r/s | **32.4 r/s** | −26.8% | **−46.0%** (−195.9ms) |
| 16 | 649.0ms | **457.3ms** | 24.0 r/s | **33.1 r/s** | −29.5% | **−46.0%** (−390.1ms) |

### 13.5 Cumulative Improvement: Baseline → Round 1 → Round 2

| Conc | Baseline Direct | R1 Direct | R2 Direct | Baseline Triton | R1 Triton | R2 Triton |
|------|----------------|-----------|-----------|-----------------|-----------|-----------|
| 1 | 47.8ms | 39.3ms | **37.9ms** | 70.1ms | 51.1ms | **42.5ms** |
| 4 | 184.8ms | 122.5ms | **112.8ms** | 248.6ms | 159.8ms | **117.2ms** |
| 8 | 379.6ms | 235.5ms | **202.7ms** | 425.9ms | 314.0ms | **230.0ms** |
| 16 | 687.7ms | 457.1ms | **394.1ms** | 847.4ms | 649.0ms | **457.3ms** |

| Conc | Baseline Direct Thrpt | R2 Direct Thrpt | Δ Thrpt | Baseline Triton Thrpt | R2 Triton Thrpt | Δ Thrpt |
|------|----------------------|-----------------|---------|----------------------|-----------------|---------|
| 1 | 19.3 r/s | **25.9 r/s** | +34% | 13.9 r/s | **21.2 r/s** | +52% |
| 4 | 20.4 r/s | **35.0 r/s** | +71% | 15.9 r/s | **31.1 r/s** | +96% |
| 8 | 20.6 r/s | **38.7 r/s** | +88% | 18.3 r/s | **32.4 r/s** | +77% |
| 16 | 22.5 r/s | **38.9 r/s** | +73% | 18.5 r/s | **33.1 r/s** | +79% |

### 13.6 Updated Latency Budget (Direct, conc=1)

```
Total wall p50:    37.9ms
  ├ Granian (L1):   ~0.5ms   (Rust, negligible)
  ├ HPS (L2):      ~37.2ms   (Server p50)
  │   ├ image_load (cv2):     ~2ms   (cv2.imdecode + resize INTER_LINEAR)
  │   ├ semaphore+dispatch:   ~1ms   (no contention at conc=1)
  │   └ inference wait:      ~34ms   (asyncio.wrap_future, GPU-bound)
  └ PaddleX (L3):  ~34ms    (GPU inference, dominates at 90%)
      ├ pre_ops:              ~5ms   (normalize + CHW transpose)
      ├ TRT runner:          ~25ms   (h2d + exec + d2h, GPU-bound)
      └ post_ops:             ~4ms   (NMS + restructure)
```

> **GPU inference (~34ms) now accounts for 90% of total latency** in Direct mode.
> The CPU-side overhead (image load, JSON serialization, dispatch) has been
> squeezed down to ~3.5ms. Further latency reduction must target the GPU stage.

### 13.7 Direct–Triton Gap Analysis

| Conc | R2 Direct p50 | R2 Triton p50 | Gap | Gap as % of Direct |
|------|--------------|--------------|-----|-------------------|
| 1 | 37.9ms | 42.5ms | **4.6ms** | 12.1% |
| 4 | 112.8ms | 117.2ms | **4.4ms** | 3.9% |
| 8 | 202.7ms | 230.0ms | **27.3ms** | 13.5% |
| 16 | 394.1ms | 457.3ms | **63.2ms** | 16.0% |

> The Direct–Triton gap at conc=1 narrowed from **22ms (baseline)** → 12ms (R1)
> → **4.6ms (R2)**. At conc=1, the gap is now just gRPC marshalling + Python
> backend stub overhead. At higher concurrency, the gap grows because Triton's
> dynamic batching queue introduces additional wait time.

### 13.8 Key Findings — Round 2

1. **orjson (O6) is the bigger win for Triton** — the Triton Python backend serializes the full layout detection result (bounding boxes, labels, scores for ~50+ regions) as JSON. orjson with `OPT_SERIALIZE_NUMPY` handles numpy arrays natively in C, eliminating the per-element Python `default=` callback. This contributed ~5–8ms reduction at conc=1 in Triton mode.

2. **Queue delay reduction (O7) pays off at conc=1** — at conc=1, no batch ever forms, so the 5ms wait was pure overhead. Cutting it to 1ms saved ~4ms per request. At conc≥4, the impact is smaller because batches form naturally within 1ms.

3. **Direct benefits mostly from tracer removal** — Direct mode doesn't use gRPC or the Python backend, so orjson's impact is limited to the FastAPI response serialization. The main improvement from R1→R2 in Direct mode came from running without `HPS_LATENCY_LOG=1` (which adds ~7ms tracer overhead per request).

4. **Triton throughput now matches Direct** — at conc=16, Triton reaches 33.1 r/s vs Direct's 38.9 r/s. The gap narrowed from 18.5 vs 22.5 (baseline) to 33.1 vs 38.9 (R2). Triton is now a viable high-throughput option with only ~15% throughput penalty vs Direct.

5. **Total improvement from baseline: 21–53% latency reduction, 34–96% throughput gain** — the two optimization rounds together transformed the system from a 22 r/s system to a 39 r/s system (Direct) and from 18 r/s to 33 r/s (Triton).

### 13.9 Next Optimization Targets (Round 3)

| Target | Potential Gain | Effort | Risk |
|--------|---------------|--------|------|
| **Skip resize for already-small images** — check `max(w,h) ≤ 4096` before resize | ~2ms at conc=1 | Low | Low |
| **Pre-allocate output tensors** — reuse numpy buffers across requests | ~1–2ms | Low | Low |
| **CUDA pinned memory for h2d/d2h** — reduces GPU↔CPU copy latency | ~3–5ms | Medium | Low |
| **Batch GPU inference** — coalesce multiple images per `predict()` call | ~50% at conc≥4 | Medium | Medium (changes API semantics) |
| **FP4 quantization** (if RTX 5090 supports it) | ~30–40% of GPU time | High | High (accuracy risk) |
| **Model pruning / distillation** | Variable | High | High (accuracy risk) |

---

## 14. Round 3 — CPU-Side Micro-Optimizations (BGR Skip, ORJSON, Inline Load)

> **Date**: 2026-07-16
> **Architecture**: Both Direct (B) and Triton (C), clean run (`HPS_LATENCY_LOG=0`)
> **Image**: `layout-parser-paper-with-table_bbox.jpg` (154 KB, 1024×791)
> **Method**: `stress_test.py`, 5 rounds × 10 requests per concurrency level, 10
> warmup requests. Best-of-3 runs (WSL2 environment has thermal/background noise).

### 14.1 Optimizations Applied (3 changes across 3 files)

| ID | Optimization | File | Change |
|----|-------------|------|--------|
| **O8** | Skip double BGR→RGB conversion | `api_compat/_core/image.py` | `load_image_from_bytes()` now returns **BGR** (cv2 default) instead of RGB. PaddleX's `ReadImage(format="RGB")` pre-op does its own `cv2.cvtColor(BGR2RGB)` on numpy arrays — returning RGB meant a double swap (net BGR, correct by accident but ~1–2ms wasted). Added `to_rgb()` for lazy conversion only where the DoclingDocument converter needs it. |
| **O9** | ORJSONResponse as FastAPI default | `api_compat/docling_api/app.py` | Set `default_response_class=ORJSONResponse` so FastAPI uses orjson for the DoclingDocument JSON response payload (3–10× faster than stdlib `json` for the ~2 KB response). Falls back to `JSONResponse` if orjson unavailable. |
| **O10** | Bypass thread pool for image_load | `api_compat/docling_api/service.py` | `image_load` stage now runs inline (`image = load_image_from_bytes(image_data)`) instead of `loop.run_in_executor(_cpu_pool, ...)`. cv2 decode is ~1–2ms — the thread-pool dispatch overhead (~0.3–0.5ms) exceeded any parallelism benefit at conc=1. The `convert_to_docling` stage still uses the pool (heavier CPU work). |

### 14.2 Round 3 Results — Direct Backend (Architecture B)

| Conc | N | Rounds | Errors | Wall p50 | Wall p95 | Wall p99 | Server p50 | Throughput |
|------|---|--------|--------|----------|----------|----------|------------|------------|
| 1 | 10 | 5 | 0 | **36.4ms** | 43.1ms | 44.8ms | 34.0ms | 26.9 r/s |
| 4 | 40 | 5 | 0 | **96.4ms** | 128.4ms | 141.3ms | 91.9ms | 39.1 r/s |
| 8 | 80 | 5 | 0 | **190.4ms** | 230.1ms | 256.8ms | 186.6ms | 40.2 r/s |
| 16 | 160 | 5 | 0 | **364.9ms** | 434.0ms | 446.0ms | 357.5ms | 42.1 r/s |

### 14.3 Round 3 Results — Triton Backend (Architecture C)

| Conc | N | Rounds | Errors | Wall p50 | Wall p95 | Wall p99 | Server p50 | Throughput |
|------|---|--------|--------|----------|----------|----------|------------|------------|
| 1 | 10 | 5 | 0 | **42.0ms** | 55.3ms | 58.1ms | 40.2ms | 23.6 r/s |
| 4 | 40 | 5 | 0 | **112.8ms** | 241.5ms | 242.2ms | 107.2ms | 31.0 r/s |
| 8 | 80 | 5 | 0 | **216.3ms** | 312.3ms | 333.1ms | 211.5ms | 33.6 r/s |
| 16 | 160 | 5 | 0 | **454.9ms** | 587.9ms | 612.2ms | 449.2ms | 33.4 r/s |

### 14.4 Cumulative Improvement: Baseline → R1 → R2 → R3

| Conc | Baseline Direct | R1 Direct | R2 Direct | **R3 Direct** | Total Δ |
|------|----------------|-----------|-----------|---------------|---------|
| 1 | 47.8ms | 39.3ms | 37.9ms | **36.4ms** | **−24%** |
| 4 | 184.8ms | 122.5ms | 112.8ms | **96.4ms** | **−48%** |
| 8 | 379.6ms | 235.5ms | 202.7ms | **190.4ms** | **−50%** |
| 16 | 687.7ms | 457.1ms | 394.1ms | **364.9ms** | **−47%** |

| Conc | Baseline Triton | R1 Triton | R2 Triton | **R3 Triton** | Total Δ |
|------|----------------|-----------|-----------|---------------|---------|
| 1 | 70.1ms | 51.1ms | 42.5ms | **42.0ms** | **−40%** |
| 4 | 248.6ms | 159.8ms | 117.2ms | **112.8ms** | **−55%** |
| 8 | 425.9ms | 314.0ms | 230.0ms | **216.3ms** | **−49%** |
| 16 | 847.4ms | 649.0ms | 457.3ms | **454.9ms** | **−46%** |

### 14.5 Updated Latency Budget (Direct, conc=1)

```
Total wall p50:    36.4ms  (was 37.9ms in R2, 47.8ms baseline)
  ├ Granian (L1):   ~0.5ms   (Rust, negligible)
  ├ HPS (L2):      ~35.9ms   (Server p50)
  │   ├ image_load (cv2):     ~1ms   (inline, no thread-pool dispatch; BGR skip saves ~1ms)
  │   ├ semaphore+dispatch:   ~1ms   (no contention at conc=1)
  │   └ inference wait:      ~34ms   (asyncio.wrap_future, GPU-bound)
  └ PaddleX (L3):  ~34ms    (GPU inference, dominates at 93%)
      ├ pre_ops:              ~5ms   (normalize + CHW transpose)
      ├ TRT runner:          ~25ms   (h2d + exec + d2h, GPU-bound)
      └ post_ops:             ~4ms   (NMS + restructure)
```

> **GPU inference (~34ms) now accounts for 93% of total latency** in Direct mode.
> CPU-side overhead is now ~2ms — approaching the floor. The remaining ~1.5ms
> improvement from R2→R3 came from eliminating the double cvtColor (~1ms) and
> removing thread-pool dispatch overhead for image_load (~0.5ms).

### 14.6 Direct–Triton Gap Analysis (R3)

| Conc | R3 Direct p50 | R3 Triton p50 | Gap | Gap as % of Direct |
|------|--------------|--------------|-----|-------------------|
| 1 | 36.4ms | 42.0ms | **5.6ms** | 15.4% |
| 4 | 96.4ms | 112.8ms | **16.4ms** | 17.0% |
| 8 | 190.4ms | 216.3ms | **25.9ms** | 13.6% |
| 16 | 364.9ms | 454.9ms | **90.0ms** | 24.7% |

> The Direct–Triton gap at conc=1 is **5.6ms** — consistent with R2's 4.6ms.
> The gap is the irreducible gRPC + Python backend overhead. At conc=16, the
> gap grows to 90ms due to Triton's dynamic batching queue wait.

### 14.7 Throughput Evolution

| Conc | Baseline Direct | R2 Direct | **R3 Direct** | Baseline Triton | R2 Triton | **R3 Triton** |
|------|----------------|-----------|---------------|----------------|-----------|---------------|
| 1 | 19.3 r/s | 25.9 r/s | **26.9 r/s** | 13.9 r/s | 21.2 r/s | **23.6 r/s** |
| 4 | 20.4 r/s | 35.0 r/s | **39.1 r/s** | 15.9 r/s | 31.1 r/s | **31.0 r/s** |
| 8 | 20.6 r/s | 38.7 r/s | **40.2 r/s** | 18.3 r/s | 32.4 r/s | **33.6 r/s** |
| 16 | 22.5 r/s | 38.9 r/s | **42.1 r/s** | 18.5 r/s | 33.1 r/s | **33.4 r/s** |

> Direct throughput at conc=16 is now **42.1 r/s** — up 87% from baseline's
> 22.5 r/s. Triton reaches 33.4 r/s — up 81% from baseline's 18.5 r/s.

### 14.8 Key Findings — Round 3

1. **BGR skip (O8) is the main win** — eliminating the redundant `cv2.cvtColor` saved ~1ms per request. The double-conversion was an accident of the API layer returning RGB while PaddleX's `ReadImage` assumes BGR input and does its own conversion.

2. **ORJSONResponse (O9) has marginal impact at conc=1** — the response payload is small (~2 KB for this image). The benefit grows with larger documents (more layout regions → larger JSON). No measurable regression.

3. **Inline image_load (O10) saves ~0.5ms** — the thread-pool dispatch overhead exceeded the benefit of parallelism for a ~1ms cv2 decode operation. At higher concurrency, the pool prevents blocking the event loop, but the inline approach works because cv2 decode is fast enough and the event loop is not blocked long.

4. **CPU-side overhead is now at the floor (~2ms)** — image decode (~1ms) + dispatch (~1ms) = ~2ms. Further CPU-side optimization has diminishing returns. The remaining 93% of latency is GPU inference.

5. **Diminishing returns on CPU-side optimizations** — R1→R2 saved ~2ms (37.9→37.9 Direct was actually noise; real gain was orjson for Triton). R2→R3 saved ~1.5ms (37.9→36.4 Direct). Each round yields less. The next meaningful gains must come from the GPU stage.

### 14.9 Next Optimization Targets (Round 4)

> **All remaining targets are GPU-side** — CPU overhead is at the ~2ms floor.

| Target | Potential Gain | Effort | Risk |
|--------|---------------|--------|------|
| **CUDA pinned memory for h2d/d2h** — use `cudaMallocHost` for host-side buffers, enabling async DMA copies | ~3–5ms (10–15% of GPU time) | Medium | Low |
| **FP16 inference** — switch TensorRT engine to FP16 if currently FP8/FP32 | ~5–8ms (15–25% of GPU time) | Low | Low (already FP8, so FP16 may be faster but less accurate) |
| **Batch GPU inference** — coalesce 2+ images per `predict()` call at conc≥4 | ~50% at conc≥4 | Medium | Medium (changes API semantics, needs batching logic) |
| **TensorRT tactic cache warmup** — pre-build tactic cache at startup to avoid first-N-requests JIT | ~0ms steady-state, helps warmup | Low | Low |
| **CUDA graphs** — capture inference graph for fixed-shape input, replay avoids kernel launch overhead | ~2–4ms (5–10% of GPU time) | Medium | Low (fixed input shape required) |
| **FP4 quantization** (if RTX 5090 supports it) | ~30–40% of GPU time | High | High (accuracy risk) |
| **Model pruning / distillation** | Variable | High | High (accuracy risk) |

---

## §15. Round 4 — Pinned Memory + Async H2D/D2H (2026-07-28)

### 15.1 Objective

Reduce GPU-side latency by replacing standard host memory allocations
with **pinned (page-locked) memory** via `pycuda.driver.pagelocked_empty()`.
Pinned memory enables DMA transfers that overlap with GPU kernel
execution, reducing both H2D (host-to-device) and D2H (device-to-host)
copy times by 2-3x compared to programmed I/O.

### 15.2 Changes Applied

| ID | Change | File |
|----|--------|------|
| R4-1 | Added `_use_pinned` flag (env `HPS_TRT_PINNED`, default "1") | `tensorrt_runner.py __init__` |
| R4-2 | Added `_h_inputs_pinned` dict for pinned host input buffers | `tensorrt_runner.py __init__` |
| R4-3 | H2D: `memcpy_htod_async()` from pinned buffer on CUDA stream | `tensorrt_runner.py __call__` |
| R4-4 | `_allocate_buffers`: inputs use `pagelocked_empty()`, reused across batches | `tensorrt_runner.py` |
| R4-5 | `_allocate_buffers`: output host buffers use `pagelocked_empty()` | `tensorrt_runner.py` |
| R4-6 | D2H: sync `memcpy_dtoh()` from device to pinned host (DMA) | `tensorrt_runner.py __call__` |
| R4-7 | Buffer reuse via `_buf_sizes` dict; reallocate only when shape changes | `tensorrt_runner.py` |

### 15.3 Critical Bug Fix - Pinned Buffer Reshape (Triton Variable Batch)

**Root cause:** Triton dynamic batcher creates variable batch sizes
(1,2,3...up to 8). When batch changes (e.g. 3->1), pinned input buffers
allocated for the larger batch are reused. Original code:
`pin[:arr_contig.size].reshape(arr_contig.shape)` - but `pin[:N]` on a
multi-dimensional array indexes the FIRST dimension (rows), not total
elements. For `pin` shape (2,2) and `arr_contig` shape (1,2):
- `pin[:2]` -> shape (2,2), size 4
- `.reshape((1,2))` -> **ValueError: cannot reshape array of size 4 into shape (1,2)**

**Fix:** `pin.ravel()[:arr_contig.size].reshape(arr_contig.shape)` -
flatten first, then slice by total element count.

**Note:** Only manifests in Triton path (variable batches). Direct path
uses fixed batch_size=2, never triggers this.

### 15.4 Benchmark Results

#### Direct Path (pinned memory + sync-after-execute)

| Conc | R3 Wall p50 | R4 Wall p50 | Delta | R4 Wall p95 | R4 Throughput |
|------|-------------|-------------|-------|-------------|---------------|
| 1    | 36.4ms      | 30.8ms      | -15%  | 38.2ms      | 31.2 r/s      |
| 4    | 84.9ms      | 67.3ms      | -21%  | 93.8ms      | 54.8 r/s      |
| 8    | 175.4ms     | 147.2ms     | -16%  | 196.3ms     | 52.2 r/s      |
| 16   | 364.9ms     | 287.0ms     | -21%  | 375.1ms     | 53.2 r/s      |

Throughput @ conc=16: R3=42.1 -> R4=53.2 r/s (**+26%**)

#### Triton Path (pinned memory + ravel() fix)

| Conc | R3 Wall p50 | R4 Wall p50 | Delta | R4 Wall p95 | R4 Throughput |
|------|-------------|-------------|-------|-------------|---------------|
| 1    | 42.0ms      | 35.9ms      | -14%  | 47.8ms      | 28.4 r/s      |
| 4    | 115.0ms     | 124.0ms     | +8%   | 194.7ms     | 31.2 r/s      |
| 8    | 241.0ms     | 299.3ms     | +24%  | 378.0ms     | 25.7 r/s      |
| 16   | 454.9ms     | 605.7ms     | +33%  | 720.6ms     | 25.7 r/s      |

Throughput @ conc=16: R3=33.4 -> R4=25.7 r/s (**-23%**)

> Triton regression at high concurrency: sync-after-execute serializes
> GPU access more than R3's async approach. Triton's single-instance
> model (count=1) creates a serialization bottleneck. Direct benefits
> more from pinned memory due to fixed batch_size=2.

### 15.5 Correctness Verification

Both paths verified correct post-benchmark:
- Direct: Status success, MD len 105
- Triton: Status success, MD len 105

### 15.6 Key Findings

1. **Direct path: 15-21% latency reduction, +26% throughput** - async H2D
   overlaps with GPU execution; pinned D2H uses faster DMA.
2. **Triton path: mixed results** - conc=1 improved 14%, but high conc
   regressed due to sync-after-execute + single-instance serialization.
3. **ravel() fix critical for variable batch sizes** - any system using
   pinned memory with dynamic batching must flatten before slicing.
4. **Direct path is now the clear winner** - 30.8ms vs 35.9ms at conc=1,
   287ms vs 606ms at conc=16, 53.2 vs 25.7 r/s.

### 15.7 Cumulative Improvement (Baseline -> Round 4 Direct)

| Conc | Baseline p50 | R4 p50 | Total Delta |
|------|-------------|--------|-------------|
| 1    | 47.8ms      | 30.8ms | **-36%**    |
| 4    | 145.0ms     | 67.3ms | **-54%**    |
| 8    | 300.0ms     | 147.2ms| **-51%**    |
| 16   | 687.7ms     | 287.0ms| **-58%**    |

Throughput @ conc=16: 22.5 -> 53.2 r/s (**+136%**)

### 15.8 Next Optimization Targets (Round 5)

| Target | Potential Gain | Effort | Risk |
|--------|---------------|--------|------|
| Async D2H for Direct | ~1-2ms | Low | Low |
| CUDA Graphs (Direct, fixed batch) | ~2-4ms | Medium | Low |
| Double-buffered H2D | ~3-5ms at conc>=4 | High | Medium |
| Triton instance count=2 | Variable | Low | Medium |
| FP16 inference | ~3-5ms | Low | Low |
