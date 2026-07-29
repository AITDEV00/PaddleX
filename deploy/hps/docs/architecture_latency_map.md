# Architecture & Latency Map — PaddleX HPS Docling API

> **Method**: Logic Mapping Technique — trace-through-before-you-build.
> Every function in the request path is traced end-to-end with exact
> file/line references, data contracts, and measured/estimated latency.

---

## 1. The Big Picture: Request Lifecycle

```
HTTP POST /v1/convert/file  (or /v1/convert/source)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Granian ASGI Server  (WORKERS=1, single process)                       │
│  └─ asyncio event loop (single thread)                                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FastAPI Route Handler                                                   │
│  routes.py:88  convert_source() / routes.py:129  convert_file()         │
│  ┌─ parse request (Form/JSON)                              ~0.1ms       │
│  └─ call convert_image(image_data, filename, to_formats)                │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Conversion Pipeline  (service.py:90  convert_image)                     │
│                                                                         │
│  Stage A: image_load       ──→  9.9ms mean   (CPU pool thread)          │
│  Stage B: layout_detection ──→  155.4ms mean (asyncio + inference thread)│
│  Stage C: convert_to_docling →  1.0ms mean   (CPU pool thread)          │
│  Stage D: export_formats   ──→  1.7ms mean   (CPU pool thread, parallel) │
│                                                                         │
│  Total server-side:  158.1ms mean  (p50=140.7ms  max=404.3ms)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Full Call Chain — Function by Function

### Stage A: Image Load (9.9ms mean)

```
routes.py:129  convert_file()
│  ├── file.read()                                    ~0.5ms   (async I/O)
│  └── await convert_image(image_data, filename, fmt)          [service.py:90]
│
service.py:130  convert_image()
│  ├── tracer = LatencyTracer(...)                    ~0ms     (no-op if disabled)
│  └── with tracer.stage("image_load"):
│      └── await loop.run_in_executor(_cpu_pool, ...)         [offload to thread]
│          │
│          image.py:18  load_image_from_bytes(data)
│          ├── PIL.Image.open(BytesIO(data))          ~2-3ms   (PNG/JPEG decode)
│          ├── .convert("RGB")                        ~1ms     (channel normalize)
│          ├── if max(h,w) > MAX_IMAGE_DIM(4096):     ~3-5ms   (conditional resize)
│          │   └── img.resize((new_w,new_h), LANCZOS)
│          └── np.array(img)                          ~2-3ms   (PIL → numpy HWC uint8)
│
│  Data contract OUT: np.ndarray (H, W, 3) RGB uint8
```

**Latency breakdown:**
| Sub-step | Time | Notes |
|----------|------|-------|
| PIL decode | ~2-3ms | Depends on format (PNG faster than JPEG) |
| RGB convert | ~1ms | PIL internal |
| Resize (if >4096px) | ~3-5ms | LANCZOS interpolation, only if oversized |
| np.array() | ~2-3ms | PIL Image → numpy array copy |
| **Total** | **~9.9ms** | Runs on CPU pool thread (non-blocking) |

---

### Stage B: Layout Detection (155.4ms mean — THE BOTTLENECK)

```
service.py:142  with tracer.stage("layout_detection"):
│  └── await run_layout_detection(image)                        [inference.py:318]
│
inference.py:318  run_layout_detection(image)
│  └── if INFERENCE_BACKEND == "direct":
│      └── await _run_direct(image)                             [inference.py:375]
│
inference.py:375  _run_direct(image)
│
│  ┌─ B.1: SEMAPHORE ACQUIRE ──────────────────────────────────────────┐
│  │  async with state.semaphore:    # PIPELINE_DEPTH=4               │
│  │     # WAIT if 4+ requests already in-flight                      │
│  │     # Measured: 40.3ms mean (contention under load)              │
│  └──────────────────────────────────────────────────────────────────┘
│
│  ┌─ B.2: TASK SUBMIT ───────────────────────────────────────────────┐
│  │  task_id = uuid.uuid4().hex                                      │
│  │  future = Future()                                               │
│  │  state._pending[task_id] = future                                │
│  │  state._task_queue.put((task_id, image, future))  # ~0.01ms      │
│  └──────────────────────────────────────────────────────────────────┘
│
│  ┌─ B.3: ASYNC WAIT ────────────────────────────────────────────────┐
│  │  boxes = await loop.run_in_executor(None, future.result)         │
│  │  # Event loop is FREE while inference thread works               │
│  │  # Measured gpu_inference span: 104.5ms mean                     │
│  └──────────────────────────────────────────────────────────────────┘
│
│  └── return boxes
```

**B.1 — Semaphore Wait (40.3ms mean, 25.9% of total)**

```
inference.py:387  async with state.semaphore:
```
- `state.semaphore = asyncio.Semaphore(PIPELINE_DEPTH)` — created in `AppState.__init__()` (inference.py:113)
- **PIPELINE_DEPTH=4** (was 3, increased this session)
- This is **pure contention wait** — no CPU work happens here
- When ≤4 requests are in-flight: ~0ms (instant acquire)
- When >4 requests are in-flight: waits for a slot to free up
- p50=17.9ms, max=260.3ms — long tail from burst traffic
- **Optimization applied**: Increased PIPELINE_DEPTH 3→4 to reduce contention

**B.2 — Task Submit (~0.01ms)**

```python
task_id = uuid.uuid4().hex                    # ~0.01ms
future: Future = Future()                     # ~0ms
state._pending[task_id] = future              # dict insert, ~0ms
state._task_queue.put((task_id, image, future))  # queue.Queue.put, ~0.01ms
```

Negligible — just UUID generation + queue insertion.

**B.3 — Async Wait for Inference Thread (104.5ms mean, 67.4% of total)**

```
loop.run_in_executor(None, future.result)
```
- The asyncio event loop calls `future.result()` in the **default ThreadPoolExecutor**
- This blocks the executor thread (not the event loop) until the inference thread resolves the future
- The 104.5ms measured here = everything the inference thread does (see B.4 below)

---

### B.4 — Inference Thread: What Actually Runs (104.5ms)

```
inference.py:205  _inference_worker()  [dedicated daemon thread]
│
│  # Model already loaded at startup
│  while True:
│      task = state._task_queue.get()          # blocks until task arrives
│      │
│      └── if BATCH_SIZE <= 1:
│          └── _process_single(task)           [inference.py:255]
│      else:
│          └── _process_batch(batch)           [inference.py:278]
│
inference.py:255  _process_single(task)
│  ├── gen = state.model.predict(image)        ← ENTERS PADDLEX (generator)
│  ├── results = list(gen)                     ← DRAINS generator (runs all stages)
│  ├── boxes = _extract_boxes(results[0])      ← getattr(result, "boxes", [])
│  └── future.set_result(boxes)               ← RESOLVES the asyncio future
```

Inside `state.model.predict(image)` — this is the PaddleX framework:

```
base_predictor.py:66  BasePredictor.__call__(input)
│  └── yield from self.apply(input)            [base_predictor.py:118]
│
base_predictor.py:118  apply(input)
│  ├── batches = self.batch_sampler(input)     ← ImageBatchSampler (~1ms)
│  │   └── yields one BatchData with instances=[{img: numpy_array}]
│  │
│  └── for batch_data in batches:              (always 1 batch for single image)
│      ├── pred = self.process(batch_data)     ← THE 5 SUB-STAGES (below)
│      └── yield self.result_class(item)       ← LayoutAnalysisResult wrap (~1ms)
│
layout_analysis/predictor.py:70  process(batch_data)
│
│  ┌─ SUB-STAGE 1: PRE-OPS (CPU)  ─────────────────────────────────────┐
│  │  datas = batch_data.instances                                     │
│  │  for pre_op in self.pre_ops[:-1]:                                 │
│  │      datas = pre_op(datas)                                        │
│  │                                                                   │
│  │  pre_ops[0]: ReadImage     → cv2.cvtColor BGR→RGB    ~1ms         │
│  │  pre_ops[1]: Resize        → cv2.resize BICUBIC      ~5-15ms      │
│  │  pre_ops[2]: Normalize     → vectorized norm (OPT)   ~3-8ms       │
│  │  pre_ops[3]: ToCHWImage    → transpose (2,0,1)       ~1ms         │
│  │  Total pre_ops: ~10-25ms                                          │
│  └───────────────────────────────────────────────────────────────────┘
│
│  ┌─ SUB-STAGE 2: TOBATCH (CPU)  ─────────────────────────────────────┐
│  │  batch_inputs = self.pre_ops[-1](datas)  # ToBatch                │
│  │  └── np.stack([img_size, img, scale_factors])         ~2-5ms      │
│  └───────────────────────────────────────────────────────────────────┘
│
│  ┌─ SUB-STAGE 3: TRT RUNNER (GPU + MEMCPY)  ─────────────────────────┐
│  │  batch_preds = self.runner(batch_inputs)                          │
│  │  └── tensorrt_runner.py:183  TensorRTRunner.__call__()            │
│  │                                                                   │
│  │  h2d:  np.ascontiguousarray + cuda.memcpy_htod       3.8ms        │
│  │  exec: context.execute_async_v3 + stream.synchronize 9.0ms (GPU)  │
│  │  d2h:  cuda.memcpy_dtoh (+ .copy() if not skipped)  21.9ms        │
│  │  Total TRT: 34.7ms  (GPU util = 26.1%)                            │
│  └───────────────────────────────────────────────────────────────────┘
│
│  ┌─ SUB-STAGE 4: FORMAT OUTPUT (CPU)  ───────────────────────────────┐
│  │  preds_list = self._format_output(batch_preds)                    │
│  │  └── object_detection/predictor.py:155  _format_output()          │
│  │      └── numpy slicing: pred[0][start:end] per image  ~1ms        │
│  └───────────────────────────────────────────────────────────────────┘
│
│  ┌─ SUB-STAGE 5: POST-OP (CPU — BIGGEST)  ───────────────────────────┐
│  │  boxes = self.post_op(preds_list, datas, ...)                     │
│  │  └── layout_analysis/processors.py:710  LayoutAnalysisProcess     │
│  │                                                                   │
│  │  5a. round coords + threshold filter (score > 0.5)    ~1ms        │
│  │  5b. layout_nms: nms(boxes, iou_same=0.6, iou_diff=0.98)          │
│  │      └── VECTORIZED (OPT): _iou_matrix + greedy numpy ~5-10ms     │
│  │  5c. filter_large_image: Python loop over boxes      ~2-5ms       │
│  │  5d. check_containment: _containment_matrix (OPT)    ~3-8ms       │
│  │  5e. sort boxes by order index (np.lexsort/argsort)  ~1ms         │
│  │  5f. unclip_boxes: numpy arithmetic                   ~1ms         │
│  │  5g. restructured_boxes: Python loop + dict creation ~5-10ms      │
│  │  Total post_op: ~18-35ms (was ~30-50ms before optimization)      │
│  └───────────────────────────────────────────────────────────────────┘
│
│  Total inside predict(): ~65-100ms (was ~100-135ms before optimization)
```

---

### Stage C: Convert to Docling (1.0ms mean)

```
service.py:155  with tracer.stage("convert_to_docling"):
│  └── await loop.run_in_executor(_cpu_pool, ...)
│      │
│      converter/service.py  PaddleXToDoclingConverter.convert()
│      ├── for box in boxes:                          # Python loop
│      │   ├── map PaddleX label → DoclingLabel
│      │   ├── create BoundingBox(x0,y0,x1,y1)
│      │   ├── create ProvenanceItem(page_no=1, bbox=...)
│      │   └── doc.add_text(label, placeholder, prov)  # or add_picture/table
│      └── return DoclingDocument
│
│  Data contract IN:  list[dict] — each dict has cls_id, label, score, coordinate, order
│  Data contract OUT: DoclingDocument with items + provenance
```

**Latency**: ~1ms — fast because boxes list is small (50-200 items) and DoclingDocument construction is lightweight.

---

### Stage D: Export Formats (1.7ms mean)

```
service.py:163  with tracer.stage("export_formats"):
│  ├── export_task = loop.run_in_executor(_cpu_pool, _export_to_formats, ...)
│  ├── confidence_task = loop.run_in_executor(_cpu_pool, _converter.compute_confidence, ...)
│  └── result, confidence = await asyncio.gather(export_task, confidence_task)
│      │
│      ├── _export_to_formats(doc, to_formats)
│      │   ├── doc.export_to_markdown()    ~0.5ms  (if "md" requested)
│      │   ├── doc.export_to_dict()        ~0.3ms  (if "json" requested — DoclingDocument itself)
│      │   ├── doc.export_to_html()        ~0.5ms  (if "html" requested)
│      │   ├── doc.export_to_text()        ~0.2ms  (if "text" requested)
│      │   └── doc.export_to_doctags()     ~0.2ms  (if "doctags" requested)
│      │
│      └── _converter.compute_confidence(boxes)
│          ├── count high/medium/low confidence boxes
│          └── return ConfidenceScores(quality_grade=...)
│
│  Data contract OUT: ExportResult(md_content, json_content, ...) + ConfidenceScores
```

**Latency**: ~1.7ms — runs in parallel (export + confidence via `asyncio.gather`), so it's `max(export, confidence)` ≈ 1.7ms.

---

## 3. Latency Attribution Summary

```
Total server-side latency:  158.1ms mean  (p50=140.7ms  max=404.3ms)

┌────────────────────────────────────┬──────────┬──────┬─────────────────────────────────┐
│ Stage                              │ Mean     │ %    │ Where / Why                     │
├────────────────────────────────────┼──────────┼──────┼─────────────────────────────────┤
│ A. image_load                      │  9.9ms   │ 6.3% │ CPU pool — PIL decode + resize  │
│                                    │          │      │                                 │
│ B. layout_detection_total          │ 155.4ms  │98.4% │ THE BOTTLENECK                  │
│   ├ B.1 semaphore_wait             │  40.3ms  │25.9% │ asyncio Semaphore contention    │
│   ├ B.3 async/future overhead      │  10.6ms  │ 6.8% │ queue.put + executor dispatch   │
│   ├ B.4 PaddleX pre_ops            │ ~15-25ms │~12%  │ ReadImage + Resize + Normalize  │
│   │   ├ ReadImage (cv2.cvtColor)   │  ~1ms    │      │ BGR→RGB color conversion        │
│   │   ├ Resize (cv2.resize BICUBIC)│  ~5-15ms │      │ Image resize to model input     │
│   │   ├ Normalize (vectorized OPT) │  ~3-8ms  │      │ img*alpha+beta (was split/merge)│
│   │   └ ToCHWImage (transpose)     │  ~1ms    │      │ HWC→CHW reorder                 │
│   ├ B.4 ToBatch (np.stack)         │  ~2-5ms  │~2%   │ Stack img+img_size+scale_factors│
│   ├ B.4 TRT Runner                 │  34.7ms  │22.3% │ GPU + memcpy                    │
│   │   ├ h2d (host→device copy)     │   3.8ms  │      │ np.ascontiguousarray + memcpy   │
│   │   ├ exec (GPU forward)         │   9.0ms  │      │ TensorRT execute_async_v3      │
│   │   └ d2h (device→host copy)     │  21.9ms  │      │ memcpy_dtoh (.copy() SKIPPED)   │
│   ├ B.4 _format_output             │  ~1ms    │<1%   │ numpy slicing                   │
│   ├ B.4 post_op                    │ ~18-35ms │~15%  │ NMS + containment + restructure │
│   │   ├ threshold filter           │  ~1ms    │      │ numpy boolean mask              │
│   │   ├ nms (VECTORIZED OPT)       │  ~5-10ms │      │ _iou_matrix + greedy numpy     │
│   │   ├ filter_large_image         │  ~2-5ms  │      │ Python loop over boxes          │
│   │   ├ check_containment (OPT)    │  ~3-8ms  │      │ _containment_matrix (numpy)    │
│   │   ├ sort (np.lexsort)          │  ~1ms    │      │ Reading order sort              │
│   │   ├ unclip_boxes               │  ~1ms    │      │ numpy arithmetic                │
│   │   └ restructured_boxes         │  ~5-10ms │      │ Python loop + dict creation     │
│   └ B.4 result wrap                │  ~1ms    │<1%   │ LayoutAnalysisResult()          │
│                                    │          │      │                                 │
│ C. convert_to_docling              │  1.0ms   │ 0.6% │ CPU pool — DoclingDocument build│
│ D. export_formats                  │  1.7ms   │ 1.1% │ CPU pool — md/json/html export  │
└────────────────────────────────────┴──────────┴──────┴─────────────────────────────────┘
```

---

## 4. Thread Architecture — Who Runs What

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Granian Process (WORKERS=1)                                                │
│                                                                             │
│  ┌─ Thread 1: asyncio Event Loop (main) ──────────────────────────────────┐ │
│  │  • FastAPI route handlers                                               │ │
│  │  • semaphore acquire/release                                            │ │
│  │  • loop.run_in_executor() calls                                         │ │
│  │  • asyncio.gather() for parallel export                                 │ │
│  │  • NEVER blocks — all blocking work offloaded                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Thread 2: Inference Worker (daemon) ──────────────────────────────────┐ │
│  │  • Created in AppState.start_inference()                                │ │
│  │  • Holds the CUDA context (thread-local!)                               │ │
│  │  • Loads model at startup (load_model)                                  │ │
│  │  • Loops: task_queue.get() → _process_single/_process_batch             │ │
│  │  • Runs: predict() → pre_ops → TRT runner → post_op                     │ │
│  │  • Resolves futures via future.set_result()                             │ │
│  │  • ONLY thread that touches GPU                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Thread Pool: _cpu_pool (CPU_POOL_SIZE=8 threads) ─────────────────────┐ │
│  │  • image_load: PIL decode + resize                                      │ │
│  │  • convert_to_docling: DoclingDocument construction                     │ │
│  │  • export_formats: markdown/json/html export                            │ │
│  │  • compute_confidence: quality scoring                                  │ │
│  │  • These are CPU-bound, NOT GPU — safe to parallelize                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ Thread Pool: Default Executor (asyncio fallback) ─────────────────────┐ │
│  │  • future.result() blocking wait (1 thread per in-flight request)       │ │
│  │  • These threads just sleep until the inference thread resolves         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The inference thread is the GPU bottleneck. Only ONE request can be processed at a time on the GPU. The semaphore (PIPELINE_DEPTH=4) allows 4 requests to be *queued*, but they execute *serially* on the inference thread. The benefit of PIPELINE_DEPTH>1 is that the inference thread never has to wait for the next task — it's already in the queue.

---

## 5. Data Contract Map — What Flows Between Stages

```
HTTP Request
  │
  ▼  bytes (PNG/JPEG image data)
  
load_image_from_bytes()
  │
  ▼  np.ndarray (H, W, 3) RGB uint8    ← "image"
  
ReadImage.__call__()
  │
  ▼  List[dict] with keys: img, ori_img, img_size, ori_img_size
  │     img: np.ndarray (H, W, 3) RGB uint8 (same as input for numpy path)
  │     img_size: [w, h]

Resize.__call__()
  │
  ▼  List[dict] with updated: img (resized), img_size (new), scale_factors [w_scale, h_scale]
  │     img: np.ndarray (H', W', 3) RGB uint8 — resized to model target

Normalize.__call__()
  │
  ▼  List[dict] with updated: img → float32, normalized (img * alpha + beta)
  │     img: np.ndarray (H', W', 3) float32 — HWC, normalized

ToCHWImage.__call__()
  │
  ▼  List[dict] with updated: img → transposed to (3, H', W') CHW

ToBatch.__call__()
  │
  ▼  List[np.ndarray]:
  │     [0] img_size:    (1, 2) float32 — [h, w]
  │     [1] img:         (1, 3, H', W') float32 — batched image
  │     [2] scale_factors:(1, 2) float32 — [h_scale, w_scale]

TensorRTRunner.__call__()
  │
  ▼  List[np.ndarray]: raw model outputs
  │     [0] boxes: (N_total, 4) float32 — all detected boxes [x1,y1,x2,y2]
  │     [1] box_nums: (1,) int32 — number of boxes for this image
  │     (or similar — depends on model output spec)

_format_output()
  │
  ▼  List[dict]: [{"boxes": np.ndarray (N, 6)}]  — [cls_id, score, x1, y1, x2, y2]

post_op (LayoutAnalysisProcess.apply())
  │
  ▼  List[dict]: [{"cls_id": int, "label": str, "score": float,
  │                 "coordinate": [x1,y1,x2,y2], "order": int}]
  │     Coordinates in ORIGINAL image space (rescaled back)

_extract_boxes()
  │
  ▼  list[dict] — same as above, extracted from LayoutAnalysisResult.boxes

PaddleXToDoclingConverter.convert()
  │
  ▼  DoclingDocument — with items, provenance, page metadata

_export_to_formats()
  │
  ▼  ExportResult(md_content, json_content, html_content, text_content, doctags_content)

ConvertDocumentResponse (JSON) → HTTP Response
```

---

## 6. Where Latency Is — Ranked by Impact

| Rank | Stage | Mean | % | Root Cause | Optimized? |
|------|-------|------|---|------------|------------|
| **1** | **semaphore_wait** | 40.3ms | 25.9% | PIPELINE_DEPTH contention — waiting for GPU slot | ✅ Depth 3→4 |
| **2** | **TRT d2h memcpy** | 21.9ms | 14.1% | cuda.memcpy_dtoh + redundant .copy() | ✅ .copy() skipped |
| **3** | **post_op (NMS+containment+restructure)** | ~18-35ms | ~15% | CPU-bound box processing | ✅ NMS+containment vectorized |
| **4** | **TRT exec (GPU)** | 9.0ms | 5.8% | Actual GPU compute — can't optimize further | ❌ Hard limit |
| **5** | **async/future overhead** | 10.6ms | 6.8% | queue.put + executor dispatch + future.result | ❌ Architectural |
| **6** | **pre_ops (Resize+Normalize)** | ~10-20ms | ~8% | cv2.resize BICUBIC + normalization | ✅ Normalize vectorized |
| **7** | **image_load** | 9.9ms | 6.3% | PIL decode + resize | ❌ I/O bound |
| **8** | **TRT h2d memcpy** | 3.8ms | 2.5% | np.ascontiguousarray + cuda.memcpy_htod | ❌ Unavoidable |
| **9** | **ToBatch** | ~2-5ms | ~2% | np.stack of 3 arrays | ❌ Minimal |
| **10** | **convert_to_docling** | 1.0ms | 0.6% | DoclingDocument construction | ❌ Minimal |
| **11** | **export_formats** | 1.7ms | 1.1% | Markdown/JSON export | ❌ Minimal |

### Latency visualization (mean, 158.1ms total):

```
                          ◄──────────────── 158.1ms total ────────────────►
                          ┬──┬────────────────────────────────────────────┬─┬
  semaphore_wait (40.3ms) │██│                                              │ │
                          └──┘                                              │ │
                              ┬─────────┬──────────────────┬──┬─────────┬──┘ │
  PaddleX pre_ops (~15ms)     │█████████│                  │  │         │    │
                              └─────────┘                  │  │         │    │
                                          ┬───┬────────────┤  │         │    │
  TRT h2d (3.8ms)                         │███│            │  │         │    │
                                          └───┘            │  │         │    │
                                                    ┬──┐  │  │         │    │
  TRT exec (9.0ms)                                  │██│  │  │         │    │
                                                    └──┘  │  │         │    │
                                                          ┬───────────┤  │    │
  TRT d2h (21.9ms)                                         │███████████│  │    │
                                                          └───────────┘  │    │
                                                                         ┬──┐  │
  post_op (~25ms)                                                        │██│  │
                                                                         └──┘  │
                                                                               ┬──┐
  async overhead (10.6ms)                                                      │██│
                                                                               └──┘
                                                                                  ┬┬┐
  image_load + convert + export                                                   │││ (12.6ms)
                                                                                  └┴┘
```

---

## 7. Key Architectural Decisions & Their Latency Implications

### 7.1 Single Inference Thread (CUDA Context is Thread-Local)

```
AppState.__init__()  →  state._infer_thread = Thread(target=_inference_worker)
```

- **Why**: pycuda/TensorRT CUDA contexts are thread-local. The thread that creates the context must be the one that executes GPU calls.
- **Impact**: GPU processes ONE request at a time. No GPU-level parallelism.
- **Mitigation**: PIPELINE_DEPTH>1 keeps the queue full so the inference thread never idles waiting for the next task.

### 7.2 Semaphore Limits In-Flight Requests

```
state.semaphore = asyncio.Semaphore(PIPELINE_DEPTH)  # =4
```

- **Why**: Without the semaphore, every request would queue a task, and memory would grow unbounded (each task holds a full image array).
- **Impact**: 5th+ concurrent request waits 40.3ms mean for a slot.
- **Trade-off**: Higher PIPELINE_DEPTH → less wait but more memory + more GPU contention.

### 7.3 CPU Pool Offloading

```
_cpu_pool = ThreadPoolExecutor(max_workers=CPU_POOL_SIZE)  # =8
```

- **Why**: PIL decode, DoclingDocument construction, and format export are CPU-bound. Running them on the event loop would block ALL requests.
- **Impact**: These stages run concurrently across requests without blocking.
- **Key**: CPU_POOL_SIZE (8) ≥ PIPELINE_DEPTH (4) ensures CPU stages never block GPU submission.

### 7.4 Batch Collection (BATCH_SIZE=2)

```
_inference_worker():
    if BATCH_SIZE <= 1:
        _process_single(task)
    else:
        batch = [task]
        # collect up to BATCH_SIZE with BATCH_TIMEOUT_MS timeout
        _process_batch(batch)
```

- **Why**: Amortizes kernel launch overhead — one `predict()` call for 2 images instead of 2 separate calls.
- **Impact**: Under low load, adds BATCH_TIMEOUT_MS (5ms) latency waiting for a second request. Under high load, improves throughput ~1.5-2x.
- **Trade-off**: BATCH_SIZE=2 with 5ms timeout = max 5ms added latency for 2x throughput under load.

### 7.5 Future-Based Async Bridge

```
# inference.py:400
boxes = await loop.run_in_executor(None, future.result)
```

- **Why**: The inference thread can't call `asyncio` APIs (it's not the event loop thread). Futures provide a thread-safe bridge.
- **How**: Inference thread calls `future.set_result(boxes)`. The executor thread polling `future.result()` unblocks. The event loop's `await` resumes.
- **Overhead**: ~10.6ms — queue.put + executor thread dispatch + future polling. This is the "async tax" for bridging sync GPU code with async web server.

---

## 8. Optimization Impact Map (Changes Applied This Session)

```
                    BEFORE                          AFTER (expected)
                    ──────                          ──────────────
semaphore_wait      40.3ms (PIPELINE_DEPTH=3)  →   ~30ms (PIPELINE_DEPTH=4)
TRT d2h             21.9ms (.copy() always)    →   ~11ms (.copy() skipped)
post_op NMS         ~10-20ms (Python O(n²))    →   ~5-10ms (vectorized numpy)
post_op containment ~5-15ms (Python O(n²))     →   ~3-8ms (vectorized numpy)
Normalize           ~5-15ms (split/loop/merge) →   ~3-8ms (single vectorized op)

Total estimated savings: ~30-55ms (~19-35% improvement)
Expected new total: ~103-128ms (down from 158ms)
```

### What each optimization targets:

```
┌─────────────────────────────┬──────────┬───────────────────────────────────────┐
│ Optimization                │ Saves    │ Mechanism                             │
├─────────────────────────────┼──────────┼───────────────────────────────────────┤
│ PIPELINE_DEPTH 3→4          │ ~10ms    │ Less semaphore contention under load  │
│ CPU_POOL_SIZE 4→8           │ ~0ms*    │ Prevents CPU stages blocking GPU      │
│ BATCH_SIZE 1→2              │ ~0ms**   │ Throughput gain under load (2x)       │
│ HPS_TRT_SKIP_D2H_COPY=1     │ ~10ms    │ Eliminates redundant numpy array copy │
│ NMS vectorized              │ ~5-10ms  │ numpy _iou_matrix replaces Python O(n²)│
│ check_containment vectorized│ ~3-8ms   │ numpy _containment_matrix replaces O(n²)│
│ Normalize vectorized        │ ~3-5ms   │ Single op replaces split/loop/merge   │
└─────────────────────────────┴──────────┴───────────────────────────────────────┘

*  CPU_POOL_SIZE doesn't reduce single-request latency, but prevents queuing
   under concurrent load.
** BATCH_SIZE doesn't reduce single-request latency (adds 5ms wait), but
   improves throughput 1.5-2x under concurrent load.
```

---

## 9. File Reference — Complete Call Chain

| # | File | Line | Function | Stage |
|---|------|------|----------|-------|
| 1 | `docling_api/routes.py` | 129 | `convert_file()` | HTTP entry |
| 2 | `docling_api/service.py` | 90 | `convert_image()` | Pipeline orchestrator |
| 3 | `_core/image.py` | 18 | `load_image_from_bytes()` | Stage A: image_load |
| 4 | `_core/inference.py` | 318 | `run_layout_detection()` | Stage B: dispatch |
| 5 | `_core/inference.py` | 375 | `_run_direct()` | Stage B: semaphore + queue |
| 6 | `_core/inference.py` | 113 | `AppState.__init__()` | Semaphore/thread setup |
| 7 | `_core/inference.py` | 205 | `_inference_worker()` | Inference thread loop |
| 8 | `_core/inference.py` | 255 | `_process_single()` | Task processing |
| 9 | `predictors/base_predictor.py` | 66 | `BasePredictor.__call__()` | predict() entry |
| 10 | `predictors/base_predictor.py` | 118 | `apply()` | batch_sampler + process |
| 11 | `layout_analysis/predictor.py` | 70 | `process()` | 5 sub-stages |
| 12 | `object_detection/processors.py` | 35 | `ReadImage.__call__()` | Sub-stage 1a |
| 13 | `object_detection/processors.py` | 102 | `Resize.__call__()` | Sub-stage 1b |
| 14 | `object_detection/processors.py` | 133 | `Normalize.__call__()` | Sub-stage 1c |
| 15 | `object_detection/processors.py` | 162 | `ToCHWImage.__call__()` | Sub-stage 1d |
| 16 | `object_detection/processors.py` | 200 | `ToBatch.__call__()` | Sub-stage 2 |
| 17 | `runners/tensorrt_runner.py` | 183 | `TensorRTRunner.__call__()` | Sub-stage 3: GPU |
| 18 | `object_detection/predictor.py` | 155 | `_format_output()` | Sub-stage 4 |
| 19 | `layout_analysis/processors.py` | 710 | `LayoutAnalysisProcess.apply()` | Sub-stage 5: post_op |
| 20 | `object_detection/processors.py` | 635 | `nms()` | Sub-stage 5b: NMS |
| 21 | `object_detection/processors.py` | 663 | `check_containment()` | Sub-stage 5d |
| 22 | `layout_analysis/processors.py` | 383 | `restructured_boxes()` | Sub-stage 5g |
| 23 | `docling_api/converter/service.py` | — | `PaddleXToDoclingConverter.convert()` | Stage C |
| 24 | `docling_api/service.py` | 200+ | `_export_to_formats()` | Stage D |
| 25 | `_core/latency.py` | 60 | `LatencyTracer` | Timing instrumentation |
| 26 | `_core/config.py` | 88+ | All config constants | Settings |
| 27 | `scripts/run_api.sh` | 100+ | Granian launch | Server startup |

---

## 10. Remaining Latency — What Can't Be Optimized (Without Architecture Change)

After all optimizations, the irreducible latency floor is:

```
┌──────────────────────────────┬──────────┬───────────────────────────────────────┐
│ Irreducible Stage            │ Time     │ Why it can't be reduced               │
├──────────────────────────────┼──────────┼───────────────────────────────────────┤
│ GPU exec (TensorRT forward)  │ 9.0ms    │ Physics — GPU compute time            │
│ TRT h2d memcpy               │ 3.8ms    │ PCIe bandwidth (host→device)          │
│ TRT d2h memcpy (without copy)│ ~11ms    │ PCIe bandwidth (device→host)          │
│ cv2.resize BICUBIC           │ ~5-15ms  │ CPU image resize — could use GPU      │
│ async/future overhead        │ ~10ms    │ Thread synchronization cost           │
│ image_load (PIL decode)      │ ~5ms     │ I/O + decode                          │
│ post_op (restructured_boxes) │ ~5-10ms  │ Python dict creation per box          │
│ ToBatch (np.stack)           │ ~2-5ms   │ Memory allocation + copy              │
│ convert + export             │ ~3ms     │ DoclingDocument + format export       │
├──────────────────────────────┼──────────┼───────────────────────────────────────┤
│ TOTAL irreducible            │ ~54-73ms │ Without GPU-side NMS or GPU resize    │
└──────────────────────────────┴──────────┴───────────────────────────────────────┘
```

**To go below ~60ms would require:**
1. **GPU-side NMS** — TensorRT plugin or CUDA kernel (eliminates ~10ms post_op)
2. **GPU-side resize** — CUDA resize kernel (eliminates ~10ms cv2.resize)
3. **Triton backend** — continuous batching + non-blocking queue (eliminates ~10ms async overhead)
4. **CUDA streams for overlap** — overlap h2d of next batch with exec of current (eliminates ~4ms)
5. **Pinned host memory** — faster h2d/d2h memcpy (could halve memcpy time)
