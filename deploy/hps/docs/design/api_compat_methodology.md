# API Compatibility Layer — Architecture & Implementation Methodology

> **Location**: `PaddleX/deploy/hps/api_compat/`
> **Purpose**: Expose PaddleX inference through third-party API protocols (Docling, Unstructured IO, etc.)

---

## 1. Big Picture: What This Codebase Does

The `api_compat` package is a **thin FastAPI serving layer** that takes PaddleX's
layout detection model (PP-DocLayoutV3 TensorRT) and exposes it through API
protocols that third-party tools already know how to talk to.

```
┌─────────────────────────────────────────────────────┐
│                   Client Request                     │
│  (Docling protocol: POST /v1/convert/file)          │
│  (multipart: file=<binary>, to_formats=md)          │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   docling_api/routes.py  │  ← Protocol-specific HTTP layer
          │   (parse request shape)  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  docling_api/service.py  │  ← Pipeline orchestration
          │  (load → detect → build) │
          └────────────┬────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
┌────────▼───┐ ┌───────▼──────┐ ┌───▼──────────────────┐
│ _core/     │ │ _core/       │ │ docling_api/         │
│ image.py   │ │ inference.py │ │ converter/service.py │
│ (decode)   │ │ (GPU detect) │ │ (build DoclingDoc)   │
└────────────┘ └──────────────┘ └──────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  docling_api/service.py  │  ← Export to N formats
          │  (_export_to_formats)    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Client Response        │
          │  (Docling protocol shape)│
          └─────────────────────────┘
```

**Key insight**: `_core/` is generic infrastructure shared across ALL API
compatibility layers. `docling_api/` is the protocol-specific slice. Future
layers (e.g., `unstructured_api/`) reuse `_core/` and add their own slice.

---

## 2. Package Structure (18 Files)

```
api_compat/
├── __init__.py                      # Package docstring
├── _core/                           # ── GENERIC (shared by all layers) ──
│   ├── __init__.py
│   ├── config.py                    # Env vars: single source of truth
│   ├── engine.py                    # TRT engine caching (avoid 3-min rebuild)
│   ├── inference.py                 # Model loading + dedicated inference thread
│   ├── image.py                     # Image decode from bytes / fetch from URL
│   └── health/
│       ├── __init__.py
│       ├── schema.py                # HealthCheckResponse, ReadinessResponse
│       └── routes.py               # /health, /health-check, /ready
└── docling_api/                     # ── DOCLING-SPECIFIC SLICE ──
    ├── __init__.py
    ├── app.py                       # FastAPI factory + lifespan + CORS
    ├── routes.py                    # /v1/convert/source, /v1/convert/file
    ├── schema.py                    # Docling request/response models
    ├── service.py                   # Pipeline orchestration + export
    └── converter/
        ├── __init__.py
        ├── labels.py                # PaddleX→Docling label mapping (25 cats)
        ├── schema.py                # ConfidenceScores, QualityGrade
        └── service.py               # PaddleXToDoclingConverter class
```

---

## 3. Function-Level Logic Map

### 3.1 `_core/config.py` — Configuration (zero functions, all constants)

Reads environment variables **once** at import time. Every other module imports
from here — no `os.environ` calls anywhere else.

| Constant | Env Var | Default | Purpose |
|----------|---------|---------|---------|
| `MODEL_NAME` | `HPS_API_MODEL` | `PP-DocLayoutV3` | PaddleX model name |
| `MODEL_PRECISION` | `HPS_API_PRECISION` | `fp8` | TRT precision |
| `MODEL_DEVICE_ID` | `HPS_API_DEVICE_ID` | `0` | GPU device |
| `ENGINE_DIR` | `HPS_API_ENGINE_DIR` | `/tmp/paddlex-engines` | Writable cache |
| `BUILTIN_ENGINE_DIRS` | (derived) | 3 paths + PADDLEX_HOME | Search order |
| `MAX_IMAGE_DIM` | `HPS_API_MAX_IMAGE_DIM` | `4096` | Resize limit |
| `DEFAULT_TIMEOUT` | `HPS_API_TIMEOUT` | `30` | HTTP fetch timeout |
| `STARTUP_TIMEOUT` | `HPS_API_STARTUP_TIMEOUT` | `300` | Model load deadline |
| `PIPELINE_DEPTH` | `HPS_API_PIPELINE_DEPTH` | `4` | Max concurrent in-flight inference tasks (1 = serial) |
| `CPU_POOL_SIZE` | `HPS_API_CPU_POOL_SIZE` | `8` | Thread pool size for CPU-bound stages (image decode, export) |
| `INFERENCE_BACKEND` | `HPS_API_BACKEND` | `triton` | Inference backend: `triton` (continuous batching via Triton) or `direct` (in-process micro-batching) |
| `BATCH_SIZE` | `HPS_API_BATCH_SIZE` | `2` | Direct-backend only: GPU micro-batch size (1 = no batching, 2 = balanced latency/throughput) |
| `BATCH_TIMEOUT_MS` | `HPS_API_BATCH_TIMEOUT_MS` | `5` | Direct-backend only: max wait (ms) for batch to fill before flushing partial batch |
| `TRITON_URL` | `HPS_TRITON_URL` | `localhost:8001` | Triton Inference Server gRPC URL |
| `TRITON_MODEL_NAME` | `HPS_TRITON_MODEL_NAME` | `doclayout-v3` | Triton model name (in model repository) |
| `TRITON_REQUEST_TIMEOUT` | `HPS_TRITON_TIMEOUT` | `30` | gRPC request timeout (seconds) |

**TRT runner env vars** (read by `paddlex/inference/models/runners/tensorrt_runner.py`):

| Env Var | Default | Purpose |
|---------|---------|---------|
| `HPS_TRT_SKIP_D2H_COPY` | `0` | Skip `.copy()` on D2H outputs (return views). Safe for single-threaded direct backend. |
| `HPS_TRT_SKIP_D2H_OUTPUTS` | (empty) | Skip D2H copy for specified outputs. `auto` = skip any output >10MB. Comma-separated names for explicit list (e.g., `fetch_name_2`). **#13 optimization — skips 96MB masks D2H.** |
| `HPS_TRT_PINNED` | `1` | Use pinned/page-locked host output buffers for async DMA. Set `0` for fallback/debugging. |
| `HPS_LATENCY_LOG` | `0` | Enable per-stage JSON timing traces (`trt_runner`, `det_process` events in stdout). |

Logging is configured via `setup_logging()` (called from `lifespan()`) using
`HPS_API_LOG_LEVEL` (default `INFO`).

### 3.2 `_core/engine.py` — TRT Engine Caching

**Function**: `prepare_engine() -> str`

```
prepare_engine()
  ├── Check /tmp/paddlex-engines/inference_fp8.trt exists?
  │     YES → return it (instant load)
  │     NO  → search BUILTIN_ENGINE_DIRS for pre-built engine
  │           ├── Found? → copy to /tmp, return /tmp path
  │           └── Not found? → return /tmp path anyway
  │               (PaddleX will auto-build from ONNX, ~3 min)
  └── Return engine path string
```

### 3.3 `_core/inference.py` — Model + Inference Thread

**Class**: `AppState` (singleton `state`)
**Functions**: `load_model()`, `_inference_worker()`, `run_layout_detection()`

```
AppState (singleton)
  ├── model: Any                    # PaddleX model instance
  ├── semaphore: asyncio.Semaphore(PIPELINE_DEPTH)  # Configurable depth (default 4)
  ├── _ready: bool                  # Model loaded?
  ├── _load_error: Exception?       # Capture crash from thread
  ├── _task_queue: queue.Queue      # Async → thread communication
  ├── _pending: dict[str, Future]   # task_id → Future (thread → async)
  ├── _pending_lock: threading.Lock  # Protects _pending dict
  ├── _load_done: threading.Event   # Startup sync
  │
  ├── start_inference()             # Launch _inference_worker thread
  ├── wait_ready(timeout)           # Block until loaded or timeout
  └── shutdown()                    # Send None sentinel → thread exits

load_model()                        # Called INSIDE inference thread
  ├── prepare_engine()              # Get TRT engine path
  ├── create_model(MODEL_NAME, engine="tensorrt", engine_config={...})
  └── state._ready = True

_inference_worker()                 # Thread target (daemon)
  ├── try: load_model()
  │     except: capture _load_error, set _load_done, return
  ├── set _load_done
  └── loop forever:
        ├── task_id, image, future = _task_queue.get()   # Block waiting
        ├── if task_id is None: break                      # Shutdown sentinel
        ├── try:
        │     ├── results = list(model.predict(image))
        │     ├── boxes = extract from results[0]
        │     └── future.set_result(boxes)   # Resolve matching Future
        ├── except: future.set_exception(e)
        └── finally: _pending.pop(task_id)   # Clean up

run_layout_detection(image)         # ASYNC — called from route handlers
  ├── async with semaphore:         # Limit in-flight to PIPELINE_DEPTH (4)
  │     ├── task_id = uuid4().hex
  │     ├── future = Future()
  │     ├── _pending[task_id] = future  (under lock)
  │     ├── _task_queue.put((task_id, image, future))
  │     └── await asyncio.wait_for(
  │           loop.run_in_executor(None, future.result),
  │           timeout=120)
  └── return boxes (or raise error/timeout)
```

**Critical design**: CUDA contexts are thread-local (pycuda). The model MUST be
loaded and used in the SAME thread. The task queue bridges async FastAPI →
sync CUDA thread without context leaks.

**Pipeline depth** (`HPS_API_PIPELINE_DEPTH`, default 4): Controls how many
requests can be in-flight simultaneously. The inference thread still processes
one `predict()` at a time (GPU is serial), but the asyncio layer can queue
multiple tasks. Results are matched by `task_id` via per-task Futures — no
head-of-line blocking. With `DEPTH=1`, this degenerates to the original
fully-serial behavior.

### 3.4 `_core/image.py` — Image Loading

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `load_image_from_bytes(data)` | `bytes` | `np.ndarray (H,W,3) RGB uint8` | Resizes if > MAX_IMAGE_DIM |
| `fetch_image_from_url(url, headers)` | `str, dict` | `bytes` | httpx async, follows redirects |

### 3.5 `_core/health/` — Health Endpoints

**Routes**: `/health`, `/health-check` (liveness), `/ready` (readiness)

```
/health        → {"status": "ok"}           # Always (process alive)
/health-check  → {"status": "ok"}           # Alias
/ready         → {"status": "ok"} or 503    # Only if state.ready == True
```

### 3.6 `docling_api/app.py` — App Factory

**Functions**: `create_app()`, `lifespan()`

```
create_app()
  ├── FastAPI(title="PaddleX HPS Docling API", lifespan=lifespan)
  ├── add CORSMiddleware (allow all)
  ├── include_router(health_router)      # from .._core.health
  ├── include_router(convert_router)     # from .routes
  └── return app

lifespan(app)  # async context manager
  ├── STARTUP:
  │     ├── state.start_inference()          # Launch thread
  │     ├── state.wait_ready(STARTUP_TIMEOUT) # Block up to 300s
  │     └── if not ready: shutdown(), raise RuntimeError
  ├── yield  # App running
  └── SHUTDOWN:
        └── state.shutdown()                 # Stop thread
```

### 3.7 `docling_api/routes.py` — HTTP Handlers

**Routes**: `POST /v1/convert/source`, `POST /v1/convert/file`,
`POST /v1/convert/source/async` (501), `POST /v1/convert/file/async` (501),
`POST /v1/convert/source/batch` (501)

```
POST /v1/convert/source (JSON body)
  ├── request.sources[0]  # Only first source processed
  ├── if kind == "http":
  │     ├── filename = URL path basename
  │     ├── image_data = await fetch_image_from_url(url, headers)
  │     └── on error: return error_response(ERR_FETCH=SOURCE_UNAVAILABLE, ...)
  ├── if kind == "file":
  │     ├── image_data = base64decode(source.base64_string)
  │     ├── filename = source.filename
  │     └── on error: return error_response(ERR_DECODE=USER_INPUT, ...)
  ├── else (s3/azure/gcs/googledrive): 501 Not Implemented
  └── return await convert_image(image_data, filename, to_formats)

POST /v1/convert/file (multipart upload)
  ├── Validate filename present (else 400)
  ├── Parse to_formats FORM FIELDS via _parse_to_formats():
  │     ├── None/empty → [MARKDOWN] (default)
  │     ├── ['md', 'json'] → [MARKDOWN, JSON] (upstream repeated fields)
  │     ├── ['md,json'] → [MARKDOWN, JSON] (defensive CSV-in-element)
  │     └── Invalid format → 400 with valid options listed
  ├── image_data = await file.read()
  └── return await convert_image(image_data, file.filename, formats)

_parse_to_formats(to_formats: list[str] | None) → list[OutputFormat]
  ├── if not to_formats → [OutputFormat.MARKDOWN]
  ├── Flatten: each element may contain comma-separated values
  │     raw_values = [part for item in to_formats for part in item.split(",")]
  ├── if not raw_values → [OutputFormat.MARKDOWN]
  ├── return [OutputFormat(v) for v in raw_values]
  └── on ValueError → HTTPException(400)
```

**Protocol note**: ``to_formats`` is a **list-typed multipart form field**
(``list[str] | None = Form(default=None)``), not a query parameter and not a
JSON-serialized string. This matches upstream docling-serve's
``FormDepends(ConvertDocumentsOptions)`` pattern: ``is_json_field()`` only
handles ``dict`` origin (not ``list``), so ``list[OutputFormat]`` is kept as a
list type. Clients like httpx send **repeated form fields**
(``to_formats=md&to_formats=json``), which FastAPI collects into ``list[str]``.

### 3.8 `docling_api/service.py` — Pipeline Orchestration

**Functions**: `error_response()`, `convert_image()`, `_export_to_formats()`

```
convert_image(image_data, filename, to_formats)   # ASYNC
  │
  ├── t0 = perf_counter()
  │
  ├── Step 1: Load image (CPU — offloaded to thread pool)
  │     image = await loop.run_in_executor(_cpu_pool, load_image_from_bytes, ...)
  │     └── on error: return error_response(ERR_IMAGE_LOAD=INFERENCE_FAILURE, ...)
  │
  ├── Step 2: Layout detection (GPU — via inference thread + semaphore)
  │     boxes = await run_layout_detection(image)
  │     └── on error: return error_response(ERR_LAYOUT_DETECTION=INFERENCE_FAILURE, ...)
  │
  ├── layout_time = perf_counter() - t0
  │
  ├── Step 3: Build DoclingDocument (CPU — offloaded to thread pool)
  │     doc = await loop.run_in_executor(_cpu_pool, _converter.convert, ...)
  │
  ├── Step 4+5: Export & confidence (CPU — parallel, offloaded to thread pool)
  │     result, confidence = await asyncio.gather(
  │       loop.run_in_executor(_cpu_pool, _export_to_formats, ...),
  │       loop.run_in_executor(_cpu_pool, _converter.compute_confidence, ...),
  │     )
  │     └── Each format in try/except → errors collected, not fatal
  │
  ├── status = SUCCESS (no errors) or PARTIAL_SUCCESS (some errors)
  │
  └── return ConvertDocumentResponse(
          document=ExportDocumentResponse(filename, md, json, html, ...),
          status, errors, processing_time, timings, confidence
      )

_export_to_formats(doc, to_formats)
  ├── Pre-compute markdown once if MARKDOWN requested (reuse as cached_md)
  └── For each fmt in to_formats (via _EXPORTERS dispatch table):
        ├── MARKDOWN → return cached_md (pre-computed above)
        ├── JSON     → return doc itself (DoclingDocument)
        ├── TEXT     → doc.export_to_text() via _call_optional_export
        ├── HTML     → doc.export_to_html() via _call_optional_export
        ├── DOCTAGS  → doc.export_to_doctags() via _call_optional_export
        ├── DOCLANG  → doc.export_to_doclang() via _call_optional_export
        └── on Exception → append ErrorItem(category=INTERNAL, error_message=...)

  _call_optional_export(doc, method_name)
        ├── getattr(doc, method_name, None)
        └── return method() if method exists, else None

  _FIELD_MAP maps each OutputFormat → ExportResult attribute for setattr
```

### 3.9 `docling_api/converter/` — PaddleX → DoclingDocument

**Class**: `PaddleXToDoclingConverter`

```
PaddleXToDoclingConverter.convert(boxes, image, filename, page_no)
  ├── h, w = image.shape[:2]
  ├── doc = DoclingDocument(name=filename)
  ├── doc.add_page(page_no, size=Size(w, h))
  ├── sorted_boxes = _sort_by_reading_order(boxes)
  └── for each box:
        ├── label_str → PADDLEX_TO_DOCLING[label_str] (default: TEXT)
        ├── bbox = BoundingBox(xmin, ymin, xmax, ymax, TOPLEFT)
        ├── prov = ProvenanceItem(page_no, bbox, charspan=(0,0))
        └── dispatch by label type:
              ├── _SPECIAL_LABELS (PICTURE, TABLE, FORMULA)
              │     → _add_special_item():
              │           PICTURE  → doc.add_picture(prov)
              │           TABLE    → doc.add_table(data=TableData(0,0), prov)
              │           FORMULA  → doc.add_text(label=FORMULA, text="", prov)
              └── else → _add_text_item():
                    TITLE          → doc.add_title(text="[label]", prov)
                    SECTION_HEADER → doc.add_heading(text="[label]", level=1, prov)
                    default        → doc.add_text(label, text="[label]", prov)

PaddleXToDoclingConverter.compute_confidence(boxes)
  ├── if empty boxes → return (0.0, 0.0, LOW)
  ├── scores = [box.score for box in boxes]
  ├── mean = sum(scores) / len(scores)
  ├── grade = HIGH (≥0.85) | MEDIUM (≥0.6) | LOW (<0.6)
  └── return ConfidenceScores(layout_score=mean, mean_score=mean, mean_grade=grade)

_sort_by_reading_order(boxes)  # static method
  └── Sort by tuple key: (0, order) if order present, else (1, y_coord)
      → Boxes with an order field sorted first (group 0), then
        boxes without order sorted by y-coordinate (group 1, top-to-bottom)
```

### 3.10 `docling_api/schema.py` — Protocol Models (Pure Re-Export Layer)

**Design**: ``schema.py`` contains **zero custom models**. Every symbol is a
re-export from the installed upstream ``docling`` package (docling-slim
2.114.0 + docling-core 2.87.1). This guarantees 1:1 compatibility by
construction — if upstream adds/renames a field, our API breaks loudly at
import time rather than silently at runtime.

Re-exported symbols (50+):

| Model / Enum | Source | Role |
|--------------|--------|------|
| `OutputFormat` enum | `docling.datamodel.base_models` | 11 members: md, json, yaml, html, html_split_page, text, doctags, vtt, doclang, dclx, chunks |
| `QualityGrade` enum | `docling.datamodel.base_models` | EXCELLENT, GOOD, FAIR, POOR, LOW (we use 3 by design) |
| `ConvertDocumentsOptions` | `docling.datamodel.pipeline_options` | `to_formats`, `do_ocr`, `table_mode`, etc. (extra="allow") |
| `ConvertSourcesRequest` | `docling.datamodel.service.requests` | Top-level JSON body: `sources[]` + `options` + `target` |
| `ConvertDocumentResponse` | `docling.datamodel.service.responses` | Full response: document + status + errors + timings + confidence |
| `ExportDocumentResponse` | `docling.datamodel.service.responses` | The document content: filename + md/json/html/text/doctags/doclang |
| `ErrorItem` | `docling.datamodel.service.responses` | Error details: component_type, module_name, error_message, category |
| `ConfidenceScores` | `docling.datamodel.base_models` | layout_score, mean_score, mean_grade, low_score, low_grade |
| `FailureCategory` enum | `docling.datamodel.base_models` | INTERNAL, USER_INPUT, SOURCE_UNAVAILABLE, INFERENCE_FAILURE, etc. |
| `ConversionStatus` enum | `docling.datamodel.base_models` | SUCCESS, PARTIAL_SUCCESS, FAILURE, PENDING |
| `ProfilingItem` / `ProfilingScope` | `docling.datamodel.base_models` | Per-stage timing data |
| `DoclingComponentType` enum | `docling.datamodel.base_models` | MODEL, PIPELINE, etc. |

All re-exports verified against installed upstream in L3 audit (see §10).

---

## 4. End-to-End Request Flow (Concrete Example)

**Request**: `POST /v1/convert/file` with a multipart form upload (file + `to_formats` form field)

```
1.  Granian receives HTTP request
2.  FastAPI routes to convert_file() in routes.py
3.  convert_file():
      - file.filename → "invoice.png"
      - to_formats=Form(['md']) → _parse_to_formats(['md']) → [OutputFormat.MARKDOWN]
      - image_data = await file.read()  → bytes
4.  convert_image(image_data, "invoice.png", [MARKDOWN])  in service.py
5.    load_image_from_bytes(image_data)  in _core/image.py
        - PIL.open → convert("RGB") → np.array
        - If >4096px: resize with LANCZOS
        - Returns: np.ndarray (H, W, 3)
6.    run_layout_detection(image)  in _core/inference.py
        - async with semaphore (serial GPU access)
        - task_id = uuid4()
        - _task_queue.put((task_id, image))
        - await _result_queue.get()  → (task_id, boxes, None)
        - boxes = [{"label":"text","score":0.95,"coordinate":[x,y,x,y],...}, ...]
7.    _converter.convert(boxes, image, "invoice.png", page_no=1)
        - Creates DoclingDocument
        - Maps each box label via PADDLEX_TO_DOCLING
        - Adds picture/table/text/title items with provenance
        - Returns: DoclingDocument
8.    _export_to_formats(doc, [MARKDOWN])
        - doc.export_to_markdown() → "# [doc_title]\n\n[text]\n..."
        - Returns: ExportResult(md_content="# [doc_title]...")
9.    _converter.compute_confidence(boxes)
        - mean score = 0.95 → grade=HIGH
10.   Return ConvertDocumentResponse(
        document=ExportDocumentResponse(filename="invoice.png", md_content="..."),
        status=SUCCESS,
        processing_time=0.12,
        timings={"layout": ProfilingItem(time=0.08)},
        confidence=ConfidenceScores(layout_score=0.95, mean_grade=HIGH)
      )
11. FastAPI serializes to JSON → HTTP 200 → Client
```

---

## 5. Vertical Slice Architecture — How to Build a New Layer

This is the **methodology** for adding a new API compatibility layer (e.g.,
`unstructured_api/`). Follow these steps **in order** — each step is a
vertical slice that compiles and can be smoke-tested independently.

### Step 0: Understand the Protocol

Before writing any code, read the target API's documentation:
- What endpoints does it expose? (paths, methods)
- What is the request schema? (JSON body? multipart? query params?)
- What is the response schema? (nested structure? specific field names?)
- What output formats does it support?
- What error shapes does it return?

### Step 1: Create the Package Skeleton

```
api_compat/<new_api>/
├── __init__.py          # Package docstring
├── app.py               # Will hold create_app() + lifespan
├── routes.py            # Will hold protocol-specific routes
├── schema.py            # Will hold protocol-specific models
└── service.py           # Will hold pipeline orchestration
```

**Rule**: Every layer has the same 5-file shape. This makes navigation predictable.

### Step 2: Define the Schema (`schema.py`)

Define Pydantic models that mirror the target protocol's request/response shapes.

```python
# Example: unstructured_api/schema.py
class Element(BaseModel):
    type: str           # "Text", "Title", "Image", etc.
    text: str = ""
    metadata: dict = {}

class PartitionResponse(BaseModel):
    elements: list[Element]
```

**Smoke test**: Import the module, instantiate models, verify validation works.

### Step 3: Define the Converter (`converter/`)

If the target protocol needs a different document representation than
DoclingDocument, create a `converter/` sub-package:

```
<new_api>/converter/
├── __init__.py
├── labels.py            # PaddleX → target protocol label mapping
├── schema.py            # Converter-specific models (if any)
└── service.py           # Converter class
```

The converter takes PaddleX boxes + image and produces the target format.

**Smoke test**: Call `converter.convert(boxes, image)` with mock data, verify output.

### Step 4: Build the Service (`service.py`)

Wire the pipeline: image loading → inference → conversion → export.

```python
async def convert_image(image_data, filename, to_formats):
    image = load_image_from_bytes(image_data)      # _core
    boxes = await run_layout_detection(image)       # _core
    doc = _converter.convert(boxes, image, filename)
    result = _export_to_formats(doc, to_formats)
    return build_response(result)
```

**Key**: The service imports from `.._core.*` for infrastructure and from
`.converter.*` for protocol-specific conversion. It does NOT import from
`.._core.inference` directly for model creation — only `run_layout_detection`.

**Smoke test**: Call `convert_image()` with real image bytes, verify response.

### Step 5: Build the Routes (`routes.py`)

Create `APIRouter` with the protocol's endpoints. Routes are thin — they
parse the request, call `service.convert_image()`, and return the response.

```python
router = APIRouter()

@router.post("/v1/partition", response_model=PartitionResponse)
async def partition(request: PartitionRequest):
    # Parse request → get image_data + filename
    return await convert_image(image_data, filename, request.to_formats)
```

**Smoke test**: Mount router on a test FastAPI app, send requests via httpx.

### Step 6: Build the App (`app.py`)

Factory function that creates FastAPI, adds CORS, mounts health router from
`_core` and the protocol-specific router from `.routes`.

```python
def create_app() -> FastAPI:
    app = FastAPI(title="PaddleX HPS <Protocol> API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, ...)
    app.include_router(health_router)     # from .._core.health
    app.include_router(partition_router)  # from .routes
    return app

app = create_app()  # Module-level for Granian
```

**Smoke test**: Start Granian with `<new_api>.app:app`, hit `/health` + endpoints.

### Step 7: Deploy

```bash
podman exec paddlex-hps mkdir -p /opt/api_compat/<new_api>
podman cp ~/ADEO/OCR/PaddleX/deploy/hps/api_compat/<new_api>/. \
    paddlex-hps:/opt/api_compat/<new_api>/
podman exec paddlex-hps find /opt/api_compat/<new_api> -type d -name __pycache__ -exec rm -rf {} +
```

Start with: `api_compat.<new_api>.app:app`

---

## 6. Design Principles (The "Why")

### 6.1 Why a Dedicated Inference Thread?

```
FastAPI (async)  ──queue──►  Inference Thread (sync CUDA)
     ↑                            │
     └──────result queue──────────┘
```

- **pycuda CUDA contexts are thread-local** — a context created in thread A
  cannot be used in thread B
- PaddleX's `model.predict()` internally creates/uses a CUDA context
- FastAPI runs handlers in an event loop (single thread, async)
- If we called `predict()` directly, we'd block the event loop AND potentially
  create the context in the wrong thread
- **Solution**: dedicated daemon thread that owns the CUDA context, communicates
  via queues

### 6.2 Why `asyncio.Semaphore(PIPELINE_DEPTH)`?

- Single GPU → inference thread processes one `predict()` at a time (serial GPU)
- `PIPELINE_DEPTH` (default 4, env: `HPS_API_PIPELINE_DEPTH`) controls how many
  requests can be **in-flight** simultaneously in the asyncio layer
- With depth > 1, the inference thread always has work queued — no GPU idle gap
  between requests (request B's task is already on the queue when A finishes)
- Results are matched by `task_id` via per-task `Future` objects, not by queue
  order — no head-of-line blocking (if the thread finishes B before A, B's
  future resolves immediately without waiting for A)
- With `DEPTH=1`, this degenerates to the original fully-serial behavior

**CPU stage offloading**: CPU-bound work (image decode, DoclingDocument
conversion, format export) is offloaded to a dedicated thread pool
(`HPS_API_CPU_POOL_SIZE`, default 8) via `loop.run_in_executor()`. This prevents
one request's CPU work from blocking the event loop and stalling all others.
Export and confidence computation run in parallel via `asyncio.gather()`.

### 6.3 Why `_core/` Separate from `docling_api/`?

- **Reuse**: `unstructured_api/` will need the exact same config, engine,
  inference, image, and health code
- **Separation of concerns**: `_core/` = "how to serve PaddleX" (infrastructure);
  `docling_api/` = "how to speak Docling protocol" (business logic)
- **Testability**: `_core/` can be tested without any protocol layer

### 6.4 Why a Converter Sub-package?

- The label mapping (25 PaddleX categories → Docling labels) is data, not logic
- Confidence thresholds are tunable parameters
- Keeping them in `labels.py` makes them easy to audit and modify
- The converter `service.py` is stateless and safe to share across requests

### 6.5 Why `extra="allow"` on `ConvertDocumentsOptions`?

- The real Docling API has many options we don't support (OCR engine, table mode, etc.)
- `extra="allow"` lets clients send the full Docling request without 422 errors
- We simply ignore unsupported fields — forward compatibility

### 6.6 Why Pre-compute Markdown?

- Both `MARKDOWN` and `TEXT` formats need markdown as a base
- `TEXT` is just markdown with syntax stripped: `re.sub(r'[#*_`>|\-]', '', md)`
- Computing markdown once and reusing avoids double export

---

## 7. Dependency Graph (Import Direction)

```
                    config.py  ◄──── (no deps, reads env vars)
                       │
                    engine.py  ◄──── config
                       │
                   inference.py ◄── config, engine, paddlex
                       │
                    image.py   ◄── config
                       │
                   health/routes.py ◄── inference, health/schema
                       │
          ┌────────────┼─────────────────┐
          │            │                 │
   docling_api/app.py  │          (future layers)
     ├── _core.health.routes
     ├── _core.inference.state
     ├── _core.config.STARTUP_TIMEOUT
     └── docling_api.routes
                │
   docling_api/routes.py
     ├── _core.image.fetch_image_from_url
     ├── docling_api.schema
     └── docling_api.service
                │
   docling_api/service.py
     ├── _core.image.load_image_from_bytes
     ├── _core.inference.run_layout_detection
     ├── docling_api.converter.service
     └── docling_api.schema
                │
   docling_api/converter/service.py
     ├── converter.labels
     └── converter.schema
```

**Rule**: Dependencies flow downward. `_core/` never imports from any `*_api/`
package. Protocol layers import from `_core/` and from their own sub-packages.

---

## 8. Adding the Unstructured IO Layer — Concrete Plan

Based on the [unstructured-api](https://github.com/Unstructured-IO/unstructured-api)
protocol:

### 8.1 Endpoints to Implement

| Endpoint | Purpose |
|----------|---------|
| `POST /general/v1/general` | Main partition endpoint (multipart file upload) |
| `POST /clean/v1/clean` | (future) Data cleaning |

### 8.2 Response Shape

Unstructured returns a **flat list of elements** (not a document tree):

```json
[
  {"type": "Title", "text": "...", "metadata": {"page_number": 1, ...}},
  {"type": "Text", "text": "...", "metadata": {...}},
  {"type": "Image", "text": "", "metadata": {...}}
]
```

### 8.3 Files to Create

```
api_compat/unstructured_api/
├── __init__.py
├── app.py               # create_app() with health + partition router
├── routes.py            # POST /general/v1/general
├── schema.py            # Element, PartitionResponse, PartitionParameters
└── service.py           # Pipeline: image → detect → convert to elements list
```

### 8.4 Label Mapping (PaddleX → Unstructured)

| PaddleX Label | Unstructured Type |
|---------------|-------------------|
| `doc_title` | `Title` |
| `text`, `content`, `abstract` | `Text` / `NarrativeText` |
| `paragraph_title` | `Header` |
| `image`, `header_image`, `footer_image`, `seal` | `Image` |
| `table` | `Table` |
| `chart` | `TableChart` (or `Image`) |
| `footer` | `Footer` |
| `header` | `Header` |
| `footnote` | `Footnote` |
| `display_formula`, `inline_formula` | `Formula` (custom type) |

### 8.5 Differences from Docling Layer

| Aspect | Docling | Unstructured |
|--------|---------|--------------|
| Response | Single document object | Flat list of elements |
| Document model | DoclingDocument (tree) | List of dicts (flat) |
| Export formats | markdown, json, html, text, doctags | Always JSON element list |
| Converter | Build DoclingDocument tree | Build flat element list |
| Confidence | Nested ConfidenceScores object | `metadata.confidence` per element |

### 8.6 What Reuses `_core/`

**Everything in `_core/`** — config, engine, inference, image, health:
- Same model, same GPU, same inference thread
- Same image loading and resizing
- Same health endpoints
- Only the converter and response shape differ

---

## 9. Quick Reference: Deployment Commands

### Docker build (standalone image)

```bash
# Build from PaddleX repo root
docker build -f deploy/hps/docker/cuda13/full/Dockerfile -t paddlex-hps:layout-cu13-full .
docker run --gpus all -p 8080:8080 paddlex-hps:layout-cu13-full
```

### Container deployment (dev/iteration)

```bash
# Deploy code to container
podman exec paddlex-hps mkdir -p /opt/api_compat
podman cp ~/ADEO/OCR/PaddleX/deploy/hps/api_compat/. paddlex-hps:/opt/api_compat/
podman exec paddlex-hps find /opt/api_compat -type d -name __pycache__ -exec rm -rf {} +

# Kill old server
podman exec paddlex-hps bash -c 'pkill -9 -f granian; sleep 2'

# Start server (Docling layer)
podman exec -d paddlex-hps bash -c \
  'cd /opt && PYTHONPATH=/opt python3 -m granian \
   --interface asgi --host 0.0.0.0 --port 8080 --workers 1 \
   "api_compat.docling_api.app:app" > /tmp/granian.log 2>&1'

# Future: Start server (Unstructured layer)
podman exec -d paddlex-hps bash -c \
  'cd /opt && PYTHONPATH=/opt python3 -m granian \
   --interface asgi --host 0.0.0.0 --port 8081 --workers 1 \
   "api_compat.unstructured_api.app:app" > /tmp/granian-unstructured.log 2>&1'
```

### Smoke Test (Docling)

```bash
# Health
curl -s http://localhost:8080/health
curl -s http://localhost:8080/ready

# Convert file (to_formats as repeated form fields — upstream FormDepends pattern)
curl -s -X POST "http://localhost:8080/v1/convert/file" \
  -F "file=@/tmp/test_doc.png" \
  -F "to_formats=md" \
  -F "to_formats=json" | python3 -m json.tool

# Convert file (comma-separated single field — defensive fallback)
curl -s -X POST "http://localhost:8080/v1/convert/file" \
  -F "file=@/tmp/test_doc.png" \
  -F "to_formats=md,json" | python3 -m json.tool

# Convert source (URL)
curl -s -X POST http://localhost:8080/v1/convert/source \
  -H "Content-Type: application/json" \
  -d '{"sources":[{"kind":"http","url":"https://example.com/image.png"}],"options":{"to_formats":["markdown"]}}'
```

---

## 10. API Compatibility Audit (L1–L4 Code Smell Detection)

> **Audit date**: 2026-07-23
> **Technique**: L1–L4 code smell detection + logic mapping
> **Scope**: All 18 source files in `api_compat/`
> **Upstream reference**: docling-slim 2.114.0, docling-core 2.87.1,
> docling-serve v1.27.0 (branch `jya0-v1.27.0`)

### 10.1 Logic Map (Prerequisite for L2/L3)

Before auditing, a full function-level logic map was built (see §3 above).
This map traces every import chain, dispatch table, and data flow path,
enabling systematic L2 (per-file) and L3 (cross-reference) analysis.

### 10.2 L1 — Automated Static Analysis

| Tool | Command | Result |
|------|---------|--------|
| **pyflakes** | `pyflakes api_compat/` | ✅ 0 findings |
| **vulture** | `vulture api_compat/ --min-confidence 80` | ✅ 0 findings (conf ≥ 80) |
| **ruff** | `ruff check api_compat/ --select=F,E9,PL,E,W --ignore=PLR0913,PLR2004,PLR0915,PLR0912,PLC0414` | ✅ 0 findings |

Environment: Python 3.14, `/tmp/lint-env` virtualenv with ruff, pyflakes, vulture,
docling-slim, docling-core, scipy, httpx, fastapi, python-multipart.

### 10.3 L2 — Per-File Semantic Checklist

All 18 source files read and analyzed:

- **Imports**: Every import verified to resolve (no unused, no missing)
- **Comments**: All docstrings match function behavior
- **Functions**: All signatures match their call sites
- **Data structures**: `_EXPORTERS` (6 entries), `_FIELD_MAP` (6 entries),
  `PADDLEX_TO_DOCLING` (25 entries), `_SPECIAL_LABELS`, `_TEXT_VALID_LABELS`
- **Error handling**: `ERR_IMAGE_LOAD`, `ERR_LAYOUT_DETECTION`, `ERR_FETCH`,
  `ERR_DECODE` — all map to valid `FailureCategory` enum values

### 10.4 L3 — Cross-Reference Against Upstream Contracts

| Check | Verified Against | Result |
|-------|-----------------|--------|
| 50+ schema re-exports | Installed `docling-slim 2.114.0` | ✅ All exist |
| `OutputFormat` 11 members | `docling.datamodel.base_models` | ✅ Exact match |
| `ErrorItem` fields | `docling.datamodel.service.responses` | ✅ All 6 fields match |
| `ConvertDocumentResponse` fields | `docling.datamodel.service.responses` | ✅ All 8 fields match |
| `ExportDocumentResponse` fields | `docling.datamodel.service.responses` | ✅ All 8 fields match |
| `ConfidenceScores` fields | `docling.datamodel.base_models` | ✅ All 5 fields match |
| `QualityGrade` 5 members | `docling.datamodel.base_models` | ✅ Match (we use 3 by design) |
| `FailureCategory` enum | `docling.datamodel.base_models` | ✅ All values valid |
| `ConversionStatus` enum | `docling.datamodel.base_models` | ✅ All values valid |
| 5 `DoclingDocument.export_to_*` methods | `docling-core 2.87.1` | ✅ All exist |
| 25 `PADDLEX_TO_DOCLING` label values | `docling_core.types.doc.labels.DocItemLabel` | ✅ All are valid `DocItemLabel` |
| `converter/schema.py` re-exports | Same object identity (`is` check) | ✅ Not copies — `CS2 is ConfidenceScores` |

### 10.5 Protocol Drift Found and Fixed

**Issue (round 1)**: `/v1/convert/file` accepted `to_formats` as a **query parameter**
(`to_formats: str | None = None`), but upstream docling-serve uses
`FormDepends(ConvertDocumentsOptions)` which flattens Pydantic model fields
into **multipart form fields**. Fixed by changing to `Form(default=None)`.

**Issue (round 2 — critical)**: The round-1 fix used `to_formats: str | None = Form(default=None)`
(a single string), but upstream `FormDepends` keeps `list[OutputFormat]` as a
**list type** — `is_json_field()` only handles `dict` origin, not `list`. This
means httpx sends **repeated form fields** (`to_formats=md&to_formats=json`),
and a `str` parameter silently receives only the first value, dropping all
others. A docling-serve client requesting both Markdown and JSON would get
only Markdown.

**Root cause**: Unit tests called `_parse_to_formats()` directly with string
arguments, giving false confidence. HTTP-level tests (using `httpx.ASGITransport`
to send real multipart form data) revealed the bug: the OpenAPI schema showed
`to_formats` as `anyOf: [{type: string}, {type: null}]` instead of
`anyOf: [{type: array, items: {type: string}}, {type: null}]`.

**Fix applied** in `routes.py`:
1. Changed `to_formats: str | None = Form(default=None)` →
   `to_formats: list[str] | None = Form(default=None)` (list type)
2. Changed `convert_file_async` stub parameter to match (`list[str] | None`)
3. Rewrote `_parse_to_formats()` to accept `list[str] | None`:
   - Flattens elements (each may contain comma-separated values for defensive
     backward compatibility with simple clients)
   - Validates each value against `OutputFormat` enum
   - Raises `HTTPException(400)` on invalid format
4. Removed unused `import json` (no longer needed — no JSON array parsing)
5. Added 6 HTTP-level tests using `httpx.ASGITransport` that send real
   multipart form data exactly like upstream docling-serve tests
6. Added 3 official-client compatibility tests using the real
   `DoclingServiceClient._form_encode_options()` from `docling-slim` to
   verify our endpoint is wire-compatible with the official Python client

### 10.6 L4 — Recheck After Fix

| Tool | Command | Result |
|------|---------|--------|
| **pyflakes** | `pyflakes api_compat/docling_api/routes.py` | ✅ 0 findings |
| **ruff** | `ruff check api_compat/docling_api/routes.py tests/test_api_compat.py` | ✅ 0 findings |
| **vulture** | `vulture ... --min-confidence 60` | ✅ 0 real findings (route handlers = false positives) |

### 10.7 Test Results

**Test file**: `tests/test_api_compat.py` (69 contract tests: 20 unit + 6 HTTP-level + 3 original official client + 40 comprehensive official client)

**Test categories**:

| Category | Count | What it verifies |
|----------|-------|------------------|
| Schema re-exports | 5 | OutputFormat, ErrorItem, ConvertDocumentResponse, exporter/field-map consistency |
| Service dispatch | 2 | error_response builder, _export_to_formats dispatch |
| Routes parsing | 7 | _parse_to_formats: defaults, single/multiple/CSV, invalid→400 |
| Converter | 4 | Label mapping (25 cats), QualityGrade thresholds, confidence, reading order |
| Health schema | 1 | HealthCheckResponse defaults to 'ok' |
| Re-exports | 2 | converter/schema.py thin layer, DoclingDocument methods |
| Requests | 1 | ConvertSourcesRequest.options defaults to [MARKDOWN] |
| Router | 1 | All 5 convert endpoints registered |
| HTTP-level | 6 | Real multipart form data through ASGI app |
| Original official client | 3 | Wire-format compatibility with DoclingServiceClient |
| Comprehensive official client | 40 | Full encoding pipeline, all option types, all endpoints, response parsing |

```
==================================================
RESULTS: 69 passed, 0 failed
==================================================
```

**HTTP-level tests** use `httpx.ASGITransport(app=app)` to send real multipart
form data through the FastAPI app — exactly like upstream docling-serve tests.
They verify that `to_formats` sent as repeated form fields
(`data={"to_formats": ["md", "json"]}`) is correctly received as a list with
all values, not silently truncated to the first value.

**Official client tests** use the real `DoclingServiceClient` from
`docling.service_client` (shipped inside `docling-slim`) to verify
wire-format compatibility with the official Python client SDK.

The **comprehensive official client suite** (40 tests) covers:

1. **Options encoding pipeline** — `_form_encode_options()` + `_serialize_convert_options()`:
   - Primitives (bool, str, int, float) pass through as-is
   - Primitive lists (to_formats, ocr_lang) pass through as repeated form fields
   - Dicts (ocr_custom_config) are JSON-encoded
   - Nested models (picture_description_api, vlm_pipeline_model_api) are JSON-encoded
   - Tuples (page_range) become lists
   - Enums become string values
   - `exclude_none=True` omits unset fields
   - All 45+ ConvertDocumentsOptions fields survive round-trip

2. **Endpoint coverage** — all 7 routes tested:
   - `/v1/convert/file` (multipart POST) — accepts full 25+ field payload
   - `/v1/convert/source` (JSON body POST) — ConvertSourcesRequest with file/http sources
   - `/v1/convert/file/async` → 501 Not Implemented
   - `/v1/convert/source/async` → 501 Not Implemented
   - `/v1/convert/source/batch` → 501 Not Implemented (requires cloud sources + target)
   - `/health`, `/health-check`, `/livez`, `/ready`, `/readyz`, `/version`

3. **Response model compatibility** — our responses parsed with official models:
   - `ConvertDocumentResponse.model_validate()` on our JSON output
   - `ExportDocumentResponse` fields: filename, md_content, json_content, etc.
   - `HealthCheckResponse` with `status` field
   - `ConversionStatus` enum values match
   - Processing time, timings, confidence fields

4. **Wire-format edge cases**:
   - FastAPI silently ignores unknown multipart form fields (no 422) — our endpoint
     only declares `file` + `to_formats`, but accepts the full 25+ field payload
   - `to_formats` sent as repeated form fields (not JSON array)
   - Bool fields sent as Python `True`/`False` (not strings)
   - Float fields (images_scale, picture_description_area_threshold) preserved
   - Page range tuple → list encoding
   - Content-Type: application/json on responses

**Key discovery**: FastAPI silently ignores unknown form fields — sending 25+
fields to an endpoint that only declares `file` + `to_formats` returns 200,
NOT 422. This means our endpoint IS wire-compatible with the full official
client payload without needing to declare every field.

**Note**: Tests run without GPU/PaddleX — `paddlex` is stubbed via
`sys.modules` since contract tests verify protocol shapes, not inference.

### 10.8 Audit Conclusion

The `api_compat` package is **fully Docling API compatible** after the fix:
- ✅ All 50+ schema re-exports match upstream docling-slim 2.114.0
- ✅ All response/request field names match upstream contracts
- ✅ All 11 OutputFormat members match
- ✅ All 5 DoclingDocument export methods exist and are dispatched correctly
- ✅ All 25 PaddleX→Docling label mappings use valid DocItemLabel values
- ✅ Protocol drift (to_formats str→list[str]) found and fixed
- ✅ 69/69 contract tests pass (20 unit + 6 HTTP-level + 3 original + 40 comprehensive official client)
- ✅ Full wire-format compatibility with official DoclingServiceClient (45+ option fields, 7 endpoints)
- ✅ Response JSON parses cleanly with official ConvertDocumentResponse model
- ✅ L1–L4 code smell detection: zero findings

---

## 11. Latency Tracing & Bottleneck Identification

### 11.1 Design: Easy-to-Remove Latency Flag

A single environment variable controls all latency instrumentation:

```
HPS_LATENCY_LOG=1   →  enable structured per-stage timing logs
HPS_LATENCY_LOG=0   →  disable (default — zero overhead)
```

When **disabled** (default), `LatencyTracer` methods are no-ops: no
`time.perf_counter()` calls, no dict allocations, no log output. The
overhead is a single boolean check per stage.

When **enabled**, each request logs a JSON line with the full stage
breakdown:

```json
{"event":"latency_trace","request_id":"abc123","filename":"doc.png",
 "stages":{"image_load":0.003,"layout_detection":0.045,
           "convert_to_docling":0.002,"export_formats":0.001},
 "total":0.051,"to_formats":["md","json"]}
```

### 11.2 Implementation

**Module**: `api_compat/_core/latency.py`

| Component | Description |
|-----------|-------------|
| `is_latency_logging_enabled()` | Re-reads `HPS_LATENCY_LOG` env var (supports runtime toggle) |
| `set_latency_logging(bool)` | Sets the env var (for tests/debugging) |
| `LatencyTracer` | Per-request tracer with `.stage(name)` context manager, `.record(name, elapsed)`, `.finish()` |

**Instrumented stages in `service.py:convert_image()`**:
1. `image_load` — PIL decode + resize
2. `layout_detection` — GPU inference via `run_layout_detection()`
3. `convert_to_docling` — PaddleX → DoclingDocument
4. `export_formats` — Multi-format export (md, json, html, text, doctags, doclang)

**Instrumented stages in `inference.py:run_layout_detection()`**:
1. `semaphore_wait` — Time blocked on `asyncio.Semaphore(PIPELINE_DEPTH=4)`
2. `gpu_inference` — Time waiting for the per-task Future to resolve

### 11.3 Usage

```bash
# Enable latency tracing
HPS_LATENCY_LOG=1 granian --interface asgi api_compat.docling_api.app:app

# Disable (default — no overhead)
HPS_LATENCY_LOG=0 granian --interface asgi api_compat.docling_api.app:app
```

To remove all instrumentation: delete `api_compat/_core/latency.py` and
revert the 4 `with tracer.stage(...)` blocks in `service.py` and the 2
timing blocks in `inference.py`.

---

## 12. Concurrency & Stress Testing

### 12.1 Architecture: Triton (Default) vs Direct Backend

The API supports two inference backends, selected by `HPS_API_BACKEND`:

**Triton backend** (default, `HPS_API_BACKEND=triton`):

```
  Client ──HTTP──▶ Granian (ASGI)              ┌─────────────────────────┐
                   │ api_compat (FastAPI)        │ Triton Inference Server │
                   │  triton_client.py           │  doclayout-v3 model      │
                   │                             │  dynamic_batching { }    │
                   │  await detect_layout(img)   │  model.predict(batch)    │
                   │───────── gRPC ──────────────▶│                         │
                   │◀──────── response ──────────│                         │
                   └─────────────────────────────┘                         │
                                                                  GPU       │
```

  - Triton's dynamic batcher collects concurrent requests into batches of
    up to 8, dispatches them in one GPU forward pass, and queues the next
    batch while the current executes — **continuous batching**.
  - No inference thread, no custom batching logic, no in-process model
    loading in the API layer.
  - Config: `triton/model_repo/doclayout-v3/config.pbtxt` with
    `dynamic_batching { max_queue_delay_microseconds: 5000 }`

**Direct backend** (`HPS_API_BACKEND=direct`):

  - Loads PaddleX model in-process via `create_model()`.
  - Uses custom micro-batching in a dedicated inference thread.
  - Simpler (single process), but static batching — GPU idles between
    batch collection and result splitting.
  - Used for testing without a Triton server.

### 12.2 Test File

**File**: `tests/test_concurrency.py` — 45 tests

Tests run with `HPS_API_BACKEND=direct` (forced at top of test file) so they
work without a running Triton server. Triton-specific tests stub the gRPC
client. All tests run **without GPU or PaddleX** (stubbed via `sys.modules`).

### 12.3 Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **Latency flag** | 8 | Env-var enable/disable, no-op when disabled, stage recording, JSON trace emission, pipeline integration |
| **Concurrency** | 6 | 5/20/50/100 concurrent requests all succeed, unique filenames (no cross-talk) |
| **Bottleneck ID** | 4 | Spy bypasses semaphore (concurrent), semaphore depth controls GPU concurrency, throughput, latency-under-load |
| **Pipeline overlap** | 2 | DEPTH>1 allows request overlap inside semaphore, future-based result matching (no head-of-line blocking) |
| **Micro-batching** | 7 | Direct-backend: `_process_single`, `_process_batch`, error propagation, batch collection, timeout flush |
| **Triton backend** | 9 | Config constants, client import, stubbed `detect_layout()`, error propagation, `_run_triton` dispatch, concurrent calls, `config.pbtxt` validation, `model.py` validation, entrypoint script |
| **Deadlock/race** | 2 | Mixed endpoints (file+source+health), sequential bursts (no state leak) |
| **Pipeline tracing** | 2 | Latency traces emitted with stage breakdowns when enabled, no traces when disabled |
| **Load ramp** | 2 | Gradual 1→50 ramp, slow-spy throughput |
| **Percentiles** | 2 | p50/p95/p99 with fast and slow spies |
| **Error handling** | 2 | Mixed success/error under concurrency, 1MB payloads |

### 12.4 Key Findings

1. **Configurable pipeline depth**: `HPS_API_PIPELINE_DEPTH` (default 4)
   allows multiple requests to be in-flight simultaneously. The inference
   thread still processes one `predict()` at a time (serial GPU), but the
   asyncio layer pipelines task submission. With DEPTH=3 and 50ms inference,
   5 concurrent requests complete in ~100ms (2 batches) vs ~250ms fully serial.

2. **GPU micro-batching**: `HPS_API_BATCH_SIZE` (default 2, recommended 2-4)
   enables the inference thread to collect up to N images and issue a single
   `predict(images, batch_size=N)` call. The TRT engine for PP-DocLayoutV3 is
   already optimized for batch=4 (opt-profile in `_DEFAULT_LAYOUT_PROFILE`),
   so no engine rebuild is needed. Results are split by position and delivered
   to each request's `Future`. `BATCH_TIMEOUT_MS` (default 5ms) prevents
   indefinite waiting under low load — a partial batch is flushed after the
   timeout rather than blocking.

3. **Future-based result matching**: Results are delivered to the correct
   request by `task_id` via per-task `Future` objects, not by queue order.
   No head-of-line blocking — if the thread finishes task B before A, B's
   future resolves immediately.

4. **CPU stage offloading**: Image decode, DoclingDocument conversion, and
   format export run in a dedicated thread pool (`HPS_API_CPU_POOL_SIZE`,
   default 8), preventing CPU-bound work from blocking the event loop.
   Export and confidence computation run in parallel via `asyncio.gather()`.

5. **HTTP layer is NOT the bottleneck**: When the spy bypasses the
   semaphore (replacing `convert_image` entirely), 10 concurrent 50ms
   requests complete in ~50ms (fully concurrent).

6. **Throughput scales linearly**: Load ramp from 1→50 requests shows
   throughput increasing from ~900 to ~2760 req/s (fast spy, no GPU).

7. **No deadlocks or state leaks**: Mixed endpoints, sequential bursts,
   and large payloads all handle correctly under concurrency.

### 12.4 Benchmark Results

**File**: `tests/bench_configs.py` — Configuration sweep harness

Sweeps 6 configurations with a simulated GPU model
(`latency = FIXED_MS + PER_IMAGE_MS × n`), measuring throughput, GPU call
count, and latency percentiles under concurrent load.

#### Default parameters (40ms fixed + 5ms/image, 32 requests)

| Config | DEPTH | BATCH | Wall (s) | RPS | GPU Calls | Img/Call | p99 (ms) | Speedup |
|--------|------:|------:|---------:|----:|----------:|---------:|---------:|--------:|
| serial | 1 | 1 | 1.447 | 22.1 | 32 | 1.0 | 1447 | 1.0× |
| pipeline_d3 | 3 | 1 | 1.447 | 22.1 | 32 | 1.0 | 1447 | 1.0× |
| batch_b4 | 1 | 4 | 0.483 | 66.2 | 8 | 4.0 | 483 | 3.0× |
| balanced_d3_b4 | 3 | 4 | 0.483 | 66.2 | 8 | 4.0 | 483 | 3.0× |
| **wide_d4_b8** | **4** | **8** | **0.322** | **99.3** | **4** | **8.0** | **322** | **4.5×** |
| max_d8_b8 | 8 | 8 | 0.323 | 99.0 | 4 | 8.0 | 323 | 4.5× |

#### Key observations

- **Pipeline depth alone (no batching) gives 0% speedup**: The GPU is serial,
  so DEPTH>1 without batching just queues more tasks — each still gets its own
  `predict()` call. Pipeline depth helps overlap CPU stages with GPU, but the
  GPU bottleneck remains.
- **Batching is the primary throughput lever**: BATCH=4 reduces GPU calls 4×
  (32→8), giving ~3× throughput. BATCH=8 reduces calls 8× (32→4), giving ~4.5×.
- **DEPTH matters less than BATCH** when the bottleneck is GPU: `batch_b4`
  (DEPTH=1) matches `balanced_d3_b4` (DEPTH=3) — the batch fill time dominates
  over pipeline overlap.
- **Higher fixed GPU cost → bigger batch win**: With 80ms fixed cost, BATCH=8
  gives 5.6× speedup; with 20ms fixed cost, only 2.4× — the more kernel-launch
  overhead, the more batching helps.
- **Recommended production config**: Use the **Triton backend** (default) with
  `HPS_API_PIPELINE_DEPTH=4`. Triton's continuous batching automatically
  saturates the GPU — no manual batch size tuning needed. The direct backend's
  `HPS_API_BATCH_SIZE=4` gives ~3× throughput but is inferior to Triton's
  continuous batching (which avoids GPU idle gaps between batches).

### 12.5 Test Results

```
RESULTS: 45 passed, 0 failed  (test_concurrency.py)
RESULTS: 69 passed, 0 failed  (test_api_compat.py)
```

All 114 tests pass. Lint-clean (ruff: 0 findings).

### 12.6 How to Run

```bash
cd PaddleX/deploy/hps
python3 tests/test_concurrency.py          # 45 concurrency/stress/batch/triton tests
python3 tests/test_api_compat.py           # 69 contract tests
python3 tests/bench_configs.py             # direct-backend benchmark sweep
HPS_LATENCY_LOG=1 python3 tests/test_concurrency.py  # with latency traces

# Run with the Triton backend (requires tritonserver running):
HPS_API_BACKEND=triton HPS_TRITON_URL=localhost:8001 \
    granian --interface asgi --port 8080 \
    "api_compat.docling_api.app:app"

# Run with the direct backend (no Triton needed, for development):
HPS_API_BACKEND=direct HPS_API_BATCH_SIZE=2 HPS_API_BATCH_TIMEOUT_MS=5 \
    HPS_API_PIPELINE_DEPTH=4 HPS_API_CPU_POOL_SIZE=8 \
    granian --interface asgi --port 8080 \
    "api_compat.docling_api.app:app"

# Production: dual-process entrypoint (starts both tritonserver + granian)
./scripts/run_api_triton.sh

# Benchmark direct-backend configurations
python3 tests/bench_configs.py --n 64 --fixed 80 --per-image 5
```

### 12.7 Triton Model Repository

**Directory**: `triton/model_repo/doclayout-v3/`

```
triton/model_repo/doclayout-v3/
├── config.pbtxt          ← Triton config: max_batch_size=8, dynamic_batching
└── 1/
    └── model.py          ← Python backend: loads PaddleX, batched predict()
```

`config.pbtxt` key settings:
- `max_batch_size: 8` — Triton collects up to 8 concurrent requests
- `dynamic_batching { max_queue_delay_microseconds: 5000 }` — flush partial
  batches after 5ms (low latency under light load)
- `preferred_batch_size: [1, 2, 4, 8]` — dispatch at these sizes when available
- `instance_group: KIND_GPU, gpus: [0]` — model runs on GPU 0

`model.py` lifecycle:
1. `initialize()` — Loads `paddlex.create_model("PP-DocLayoutV3", engine="tensorrt")`
2. `execute(requests)` — Triton calls this with 1–8 batched requests:
   - Decodes each image from base64
   - Calls `model.predict(images, batch_size=N)` in one GPU forward pass
   - Splits results by position, returns each to its response
3. `finalize()` — Cleanup on shutdown
