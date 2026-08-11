# PaddleX HPS — Docling-Compatible API Reference

> **Self-contained.** This document is readable from anywhere in the repo. All
> file references are given as **absolute paths** under the PaddleX workspace
> so you can jump straight to them regardless of which directory you open it from.
>
> Everything here describes the **live server code**, not a design. If a path or
> field is documented, it is implemented and tested. See
> [`deploy/hps/tests/test_api_compat.py`](../tests/test_api_compat.py).

---

## 1. What this API is

The PaddleX HPS server exposes a **Docling-compatible HTTP API**. It accepts
document/layout-conversion requests in the same shape as
[`docling-serve`](https://github.com/docling-project/docling-serve) and returns
the same response models, so an off-the-shelf Docling client can talk to it
without modification.

The conversion pipeline underneath is **layout detection with PP-DocLayoutV3**
(run on a TensorRT engine), but the API contract does not leak this — the
server just consumes an image and produces a Docling conversion response.

Three concerns are kept separate so you can build an adapter at any level:

| Concern | Where | What it is |
|---------|-------|------------|
| **HTTP contract** | `deploy/hps/api_compat/docling_api/` | FastAPI routers, schemas, request/response models |
| **Inference backend** | `deploy/hps/api_compat/_core/` | Plug-and-play `InferenceBackend` (direct vs triton) |
| **Docling conversion** | `deploy/hps/api_compat/docling_api/converter/` | PaddleX → Docling label mapping + schema building |

---

## 2. Quick start — run the server

You do **not** need Triton. The default backend is `direct` (in-process
PaddleX/TensorRT).

```bash
cd /home/jyao/ADEO/OCR/PaddleX/deploy/hps
bash scripts/run_dev_server.sh        # one-command host launch (CUDA 13 dev venv)
```

Or start the FastAPI/granian app yourself:

```bash
export HPS_API_BACKEND=direct        # "direct" | "triton"
export HPS_API_MODEL=PP-DocLayoutV3
export HPS_API_PRECISION=fp16
granian --interface asgi api_compat.docling_api.app:app --host 0.0.0.0 --port 8080
```

Once up, confirm health:

```bash
curl -s http://localhost:8080/health        # {"status":"ok",...}
curl -s http://localhost:8080/ready         # 200 when model loaded
```

> The two most useful probes are `/ready` (503 until the model is loaded) and
> `/v1/models` (inventory, Triton-compatible).

---

## 3. Every HTTP endpoint

All documented paths are implemented. The table groups them by family.

### 3.1 Health & liveness

| Method | Path | Purpose | Success |
|--------|------|---------|---------|
| GET | `/health` | Liveness/health summary | `200` |
| GET | `/health-check` | Alias of `/health` | `200` |
| GET | `/livez` | Liveness probe (k8s) | `200` |
| GET | `/ready` | Readiness — **503 until model loaded** | `200` / `503` |
| GET | `/readyz` | Alias of `/ready` | `200` / `503` |
| GET | `/version` | Server/API version string | `200` |

**Source:** `deploy/hps/api_compat/_core/health/routes.py`

### 3.2 Model inventory (Triton-compatible `/v1/models`)

These three endpoints mirror the **Triton Inference Server** HTTP API so existing
Triton tooling and probes work against this server.

| Method | Path | Purpose | Notes |
|--------|------|---------|-------|
| GET | `/v1/models` | List served models + readiness | Returns JSON array, each item `{name, version, ready, active}` |
| GET | `/v1/models/{name}` | Metadata for one model | `404` if unknown name |
| GET | `/v1/models/{name}/ready` | Readiness of one model | `200 {"status":"ok"}` / `503` / `404` |

**Model id:** the active model name from `HPS_API_MODEL` (default
`PP-DocLayoutV3`). Works identically for both backends.

**Source:** `deploy/hps/api_compat/_core/health/routes.py`
**Facade:** `deploy/hps/api_compat/_core/inference.py` → `AppState.list_models()` / `AppState.get_model_status()`

### 3.3 Conversion endpoints

| Method | Path | Body | `debug` location | Purpose |
|--------|------|------|------------------|---------|
| POST | `/v1/convert/file` | multipart form | **Form field** | Convert one uploaded image synchronously |
| POST | `/v1/convert/source` | JSON (Docling `ConvertSourcesRequest`) | **Query param** | Convert an HTTP-URL or base64 source synchronously |
| POST | `/v1/convert/file/async` | multipart form | — | Submit file → returns task id |
| POST | `/v1/convert/source/async` | JSON | — | Submit source → returns task id |
| GET | `/v1/status/poll/{task_id}` | — | — | Poll an async task's status |
| GET | `/v1/result/{task_id}` | — | — | Fetch an async task's result |
| POST | `/v1/convert/source/batch` | JSON | — | **Accepted for compat, always `501`** (single-doc only) |

**Sources:** `deploy/hps/api_compat/docling_api/routes.py`

### 3.4 OpenAPI schema

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/openapi.json` | Full OpenAPI spec (all request/response models) |
| GET | `/docs` | Swagger UI |

The OpenAPI spec is generated from the actual FastAPI models, so it is the
**single best source of truth** for field names, types, enums, and request
bodies. Point a Docling client / code generator at `/openapi.json`.

---

## 4. The `debug` parameter (two locations)

`debug=1` (or `debug=true`) switches a convert call from returning the normal
conversion JSON to returning a **PNG image** with the layout boxes drawn on it.

- **`/v1/convert/source`** — body is JSON, so `debug` is a **query** parameter:
  ```
  POST /v1/convert/source?debug=1
  Content-Type: application/json
  {"sources":[...],"options":{...}}
  ```
- **`/v1/convert/file`** — body is multipart, so `debug` is a **form** field:
  ```
  POST /v1/convert/file
  Content-Type: multipart/form-data
  files=@page.png   debug=1
  ```

Why the asymmetry? `/source` uses a JSON body so the only place for extra flags
is the URL query string; `/file` uses multipart so extra flags become form
fields. This mirrors how upstream `docling-serve` wires its
`FormDepends(ConvertDocumentsOptions)`.

When `debug=1`, the response `Content-Type` is `image/png` (not
`application/json`). The debug image shows each detected layout box with its
label and score.

**Source:** `deploy/hps/api_compat/docling_api/service.py` → `render_debug_image()`,
`deploy/hps/api_compat/_core/debug/__init__.py`

---

## 5. Response shapes (two families)

### 5.1 Normal conversion response — `ConvertDocumentResponse`

This is the Docling-standard response model (from `docling` package). Its main
fields:

| Field | Type | Notes |
|-------|------|-------|
| `document` | `ExportDocumentResponse` | Contains `filename` (and optionally `content`, `mimeType`, `chunks`) |
| `status` | `ConversionStatus` | `success` / `failure` / ... |
| `errors` | `list[ErrorItem]` | `component_type`, `module_name`, `error_message`, `category` |
| `processing_time` | `float` | Seconds |

Example (success, markdown):
```json
{
  "document": {"filename": "report.md"},
  "status": "success",
  "errors": [],
  "processing_time": 0.42
}
```

**Source:** `deploy/hps/api_compat/docling_api/schema.py` (re-exported from
`docling.datamodel.service.responses`)

### 5.2 Debug response — PNG bytes

When `debug=1` is set, the response is `image/png` (a single image with layout
boxes drawn). No JSON.

---

## 6. The backend abstraction (`direct` vs `triton`)

The **same** HTTP endpoints serve either inference backend. You pick with the
`HPS_API_BACKEND` env var:

| Value | Behavior | Requires |
|-------|----------|----------|
| `direct` **(default)** | Loads PaddleX/TensorRT **in-process**, micro-batches requests | The model engine on disk (built on first startup) |
| `triton` | Proxies inference to a Triton Inference Server via gRPC | A running Triton server (see `HPS_TRITON_URL`) |

Key files:

- **Factory:** `deploy/hps/api_compat/_core/backends/__init__.py` → `create_backend()`
- **Interface (ABC):** `deploy/hps/api_compat/_core/backends/base.py` → `InferenceBackend`
- **Direct impl:** `deploy/hps/api_compat/_core/backends/direct.py` → `DirectBackend`
- **Triton impl:** `deploy/hps/api_compat/_core/backends/triton.py` → `TritonBackend`
- **Facade:** `deploy/hps/api_compat/_core/inference.py` → `AppState` (holds the backend)

The API layer **only talks to the `InferenceBackend` interface**, so adding a
new backend (e.g. `custom`, `vllm`, `http`) never touches the HTTP routes.

### Env vars (single source of truth)

Defined in **`deploy/hps/api_compat/_core/config.py`**. Every env var read
lives there — never scatter `os.environ` elsewhere.

| Env var | Default | Meaning |
|---------|---------|---------|
| `HPS_API_BACKEND` | `direct` | `direct` / `triton` |
| `HPS_API_MODEL` | `PP-DocLayoutV3` | Model id (shown in `/v1/models`) |
| `HPS_API_PRECISION` | `fp16` | TRT engine precision |
| `HPS_API_DEVICE_ID` | `0` | GPU index |
| `HPS_API_ENGINE_DIR` | `/tmp/paddlex-engines` | Where TRT engines are built/cached |
| `HPS_API_MAX_IMAGE_DIM` | `4096` | Max input image dimension |
| `HPS_API_TIMEOUT` | `30` | Request timeout (s) |
| `HPS_API_STARTUP_TIMEOUT` | `300` | Model-load readiness timeout (s) |
| `HPS_API_PIPELINE_DEPTH` | `16` | In-flight GPU tasks (concurrency) |
| `HPS_API_CPU_POOL_SIZE` | `8` | CPU pool for decode/convert/export |
| `HPS_API_BATCH_SIZE` | `4` | Micro-batch size (direct backend only) |
| `HPS_API_BATCH_TIMEOUT_MS` | `3` | Micro-batch flush timeout (direct) |
| `HPS_API_PRE_POST_POOL_SIZE` | `4` | Pre/post processing pool (direct) |
| `HPS_TRITON_URL` | `localhost:8001` | Triton gRPC address |
| `HPS_TRITON_MODEL_NAME` | `doclayout-v3` | Model name in Triton |
| `HPS_API_LOG_LEVEL` | `INFO` | Log level |

---

## 7. How to build an adapter for a Docling client

You don't need to understand detection internals to build a client adapter. The
adapter has two jobs: **(a)** speak the Docling HTTP contract to the server, and
**(b)** read the conversion result. Read the files below **in order** to see
every seam you can plug into.

### Step 1 — Understand the wire contract (no code yet)

1. Start the server (Section 2).
2. `curl http://localhost:8080/openapi.json` and read the generated spec.
   This is authoritative for request bodies and response models.
3. Compare with upstream Docling: the request/response models are **the same
   `docling` classes**, so any Docling client model works.

### Step 2 — Read the routing layer (what endpoints exist)

- **`src/router`** → `deploy/hps/api_compat/docling_api/routes.py`
  All `/v1/*` handlers, parameter extraction (`Form`/`Query`/`File`), error
  mapping, the `debug` handling, and the async task flow.

### Step 3 — Read the schema layer (the exact data shapes)

- **Schema** → `deploy/hps/api_compat/docling_api/schema.py`
  Re-exports every Docling request/response/enum. If a type you need is missing
  here it's also not in upstream `docling-slim`.

### Step 4 — Read the service layer (what happens to a request)

- **Service** → `deploy/hps/api_compat/docling_api/service.py`
  Orchestrates: fetch/decode image → run detection → render debug → convert to
  Docling → export to requested formats → build `ConvertDocumentResponse`.

### Step 5 — Read the converter (PaddleX boxes → Docling doc)

- **Converter** → `deploy/hps/api_compat/docling_api/converter/`
  - `labels.py` — PaddleX label → Docling label mapping.
  - `schema.py` — `ConfidenceScores`, `QualityGrade`, etc.
  - `service.py` — `PaddleXToDoclingConverter` builds the Docling doc.

### Step 6 — Read the inference backend (swap-able engine)

- **Backends** → `deploy/hps/api_compat/_core/backends/`
  Only touch this if you are adding a **new engine backend**, not a new client.

### Adapter reference — minimal implementation sketch

A minimal Python client (using `httpx`) that uploads an image and gets the
markdown filename back:

```python
import httpx

SERVER = "http://localhost:8080"

def convert_file(path: str, to_formats=("md",), debug=False):
    with open(path, "rb") as f:
        data = {
            "files": f,  # the field is "files" (plural), NOT "file"
        }
        if to_formats:
            data["to_formats"] = list(to_formats)
        if debug:
            data["debug"] = "1"
        r = httpx.post(f"{SERVER}/v1/convert/file", files=data, timeout=120)
    if debug:
        return r.content, r.headers.get("content-type")  # image/png
    body = r.json()
    return body["document"]["filename"], body["status"], body["errors"]

def async_flow(path: str):
    """Follow the official client's submit→poll→result pattern."""
    with open(path, "rb") as f:
        r = httpx.post(f"{SERVER}/v1/convert/file/async",
                       files={"files": f}, timeout=10)
    task_id = r.json()["task_id"]
    # task already complete (synchronous during submission)
    return httpx.get(f"{SERVER}/v1/result/{task_id}", timeout=10).json()
```

---

## 8. Full file map (what to read when)

| What you want | Read this (in order) |
|---------------|----------------------|
| Every HTTP path | `docling_api/routes.py` |
| The exact JSON models | `docling_api/schema.py` |
| Request→response flow | `docling_api/service.py` |
| PaddleX→Docling mapping | `docling_api/converter/*` |
| Inference backend contract | `_core/backends/base.py` + `_core/backends/__init__.py` |
| `direct` engine impl | `_core/backends/direct.py` |
| Triton engine impl | `_core/backends/triton.py` + `_core/triton_client.py` |
| Backend facade (`AppState`) | `_core/inference.py` |
| Env vars (single source) | `_core/config.py` |
| Health `/v1/models` routes | `_core/health/routes.py` |
| Debug renderer seam | `_core/debug/__init__.py` |
| Image fetch/decoding helpers | `_core/image.py` |
| Request/response (async) | `api_compat/docling_api/task_store.py` |
| Protocol contract tests | `deploy/hps/tests/test_api_compat.py` |

---

## 9. Quick start (codeless)

```bash
# health
curl -s localhost:8080/health
# readiness (503 until model ready)
curl -s -w '%{http_code}\n' localhost:8080/ready
# inventory
curl -s localhost:8080/v1/models
# {"models":[{"name":"PP-DocLayoutV3","version":"1","ready":true,"active":true}]}
# convert a file, get markdown filename
curl -s -F files=@page.png -F to_formats=md localhost:8080/v1/convert/file
# debug (returns PNG)
curl -s -F files=@page.png -F debug=1 localhost:8080/v1/convert/file -o out.png
# OpenAPI spec
curl -s localhost:8080/openapi.json | jq '.paths | keys'
```