# Mistral OCR API Layer — Design & Enable/Disable Selection

**Status**: Proposal (pending review)
**Scope**: Add a Mistral-compatible `/v1/ocr` API layer to
`deploy/hps/api_compat/`, alongside the existing Docling layer, with
per-layer runtime enable/disable selection.

---

## 1. Motivation

The existing `api_compat/` exposes a single protocol: Docling (`/v1/convert/*`).
Two gaps drive this design:

1. **Model-id selection.** The Docling convention has **no model-id** in any
   route. The model is fixed server-side via `HPS_API_MODEL`/`HPS_API_BACKEND`.
   `GET /v1/models` exists purely as a Triton-compatible **inventory**
   (discovery/readiness) — it is **not** routing. If we want to serve
   multiple models (PP-DocLayoutV3, PP-StructureV3, PaddleOCR-VL) behind one
   gateway and let clients pick per request, we need a protocol that carries a
   model id. Mistral OCR does: `request.model`.

2. **Reuse official typed DTOs.** Mistral's generated SDK ships
   `mistralai.client.models.OCRRequest` / `OCRResponse` / `OCRPageObject` /
   `OCRTextBlock` etc. Importing these directly means we **do not maintain our
   own DTOs** — the request/response contract is identical to what Mistral's
   own generated client produces, which makes end-to-end compatibility testing
   trivial (point the official `Mistral` client at our server via
   `server_url=`).

This design follows the **vertical-slice methodology** already documented in
[`api_compat_methodology.md`](api_compat_methodology.md) (§5, §7, §9): a new
layer reuses everything in `_core/` (config, engine, inference, image, health)
and only adds protocol-specific schema + converter + service + routes.

---

## 2. Decision: What to build

We add a **Mistral OCR-compatible layer** mounted under `/v1/ocr`, reusing the
existing `_core` inference backend. In addition, we introduce a **mode switch**
so each API layer can be enabled or disabled at startup (see §6).

| Aspect | Docling layer (existing) | Mistral OCR layer (new) |
|--------|--------------------------|--------------------------|
| Protocol | `docling-serve` `/v1/convert/*` | Mistral `/v1/ocr` |
| Model in request | ❌ (fixed via env) | ✅ `request.model` |
| DTOs | docling pydantic re-exports | `mistralai.client.models.*` |
| Output | DoclingDocument → md/json/html/text | Mistral `OCRResponse` (pages, blocks) |
| Response "model" field | — | echoes `request.model` |

**Do NOT replace the Docling layer.** It is the layout-detection protocol that
already works. The Mistral layer is an **additional** convention for model-id
routing and OCR-style clients. Both layers share one GPU model and one
`_core` backend; enabling both means both are served on their own route
prefixes (see §6.2 for the multi-layer app decision).

## 3. Protocol reference (Mistral OCR)

### 3.1 Endpoint

```
POST /v1/ocr
```

Request body (`OCRRequest`, subset of fields we support):

```jsonc
{
  "model": "PaddlePaddle/PP-DocLayoutV3",
  "document": { "type": "document_url", "document_url": "https://..." },
  // or { "type": "image_url", "image_url": "https://..." }
  // or { "type": "file", "file_name": "x.pdf", "file_data": "<base64>" }
  "include_blocks": true,
  "image_limit": 10,
  "image_min_size": 100,
  "pages": null,
  "bbox_annotation_format": null,
  "table_format": null,
  "extract_header": false,
  "extract_footer": false,
  "confidence_scores_granularity": null
}
```

Response (`OCRResponse`):

```jsonc
{
  "model": "PaddlePaddle/PP-DocLayoutV3",
  "pages": [
    {
      "index": 0,
      "markdown": "...",
      "images": [],
      "tables": [],
      "hyperlinks": [],
      "header": null,
      "footer": null,
      "dimensions": { "width": 1000, "height": 1400, "dpi": 300 },
      "confidence_scores": { "page_number": 0, "overall_confidence": 0.9 },
      "blocks": [
        {
          "type": "text",
          "top_left_x": 120, "top_left_y": 200,
          "bottom_right_x": 920, "bottom_right_y": 420,
          "content": "...",
          "confidence": 0.92
        }
      ]
    }
  ],
  "usage_info": { "pages_processed": 1, "doc_size_bytes": 0 }
}
```

### 3.2 SDK types we use (from `mistralai`)

- `models.OCRRequest` — `model`, `document: DocumentUnion`, `pages`,
  `include_image_base64`, `image_limit`, `image_min_size`,
  `bbox_annotation_format`, `document_annotation_format`,
  `document_annotation_prompt`, `table_format`, `extract_header`,
  `extract_footer`, `include_blocks`, `confidence_scores_granularity`.
- `models.DocumentUnion` = `FileChunk | DocumentURLChunk | ImageURLChunk`.
- `models.OCRResponse` — `pages: list[OCRPageObject]`, `model`, `usage_info`.
- `models.OCRPageObject` — `index`, `markdown`, `images`, `tables`,
  `hyperlinks`, `header`, `footer`, `dimensions`, `confidence_scores`, `blocks`.
- `models.Block` is a **union** of concrete classes — never instantiate
  `Block(...)`; build the concrete type:
  `OCRTextBlock | OCRTitleBlock | OCRTableBlock | OCRImageBlock | OCRListBlock |
  OCREquationBlock | OCREaptionBlock | OCRCodeBlock | OCRHeaderBlock |
  OCRFooterBlock | OCRSignatureBlock | OCRReferencesBlock | OCRAsideTextBlock`.

**Caution**: `mistralai` DTOs are generated. Pin an exact version
(e.g. `mistralai==<tested>`) in `deploy/hps/requirements*.txt` and gate the
import so the layer is optional at import time (§6.1).

## 4. Files to create

Following the **layer's "5-file slice" shape**:

```
api_compat/
├── __init__.py                 # (edit) update package docstring
├── _core/
│   └── config.py               # (edit) HPS_API_LAYERS env var
├── web_app.py                  # (new) master app — mounts enabled layers
├── mistral_ocr_api/
│   ├── __init__.py             # package docstring
│   ├── app.py                  # create_app() + lifespan (mirrors docling_api/app.py)
│   ├── routes.py               # POST /v1/ocr
│   ├── schema.py               # thin aliases to mistralai.models
│   ├── service.py              # business logic: doc → image → detect → OCRResponse
│   └── converter/
│       ├── __init__.py
│       ├── labels.py           # PaddleX → Mistral block type mapping
│       └── service.py          # PaddleX boxes → list[OCRPageObject]
└── (existing) docling_api/      # unchanged
```

### 3.4 (Addendum) Import path note

The methodology doc's earlier plans referenced `api_compat.unstructured_api`.
The concrete proposal here uses `api_compat.mistral_ocr_api` for Mistral OCR.
The same 5-file slice rule applies; only the package name differs.

## 4. Per-file responsibilities

### `mistral_ocr_api/schema.py`

Thin module. Rather than hand-writing DTOs, **re-export** the official types so
`routes.py`/`service.py` read cleanly:

```python
"""Request/response models for the Mistral OCR layer.

Re-exports the official `mistralai` generated DTOs so we never drift from
the wire contract Mistral's own client produces.
"""
from mistralai.client import models  # pinned version, see requirements

OCRRequest = models.OCRRequest
OCRResponse = models.OCRResponse
DocumentUnion = models.DocumentUnion
DocumentURLChunk = models.DocumentURLChunk
ImageURLChunk = models.ImageURLChunk
FileChunk = models.FileChunk
```

### `mistral_ocr_api/converter/labels.py`

Maps PP-DocLayoutV3's 25 labels to Mistral block types:

| PaddleX label | Mistral block type |
|---------------|--------------------|
| `text`, `content`, `abstract`, `number`, `reference`, `reference_content`, `formula_number`, `aside_text`, `vertical_text` | `OCRTextBlock` |
| `doc_title`, `paragraph_title`, `figure_title` | `OCRTitleBlock` |
| `table` | `OCRTableBlock` |
| `image`, `header_image`, `footer_image`, `seal` | `OCRImageBlock` |
| `chart` | `OCRImageBlock` (fallback) |
| `display_formula`, `inline_formula` | `OCREquationBlock` |
| `header` | `OCRHeaderBlock` |
| `footer` | `OCRFooterBlock` |
| `footnote`, `vision_footnote` | `OCRCaptionBlock` (fallback) |
| `algorithm` | `OCRCodeBlock` |

The exact set is data, not logic — kept in `labels.py` for auditability,
mirroring `docling_api/converter/labels.py`.

### `mistral_ocr_api/converter/service.py`

Stateless `PaddleXToMistralConverter`:

- `convert(boxes, image, page_no, include_blocks)` → `OCRPageObject`.
- Sort boxes by reading order (`_sort_by_reading_order`, same logic as docling
  converter).
- For each box, `xmin,ymin,xmax,ymax = box["coordinate"]` → build the concrete
  block (`OCRTextBlock`, `OCRTitleBlock`, ...) with `top_left_*`, `bottom_right_*`,
  `content`, `confidence=box.get("score")`.
- `content`: for layout-only we have no OCR text; emit the placeholder text
  (e.g. `[text]`, `[title]`, `[table]`) the same way the docling converter does.
  Future PaddleOCR-VL can fill real text here.
- Populate `dimensions` from `image.shape`.
- Populate `confidence_scores` from box scores (mean).

### `mistral_ocr_api/service.py`

Orchestration (mirrors `docling_api/service.py`):

```python
async def process_ocr(request) -> OCRResponse:
    # 1. Resolve document → image bytes
    doc = request.document
    if isinstance(doc, (dict, DocumentURLChunk)):
        image_bytes = await fetch_image_from_url(doc["document_url"])
    elif ImageURLChunk: ...
    elif FileChunk: image_bytes = base64.b64decode(doc["file_data"])
    else: raise HTTPException(400, "unsupported document type")

    # 2. Decode + resize (reuses _core.image)
    image = await loop.run_in_executor(pool, load_image_from_bytes, image_bytes)

    # 3. Detect (reuses _core.inference, honors request.model via resolver)
    model = resolve_model(request.model)          # §5
    boxes = await run_layout_detection(image)     # same backend

    # 4. Convert to OCRResponse
    page = converter.convert(boxes, image, page_no=0, include_blocks=request.include_blocks)
    return OCRResponse(model=request.model or DEFAULT_MODEL, pages=[page], usage_info=...)
```

### `mistral_ocr_api/routes.py`

```python
router = APIRouter(tags=["ocr"])

@router.post("/v1/ocr", response_model=OCRResponse)
async def ocr(request: OCRRequest) -> OCRResponse:
    if not state.ready:
        raise HTTPException(503, "model not ready")
    return await process_ocr(request)
```

### `mistral_ocr_api/app.py`

`create_app()` + `lifespan` — a near-copy of `docling_api/app.py`: same
`lifespan` (starts/shuts the shared `_core.inference.state`), same CORS,
mounts `health_router` + `ocr_router`.

## 5. Model-id dispatch (`request.model`)

**Decision (confirmed)**: The backend is a **single PaddleX model** today
(`HPS_API_MODEL`). `request.model` does **not** yet route to different
backends — it is **validated + echoed**, never rejected. On a mismatch we log
a warning and serve the configured model. This keeps clients written against a
model id working even before multi-model registration exists.

```python
def resolve_model(model: str | None) -> str:
    """Return the effective backend model for a request.

    Today: single configured model — any non-empty model id is accepted and
    echoed back. When multiple models/backends exist, this becomes a registry
    lookup (name → InferenceBackend instance) and run_layout_detection gains a
    backend selector.
    """
    configured = MODEL_NAME
    if model and model != configured:
        logger.warning("model=%r requested, serving %r", model, configured)
    return configured
```

Future multi-model (out of scope): a `model_registry` in `_core` mapping
`model id → InferenceBackend`, and `run_layout_detection(image, model=...)`.
The Mistral layer already carries the id end-to-end, so the router is ready.

## 6. Enable / disable API layers

### 6.1 Config: `HPS_API_LAYERS`

Add to `_core/config.py` (the single source of truth for env reads):

```python
# Which API layers to mount.  Comma-separated subset of:
#   docling   -> POST /v1/convert/*        (default, always-safe)
#   mistral   -> POST /v1/ocr              (requires mistralai installed)
#   (future)  -> unstructured, triton-infer, openai-chat, ...
# Empty/absent -> "docling" (backwards compatible).
API_LAYERS = [
    s.strip() for s in os.environ.get("HPS_API_LAYERS", "docling").split(",")
    if s.strip()
]
```

Usage examples:

```bash
# Docling only (today's default — no code change for existing deploys)
HPS_API_LAYERS=docling

# Docling + Mistral OCR
HPS_API_LAYERS=docling,mistral

# Mistral only
HPS_API_LAYERS=mistral
```

### 6.2 Single app, conditional routers (DECISION: this is the default)

**Decision (confirmed)**: keep **one** master app that mounts routers
conditionally. This preserves the existing single-port/single-process
deployment (health + whichever layers) and avoids running N Granian processes
per protocol.

```python
# api_compat/web_app.py  (new) — the "master" app
def create_app() -> FastAPI:
    app = FastAPI(...)
    app.include_router(health_router)
    for layer in API_LAYERS:
        if layer == "docling":
            from .docling_api.routes import router as r
            app.include_router(r)
        elif layer == "mistral":
            from .mistral_ocr_api.routes import router as r
            app.include_router(r)
        else:
            logger.warning("unknown API layer %r, ignoring", layer)
    return app

app = create_app()  # module-level for Granian
```

**Missing-dependency semantics (decision)**: `mistral_ocr_api/schema.py`
imports `mistralai` lazily in a `try/except`. If `mistralai` is **not installed**
and `mistral` is in `HPS_API_LAYERS`, the master app **fails fast at startup**
with a clear error (`ImportError: HPS_API_LAYERS includes 'mistral' but
'mistralai' is not installed`). It must **not** silently drop the layer — that
would hide a misconfiguration.

### 6.3 Per-layer apps still available

Keep `api_compat/<layer>/app.py:create_app()` for those who want a dedicated
process/port per protocol (e.g. different scaling). The run script defaults to
the master app but allows override (§6.4).

### 6.4 Run script (`scripts/run_api.sh`)

Add an `HPS_API_LAYER`/`HPS_API_LAYERS` passthrough and a target-app override:

```bash
# Existing default (docling) stays the same:
exec granian ... "api_compat.web_app.app:app"

# To run Mistral in its own process on another port:
HPS_API_LAYERS=mistral PORT=8081 granian ... "api_compat.mistral_ocr_api.app:app"
```

The master app is backward-compatible: with `HPS_API_LAYERS` unset it behaves
exactly like today's docling app.

## 7. Compatibility test (using the official client)

Place in `deploy/hps/tests/test_mistral_compat.py`:

```python
from mistralai import Mistral

client = Mistral(
    api_key="unused",
    server_url="http://localhost:8080",   # our server, NOT Mistral's
)

res = client.ocr.process(
    model="PaddlePaddle/PP-DocLayoutV3",
    document={"type": "image_url", "image_url": "https://example.com/a.png"},
    include_blocks=True,
)
assert res.model == "PaddlePaddle/PP-DocLayoutV3"
assert len(res.pages) >= 1
if res.pages[0].blocks:
    b = res.pages[0].blocks[0]
    assert b.top_left_x is not None
```

This proves our server speaks the exact Mistral wire contract, since it is the
official SDK hitting us via `server_url`.

## 8. Design principles (why this is clean)

- **No hand-written DTOs** — import `mistralai.client.models` (pinned) so the
  contract matches Mistral's generated client by construction.
- **Reuses `_core` wholesale** — config, engine, image, inference, health. Only
  the converter + routes are protocol-specific.
- **Single backend, many protocols** — both layers drive the same
  `_core.state`/`run_layout_detection`; enabling Mistral does not add a second
  model or GPU context.
- **Opt-in** — `HPS_API_LAYERS` defaults to `docling`, so existing deploys and
  the current smoke tests are unaffected.
- **Forward-compatible dispatch** — `request.model` is threaded through now so
  future multi-model registration needs no API change.

## 9. Rollout checklist

- [ ] Pin `mistralai==<tested>` in `deploy/hps/requirements.txt`
- [ ] `_core/config.py`: add `API_LAYERS`
- [ ] `api_compat/web_app.py`: conditional master app (default layer app)
- [ ] `mistral_ocr_api/` slice (schema, converter, service, routes, app)
- [ ] `scripts/run_api.sh`: passthrough + target-app override
- [ ] Smoke: `HPS_API_LAYERS=docling,mistral granian ... api_compat.web_app.app:app`
- [ ] `/health`, `/v1/ocr` sanity; `/v1/convert/file` still works
- [ ] Official-client compat test (with `server_url` override)
- [ ] Update `docs/API_REFERENCE.md` + methodology doc

## 10. Decisions log

| # | Decision | Value |
|---|----------|-------|
| D1 | Default enable | `HPS_API_LAYERS` defaults to `docling` (backward compatible) |
| D2 | Layer mounting | Single master app `web_app.py` mounts routers conditionally (default); per-layer apps remain for dedicated ports |
| D3 | `request.model` mismatch | warn + echo configured model; never reject |
| D4 | Missing `mistralai` | fail fast at startup if `mistral` requested but not installed |
| D5 | DTOs | import `mistralai.client.models` (pinned), never hand-write |
| D6 | Block construction | concrete block classes; never `Block` union |

Live progress is tracked in
`docs/scratchpads/mistral_ocr_layer_session.md`.