# Session Scratchpad — Mistral OCR API Layer

Tracking doc for implementing the Mistral OCR API layer in
`deploy/hps/api_compat/`. Update as work progresses.

## Goal
Add a Mistral-compatible `/v1/ocr` API layer with enable/disable selection,
reusing `_core` infrastructure. Design doc:
`docs/design/mistral_ocr_api_layer.md`.

## Decisions (confirmed)
1. **Master app** `api_compat/web_app.py` mounts health + layer routers
   conditionally based on `HPS_API_LAYERS`. Default = `docling` (backward
   compatible).
2. **`request.model` mismatch**: **warn + echo** (do NOT reject) — single
   configured model today. Isolate in `resolve_model()` for future registry.
3. **DTOs**: import official `mistralai.client.models`, pinned. Gate import so
   layer is optional at import time (missing `mistralai` → layer unavailable).
4. **App factory shape**: `create_app()` + module-level `app` (Granian).
5. **Mistral block types**: build concrete classes, never `Block` union.
6. **Docling layer keeps its own `app.py`** (unchanged). `web_app.py` is the
   new master.

## File map
```
_core/config.py                  (edit) add API_LAYERS
api_compat/web_app.py            (new) master app
api_compat/__init__.py           (edit) docstring
api_compat/mistral_ocr_api/
    __init__.py                  (new)
    app.py                       (new) create_app + lifespan
    routes.py                    (new) POST /v1/ocr
    schema.py                    (new) re-export mistralai models
    service.py                   (new) orchestration
    converter/
        __init__.py
        labels.py                (new) PaddleX->Mistral type map
        service.py               (new) boxes -> OCRPageObject
scripts/run_api.sh               (edit) passthrough + app override
deploy/hps/requirements.txt      (edit) pin mistralai
```

## Key API facts (from reading source)
- `_core/inference.py`:
  - `state` = AppState singleton (`state.start_inference()`, `state.wait_ready(timeout)`, `state.shutdown()`, `state.ready`, `state.backend.detect(image)`)
  - `run_layout_detection(image) -> list[dict]` where each dict has
    `label`, `score`, `coordinate:[xmin,ymin,xmax,ymax]`, `order`, `cls_id`, `polygon_points`.
- `_core/image.py`: `load_image_from_bytes(bytes) -> np.ndarray`, `fetch_image_from_url(url) -> bytes`.
- `_core/config.py`: `MODEL_NAME`, `CPU_POOL_SIZE`, `STARTUP_TIMEOUT`, `INFERENCE_BACKEND`, `setup_logging()`.
- `_core/health/routes.py`: `router` with /health /ready /v1/models.
- `docling_api/converter/service.py`: `_sort_by_reading_order(boxes)` logic, PADDLEX label handling.
- `docling_api/app.py`: lifespan pattern to copy.

## Gotchas
- `mistralai` DTO import must be lazy/optional; wrap in try/except in schema.py.
- `resolve_model`: warn+echo, never reject.
- Converter `content`: layout-only → placeholder text like docling (`[text]`).
- Keep single module-level `app = create_app()` per app file for Granian.

## Status
- [x] scratchpad created
- [x] design doc adjusted
- [x] config API_LAYERS
- [x] web_app.py
- [x] mistral slice
- [x] run_api.sh
- [x] __init__ docstring
- [x] requirements pin (mistralai==2.9.3)
- [x] verify imports — master app builds (docling default OK, docling,mistral OK),
      fail-fast ImportError when mistralai missing, all schema/converter tests pass
- [x] end-to-end server smoke test (2026-08-18) — mistral-only server (HPS_API_LAYERS=mistral,
      HPS_API_TARGET=api_compat.mistral_ocr_api.app:app, direct backend, fp16). Engine built
      fresh (54s, 70.6MB). Official mistralai client (`from mistralai.client import Mistral`)
      works: client.ocr.process(model=..., document=DocumentURLChunk(...)).
      Verified: docling routes absent (404); model warn+echo; include_blocks=False; image_url
      chunk; unsupported type → 422; FileChunk (file_id) → 400 "file_data missing".
- [ ] (gap) FileChunk: server expects inline `file_data` base64, but official client sends
      `file_id` reference (needs a files-upload endpoint we don't expose). document_url +
      image_url both work today.

## E2E test client
`tests/test_mistral_ocr_client.py` — uses official `mistralai` client. Run with
`.venv-cuda13-py310/bin/python tests/test_mistral_ocr_client.py --url http://localhost:8080`.
Needs a local image server (e.g. `python3 -m http.server 9090` in tests/mig_inference/input).
Note: `Mistral` is at `mistralai.client.Mistral` (NOT `from mistralai import Mistral`); the
OCR sub-SDK method is `client.ocr.process(...)`.