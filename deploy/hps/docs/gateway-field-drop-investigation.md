# Gateway Field-Drop Investigation — why LiteLLM drops the PaddleX extensions

**Status:** OPEN · needs human/operator inspection of the LiteLLM deployment
**Date:** 2026-08-20
**Author:** HPS mistral-OCR integration
**Scope:** this document is a **problem statement + inspection checklist**. It
does **not** contain conclusions about the gateway internals — those must be
verified by the reader against the live LiteLLM deployment at
`/home/jyao/ADEO/service/litellm` and the running proxy.

---

## 1. Problem statement

When a request reaches the PaddleX HPS backend **directly** (no gateway), all
three PaddleX-extension behaviors work:

| Field | Direct backend result |
|-------|----------------------|
| `confidence_scores_granularity="block"` (stock Mistral) | 16/16 blocks carry scores |
| `include_native_labels=true` (extension) | 16/16 blocks carry a native `label` |
| `threshold=0.3..0.99` (extension) | block count changes with the value |

When the **same request** is sent through the LiteLLM gateway
(`https://litellm.ecouncil.ae/v1/ocr`, key `sk-05132025`), the result differs:

| Field | Via gateway result | Behaviour |
|-------|--------------------|-----------|
| `confidence_scores_granularity="block"` (stock Mistral) | 16/16 blocks carry scores | **FORWARDED** |
| `include_native_labels=true` (extension) | 0/16 blocks carry a `label` | **DROPPED** |
| `threshold=0.3/0.5/0.7/0.8/0.99` (extension) | 16 blocks at every value | **DROPPED** |

**Concluding statement:** the gateway drops the two PaddleX-extension top-level
request fields (`include_native_labels`, `threshold`), while it forwards the
stock Mistral field (`confidence_scores_granularity`). Because the backend
honours all three when hit directly, the drop is **attributable to the LiteLLM
gateway layer**, not the backend. The exact mechanism is **not** determined
here; the goal of this document is to let an operator find it.

---

## 2. What we already know (facts, verified directly against the backend)

1. The backend `deploy/hps/api_compat/mistral_ocr_api/schema.py` declares these
   fields on `PaddleXOCRRequest`:
   - `threshold: Optional[Any]` (default `None`)
   - `include_native_labels: Optional[bool]` (default `False`)
   - `layout_*`, `include_paddlex_metadata`, `filter_overlap_boxes`, etc.
2. `service.process_ocr` reads them via `as_optional(getattr(request, ...))`
   and threads them into `ConvertOptions` / `run_layout_detection`.
3. Stock Mistral field `confidence_scores_granularity` IS part of the official
   `mistralai` `OCRRequest` DTO (verified in
   `mistralai/client/models/ocrrequest.py`), so it is expected to pass through
   a schema-aware gateway.
4. The extension fields are **not** part of the official `OCRRequest` DTO. A
   LiteLLM proxy that validates/strips the request body against the official
   schema would therefore remove them before forwarding to the backend.

---

## 3. Hypotheses to confirm/reject (each with a way to test)

| # | Hypothesis | How to confirm |
|---|------------|----------------|
| H1 | LiteLLM validates the request body against the stock Mistral OCR schema and strips unknown top-level fields before forwarding. | Inspect whether the `/v1/ocr` route applies a pydantic model/schema to the request body, and whether it serializes back only known fields. |
| H2 | The `PP-DocLayoutV3` route/model entry has a passthrough or "allowed params" allow-list that excludes the extensions. | Inspect the LiteLLM config (`config.yaml` / model_list entry for `PP-DocLayoutV3`) for an allow-list, `allowed_*`, or passthrough keys. |
| H3 | A LiteLLM middleware / `convert_*` hook rewrites or prunes the body before the request reaches the provider call. | Grep the codebase for where the `/v1/ocr` request body is read and where it is forwarded, and whether any transform drops keys. |
| H4 | The gateway is not actually running our latest backend image (the one with `include_native_labels`) — but this is UNLIKELY since `confidence_scores_granularity` works. | Confirm the backend image/digest the gateway routes to and that it includes the Option-A code. |
| H5 | LiteLLM requires the extension fields to be declared in the model's `request_options`/custom `request_*` config to pass them through. | Check the `PP-DocLayoutV3` model config for `request_options`/custom body overrides. |

---

## 4. Inspection checklist (run on `/home/jyao/ADEO/service/litellm`)

### 4.1 Locate the OCR route handler
- Find where `/v1/ocr` is routed: grep `aocr`, `ocr`, `"/v1/ocr"` in
  `litellm/main.py` and `litellm/router.py`.
- Identify the request model class used for the OCR route (if any). It likely
  lives under `litellm/ocr/` (e.g. `ocr/request.py`, `ocr/types/...`).
- Print the fields of that request model:
  ```bash
  grep -rn "class.*OCRRequest\|class.*OCROcrV1\|include_native_labels\|confidence_scores_granularity" litellm/ocr/
  ```

### 4.2 Determine whether unknown fields are stripped
- Look at how the request model is converted to a dict for the upstream call
  (`model_dump(..., exclude_unset=...)`, `exclude_none=`, or manual field
  assembly). If it uses an explicit field list or `exclude_unset`, extension
  fields sent by the caller are dropped unless they are model fields.
- Search for `exclude_unset`, `exclude_none`, `model_dump`, `.get(` on the
  OCR request object in the handler.

### 4.3 Check the `PP-DocLayoutV3` model config on the gateway
- Open the config file the proxy loaded (e.g. `config.yaml`, or whatever
  `--config` points to at startup).
- Locate the `PP-DocLayoutV3` model group entry.
- Look for: `allowed_parameters` / `allowed_arguments`, `passthrough`,
  `request_options`, `additional_headers`, `drop_params`, or any body
  transformation. Note whether it differs from the `Qwen/...` entries that
  succeed with extra fields.

### 4.4 Enable debug logging / trace the request
- Run (or inspect) the proxy with `LITELLM_LOG=DEBUG` (or `--detailed_debug`).
- Send one OCR request with `include_native_labels=true` and `threshold=0.3`,
  then read the logs for the incoming body and the body sent upstream to
  `PP-DocLayoutV3`. Diff the two: the field that disappears is the one LiteLLM
  dropped and the log line naming it will show the mechanism (validation vs.
  allow-list vs. transform).

### 4.5 Verify the backend version behind the gateway
- Confirm the gateway forwards to an HPS pod running the image that includes
  the Option-A schema. Quick check: hit the backend `/v1/ocr` directly (bypass
  gateway) with `include_native_labels=true` — if it returns labels, the
  backend is fine and the drop is upstream.

---

## 5. Reproducer (for whoever inspects)

Save the image to a file first, then send the two field-sets through the
gateway:

```bash
# image path: deploy/hps/tests/mig_inference/input/08_document_scan.jpg
DATA_URI="data:image/jpeg;base64,$(base64 -w0 deploy/hps/tests/mig_inference/input/08_document_scan.jpg)"

# STOCK field (expected: forwarded -> 16/16 with confidence)
curl -sk https://litellm.ecouncil.ae/v1/ocr -H 'Authorization: Bearer sk-05132025' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"PP-DocLayoutV3\",\"document\":{\"type\":\"image_url\",\"image_url\":\"$DATA_URI\"},\"include_blocks\":true,\"confidence_scores_granularity\":\"block\"}"

# EXTENSION field (expected: dropped -> 0/16 labels)
curl -sk https://litellm.ecouncil.ae/v1/ocr -H 'Authorization: Bearer sk-05132025' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"PP-DocLayoutV3\",\"document\":{\"type\":\"image_url\",\"image_url\":\"$DATA_URI\"},\"include_blocks\":true,\"include_native_labels\":true}"

# EXTENSION field (expected: dropped -> 16 blocks)
curl -sk https://litellm.ecouncil.ae/v1/ocr -H 'Authorization: Bearer sk-05132025' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"PP-DocLayoutV3\",\"document\":{\"type\":\"image_url\",\"image_url\":\"$DATA_URI\"},\"include_blocks\":true,\"threshold\":0.99}"
```

The reusable probe is also committed at `deploy/hps/tests/test_gateway_ocr.py`.

---

## 6. What a fix will look like (once the mechanism is found)

- If **H1/H3 (schema validation strips unknowns)**: add the extension fields
  to the OCR request model LiteLLM uses (or switch to a passthrough body for
  the `PP-DocLayoutV3` route) so `include_native_labels`/`threshold` survive.
- If **H2 (allow-list)**: add the fields to the model entry's allow-list.
- If **H5 (request_options)**: declare the fields under the route's
  `request_options` so LiteLLM forwards them verbatim.

The backend needs **no change**; it already declares and honours the fields.

---

## 7. Blocking question for the operator
Does the LiteLLM gateway that serves `https://litellm.ecouncil.ae/v1/ocr`
intend to pass arbitrary PaddleX-extension request fields through to the
`PP-DocLayoutV3` backend? If yes, the extension fields must be allow-listed /
added to the OCR request schema at the gateway. If no, callers must hit the
backend directly (or via a gateway that is configured to passthrough) to use
`include_native_labels`/`threshold`.