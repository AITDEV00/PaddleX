# Feasibility — extend Mistral OCR request/response models via inheritance + extra fields

**Date:** 2026-08-20 · **mistralai SDK:** 2.9.3 (pydantic v2, Speakeasy-generated) · **Server:** PaddleX HPS `mistral_ocr_api`

**Question:** Can we *inherit and extend* the official Mistral `OCRRequest` / `OCRResponse`
models with extra fields (threshold, native `label`, `score`), pass them through the official
Mistral client, and stay wire-compatible?

**Short answer: Partially. It works cleanly on the server (output) side. It does NOT work on the
official-client (input) side for arbitrary fields. But one of the three blockers — confidence
scores — has a fully native, zero-client-change solution the requester appears not to know about.**

---

## 0. What the DTOs actually are

`OCRRequest`, `OCRResponse`, `OCRPageObject`, and every block class are **pydantic v2
`BaseModel`s** with `model_config.extra = None` (i.e. unknown fields are **ignored**, not kept).

```python
OCRRequest MRO: ['OCRRequest', 'BaseModel', 'BaseModel', 'object']
extra: None
```

Two SDK facts dominate the whole analysis:

1. **`OCRRequest` carries a hand-written `@model_serializer`** that iterates
   `type(self).model_fields` and emits **subclass-added fields too**.
2. **The OCR *resource* is what you call, not the model class.** The wire request body is built
   by `Ocr.process()` from a **fixed keyword signature** — it constructs `models.OCRRequest(...)`
   internally from scalar args. There is **no `extra_body` / raw dict / subclass passthrough**.

---

## Direction 1 — Client → Server (sending custom params like `threshold`)

### Finding: NOT possible with the unmodified official client

`Mistral.ocr.process()` has a fixed signature (16 keyword params + retries/server_url/timeout/http_headers)
and builds the request internally:

```python
request = models.OCRRequest(model=model, document=..., pages=..., ...)  # fixed
req = self._build_request(..., request=request, ...)
```

- There is **no `**kwargs` / `extra_body` / arbitrary-dict escape hatch**.
- Even if we subclass `OCRRequest` and build it ourselves, **we cannot hand it to `process()`**
  — the method re-constructs a base `OCRRequest` from scalars and discards ours.

**Subclassing the request model therefore buys nothing on the input side.** Tested: a subclass
instance serialized through the SDK's `marshal_json` loses its set custom values (they appear as
`null`) — it is validated back to the base `OCRRequest`.

### What it would take (all require a client-side change; none are "free")

| Option | Description | Client change |
|---|---|---|
| **A. Vendor/patch `Ocr.process`** | Fork or monkeypatch the resource method to inject `threshold` into the request before `_build_request`. | Patch/overlay the SDK in ECAS |
| **B. Call raw HTTP** | Drop the SDK for this call; send the JSON body yourself (the report already shows a working curl). | Replace `Mistral.ocr.process` with `httpx.post` |
| **C. LiteLLM native SDK** | Use LiteLLM's own client which supports `drop_params` / extra body. | Switch client library |
| **D. Deployment default** | Set `threshold` server-side at model load (per the earlier finding, this is where it's hardcoded). | None for client — but per-request control is lost |

> The requester explicitly wants **per-request** thresholds (different document classes, different
> operating points), which rules out **D** alone. That forces **A/B/C** — a real client-side change.

---

## 2. Response outbound — Server produces extra fields

### Finding: fully supported, trivially

On the **server** side we construct the response objects ourselves, so subclassing works cleanly:

```python
class RichTextBlock(models.OCRTextBlock):
    label: str | None = None
    score: float | None = None
```

This serializes correctly: `{"top_left_x":...,"type":"text","label":"chart","score":0.91}`.
The custom `@model_serializer` iterates `type(self).model_fields`, so subclass fields flow out.

**Compatibility:** the official client **parses it without error** and keeps all *known* fields.
It **drops the unknown extras** (silently — `__pydantic_extra__` stays empty, `extra=None`).

**Consequence:** if ECAS uses the **unmodified official client**, extra `label`/`score` fields are
stripped before they reach the caller. They would only appear if ECAS also vendors/patchs the
client to `extra="allow"` (or reads raw JSON / `UnknownBlock`).

---

## 3 — Response inbound — Official client reads the extra fields

Three distinct sub-cases, each tested empirically:

### 3a. Custom `label`/`score` as **extra** block fields
- Server emits `"label":"chart","score":0.91` on a `type:"text"` block.
- Official client parses to `OCRTextBlock`, **silently drops** `label`/`score`
  (`__pydantic_extra__` empty). ❌ Not visible to ECAS without a client patch.

### 3b. Native **`confidence_scores`** block/page field — ✅ **Fully round-trips, no client change**

`OCRBlockConfidenceScores` and `OCRPageConfidenceScores` are **first-class Mistral fields**:

```python
OCRBlockConfidenceScores:
    average_content_confidence_score: float | None
    minimum_content_confidence_score: float | None
    block_type_confidence_score:    float | None    # ← maps to PaddleX per-box `score`
```

Tested through the official client parser:
```
parsed block: OCRTextBlock
has confidence_scores: True   (block_type_confidence_score=0.88)   # ✅ KEPT
```
This is the **native, lossless path for Blocker 2**. The official `confidence_scores_granularity`
request field even documents: `"block"` = per-block scores, `"page"` = aggregate.

> Note: for layout-only models there are no *word* scores; we'd populate
> `block_type_confidence_score` (per box) and optionally `average/min` per page.

### 3c. Custom `type` value (e.g. `"chart"`) to carry native label — ✅ parses, but shape changes

Sending `"type":"chart"` (not in Mistral's union) → official client returns an **`UnknownBlock`**:

```python
type: "UNKNOWN"
raw: {"type":"chart","top_left_x":...,"content":"[chart]","score":0.91,"label":"chart"}
is_unknown: True
```

- No parse error — officially tolerated.
- But the block shape collapses to `UnknownBlock.raw` (a `Any` dict), so it is **not a drop-in
  `OCRTextBlock`** and would require every consumer to handle unknown blocks. Effectively a
  **breaking** response shape. Not recommended for the label distinction.

---

## 4. Requirement-by-requirement verdict

| Report blocker | Desired | Native Mistral path? | Feasible via inherit+extra? | Verdict |
|---|---|---|---|---|
| **1. threshold 0.3** | per-request detection threshold | No field exists | **No** on client side (fixed `process()` sig) | **Needs client change** (patch/vendor `process`, raw HTTP, or LiteLLM SDK) |
| **2. per-block score** | numeric `score` on blocks | ✅ **Yes** — `confidence_scores.block_type_confidence_score` (round-trips via official client) | Possible but unnecessary | **Recommended: native `confidence_scores`** — zero client change |
| **3. native label** (`chart`≠`image`, `number`≠`text`) | native label on block | No clean native field | Server-side easy; **stripped by official client** | **Needs client patch** (`extra="allow"`) or `UnknownBlock` (breaking) |

---

## 5. Conclusion / recommendation

**"Inherit and extend" is the right pattern for the *server's* output construction, but it is
*not* sufficient to get the data to a stock official Mistral client.**

Specifically:

1. **Do use subclassing on the server** to emit `label` and `score` (and pass `threshold` through
   the request schema via a server-side-extended `OCRRequest`). It's clean, safe, and lets the
   *same* server serve both strict-Mistral clients (they ignore extras) and extended clients.

2. **Do NOT rely on it to reach stock official clients.** Unknown fields are silently dropped,
   and the client cannot *send* custom fields either.

3. **Ship Blocker 2 through the native `confidence_scores` field** — it round-trips through the
   unmodified official client today. This is the one genuinely free win.

4. **Blocker 1 and 3 fundamentally require a client change** (vendor the SDK's `process` to
   inject `threshold` and to read `label` via `extra="allow"`), because the wire needs a custom
   field the SDK won't emit/consume on its own. The server changes alone are necessary but not
   sufficient.

### Concrete recommended implementation (server side)
- Extend the request model: `class PaddleXOCRRequest(OCRRequest): threshold: float|None = None`
  and capture it in `process_ocr()` → thread through `run_layout_detection()` →
  `_run_postprocessing()` (replace the hardcoded `predictor.threshold`).
- Populate native `confidence_scores.block_type_confidence_score` per block (solves Blocker 2
  for everyone, no client change).
- Add `label` as an **extra** block field (server-side subclass) for extended clients, keep
  `type` Mistral-compatible; document that stock clients must enable `extra="allow"`.
- Optionally expose a server-side **deployment threshold default** (env `HPS_API_THRESHOLD`) so
  teams can at least shift the operating point without per-request fields.