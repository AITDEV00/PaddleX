# PP-DocLayoutV3 OCR — Mistral Client Usage Guide

> **Endpoint:** `https://litellm.adeoaiengine.ecouncil.ae`
> **Model:** `PP-DocLayoutV3` (exposed by the PaddleX HPS backend through LiteLLM)
> **Compatible protocol:** Mistral OCR (`POST /v1/ocr`)
> **Date verified:** 2026-08-19

This guide documents how to call the PaddleX `PP-DocLayoutV3` layout-detection
service using the **official `mistralai` Python client** through the LiteLLM
gateway, and which Mistral OCR capabilities are (and are not) supported.

---

## 1. Prerequisites

```bash
pip install mistralai httpx
```

You need a LiteLLM **bearer token** (an API key) for the gateway. Replace
`<your api key here>` in every example below with your actual key.

> **SSL note:** the gateway uses a self-signed certificate. Every example
> passes `httpx.Client(verify=False)` to the `Mistral` client, which is the
> equivalent of `curl -k`. Do **not** rely on this in production.

---

## 2. Minimum working example

```python
import httpx
from mistralai.client import Mistral
from mistralai.client.models import ImageURLChunk

GATEWAY = "https://litellm.adeoaiengine.ecouncil.ae"   # NOT ".../v1" — see §3
API_KEY = "<your api key here>"

# SSL bypass mirrors `curl -k` (self-signed gateway cert)
client = Mistral(
    api_key=API_KEY,
    server_url=GATEWAY,
    client=httpx.Client(verify=False),
)

# 1) Build a data-URI for your image (or point at an http(s):// URL)
import base64
with open("business_letter.png", "rb") as f:
    data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()

# 2) Run OCR / layout detection
resp = client.ocr.process(
    model="PP-DocLayoutV3",
    document=ImageURLChunk(image_url=data_uri),
    include_blocks=True,
)

# 3) Inspect the result
for page in resp.pages:
    print(f"page[{page.index}] {page.dimensions}")
    for block in page.blocks:
        print(
            f"  {type(block).__name__:<18} "
            f"bbox=({block.top_left_x},{block.top_left_y},"
            f"{block.bottom_right_x},{block.bottom_right_y}) "
            f"content={block.content!r}"
        )
print(resp.usage_info)
```

**Expected output (business_letter.png):**

```
page[0] dpi=300 height=580 width=626
  OCRTitleBlock    bbox=(40,79,466,148)   content='[title]'
  OCRTextBlock     bbox=(46,100,368,187)  content='[text]'
  OCRTextBlock     bbox=(128,194,555,271) content='[text]'
  OCRImageBlock    bbox=(41,194,127,260)  content='[image]'
  OCRTableBlock    bbox=(36,272,614,580)  content='[table]'
pages_processed=1 doc_size_bytes=211754
```

---

## 3. Important: `server_url` must NOT include `/v1`

The official `mistralai` SDK **always appends `/v1/ocr`** to `server_url`. If
you set `server_url = "https://litellm.adeoaiengine.ecouncil.ae/v1"`, the SDK
will POST to `https://litellm.adeoaiengine.ecouncil.ae/v1/v1/ocr` → **404**.

| `server_url` value | Resulting request | Outcome |
|---|---|---|
| `https://litellm.adeoaiengine.ecouncil.ae` | `/v1/ocr` | ✅ 200 |
| `https://litellm.adeoaiengine.ecouncil.ae/v1` | `/v1/v1/ocr` | ❌ 404 |

Always set `server_url` to the **host root** (no `/v1`).

---

## 4. How to pass the document image

The PaddleX backend accepts a URL that **the server itself resolves**. It does
**not** have a file-upload store (`/v1/files`), so you pass the image inline.

### 4.1 Data URI (recommended for uploads / small images)

```python
from mistralai.client.models import ImageURLChunk

data_uri = "data:image/png;base64," + base64.b64encode(open("img.png", "rb").read()).decode()
resp = client.ocr.process(
    model="PP-DocLayoutV3",
    document=ImageURLChunk(image_url=data_uri),
)
```

- ✅ **Base64** payloads: `data:image/png;base64,...`
- ✅ Percent-encoded payloads: `data:image/png,%89PNG...`
- The backend decodes `data:` URIs locally (added 2026-08-19); this is exactly
  what LiteLLM produces when a user uploads a file through its `/v1/ocr`.

### 4.2 `document_url` (URL the server fetches)

```python
from mistralai.client.models import DocumentURLChunk

resp = client.ocr.process(
    model="PP-DocLayoutV3",
    document=DocumentURLChunk(document_url="http://host/image.png"),
)
```

The backend fetches the URL with `httpx`. Note the server must be able to reach
the URL (a pod without egress can only reach in-cluster URLs).

### 4.3 `image_url` — use the string form

`image_url` works **as a plain string**:

```python
ImageURLChunk(image_url=data_uri)      # ✅ works (string)
```

It does **NOT** work as a dict through the LiteLLM gateway:

```python
ImageURLChunk(image_url={"url": data_uri})   # ❌ LiteLLM 500 InternalServerError
```

The dict form is a **LiteLLM-side limitation** (it mishandles `{"url": ...}`)
and never reaches PaddleX. Prefer `ImageURLChunk(image_url=<string>)` or
`DocumentURLChunk(document_url=<string>)`.

---

## 5. Supported `ocr.process` parameters

Tested against the live gateway. **All params below are accepted** (HTTP 200).
"accepted" means they do not error; whether they change output is noted.

| Parameter | Accepted | Effect on PP-DocLayoutV3 |
|---|---|---|
| `model` | ✅ | `"PP-DocLayoutV3"` (see §7 for model list) |
| `document` | ✅ | see §4 |
| `include_blocks` | ✅ | `True` → populated `blocks`; `False` → empty |
| `pages=[0]` | ✅ | single page index honored |
| `include_image_base64` | ✅ | accepted; images array is always empty (see §6) |
| `image_limit` | ✅ | accepted |
| `image_min_size` | ✅ | accepted |
| `table_format="markdown"` | ✅ | accepted |
| `table_format="html"` | ✅ | accepted |
| `extract_header` | ✅ | accepted |
| `extract_footer` | ✅ | accepted |
| `confidence_scores_granularity="word"/"page"/"block"` | ✅ | accepted |
| `document_annotation_prompt` | ✅ | accepted |
| `bbox_annotation_format` | ⚠️ | requires a `ResponseFormat` object; see §6 |
| `document_annotation_format` | ⚠️ | requires a `ResponseFormat` object; see §6 |

> ⚠️ `bbox_annotation_format` / `document_annotation_format` are typed by the
> SDK as `ResponseFormat`. Passing a raw string raises a client-side
> `ValidationError`. There is no supported value that changes PaddleX output
> today, so omit them unless you have a concrete `ResponseFormat`.

---

## 6. Capability boundaries of `PP-DocLayoutV3`

`PP-DocLayoutV3` is a **layout-detection model**. It detects **regions** (text,
titles, tables, images, equations, …) with bounding boxes. It does **not**
perform text transcription. This creates hard boundaries:

| Capability | Status | Reason |
|---|---|---|
| **Bounding-box layout blocks** | ✅ **Supported** | the core output (`blocks[].top_left_x/y, bottom_right_x/y, content, type`) |
| **Markdown text extraction** | ❌ Empty | `page.markdown` is always `""` — no text transcription |
| **Image crop extraction** | ❌ Empty | `page.images` is always `[]` (`include_image_base64` has no effect) |
| **Table structure/HTML** | ❌ Placeholder | `table_format` is accepted but output stays placeholder |
| **Confidence scores** | ❌ N/A | accepted but not populated |
| **PDF uploads** | ❌ Rejected | backend is image-only (`cv2.imdecode`); PDFs get a `400` "non-image media type" |
| **File-upload store** (`/v1/files`) | ❌ Unavailable | LiteLLM gateway has no `files_settings` configured |

> **Why:** the backend runs PaddleX `PP-DocLayoutV3` over a single image. The
> converter maps layout boxes to Mistral block types with **placeholder
> content** (`[text]`, `[title]`, `[table]`, …). Real text / images require a
> different backend (e.g. `PaddleOCR-VL`), not this one.

### Example block `content` placeholders

| Mistral block type | `content` |
|---|---|
| `OCRTextBlock` | `[text]` |
| `OCRTitleBlock` | `[title]` |
| `OCRTableBlock` | `[table]` |
| `OCRImageBlock` | `[image]` |
| `OCRListBlock` | `[list]` |
| `OCREquationBlock` | `[formula]` |
| `OCRCaptionBlock` | `[caption]` |
| `OCRCodeBlock` | `[code]` |
| `OCRHeaderBlock` | `[header]` |
| `OCRFooterBlock` | `[footer]` |
| `OCRSignatureBlock` | `[signature]` |
| `OCRReferencesBlock` | `[references]` |

---

## 7. Response shape

The gateway returns a native Mistral `OCRResponse`:

```json
{
  "pages": [
    {
      "index": 0,
      "markdown": "",
      "images": [],
      "dimensions": {"dpi": 300, "height": 580, "width": 626},
      "blocks": [
        {
          "top_left_x": 40, "top_left_y": 79,
          "bottom_right_x": 466, "bottom_right_y": 148,
          "content": "[title]",
          "type": "title"
        }
      ]
    }
  ],
  "model": "PP-DocLayoutV3",
  "usage_info": {"pages_processed": 1, "doc_size_bytes": 211754}
}
```

---

## 8. Working with bounding boxes (debug/annotate)

The `blocks` give you pixel bboxes directly. To draw/annotate:

```python
from PIL import Image, ImageDraw

img = Image.open("test.png").convert("RGB")
draw = ImageDraw.Draw(img)
for b in resp.pages[0].blocks:
    draw.rectangle(
        [(b.top_left_x, b.top_left_y), (b.bottom_right_x, b.bottom_right_y)],
        outline="red", width=3,
    )
img.save("annotated.png")
```

---

## 9. `curl` equivalent

```bash
curl -k 'https://litellm.adeoaiengine.ecouncil.ae/v1/ocr' \
  -H 'Authorization: Bearer <your api key here>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "PP-DocLayoutV3",
    "document": {
      "type": "image_url",
      "image_url": "data:image/png;base64,<base64>"
    },
    "include_blocks": true
  }'
```

---

## 10. Listing available models

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```

The gateway exposes many models (chat, embedding, STT/TTS, diffusion, OCR),
including `PP-DocLayoutV3`. Only models relevant to layout OCR are documented
here; use `client.models.list()` to see the live set.

---

## 11. Common errors & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `404` on `/v1/v1/ocr` | `server_url` included `/v1` | Use host root: `https://litellm.adeoaiengine.ecouncil.ae` |
| `500` with `MistralException` for `image_url` dict | LiteLLM mishandles `{"url": ...}` | Use `image_url=<string>` or `document_url=<string>` |
| `400` "non-image media type" | Sending a PDF / text data-URI | Send `data:image/...` only (PP-DocLayoutV3 is image-only) |
| `files.upload` `500` "files_settings is not set" | LiteLLM gateway has no file store | Pass the image via data-URI instead of uploading |
| `SSLCertVerificationError` | Self-signed cert | `client=httpx.Client(verify=False)` |
| Empty `markdown` / `images` | Model is layout-only | Use `blocks` (bboxes) instead |

---

## 12. Quick reference — import names

```python
from mistralai.client import Mistral
from mistralai.client.models import (
    ImageURLChunk,      # image_url=<string> or {"url": <string>}
    DocumentURLChunk,   # document_url=<string>
)
```

> Note: `from mistralai import Mistral` fails in 2.9.3 (namespace package).
> Always import from `mistralai.client`.