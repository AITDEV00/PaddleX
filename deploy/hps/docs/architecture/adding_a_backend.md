# Adding a New Inference Backend (Adapter Guide)

> **Self-contained.** File references use absolute paths under the PaddleX
> workspace. This guide assumes you already understand the HTTP contract —
> read [`../API_REFERENCE.md`](../API_REFERENCE.md) first.

---

## 1. The seam

Every inference engine plugs into the API layer through one abstract class:

**`deploy/hps/api_compat/_core/backends/base.py`** → `InferenceBackend`

```python
class InferenceBackend(ABC):
    name: str                                  # also the HPS_API_BACKEND value

    def start(self) -> None: ...               # acquire resources, load model
    def wait_ready(self, timeout) -> bool: ...  # block until ready
    def shutdown(self) -> None: ...             # release resources

    def detect(self, image) -> list[dict]: ...  # layout boxes for one image

    # Optional (used by /v1/models):
    def list_models(self) -> list[dict]: ...          # inventory
    def get_model_status(self, name) -> dict | None:  # single-model status
```

The API layer (`docling_api/*`, `_core/inference.py`, `_core/health/routes.py`)
**only ever talks to this interface**. That is the entire point: you add a new
engine without changing any HTTP code.

### The `detect()` contract

`detect(image)` receives a `(H, W, 3)` uint8 **BGR** numpy array and must return
a list of box dicts, each with the keys:

```python
{
    "label": str,            # layout class name (e.g. "text", "title", "table")
    "score": float,          # confidence 0..1
    "coordinate": [xmin, ymin, xmax, ymax],  # axis-aligned box
    "order": int,            # reading order index
    "cls": int,              # numeric class id
    "polygon_points": [...], # optional polygon (4 or more points)
}
```

These boxes flow into the converter (`docling_api/converter/`) which maps
`label` → Docling label and builds the document. Keep the label strings in the
same vocabulary as the existing `labels.py` mapping so conversion keeps working.

---

## 2. The factory + selection

**`deploy/hps/api_compat/_core/backends/__init__.py`** → `create_backend()`

```python
def create_backend(name=None, **kwargs):
    if backend_id == "direct":  return direct.DirectBackend(**kwargs)
    if backend_id == "triton":  return triton.TritonBackend(**kwargs)
    raise ValueError(...)
```

The `name` argument defaults to `HPS_API_BACKEND` (read in `config.py`). Add
your backend here.

### The env-var contract

Add any new env vars to **`deploy/hps/api_compat/_core/config.py`** — that file
is the single source of truth; never call `os.environ` in your backend.

---

## 3. Step-by-step: add `mybackend`

1. **Create the module**
   `deploy/hps/api_compat/_core/backends/mybackend.py`:

   ```python
   from .base import InferenceBackend

   class MyBackend(InferenceBackend):
       name = "mybackend"

       def start(self):
           # load model, set self._ready = True when done
           self._ready = True

       def wait_ready(self, timeout):
           return self._ready

       def shutdown(self):
           pass

       def detect(self, image):
           # run your engine, return [box dicts ...]
           return [...]
   ```

2. **Register in the factory**
   In `_core/backends/__init__.py`, add:
   ```python
   from . import direct, triton, mybackend   # add import
   if backend_id == "mybackend":
       return mybackend.MyBackend(**kwargs)
   ```

3. **Set the env var**
   ```bash
   export HPS_API_BACKEND=mybackend
   ```

4. **(Optional) model inventory**
   If your backend hosts a named model, implement `list_models()` and
   `get_model_status()`. If you return an empty list / `None`, the platform
   layer falls back to a static entry for `HPS_API_MODEL` (see
   `_core/inference.py::AppState.list_models`).

5. **Add requirements**
   If your engine has pip deps, add them to the right file:
   - shared API layer → `deploy/hps/requirements.base.txt`
   - direct/TRT-only → `deploy/hps/requirements.direct.txt`
   - triton-only → `deploy/hps/requirements.triton.txt`

---

## 4. What you must NOT touch

Because the HTTP layer is backend-agnostic, adding a backend should **not**
require editing:

- `deploy/hps/api_compat/docling_api/routes.py` — endpoints unchanged
- `deploy/hps/api_compat/docling_api/service.py` — orchestration unchanged
- `deploy/hps/api_compat/docling_api/converter/` — Docling mapping unchanged
- `deploy/hps/api_compat/_core/health/routes.py` — `/v1/models` unchanged

If you find yourself modifying these for a new engine, you've probably crossed
the seam — step back and put the logic in your backend class instead.

---

## 5. Testing

The contract tests in `deploy/hps/tests/test_api_compat.py` run **without GPU**
(they stub `paddlex`) and verify protocol/schema/enum contracts, not inference.
Add your own engine-level tests alongside; keep the HTTP contract tests
independent of any specific backend.