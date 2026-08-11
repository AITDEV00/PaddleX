# PaddleX HPS — Pluggable Backend Architecture (refactor 2026-08)

## Env-toggle backend selection
- `HPS_API_BACKEND` env var → `deploy/hps/api_compat/_core/backends/__init__.py::create_backend()`.
- Default is now **`direct`** (was `triton`). Change in `_core/config.py::INFERENCE_BACKEND`.
- `backends/base.py` = `InferenceBackend` ABC: `start/wait_ready/shutdown`, `detect()`, `list_models()`, `get_model_status()`.
- `backends/direct.py` = `DirectBackend` (in-process PaddleX/TensorRT micro-batch pipeline — ported from old `inference.py`).
- `backends/triton.py` = `TritonBackend` (gRPC client wrapping `_core/triton_client.py`).
- `_core/inference.py` is now a thin facade: `AppState` holds a backend; `run_layout_detection()` delegates. API layer never touches backend internals.

## Triton-compatible /v1/models (works for BOTH backends)
- Added to `_core/health/routes.py`: `GET /v1/models`, `/v1/models/{name}`, `/v1/models/{name}/ready`.
- `AppState.list_models()`/`get_model_status()` delegate to active backend; direct path reports `PP-DocLayoutV3` as the model id.
- Direct `DetectBackend.list_models()` returns `[{name,version,ready,active}]`; `get_model_status()` returns None for unknown → 404.
- Unknown model → backend returns None → route raises 404. (A stubbed test backend returning a status for any name gives 200 — that's test-only.)

## Debug vertical-slice architecture
- `_core/debug/__init__.py`: `DebugRenderer` ABC + registry (`get_debugger`, `render_debug`).
- Each model gets a subclass (e.g. `LayoutDebugRenderer.models=("PP-DocLayoutV3",...)`). Add a new model's debug slice by subclassing + the registry auto-picks it.
- HTTP: `/v1/convert/file` and `/v1/convert/source` accept `debug=1`/`debug=true` → returns PNG with boxes drawn (media_type image/png) instead of conversion.
- Debug path: `service.py::render_debug_image()` → `run_layout_detection` → `render_debug`.

## Mode-aware build (requirements split)
- `requirements.base.txt` (shared API layer) / `requirements.direct.txt` (paddlex/tensorrt/pycuda/opencv/scikit/scipy/faiss) / `requirements.triton.txt` (tritonclient[grpc]).
- Lean Dockerfile (`docker/cuda13/lean/Dockerfile`) installs base + inline direct deps; legacy Dockerfile installs base + triton.
- `requirements.api_compat.txt` still exists (referenced by full images) — NOT deleted.

## Dev venv for CUDA 13 (RTX 5090 / sm_120)
- `deploy/hps/.venv-cuda13-py310` (Python 3.10.20, uv). Works (verified: paddle 3.3.1 + tensorrt_cu13 10.16.1.11 + fastapi + granian + docling all import).
- **Important**: paddle CDN is ISP-blocked → transplant `paddle/`, `numpy/`, `nvidia/`, dist-infos from local `paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13` image into venv site-packages.
- `scripts/setup_devvenv_cuda13.sh` automates venv creation.
- LD_LIBRARY_PATH needs: `nvidia/cu13/lib:nvidia/cudnn/lib:nvidia/nccl/lib:nvidia/cusparselt/lib:paddle/libs`.
- Also installed into venv: httpx, pillow, networkx, opt_einsum==3.3.0, typing_extensions, safetensors, protobuf, decorator, astor, scipy, opencv-python-headless, orjson.
- Must use `"$V/bin/python" -m pip` (system `which pip` is `/usr/bin/pip`).

## Horizontal seam facts
- Triton convention: `/v1/models` is Triton's inventory endpoint (NOT a Docling endpoint). Docling uses `/v1/convert/*`. Our `/v1/models` reuses the same code to expose the paddle layout v3 model id on the direct path.