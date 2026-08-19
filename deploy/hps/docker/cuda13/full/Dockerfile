# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — Dockerfile (GPU, TensorRT, non-root, offline-ready)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Extends the paddlex-hps:layout-cu13-base image (NGC Triton 24.10 + PaddlePaddle 3.3.1
# + PaddleX + CUDA 13.0 + TensorRT 10.5/10.16).  Adds the api_compat Python
# service layer (FastAPI + Granian + Docling compatibility shim).
#
# Security:
#   • Non-root user "runner" (UID 10000, GID 10000)
#   • All writable state → /tmp (TRT engine cache, PaddleX home, pip cache)
#   • Model weights + TRT engines baked into the image (no network at runtime)
#
# Efficient rebuilds:
#   Layer order:  requirements → app code → models (largest, changes least)
#   BuildKit cache mounts for apt + pip.
#
# Build (from PaddleX repo root):
#   podman build \
#     -t paddlex-hps:layout-cu13-full \
#     -f deploy/hps/docker/cuda13/full/Dockerfile \
#     .
#
# Run (internet disabled, GPU enabled):
#   podman run --rm \
#     --network none \
#     --device nvidia.com/gpu=all \
#     --env NVIDIA_VISIBLE_DEVICES=0 \
#     -p 8080:8080 \
#     paddlex-hps:layout-cu13-full
#
# Or with docker:
#   docker run --rm --network none --gpus all -p 8080:8080 paddlex-hps:layout-cu13-full
# ═══════════════════════════════════════════════════════════════════════════════

FROM localhost/paddlex-hps:layout-cu13-base

# ─── Build args & environment ────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages (curl for HEALTHCHECK) ──────────────────────────────
# Layer 1: changes rarely → cached across rebuilds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── 2a. TensorRT Python bindings + quantization tooling ─────────────────────
# The base NGC image has TRT 10.5 C++ libs (libnvinfer.so.10) but NOT the
# Python "tensorrt" module. PaddleX requires importlib.find_spec("tensorrt")
# to register the "tensorrt" inference engine.
#
# PINNED to tensorrt_cu13==10.16.1.11 (serialization version 240). The
# pre-built inference_fp8.trt was built with TRT 10.16.1 — TRT engines are
# version-locked and cannot be deserialized by a different major version.
# Using >=10.16 would resolve to 11.1.0 (serialization 243), breaking the
# pre-built engine and requiring an ONNX patch + 3-min startup rebuild.
# 10.16.1 supports FP8 on Blackwell / RTX 5090 (sm_120).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu13==10.16.1.11" \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt

# ─── 2b. API Python dependencies (pinned) ───────────────────────────────────
# Layer 2b: changes only when requirements.api_compat.txt changes.
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -r /tmp/requirements.api_compat.txt \
    && rm /tmp/requirements.api_compat.txt

# ─── 2c. Patch PaddleX with custom TensorRT engine ──────────────────────────
# The installed paddlex (from PyPI wheel) does NOT include the custom
# TensorRTEngine module that was added to the jya0-v3.7.2 branch. We copy
# the new files and patch the existing __init__.py / deps.py so PaddleX
# registers the "tensorrt" inference engine.
#   NEW:   engines/tensorrt.py          — TensorRTEngine class
#   NEW:   runners/tensorrt_runner.py   — TensorRTRunner + Config
#   PATCH: engines/__init__.py          — import tensorrt engine
#   PATCH: runners/__init__.py          — import TensorRTRunner
#   PATCH: layout_analysis/__init__.py  — register tensorrt binding
#   PATCH: utils/deps.py                — is_dep_available("tensorrt") fix
#   PATCH: object_detection/processors.py — vectorized post-processing
#   PATCH: common/vision/processors.py  — vectorized Normalize
COPY paddlex/inference/models/engines/tensorrt.py \
     paddlex/inference/models/runners/tensorrt_runner.py \
     /tmp/paddlex-patch/
COPY paddlex/inference/models/object_detection/processors.py \
     /tmp/paddlex-patch/obj_det_processors.py
COPY paddlex/inference/models/object_detection/predictor.py \
     /tmp/paddlex-patch/obj_det_predictor.py
COPY paddlex/inference/models/layout_analysis/predictor.py \
     /tmp/paddlex-patch/layout_predictor.py
COPY paddlex/inference/models/common/vision/processors.py \
     /tmp/paddlex-patch/vision_processors.py
COPY deploy/hps/scripts/patch_paddlex_trt.py /tmp/patch_paddlex_trt.py
RUN python3 /tmp/patch_paddlex_trt.py \
        /tmp/paddlex-patch \
        /usr/local/lib/python3.10/dist-packages/paddlex \
    && rm -rf /tmp/paddlex-patch /tmp/patch_paddlex_trt.py

# ─── 3. Application code ────────────────────────────────────────────────────
# Layer 3: changes when you edit api_compat Python files. Small and fast.
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/scripts/run_api.sh /opt/scripts/run_api.sh
RUN chmod +x /opt/scripts/run_api.sh

# ─── 4. Model weights + TRT engines (baked in) ──────────────────────────────
# Layer 4: LARGEST layer (~500 MB). Changes rarely (only when models are
# rebuilt). Placed last so app-code edits don't invalidate this cache layer.
# Contains:
#   inference_fp8.trt  ( 55 MB)  — pre-built TRT 10.16.1 FP8 engine (loaded directly)
#   inference_fp8.onnx ( 69 MB)  — FP8-quantized ONNX (fallback if .trt missing)
#   inference.onnx     (125 MB)  — original ONNX model
#   inference.pdiparams (125 MB) — PaddlePaddle parameters
#   inference_fp16.trt ( 70 MB)  — pre-built FP16 engine (unused, kept for debug)
#   inference.json      (1.2 MB) — model metadata
#   inference.yml       (1.5 KB) — model config
#   timing.cache        (5.7 MB) — TRT timing cache (version 240, matches TRT 10.16.1)
#
# The pre-built inference_fp8.trt was built with TRT 10.16.1 (serialization
# version 240) and is loaded directly at startup — no ONNX parse, no engine
# build, no 3-minute wait. TRT is pinned to 10.16.1.11 to match.
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ─── 5. Non-root user (UID 10000) ───────────────────────────────────────────
RUN groupadd -g 10000 runner \
    && useradd -m -s /bin/bash -u 10000 -g 10000 runner

# ─── 6. Model directory accessible to runner (read-only) ────────────────────
# PaddleX searches BUILTIN_ENGINE_DIRS for pre-built engines. The first entry
# is /opt/models/PP-DocLayoutV3 — we make it readable by runner.
RUN chmod -R a+rX /opt/models /opt/api_compat /opt/scripts

# ─── 7. All writable state → /tmp ───────────────────────────────────────────
# /tmp/paddlex-engines  — TRT engine cache (copied from /opt/models at startup)
# /tmp/paddlex-home     — PADDLEX_HOME (prevents writes to /home/runner/.paddlex)
# /tmp/pip-cache        — pip cache (build-time only, but set for completeness)
RUN mkdir -p /tmp/paddlex-engines /tmp/paddlex-home \
    && chown -R runner:runner /tmp/paddlex-engines /tmp/paddlex-home \
    && chmod 1777 /tmp

# PaddleX looks for models at $PADDLEX_HOME/.paddlex/official_models/.
# We set PADDLEX_HOME=/tmp/paddlex-home so PaddleX's own writes go to /tmp.
# The api_compat engine.py reads pre-built engines from /opt/models/ and
# copies them to /tmp/paddlex-engines/ — no writes outside /tmp at runtime.
ENV PADDLEX_HOME=/tmp/paddlex-home
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines
ENV HPS_API_MODEL=PP-DocLayoutV3
ENV HPS_API_PRECISION=fp8
ENV HPS_API_DEVICE_ID=0
# Fast startup: pre-built engine loads in ~2s (no 3-min ONNX→TRT build).
# 60s is plenty for engine copy + model init.
ENV HPS_API_STARTUP_TIMEOUT=60
ENV PYTHONPATH=/opt

USER runner
WORKDIR /opt

EXPOSE 8080

# ─── Health check: /health-check (alias of /health) ──────────────────────────
# start-period=60s gives margin for TRT engine copy + model init on first boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health-check || exit 1

ENTRYPOINT ["/opt/scripts/run_api.sh"]
