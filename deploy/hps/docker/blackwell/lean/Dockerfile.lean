# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — LEAN Dockerfile (GPU, TensorRT, direct-backend-only)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Purpose: a genuinely small image (~12-15 GB) for the **direct / TensorRT**
# backend. The other Dockerfiles inherit the 26.6 GB Triton NGC base
# (paddlex-hps-ngc), so even after splitting build tooling out they land at
# ~32-37 GB. This file builds on the official Paddle image instead, which
# already ships paddle 3.3.1 + CUDA 13.0 + cuDNN 9 and NO Triton server.
#
#   Base:  docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13 (19.1 GB)
#   Target GPU: RTX 5090 (sm_120 / Blackwell / CUDA 13). For H200 (sm_90a /
#   Hopper / CUDA 12.6) use the cuda12.6 base + tensorrt_cu12.
#
# Direct backend needs (verified against api_compat source):
#   • paddlepaddle-gpu  — provided by base (3.3.1)
#   • paddlex            — model wrapper (pip install)
#   • tensorrt (cu13)    — deserialize pre-built .trt engine
#   • pycuda             — direct CUDA in-process backend
#   • docling/FastAPI deps (requirements.api_compat.txt)
#   • websockets         — docling client websocket watcher
# NOT needed: tritonserver, tritonclient, nvidia-modelopt, onnx, onnxsim,
# polygraphy, paddle2onnx (build-time only). torch only if HPS_GPU_PRE=1.
#
# TRT engines are hardware-specific: build/commit the engine ON the target
# GPU, then bake it in. The pre-built inference_fp8.trt is sm_120/5090.
#
# Build (from PaddleX repo root):
#   docker build \
#     -t paddlex-hps-api-lean \
#     -f deploy/hps/docker/blackwell/lean/Dockerfile \
#     .
#
# Run (GPU enabled, direct backend):
#   docker run -d --name paddlex-hps-lean \
#     --device nvidia.com/gpu=all \
#     --env NVIDIA_VISIBLE_DEVICES=0 \
#     --env HPS_API_BACKEND=direct \
#     --env HPS_API_PRECISION=fp8 \
#     -p 8080:8080 \
#     paddlex-hps-api-lean
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILDER: build/quantize toolchain (not shipped to runtime)
# ──────────────────────────────────────────────────────────────────────────────
FROM docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System tools to run quantization / engine build on this host GPU.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Full TensorRT Python bindings + quantization/export toolchain.
# tensorrt_cu13 pinned to 10.16.1.11 (serialization 240) to match the
# pre-built engine. Quantization tooling is BUILD-TIME only.
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu13==10.16.1.11" \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt

# Model weights + ONNX source for (re)building the engine.
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RUNTIME (final, lean)
# ═══════════════════════════════════════════════════════════════════════════════
FROM docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages (curl for HEALTHCHECK, git needed by paddle) ────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# ─── 2a. TensorRT Python bindings + pycuda (runtime only) ────────────────────
# No torch, no modelopt, no onnx/paddle2onnx/polygraphy — the runtime only
# DESERIALIZES the pre-built engine.
# tensorrt_cu13==10.16.1.11 (serialization 240) matches inference_fp8.trt.
# Using >=10.16 would resolve to 11.1.0 (243), which cannot load 10.16.x
# engines. 10.16.1 supports FP8 on Blackwell / RTX 5090 (sm_120).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu13==10.16.1.11" \
        pycuda

# LD_LIBRARY_PATH: prioritize pip tensorrt_libs, then paddle libs, then CUDA.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# ─── 2b. PaddleX (model wrapper) + runtime deps ─────────────────────────────
# Paddlex is NOT preinstalled in the official paddle base (unlike NGC), and
# neither are its runtime deps. Install paddlex + the same runtime deps the
# NGC base bundles (opencv for cv2, scikit-image/learn for post-processing,
# scipy, faiss-cpu, pycocotools, decorator, astor).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "paddlex==3.7.2" \
        "opencv-contrib-python==4.10.0.84" \
        "scikit-image==0.24.0" "scikit-learn==1.6.1" # scipy, faiss-cpu, pycocotools, decorator, astor).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "paddlex==3.7.2" \
        "opencv-contrib-python==4.10.0.84" \
        "scikit-image==0.24.0" "scikit-learn==1.6.1" "scipy==1.15.2" \
        "faiss-cpu>=1.8" "pycocotools>=2" "pypdfium2>=4.30.0" \
        decorator astor

# ─── 2b2. API Python dependencies (pinned) ──────────────────────────────────
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -r /tmp/requirements.api_compat.txt \
    && pip install --no-cache-dir "orjson>=3.10.0" \
    && rm /tmp/requirements.api_compat.txt

# ─── 2c. Patch PaddleX with custom TensorRT engine ──────────────────────────
# Copy the new TensorRT engine module + vectorized processors into the
# installed paddlex wheel and register the "tensorrt" inference engine.
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
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/scripts/run_api.sh /opt/scripts/run_api.sh
RUN chmod +x /opt/scripts/run_api.sh

# ─── 4. Model weights + pre-built TRT engine ────────────────────────────────
# Only the pre-built engine (inference_fp8.trt) + ONNX fallback + Paddle
# params. Loaded directly at startup — no build.
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ─── 5. Non-root user (UID 10000) ───────────────────────────────────────────
RUN groupadd -g 10000 runner \
    && useradd -m -s /bin/bash -u 10000 -g 10000 runner

# ─── 6. Model dir accessible to runner (read-only) ──────────────────────────
RUN chmod -R a+rX /opt/models /opt/api_compat /opt/scripts

# ─── 7. Writable state → /tmp ───────────────────────────────────────────────
RUN mkdir -p /tmp/paddlex-engines /tmp/paddlex-home \
    && chown -R runner:runner /tmp/paddlex-engines /tmp/paddlex-home \
    && chmod 1777 /tmp

# pip cache purge to strip lib bloat (cache is build-time only anyway).
RUN rm -rf /tmp/pip-cache ~/.cache/pip

ENV PADDLEX_HOME=/tmp/paddlex-home
ENV PADDLE_PDX_CACHE_HOME=/tmp/paddlex-home/.paddlex
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines
ENV HPS_API_MODEL=PP-DocLayoutV3
ENV HPS_API_PRECISION=fp8
ENV HPS_API_BACKEND=direct
ENV HPS_API_DEVICE_ID=0
# Pre-built engine loads in ~2s (no ONNX→TRT build). 60s is plenty.
ENV HPS_API_STARTUP_TIMEOUT=60
ENV PYTHONPATH=/opt

USER runner
WORKDIR /opt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health-check || exit 1

ENTRYPOINT ["/opt/scripts/run_api.sh"]