# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — Dockerfile for H200 / H100 / A100 / L40S (CUDA 12.6)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the H200 / Hopper (sm_90) variant of Dockerfile.api_compat.
# Use Dockerfile.api_compat for Blackwell / RTX 5090 (sm_120, requires CUDA 13.0).
#
# Key differences from the RTX 5090 variant:
#   • Base image: paddlex-hps:layout-cu12-base (CUDA 12.6, cuDNN 9.5) — below cu129
#   • tensorrt_cu12 (10.16.1.11) instead of tensorrt_cu13
#   • torch 2.13.0+cu126 instead of torch 2.13.0+cu130
#   • No pre-built .trt engine (RTX 5090 engines are sm_120-specific)
#     → Engine is built from ONNX at first startup (~3 min, cached after)
#   • Startup timeout 300s (engine build time)
#   • FP16 precision (modelopt FP8 ONNX has mixed-type Cast nodes that
#     TRT 10.16 STRONGLY_TYPED cannot parse; FP16 with original ONNX is
#     the reliable path.  FP8 remains available via pre-built engines on
#     sm_120 / RTX 5090 where the engine was built separately.)
#
# Target GPUs:
#   - NVIDIA H200 (sm_90a / Hopper) — FP16 (FP8-capable HW, but FP8 ONNX issue)
#   - NVIDIA H100 (sm_90a / Hopper) — FP16 (same)
#   - NVIDIA A100 (sm_80 / Ampere) — FP16
#   - NVIDIA L40S (sm_89 / Ada)    — FP16
#
# Security:
#   • Non-root user "runner" (UID 10000, GID 10000)
#   • All writable state → /tmp (TRT engine cache, PaddleX home, pip cache)
#   • Model weights baked into the image (ONNX + Paddle params, no .trt)
#
# Efficient rebuilds:
#   Layer order:  requirements → app code → models (largest, changes least)
#   BuildKit cache mounts for apt + pip.
#
# Build (from PaddleX repo root):
#   docker build \
#     -t paddlex-hps:layout-cu12-full \
#     -f deploy/hps/docker/cuda12/full/Dockerfile \
#     .
#
# Run (GPU enabled):
#   docker run -d --name paddlex-hps-h200 \
#     --device nvidia.com/gpu=all \
#     --env NVIDIA_VISIBLE_DEVICES=0 \
#     --env HPS_API_BACKEND=direct \
#     --env HPS_API_STARTUP_TIMEOUT=300 \
#     --env PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
#     --env HPS_GPU_PRE=1 \
#     --env HPS_API_BATCH_SIZE=4 \
#     -p 8080:8080 \
#     paddlex-hps:layout-cu12-full
# ═══════════════════════════════════════════════════════════════════════════════

FROM localhost/paddlex-hps:layout-cu12-base

# ─── Build args & environment ────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages (curl for HEALTHCHECK) ──────────────────────────────
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
# PINNED to tensorrt_cu12==10.16.1.11 (serialization version 240).
# 10.16.1 supports FP8 on Hopper (H200/H100, sm_90a) and Ada (L40S, sm_89).
# Using >=10.16 would resolve to 11.1.0 (serialization 243), which is
# incompatible with engines built by 10.16.x.
#
# Also install torch 2.13.0+cu126 for GPU pre-processing (Fix 8).
# torch+cu126 ships its own CUDA 12.6 runtime libs, compatible with the
# base image's CUDA 12.6. The CUDA Graph API (torch.cuda.CUDAGraph) is
# available if HPS_CUDA_GRAPHS=1 is set (experimental, see docs Section 13).
# Install torch FIRST from the cu126 index, so the correct CUDA 12.6 wheel
# is locked in before any PyPI package can pull the default cu130 torch.
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        torch==2.13.0+cu126 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir \
        "tensorrt_cu12==10.16.1.11" \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt

# ─── 2a-cleanup. Remove system TRT 10.5 C++ libs from Triton base ───────────
# The Triton base image ships TRT 10.5 C++ libs in /usr/lib/x86_64-linux-gnu/.
# The pip tensorrt_cu12==10.16.1.11 package ships its own 10.16.1 libs in
# site-packages/tensorrt_libs/. When both are present, ldconfig resolves
# libnvinfer.so.10 to the system 10.5 version, which cannot parse FP8 ONNX
# graphs. Removing the system 10.5 libs forces all TRT consumers to use the
# pip 10.16.1 version (via the tensorrt_libs package's RPATH / import side-effect).
RUN rm -f /usr/lib/x86_64-linux-gnu/libnvinfer.so \
          /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so \
          /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvonnxparser.so \
          /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource_win.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_dispatch.so \
          /usr/lib/x86_64-linux-gnu/libnvinfer_dispatch.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_dispatch.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so \
          /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10.5.0 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so \
          /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10 \
          /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.5.0 \
    && ldconfig

# LD_LIBRARY_PATH: prioritize pip tensorrt_libs, then paddle libs, then system.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# ─── 2b. API Python dependencies (pinned) ───────────────────────────────────
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -r /tmp/requirements.api_compat.txt \
    && rm /tmp/requirements.api_compat.txt

# ─── 2c. Patch PaddleX with custom TensorRT engine ──────────────────────────
# The installed paddlex (from PyPI wheel) does NOT include the custom
# TensorRTEngine module that was added to the jya0-v3.7.2 branch. We copy
# the new files and patch the existing __init__.py / deps.py so PaddleX
# registers the "tensorrt" inference engine.
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

# ─── 4. Model weights + ONNX (baked in, NO pre-built .trt) ───────────────────
# Unlike the RTX 5090 variant which bakes a pre-built inference_fp8.trt
# (sm_120-specific, 55 MB), this image only includes ONNX + Paddle params.
# The TRT engine is built from ONNX at first startup (~3 min for FP16) and
# cached in /tmp/paddlex-engines/ for subsequent restarts.
#
# Contains:
#   inference_fp8.onnx ( 69 MB)  — FP8-quantized ONNX (built by nvidia-modelopt)
#   inference.onnx     (125 MB)  — original ONNX model (used for FP16 build)
#   inference.pdiparams (125 MB) — PaddlePaddle parameters
#   inference_fp16.trt ( 70 MB)  — pre-built FP16 engine (sm_120, DELETED below)
#   inference.json      (1.2 MB) — model metadata
#   inference.yml       (1.5 KB) — model config
#   timing.cache        (5.7 MB) — TRT timing cache (version 240, TRT 10.16.1)
#
# NOTE: inference_fp8.trt and inference_fp16.trt in the model directory were
# built for sm_120 (RTX 5090). They will NOT deserialize on H200 (sm_90).
# The startup code (engine.py) looks for inference_{precision}.trt — if found
# it copies and tries to load it. On H200, deserialization will fail and TRT
# will fall back to building from ONNX. To avoid the failed-deserialization
# delay, we DELETE the sm_120 .trt files here so the engine builder goes
# straight to ONNX→TRT build.
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/
RUN rm -f /opt/models/PP-DocLayoutV3/inference_fp8.trt \
          /opt/models/PP-DocLayoutV3/inference_fp16.trt

# ─── 5. Non-root user (UID 10000) ───────────────────────────────────────────
RUN groupadd -g 10000 runner \
    && useradd -m -s /bin/bash -u 10000 -g 10000 runner

# ─── 6. Model directory accessible to runner (read-only) ────────────────────
RUN chmod -R a+rX /opt/models /opt/api_compat /opt/scripts

# ─── 7. All writable state → /tmp ───────────────────────────────────────────
# /tmp/paddlex-engines  — TRT engine cache (BUILT from ONNX at first startup)
# /tmp/paddlex-home     — PADDLEX_HOME (prevents writes to /home/runner/.paddlex)
# /tmp/pip-cache        — pip cache (build-time only)
RUN mkdir -p /tmp/paddlex-engines /tmp/paddlex-home \
    && chown -R runner:runner /tmp/paddlex-engines /tmp/paddlex-home \
    && chmod 1777 /tmp

ENV PADDLEX_HOME=/tmp/paddlex-home
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines
ENV HPS_API_MODEL=PP-DocLayoutV3
ENV HPS_API_PRECISION=fp16
ENV HPS_API_DEVICE_ID=0
# Engine build from ONNX takes ~3 min on first boot (FP16). 300s margin.
ENV HPS_API_STARTUP_TIMEOUT=300
ENV PYTHONPATH=/opt

USER runner
WORKDIR /opt

EXPOSE 8080

# ─── Health check: /health-check (alias of /health) ──────────────────────────
# start-period=300s gives margin for TRT engine build from ONNX on first boot.
# Subsequent restarts load cached engine in ~2s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8080/health-check || exit 1

ENTRYPOINT ["/opt/scripts/run_api.sh"]
