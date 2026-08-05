# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — LEAN H200 Dockerfile (minimal CUDA-runtime base)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Purpose: a MUCH smaller direct/TensorRT image for CUDA 12.6 hosts than
# Dockerfile (docker/cuda12/full) which is built on the NGC Triton base (22 GB).
# FINAL SIZE: ~7 GB (same approach as docker/cuda13/lean which reached 7.07 GB
# on RTX 5090).
#
# Target GPUs (CUDA 12.6):
#   - NVIDIA H200  (sm_90a)  — FP16
#   - NVIDIA H100  (sm_90a)  — FP16
#   - NVIDIA A100  (sm_80)   — FP16
#   - NVIDIA L40S  (sm_89)   — FP16
#   - NVIDIA RTX 4090 (sm_89) — FP16
#
# KEY DESIGN: NO pre-built .trt engine baked in. This image ships the ONNX +
# Paddle params and BUILDS the engine from ONNX on first startup (FP16, ~3 min),
# cached in /tmp for subsequent restarts. Because the engine is built at
# runtime, the image works on any CUDA 12.6-capable GPU architecture.
#
#   Because the engine is built AT RUNTIME, we must KEEP ALL the TRT builder
#   resource kernels (all sm_* archs + ptx) — they are needed to (re)build the
#   engine on whatever CUDA 12.6 GPU the image lands on. We only strip: Windows
#   DLLs, unused paddle compute kernels, C++ headers, training subpackages.
#
#   FP16 is used (not FP8): the modelopt FP8 ONNX has mixed-type Cast nodes
#   that TRT 10.16 STRONGLY_TYPED cannot parse. FP16 via the original
#   inference.onnx is the reliable path on H200. (FP8 works only on sm_120
#   where a separately-built engine is baked in.)
#
#   Build toolchain NOT needed at runtime (verified against tensorrt_runner.py):
#     - modelopt       only for int8/fp8 quantization  (we use fp16, skip)
#     - paddle2onnx    only for Paddle→ONNX export     (inference.onnx shipped)
#     - onnx/polygraphy only used during export/fold   (skipped for pre-made ONNX)
#     The engine is built from the shipped inference.onnx via trt.OnnxParser
#     (C++ libnvinfer, included in the tensorrt_cu12 wheel).
#
# Build (from PaddleX repo root):
#   docker build \
#     -t paddlex-hps-api-h200-lean \
#     -f deploy/hps/docker/cuda12/lean/Dockerfile \
#     .
#
# Run (GPU enabled, direct backend, FP16):
#   docker run -d --name paddlex-hps-h200-lean \
#     --device nvidia.com/gpu=all \
#     --env HPS_API_BACKEND=direct \
#     --env HPS_API_PRECISION=fp16 \
#     --env HPS_API_STARTUP_TIMEOUT=300 \
#     -p 8082:8080 \
#     paddlex-hps-api-h200-lean
#   (First boot builds the TRT engine from ONNX, ~3 min.)
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — PADDLE SOURCE (import-only, used to transplant runtime)
# ──────────────────────────────────────────────────────────────────────────────
FROM docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda12.6-cudnn9.5 AS paddle-base

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILDER (build-time only, not shipped)
# ──────────────────────────────────────────────────────────────────────────────
FROM paddle-base AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System tools to build pycuda (the minimal runtime base lacks CUDA headers).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# tensorrt_cu12 pinned to 10.16.1.11 (serialization 240). pycuda is built
# HERE (paddle base has full CUDA headers) and COPYed into the runtime stage.
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu12==10.16.1.11" \
        pycuda

# Model weights + ONNX (used to (re)build the engine on the target GPU).
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — RUNTIME (final, minimal CUDA-runtime base + transplanted paddle)
# ──────────────────────────────────────────────────────────────────────────────
FROM docker.io/nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages: python 3.10 + curl + git ───────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3.10-venv \
        curl ca-certificates git \
        libgomp1 libquadmath0 libgfortran5 \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─── 2. Transplant paddle's self-contained runtime from the paddle base ────
# paddle/ (2.9G) + nvidia/cudnn + cusparselt + nccl are the parts the minimal
# base does NOT already have. The nvidia/cuda:12.6.2-runtime base already
# provides cublas/cublasLt/cudart/curand/cusolver/cusparse/nvJitLink/nvrtc
# in /usr/local/cuda-12.6/targets, so we SKIP the nvidia/cu12 pip package.
#
# CRITICAL layer-semantics note: copy + strip paddle in a SINGLE RUN layer.
# Using --mount=type=bind,from=paddle-base (NOT committed to a layer) + cp -a
# + rm inside one RUN commits ONLY the final stripped files.
RUN --mount=type=bind,from=paddle-base,source=/usr/local/lib/python3.10/dist-packages/paddle,target=/paddle-src \
    mkdir -p /usr/local/lib/python3.10/dist-packages/paddle \
    && cp -a /paddle-src/. /usr/local/lib/python3.10/dist-packages/paddle/ \
    # ── Strip compute kernels NOT used by the TRT direct backend ──
    && cd /usr/local/lib/python3.10/dist-packages/paddle/libs \
    && rm -f libphi_core.so libphi_gpu.so libflashattn.so libflashattnv3.so \
          libflashmaskv2.so libcinnapi.so libmklml_intel.so libdnnl.so.3 \
    && rm -rf /usr/local/lib/python3.10/dist-packages/paddle/include \
    && rm -rf /usr/local/lib/python3.10/dist-packages/paddle/distributed \
    && du -sh /usr/local/lib/python3.10/dist-packages/paddle

# Drop the stub libcuda.so (307M) shipped in /usr/local/cuda-12.6/compat.
RUN rm -rf /usr/local/cuda-12.6/compat

# NOTE: the CUDA 12.6 paddle base ships nvidia/cublas, cuda_cccl, cuda_cupti,
# cuda_nvrtc, cuda_runtime, cudnn, cufft, cufile, curand, cusolver, cusparse,
# nccl, nvjitlink, nvtx — but NO nvidia/cusparselt (unlike the CUDA 13.0 base).
# The runtime base already provides cublas/cublasLt/cudart/curand/cusolver/
# cusparse/nvJitLink/nvrtc in /usr/local/cuda-12.6/targets. We copy only the
# packages the runtime base does NOT already have: cudnn + nccl.
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/nvidia/cudnn \
    /usr/local/lib/python3.10/dist-packages/nvidia/cudnn
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/nvidia/nccl \
    /usr/local/lib/python3.10/dist-packages/nvidia/nccl
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info \
    /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info
# numpy is a hard dep of paddle; the base's numpy is ABI-matched to it.
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/numpy \
    /usr/local/lib/python3.10/dist-packages/numpy
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/numpy.libs \
    /usr/local/lib/python3.10/dist-packages/numpy.libs

# LD_LIBRARY_PATH: prioritize tensorrt_libs, then paddle libs, then CUDA runtime.
# /usr/local/cuda-12.6/compat holds only the STUB libcuda.so (build-time). At
# runtime the REAL driver libcuda.so.1 is injected by the nvidia container
# runtime into /usr/local/nvidia/lib64 — that MUST come first.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda/lib64

# ─── 3. TensorRT Python bindings + pycuda (runtime only) ────────────────────
# tensorrt_cu12==10.16.1.11 (serialization 240) matches the H200 engines.
# 10.16.1 supports FP8 on Hopper (H200/H100, sm_90a) and Ada (L40S, sm_89).
#
# ★ UNIVERSAL design: this image builds the engine from ONNX at runtime on the
# TARGET GPU, so we KEEP ALL TensorRT builder-resource kernels (all sm_* archs
# + ptx). We cannot strip to a single arch (e.g. sm_90) — the engine must build
# on whichever CUDA 12.6 GPU it lands on (H200 sm_90a, A100 sm_80, L40S/RTX4090
# sm_89, RTX 5090 sm_120 …). We only drop the Windows DLLs (never used on
# Linux). This mirrors docker/cuda13/lean which does the same.
#
# pip 22.0.2 rejects the tensorrt wheel ("unknown metadata") — upgrade pip first.
# pycuda has no prebuilt cp310 wheel and cannot compile here (no CUDA headers);
# it is built in the BUILDER stage and COPY'd in below.
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -U "pip>=26" \
    && pip install --no-cache-dir "tensorrt_cu12==10.16.1.11" \
    # Keep ALL arch builder kernels (universal build-at-runtime); only strip
    # the Windows DLLs (never loaded on Linux, ~1.8 GB savings).
    && cd /usr/local/lib/python3.10/dist-packages/tensorrt_libs \
    && rm -f *_win_*.so.* \
    && du -sh .

# pycuda built in the builder stage (full CUDA headers). COPY the package dir
# + dist-info. It links libcuda.so.1, provided by the base's compat dir.
COPY --from=builder /usr/local/lib/python3.10/dist-packages/pycuda \
    /usr/local/lib/python3.10/dist-packages/pycuda
COPY --from=builder /usr/local/lib/python3.10/dist-packages/pycuda-*.dist-info \
    /usr/local/lib/python3.10/dist-packages/

# ─── 3b. PaddleX (model wrapper) + runtime deps ─────────────────────────────
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "paddlex==3.7.2" \
        "opencv-contrib-python==4.10.0.84" \
        "scikit-image==0.24.0" "scikit-learn==1.6.1" "scipy==1.15.2" \
        "faiss-cpu>=1.8" "pycocotools>=2" "pypdfium2>=4.30.0" \
        decorator astor

# ─── 4. API Python dependencies (pinned) ──────────────────────────────────────
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -r /tmp/requirements.api_compat.txt \
    && pip install --no-cache-dir "orjson>=3.10.0" \
    && rm /tmp/requirements.api_compat.txt

# ─── 5. Patch PaddleX with custom TensorRT engine ─────────────────────────────
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

# ─── 5. Application code ───────────────────────────────────────────────────
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/scripts/run_api.sh /opt/scripts/run_api.sh
RUN chmod +x /opt/scripts/run_api.sh

# ─── 6. Model weights + ONNX (NO pre-built .trt) ──────────────────────────
# Bake in ONNX + Paddle params. The pre-built inference_fp8.trt / fp16.trt in
# the model dir are sm_120 (RTX 5090) — they will NOT deserialize on sm_90, so
# DELETE them so the engine builder goes straight to ONNX→TRT on first boot.
# (The engine is built once from inference.onnx and cached in /tmp.)
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/
RUN rm -f /opt/models/PP-DocLayoutV3/inference_fp8.trt \
          /opt/models/PP-DocLayoutV3/inference_fp16.trt

# ─── 7. Non-root user (UID 10000) ───────────────────────────────────────────
RUN groupadd -g 10000 runner \
    && useradd -m -s /bin/bash -u 10000 -g 10000 runner

# ─── 8. Model dir accessible to runner (read-only) ──────────────────────────
RUN chmod -R a+rX /opt/models /opt/api_compat /opt/scripts

# ─── 9. Writable state → /tmp ───────────────────────────────────────────────
RUN mkdir -p /tmp/paddlex-engines /tmp/paddlex-home \
    && chown -R runner:runner /tmp/paddlex-engines /tmp/paddlex-home \
    && chmod 1777 /tmp

# pip cache purge to strip lib bloat (cache is build-time-only anyway).
RUN rm -rf /tmp/pip-cache ~/.cache/pip

ENV PADDLEX_HOME=/tmp/paddlex-home
ENV PADDLE_PDX_CACHE_HOME=/tmp/paddlex-home/.paddlex
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines
ENV HPS_API_MODEL=PP-DocLayoutV3
ENV HPS_API_PRECISION=fp16
ENV HPS_API_BACKEND=direct
ENV HPS_API_DEVICE_ID=0
# Engine build from ONNX takes ~3 min on first boot (FP16). 300s margin.
ENV HPS_API_STARTUP_TIMEOUT=300
ENV PYTHONPATH=/opt

USER runner
WORKDIR /opt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8080/health-check || exit 1

ENTRYPOINT ["/opt/scripts/run_api.sh"]