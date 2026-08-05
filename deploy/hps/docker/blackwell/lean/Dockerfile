# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — LEAN2 Dockerfile (minimal CUDA-runtime base)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Purpose: a MUCH smaller direct/TensorRT image than Dockerfile.lean (25.4 GB).
# FINAL SIZE: ~7.0 GB (validated end-to-end with the Docling client).
#
# Dockerfile.lean builds on the official Paddle image, which is a `-devel`
# image (19.1 GB): it ships the full CUDA build toolchain (nvcc, gcc-13, cmake,
# ccache, patchelf, nsight, nvvm, static .a libs) that the runtime NEVER uses.
#
# This Dockerfile (canonical lean) instead builds the RUNTIME stage on the tiny
# `nvidia/cuda:13.0.1-runtime-ubuntu22.04` base (2.4 GB), which already ships
# the core CUDA math runtime libs (cublas, cublasLt, cudart, curand, cusolver,
# cusparse, nvJitLink, nvrtc) with NO build toolchain.
#
#   Base (runtime): docker.io/nvidia/cuda:13.0.1-runtime-ubuntu22.04   (2.4 GB)
#   Base (builder): docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13
#                   (19.1 GB, devel) — used ONLY to transplant paddle's pip
#                   runtime (paddle dist-packages + nvidia/cudnn) onto the
#                   small base. Not shipped to runtime.
#
# Why transplant instead of `pip install paddlepaddle-gpu`:
#   Paddle 3.x GPU wheels are served from Paddle's own index, which redirects
#   to the bcebos CDN (paddle-whl.bj.bcebos.com). That CDN is unreachable from
#   this build host (ConnectionResetError 104 / HTTP 000), so a fresh pip
#   install of paddlepaddle-gpu==3.3.1 fails. The paddle base already has the
#   correct 3.3.1 runtime installed, so we COPY it onto the small base.
#
#   This is safe because paddle 3.3.1 is an ABI-stable, self-contained wheel:
#   its runtime lives entirely in dist-packages (paddle/ + nvidia/cudnn + the
#   nvidia math libs) and links against CUDA 13.0 runtime libs that the small
#   base provides. The 19.1 GB base's only irreplaceable value is this wheel.
#
# Direct backend needs (verified against api_compat source):
#   • paddlepaddle-gpu 3.3.1  — transplanted from the paddle base
#   • paddlex                — model wrapper (pip install, PyPI is reachable)
#   • tensorrt (cu13) 10.16.1.11 — deserialize pre-built engine
#   • pycuda                 — direct CUDA in-process backend
#   • docling/FastAPI deps (requirements.api_compat.txt)
#   • websockets             — docling client websocket watcher (test only)
# NOT needed: tritonserver, tritonclient, nvidia-modelopt, onnx, onnxsim,
# polygraphy, paddle2onnx (build-time only), nvcc / gcc / build toolchain.
#
# TRT engines are hardware-specific: build/commit the engine ON the target
# GPU, then bake it in. The pre-built inference_fp8.trt is sm_120 (RTX 5090).
# For H200 (sm_90a / CUDA 12.6) use the cuda12.6 runtime base + tensorrt_cu12.
#
# Build (from PaddleX repo root):
#   docker build \
#     -t paddlex-hps-api-lean2 \
#     -f deploy/hps/docker/blackwell/lean/Dockerfile \
#     .
#
# Run (GPU enabled, direct backend):
#   docker run -d --name paddlex-hps-lean2 \
#     --device nvidia.com/gpu=all \
#     --env HPS_API_BACKEND=direct \
#     --env HPS_API_PRECISION=fp8 \
#     -p 8082:8080 \
#     paddlex-hps-api-lean2
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — PADDLE SOURCE (import-only, used to transplant runtime)
# ──────────────────────────────────────────────────────────────────────────────
FROM docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13 AS paddle-base

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILDER (build-time only, not shipped)
# ──────────────────────────────────────────────────────────────────────────────
FROM paddle-base AS builder

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

# Full TensorRT Python bindings + quantization/export toolchain + pycuda.
# tensorrt_cu13 pinned to 10.16.1.11 (serialization 240) to match the
# pre-built engine. Quantization tooling is BUILD-TIME only.
# pycuda is built HERE (paddle base has full CUDA headers) and COPIED into the
# runtime stage — avoids recompiling on the minimal base (which lacks headers
# like cudaProfiler.h).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu13==10.16.1.11" \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt

# Model weights + ONNX source for (re)building the engine.
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RUNTIME (final, minimal CUDA-runtime base + transplanted paddle)
# ═══════════════════════════════════════════════════════════════════════════════
FROM docker.io/nvidia/cuda:13.0.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages: python 3.10 + curl + git ───────────────────────────
# This base is a bare Ubuntu 22.04 (no python). We need python3.10 + pip for
# the pip-installed paddle dist-packages and all API deps. Also curl (for
# HEALTHCHECK) and git (paddle needs it for some features).
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
# base does NOT already have. The nvidia/cuda runtime base already provides
# cublas/cublasLt/cudart/curand/cusolver/cusparse/nvJitLink/nvrtc/nvrtc-builtins
# in /usr/local/cuda-13.0/targets, so we SKIP the nvidia/cu13 pip package
# (~1.8 GB) entirely — it would only duplicate the base's own CUDA libs.
# (Verified: removing cu13 does not affect Docling inference.)
#
# We keep the nvidia/cudnn + cusparselt + nccl package dirs (which hold .so
# libs the base does not ship) and the paddle dist-info + numpy.
#
# CRITICAL layer-semantics note: we MUST copy + strip paddle in a SINGLE RUN
# layer. If we COPY the whole paddle/ dir (3.08 GB layer) and then rm files in
# a SEPARATE RUN, the second layer is only a ~20 kB whiteout — the underlying
# 3.08 GB of deleted kernels stays baked into the image. Using
# --mount=type=bind,from=paddle-base (which is NOT committed to a layer) and
# cp -a + rm inside one RUN commits ONLY the final stripped files.
RUN --mount=type=bind,from=paddle-base,source=/usr/local/lib/python3.10/dist-packages/paddle,target=/paddle-src \
    mkdir -p /usr/local/lib/python3.10/dist-packages/paddle \
    && cp -a /paddle-src/. /usr/local/lib/python3.10/dist-packages/paddle/ \
    # ── Strip compute kernels NOT used by the TRT direct backend ──
    # The api_compat "direct" backend deserializes a pre-built TensorRT engine
    # and runs inference in-process via pycuda + TensorRT. It NEVER calls
    # paddle's phi/flash-attention/MKL/CINN kernels, so paddle/libs only needs
    # the tiny loader stubs (libpaddle.so, libphi.so) + libcommon + mkl blas.
    # (Verified live: Docling SUCCESS with all of these removed.)
    && cd /usr/local/lib/python3.10/dist-packages/paddle/libs \
    && rm -f libphi_core.so libphi_gpu.so libflashattn.so libflashattnv3.so \
          libflashmaskv2.so libcinnapi.so libmklml_intel.so libdnnl.so.3 \
    # drop C++ headers + training-only distributed subpackages
    && rm -rf /usr/local/lib/python3.10/dist-packages/paddle/include \
    && rm -rf /usr/local/lib/python3.10/dist-packages/paddle/distributed \
    && du -sh /usr/local/lib/python3.10/dist-packages/paddle

# Drop the stub libcuda.so (307M) shipped in /usr/local/cuda-13.0/compat. The
# base provides it for BUILD-time linking only (pycuda was built in the
# builder stage). At runtime the real driver libcuda.so.1 is injected into
# /usr/local/nvidia/lib64 by the nvidia container runtime. (Verified live.)
RUN rm -rf /usr/local/cuda-13.0/compat

COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/nvidia/cudnn \
    /usr/local/lib/python3.10/dist-packages/nvidia/cudnn
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/nvidia/cusparselt \
    /usr/local/lib/python3.10/dist-packages/nvidia/cusparselt
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/nvidia/nccl \
    /usr/local/lib/python3.10/dist-packages/nvidia/nccl
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info \
    /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info
# numpy is a hard dep of paddle; the base's numpy is ABI-matched to it.
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/numpy \
    /usr/local/lib/python3.10/dist-packages/numpy
COPY --from=paddle-base /usr/local/lib/python3.10/dist-packages/numpy.libs \
    /usr/local/lib/python3.10/dist-packages/numpy.libs

# LD_LIBRARY_PATH: prioritize tensorrt_libs, then paddle libs, then the CUDA
# runtime libs. CRITICAL: /usr/local/cuda-13.0/compat holds only the STUB
# libcuda.so (used at BUILD time by pycuda). At RUNTIME the REAL driver
# libcuda.so.1 is injected by the nvidia container runtime into
# /usr/local/nvidia/lib64 — that MUST come before anything else, otherwise
# pycuda/TRT load the stub and fail with "no CUDA-capable device".
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64

# ─── 3. TensorRT Python bindings + pycuda (runtime only) ────────────────────
# No torch, no modelopt, no onnx/paddle2onnx/polygraphy — the runtime only
# DESERIALIZES the pre-built engine.
# tensorrt_cu13==10.16.1.11 (serialization 240) matches inference_fp8.trt.
# Using >=10.16 would resolve to 11.1.0 (243), which cannot load 10.16.x
# engines. 10.16.1 supports FP8 on Blackwell / RTX 5090 (sm_120).
#
# NOTE: the apt python3-pip is 22.0.2, which rejects the tensorrt_cu13 wheel
# ("filename has 'tensorrt-cu13', but metadata has 'unknown'"). Upgrade pip
# FIRST (matches the paddle base's pip 26.x) so the wheel resolves cleanly.
# pycuda has no prebuilt cp310 wheel and CANNOT be compiled here (the minimal
# base lacks CUDA headers like cudaProfiler.h). So it is built in the BUILDER
# stage (which has full CUDA) and COPYed in below.
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -U "pip>=26" \
    && pip install --no-cache-dir "tensorrt_cu13==10.16.1.11" \
    # Strip unused TensorRT builder resources. The wheel ships both Windows
    # (*.win_*) and Linux builder-resource kernels for MANY GPU archs. Our
    # runtime only DESERIALIZES a pre-built sm_120 (RTX 5090) engine, so:
    #   - drop the Windows DLLs (never used on Linux, ~1.8 GB)
    #   - drop builder-resource kernels for archs OTHER than sm_120 (~1.6 GB);
    #     keep sm_120 so the runtime can still (re)build an engine on the GPU.
    # Leaves core libnvinfer/plugin/onnxparser + sm120 resource (~0.9 GB).
    && cd /usr/local/lib/python3.10/dist-packages/tensorrt_libs \
    && rm -f *_win_*.so.* \
    # NOTE: the build RUN uses /bin/sh (dash), which does NOT do brace
    # expansion — so use an explicit find with !-exclude for archs we keep.
    && find . -maxdepth 1 -name 'libnvinfer_builder_resource_sm*.so.10.16.1' \
        ! -name '*_sm120.so.10.16.1' -delete \
    && rm -f libnvinfer_builder_resource_ptx.so.10.16.1 \
    && du -sh .

# pycuda built in the builder stage (full CUDA headers). COPY the package dir
# + dist-info. It links libcuda.so.1, provided by the base's compat dir.
COPY --from=builder /usr/local/lib/python3.10/dist-packages/pycuda \
    /usr/local/lib/python3.10/dist-packages/pycuda
COPY --from=builder /usr/local/lib/python3.10/dist-packages/pycuda-*.dist-info \
    /usr/local/lib/python3.10/dist-packages/

# ─── 4. PaddleX (model wrapper) + runtime deps ─────────────────────────────
# Paddlex is NOT preinstalled on this base (unlike the NGC base). Install it +
# the same runtime deps the NGC base bundles (opencv for cv2, scikit-image/
# learn for post-processing, scipy, faiss-cpu, pycocotools, decorator, astor).
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "paddlex==3.7.2" \
        "opencv-contrib-python==4.10.0.84" \
        "scikit-image==0.24.0" "scikit-learn==1.6.1" "scipy==1.15.2" \
        "faiss-cpu>=1.8" "pycocotools>=2" "pypdfium2>=4.30.0" \
        decorator astor

# ─── 4b. API Python dependencies (pinned) ──────────────────────────────────
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir -r /tmp/requirements.api_compat.txt \
    && pip install --no-cache-dir "orjson>=3.10.0" \
    && rm /tmp/requirements.api_compat.txt

# ─── 4c. Patch PaddleX with custom TensorRT engine ─────────────────────────
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

# ─── 5. Application code ───────────────────────────────────────────────────
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/scripts/run_api.sh /opt/scripts/run_api.sh
RUN chmod +x /opt/scripts/run_api.sh

# ─── 6. Model weights + pre-built TRT engine ───────────────────────────────
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

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