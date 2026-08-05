# Custom PaddleX HPS Dockerfile — NGC Triton + PaddlePaddle GPU (CUDA 12.6 variant)
#
# This is the H200 / Hopper (sm_90) variant of the Blackwell base
# (docker/blackwell/base). Use that for Blackwell / RTX 5090 (sm_120,
# requires CUDA 13.0).
#
# PaddlePaddle 3.3.1-gpu-cuda12.6-cudnn9.5 uses the SAME CUDA 12.6 + cuDNN 9.5
# as Triton 24.10-py3, so no extra runtime libraries need to be copied.
# This makes the build simpler and the image smaller than the CUDA 13.0 variant.
#
# Target GPUs:
#   - NVIDIA H200 (sm_90 / Hopper) — including MIG partitions
#   - NVIDIA H100 (sm_90 / Hopper)
#   - NVIDIA A100 (sm_80 / Ampere)
#   - NVIDIA L40S (sm_89 / Ada)
#
# Bypasses Baidu (bcebos.com) which is blocked by Etisalat.
# Uses Docker Hub paddlepaddle/paddle image to extract paddlepaddle-gpu,
# and NGC Triton 24.10-py3 as the base (Python 3.10 + CUDA 12.6 + TensorRT 10.5).
#
# Build context: PaddleX repo root (e.g. /home/jyao/ADEO/OCR/PaddleX)
#   podman build -t paddlex-hps-ngc-cuda12 -f deploy/hps/docker/hopper/base/Dockerfile .
#
# Requires BuildKit / DOCKER_BUILDKIT=1 for cache mounts.
# Podman enables BuildKit by default.

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Extract paddlepaddle-gpu from the official PaddlePaddle Docker image
# ═══════════════════════════════════════════════════════════════════════════════
# PaddlePaddle 3.3.1, CUDA 12.6, cuDNN 9.5, Python 3.10
# Has sm_90 (Hopper/H200) + sm_80 (Ampere/A100) + sm_89 (Ada/L40S) kernels
# We only COPY from this image, never rebuild this stage
FROM docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda12.6-cudnn9.5 AS paddle-source


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Base image = NGC Triton 24.10-py3
# ═══════════════════════════════════════════════════════════════════════════════
# Triton 24.10-py3 = last Triton image with Python 3.10 + CUDA 12.6
# Triton 2.51.0, Python 3.10.12, CUDA 12.6, cuDNN 9.5, TensorRT 10.5
# CUDA 12.6 + cuDNN 9.5 EXACTLY matches the PaddlePaddle image → no extra libs needed
FROM nvcr.io/nvidia/tritonserver:24.10-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── System dependencies (rarely changes → cached early) ────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libgflags2.2 \
        libgoogle-glog0v5 \
        libre2-9 \
        libb64-0d \
        libnuma1 \
        libarchive13 \
        fontconfig \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-cjk \
        fonts-wqy-microhei \
        fonts-freefont-ttf \
    && fc-cache -fv


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Copy paddlepaddle-gpu + install Python dependencies
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS deps

# Copy paddlepaddle-gpu from the PaddlePaddle image (2.9GB, ABI-compatible:
# both images use Python 3.10 + CUDA 12.6 + cuDNN 9.5 — exact match)
# No need to copy CUDA/cuDNN runtime libs: Triton 24.10 already has them.
COPY --from=paddle-source /usr/local/lib/python3.10/dist-packages/paddle /usr/local/lib/python3.10/dist-packages/paddle
COPY --from=paddle-source /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info /usr/local/lib/python3.10/dist-packages/paddlepaddle_gpu-3.3.1.dist-info

# Copy requirements files FIRST (small, rarely change → pip layer stays cached)
COPY deploy/hps/server_env/requirements/gpu.txt /tmp/requirements.txt
COPY deploy/hps/server_env/requirements/app.in /tmp/app.in

# Install all dependencies from gpu.txt, excluding the Baidu-blocked paddlepaddle-gpu line.
# Then install app.in deps + paddlepaddle-gpu's transitive deps that pip-compile omitted
# (decorator, astor — they'd normally come from the paddlepaddle-gpu wheel metadata).
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^\s*#' /tmp/requirements.txt \
    | grep -v '^\s*$' \
    | grep -v 'paddlepaddle' \
    | sed 's/[[:space:]]*#.*//' \
    | sed 's/[[:space:]]*$//' \
    > /tmp/deps.txt \
    && pip install -r /tmp/deps.txt \
    && pip install \
        "faiss-cpu>=1.8" "numpy>=1.24" "opencv-contrib-python==4.10.0.84" \
        "pycocotools>=2" "pydantic>=2" "typing-extensions>=4.11" \
    && pip install decorator astor \
    && rm /tmp/deps.txt /tmp/requirements.txt /tmp/app.in

# ─── TensorRT engine dependencies ───────────────────────────────────────────
# Native TRT runner: pycuda for GPU memory, paddle2onnx for Paddle→ONNX export,
# onnx/onnxsim/polygraphy for graph manipulation, nvidia-modelopt for INT8/FP8
# quantization.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Install PaddleX + HPS server (changes often → isolated for fast rebuild)
# ═══════════════════════════════════════════════════════════════════════════════
FROM deps AS app

# Install PaddleX from local repo (no-deps: dependencies already in deps stage)
COPY . /tmp/PaddleX
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /tmp/PaddleX \
    && rm -rf /tmp/PaddleX

# Install paddlex-hps-server
COPY deploy/hps/server_env/paddlex-hps-server /tmp/paddlex-hps-server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /tmp/paddlex-hps-server \
    && rm -rf /tmp/paddlex-hps-server


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Final runtime image
# ═══════════════════════════════════════════════════════════════════════════════
FROM app AS runtime

RUN mkdir -p /paddlex/var

COPY deploy/hps/sdk/common/server.sh /paddlex/server.sh
COPY deploy/hps/sdk/pipelines/layout_parsing/server/pipeline_config.yaml /paddlex/pipeline_config.yaml
COPY deploy/hps/sdk/pipelines/layout_parsing/server/model_repo /paddlex/model_repo
# server.sh expects config_gpu.pbtxt / config_cpu.pbtxt INSIDE each model dir in model_repo
# (normally placed there by the assemble.py SDK packaging script)
COPY deploy/hps/sdk/common/config_gpu.pbtxt /paddlex/model_repo/layout-parsing/config_gpu.pbtxt
COPY deploy/hps/sdk/common/config_cpu.pbtxt /paddlex/model_repo/layout-parsing/config_cpu.pbtxt

RUN chmod +x /paddlex/server.sh

# ─── Environment ────────────────────────────────────────────────────────────
ENV PADDLEX_HPS_DEVICE_TYPE=gpu
ENV PADDLEX_HPS_PIPELINE_CONFIG_PATH=/paddlex/pipeline_config.yaml
ENV PADDLEX_HPS_USE_HPIP=
# LD_LIBRARY_PATH: paddle's own libs + Triton's existing CUDA 12.6 / nvidia paths.
# No CUDA 13.0 compat workaround needed — CUDA 12.6 is the native version here.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

WORKDIR /paddlex

EXPOSE 8000 8001 8002

ENTRYPOINT ["/paddlex/server.sh"]
