# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — Dockerfile (Triton + Granian, GPU, continuous batching)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This image runs TWO processes:
#   1. tritonserver — PP-DocLayoutV3 with dynamic_batching (continuous batching)
#   2. granian      — FastAPI/ASGI server for Docling-compatible REST API
#
# Triton handles GPU scheduling, request batching, and model lifecycle.
# Granian handles HTTP, image loading, DoclingDocument conversion, and export.
# They communicate via gRPC on localhost:8001.
#
# Build (from PaddleX repo root):
#   podman build \
#     -t paddlex-hps-api-triton \
#     -f deploy/hps/docker/legacy/Dockerfile \
#     .
#
# Run:
#   podman run --rm --gpus all -p 8080:8080 paddlex-hps-api-triton
#
# Environment:
#   HPS_API_PIPELINE_DEPTH=3     Max concurrent in-flight gRPC requests
#   HPS_API_CPU_POOL_SIZE=4      Thread pool for CPU-bound stages
#   PORT=8080                    HTTP listen port
# ═══════════════════════════════════════════════════════════════════════════════

FROM localhost/paddlex-hps-ngc:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─── 1. System packages ──────────────────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── 2a. TensorRT Python bindings (same as Dockerfile.api_compat) ────────────
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tensorrt_cu13==10.16.1.11" \
        pycuda \
        paddle2onnx onnx onnxsim polygraphy \
        nvidia-modelopt

# ─── 2b. Triton client + API Python dependencies ─────────────────────────────
COPY deploy/hps/requirements.api_compat.txt /tmp/requirements.api_compat.txt
RUN --mount=type=cache,target=/tmp/pip-cache \
    pip install --no-cache-dir \
        "tritonclient[grpc]>=2.30.0" \
        -r /tmp/requirements.api_compat.txt \
    && rm /tmp/requirements.api_compat.txt

# ─── 2c. Patch PaddleX with custom TensorRT engine ──────────────────────────
COPY paddlex/inference/models/engines/tensorrt.py \
     paddlex/inference/models/runners/tensorrt_runner.py \
     /tmp/paddlex-patch/
COPY deploy/hps/scripts/patch_paddlex_trt.py /tmp/patch_paddlex_trt.py
RUN python3 /tmp/patch_paddlex_trt.py \
        /tmp/paddlex-patch \
        /usr/local/lib/python3.10/dist-packages/paddlex \
    && rm -rf /tmp/paddlex-patch /tmp/patch_paddlex_trt.py

# ─── 3. Application code + Triton model repository ──────────────────────────
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/triton/model_repo/ /opt/triton-model-repo/
COPY deploy/hps/scripts/run_api_triton.sh /opt/scripts/run_api_triton.sh
RUN chmod +x /opt/scripts/run_api_triton.sh

# ─── 4. Model weights + TRT engines (baked in) ──────────────────────────────
COPY models/PP-DocLayoutV3/ /opt/models/PP-DocLayoutV3/

# ─── 5. Non-root user ───────────────────────────────────────────────────────
RUN groupadd -g 10000 runner \
    && useradd -m -s /bin/bash -u 10000 -g 10000 runner

RUN chmod -R a+rX /opt/models /opt/api_compat /opt/triton-model-repo /opt/scripts

# ─── 6. Writable state → /tmp ───────────────────────────────────────────────
RUN mkdir -p /tmp/paddlex-engines /tmp/paddlex-home /tmp/triton-model-repo \
    && chown -R runner:runner /tmp/paddlex-engines /tmp/paddlex-home /tmp/triton-model-repo \
    && chmod 1777 /tmp

ENV PADDLEX_HOME=/tmp/paddlex-home
ENV PADDLE_PDX_CACHE_HOME=/tmp/paddlex-home/.paddlex
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines
ENV HPS_API_MODEL=PP-DocLayoutV3
ENV HPS_API_PRECISION=fp8
ENV HPS_API_DEVICE_ID=0
ENV HPS_API_BACKEND=triton
ENV HPS_TRITON_URL=localhost:8001
ENV HPS_TRITON_MODEL_NAME=doclayout-v3
ENV HPS_API_STARTUP_TIMEOUT=120
ENV PYTHONPATH=/opt

USER runner
WORKDIR /opt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/health-check || exit 1

ENTRYPOINT ["/opt/scripts/run_api_triton.sh"]
