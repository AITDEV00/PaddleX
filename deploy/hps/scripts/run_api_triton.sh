#!/usr/bin/env bash
#
# Launch PaddleX HPS API with Triton Inference Server + Granian (ASGI).
#
# This script starts two processes:
#   1. tritonserver  — GPU inference with dynamic batching (continuous batching)
#   2. granian       — FastAPI/ASGI server for the Docling-compatible API
#
# Triton manages the PP-DocLayoutV3 model lifecycle, dynamic batching, and
# GPU scheduling. Granian handles HTTP requests, image loading, DoclingDocument
# conversion, and format export. The two communicate via gRPC on localhost.
#
# Usage:
#   ./scripts/run_api_triton.sh              # default port 8080
#   PORT=9000 ./scripts/run_api_triton.sh    # custom port
#
# Environment variables:
#   PORT                       Listen port for Granian (default: 8080)
#   HOST                       Bind address (default: 0.0.0.0)
#   WORKERS                    Number of Granian workers (default: 1)
#   HPS_TRITON_URL             Triton gRPC URL (default: localhost:8001)
#   HPS_TRITON_MODEL_NAME      Triton model name (default: doclayout-v3)
#   HPS_API_PIPELINE_DEPTH     Max concurrent in-flight requests (default: 3)
#   HPS_API_CPU_POOL_SIZE      Thread pool for CPU stages (default: 4)
#
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
TRITON_URL="${HPS_TRITON_URL:-localhost:8001}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${HPS_ROOT}:${PYTHONPATH:-}"

# ── PaddleX model directory setup (same as run_api.sh) ────────────────────────
PADDLEX_CACHE_DIR="${PADDLE_PDX_CACHE_HOME:-/tmp/paddlex-home/.paddlex}"
MODEL_SRC="/opt/models/PP-DocLayoutV3"
MODEL_DST="${PADDLEX_CACHE_DIR}/official_models/PP-DocLayoutV3"

if [[ -d "${MODEL_SRC}" && ! -e "${MODEL_DST}" ]]; then
    mkdir -p "$(dirname "${MODEL_DST}")"
    cp -r "${MODEL_SRC}" "${MODEL_DST}"
fi
export PADDLE_PDX_CACHE_HOME="${PADDLEX_CACHE_DIR}"

# ── Triton model repository setup ─────────────────────────────────────────────
# In Docker, the model repo is at /opt/triton-model-repo.
# In dev (repo root), it's at ${HPS_ROOT}/triton/model_repo.
if [[ -d "/opt/triton-model-repo" ]]; then
    TRITON_REPO_SRC="/opt/triton-model-repo"
else
    TRITON_REPO_SRC="${HPS_ROOT}/triton/model_repo"
fi
TRITON_REPO_DST="/tmp/triton-model-repo"

rm -rf "${TRITON_REPO_DST}"
cp -r "${TRITON_REPO_SRC}" "${TRITON_REPO_DST}"

# Set environment variables for the Triton Python backend model
export HPS_API_MODEL="${HPS_API_MODEL:-PP-DocLayoutV3}"
export HPS_API_PRECISION="${HPS_API_PRECISION:-fp8}"
export HPS_API_DEVICE_ID="${HPS_API_DEVICE_ID:-0}"
export HPS_API_ENGINE_PATH="${HPS_API_ENGINE_DIR:-/tmp/paddlex-engines}/inference_${HPS_API_PRECISION}.trt"

# Tell api_compat to use the Triton backend
export HPS_API_BACKEND="triton"
export HPS_TRITON_URL="${TRITON_URL}"
export HPS_TRITON_MODEL_NAME="${HPS_TRITON_MODEL_NAME:-doclayout-v3}"

echo "============================================"
echo " PaddleX HPS Docling API (Triton + Granian)"
echo "============================================"
echo " HTTP Port:   ${PORT}"
echo " Triton URL:  ${TRITON_URL}"
echo " Model:       ${HPS_API_MODEL}"
echo " Precision:   ${HPS_API_PRECISION}"
echo " Backend:     triton (dynamic_batching)"
echo "============================================"

# ── Start Triton Inference Server ─────────────────────────────────────────────
# Triton manages the GPU model, dynamic batching, and request scheduling.
# The Python backend (model.py) loads PaddleX's create_model() and processes
# batched requests from Triton's dynamic batcher.

# Ensure libcuda.so.1 is findable by the Python backend subprocess.
# In WSL2, the driver lib is at /usr/lib/wsl/lib/ (mounted from host).
# Triton's Python backend doesn't inherit LD_LIBRARY_PATH from the parent
# unless we export it explicitly.
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

TRITON_LOG_DIR="${TRITON_LOG_DIR:-/tmp/triton-logs}"
mkdir -p "${TRITON_LOG_DIR}"

tritonserver \
    --model-repository="${TRITON_REPO_DST}" \
    --backend-config=python,shm-default-byte-size=104857600,shm-growth-byte-size=10485760 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --log-info=1 \
    --log-warning=1 \
    --log-error=1 \
    > "${TRITON_LOG_DIR}/triton.log" 2>&1 &

TRITON_PID=$!
echo "Triton started (PID=${TRITON_PID}, logs at ${TRITON_LOG_DIR}/triton.log)"

# Wait for Triton to be ready
echo "Waiting for Triton to load model..."
TRITON_READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1 && \
       curl -sf http://localhost:8000/v2/models/doclayout-v3/ready > /dev/null 2>&1; then
        TRITON_READY=true
        echo "Triton ready (after ${i}s)"
        break
    fi
    sleep 1
done

if [[ "${TRITON_READY}" != "true" ]]; then
    echo "ERROR: Triton failed to become ready within 120s"
    echo "--- Triton log (last 30 lines) ---"
    tail -30 "${TRITON_LOG_DIR}/triton.log" || true
    kill "${TRITON_PID}" 2>/dev/null || true
    exit 1
fi

# ── Start Granian (ASGI server for api_compat) ────────────────────────────────
# Granian serves the Docling-compatible REST API. It talks to Triton via gRPC.

cleanup() {
    echo "Shutting down..."
    kill "${TRITON_PID}" 2>/dev/null || true
    wait "${TRITON_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec granian \
    --interface asgi \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    "api_compat.docling_api.app:app"
