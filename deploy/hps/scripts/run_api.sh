#!/usr/bin/env bash
#
# Launch the PaddleX HPS API compatibility layer using Granian (Rust ASGI server).
#
# Usage:
#   ./scripts/run_api.sh              # default port 8080
#   PORT=9000 ./scripts/run_api.sh    # custom port
#
# Environment variables (all have sensible defaults — see _core/config.py):
#   PORT                       Listen port (default: 8080)
#   HOST                       Bind address (default: 0.0.0.0)
#   WORKERS                    Number of Granian workers (default: 1)
#
#   HPS_API_MODEL              PaddleX model name (default: PP-DocLayoutV3)
#   HPS_API_PRECISION          TRT precision: fp8/fp16/int8 (default: fp8)
#   HPS_API_DEVICE_ID          GPU device ID (default: 0)
#   HPS_API_ENGINE_DIR         Path to TRT engine cache (default: /tmp/paddlex-engines)
#   HPS_API_MAX_IMAGE_DIM      Max image dimension before resize (default: 4096)
#   HPS_API_TIMEOUT            URL fetch timeout seconds (default: 30)
#   HPS_API_STARTUP_TIMEOUT    Model load deadline seconds (default: 300)
#   HPS_API_LOG_LEVEL          Logging level (default: INFO)
#
#   ── Throughput / Latency Tuning ──────────────────────────────────────────
#   HPS_API_PIPELINE_DEPTH     Concurrent in-flight GPU tasks (default: 4)
#   HPS_API_CPU_POOL_SIZE      Thread pool for CPU-bound stages (default: 8)
#   HPS_API_BATCH_SIZE         Micro-batch size for direct backend (default: 2)
#   HPS_API_BATCH_TIMEOUT_MS   Batch collection timeout ms (default: 5)
#   HPS_TRT_SKIP_D2H_COPY      Skip extra D2H array copy (default: 1=on for direct)
#   HPS_LATENCY_LOG            Enable per-stage latency logging (default: 0)
#
# All writable state (model cache, TRT engines, logs) goes to /tmp.
#
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# PaddleX is expected to be importable (installed or on PYTHONPATH).
# Prepend our package root so api_compat.* resolves.
export PYTHONPATH="${HPS_ROOT}:${PYTHONPATH:-}"

# ── PaddleX model directory setup ────────────────────────────────────────────
# PaddleX uses PADDLE_PDX_CACHE_HOME (NOT PADDLEX_HOME) to determine its
# cache directory (CACHE_DIR in paddlex/utils/cache.py). Models are searched
# at $PADDLE_PDX_CACHE_HOME/official_models/<model>. We set this to /tmp so
# all writes stay in /tmp, and copy the baked-in model files from
# /opt/models so PaddleX finds them without downloading (critical for
# offline / --network none operation). We copy (not symlink) because the
# TRT runner writes the built engine file next to the ONNX.
PADDLEX_CACHE_DIR="${PADDLE_PDX_CACHE_HOME:-/tmp/paddlex-home/.paddlex}"
MODEL_SRC="/opt/models/PP-DocLayoutV3"
MODEL_DST="${PADDLEX_CACHE_DIR}/official_models/PP-DocLayoutV3"

if [[ -d "${MODEL_SRC}" && ! -e "${MODEL_DST}" ]]; then
    mkdir -p "$(dirname "${MODEL_DST}")"
    cp -r "${MODEL_SRC}" "${MODEL_DST}"
fi

export PADDLE_PDX_CACHE_HOME="${PADDLEX_CACHE_DIR}"

echo "============================================"
echo " PaddleX HPS Docling API (Granian)"
echo "============================================"
echo " Port:           ${PORT}"
echo " Host:           ${HOST}"
echo " Workers:        ${WORKERS}"
echo " Model:          ${HPS_API_MODEL:-PP-DocLayoutV3}"
echo " Precision:      ${HPS_API_PRECISION:-fp8}"
echo " Device:         ${HPS_API_DEVICE_ID:-0}"
echo " EngineDir:      ${HPS_API_ENGINE_DIR:-/tmp/paddlex-engines}"
echo " PipelineDepth:  ${HPS_API_PIPELINE_DEPTH:-4}"
echo " CpuPoolSize:    ${HPS_API_CPU_POOL_SIZE:-8}"
echo " BatchSize:      ${HPS_API_BATCH_SIZE:-2}"
echo " BatchTimeoutMs: ${HPS_API_BATCH_TIMEOUT_MS:-5}"
echo "============================================"

# ── Latency/throughput optimizations ─────────────────────────────────────────
# Skip the redundant D2H .copy() in the TRT runner (safe for single-threaded
# inference — the direct backend uses one inference thread, so the host buffer
# is consumed before the next call overwrites it).
export HPS_TRT_SKIP_D2H_COPY="${HPS_TRT_SKIP_D2H_COPY:-1}"

exec granian \
    --interface asgi \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    "api_compat.docling_api.app:app"
