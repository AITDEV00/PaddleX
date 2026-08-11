#!/usr/bin/env bash
#
# Launch the HPS Docling API compat server from the CUDA 13 dev venv using the
# DIRECT backend (in-process PaddleX/TensorRT). No Triton required — the
# /v1/models endpoints work from the direct backend.
#
# Usage:
#   ./scripts/run_dev.sh [port]          # default port 8080
#
# Env overrides (all optional):
#   HPS_API_BACKEND   direct|triton   (default: direct)
#   HPS_API_PRECISION fp16|fp8        (default: fp16)
#   HPS_API_DEVICE_ID 0               (default: 0)
#   PORT              8080            (default: 8080)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${HPS_DIR}/../.." && pwd)"

VENV="${HPS_DIR}/.venv-cuda13-py310"
PYTHON="${VENV}/bin/python"

PORT="${PORT:-8080}"
export HPS_API_BACKEND="${HPS_API_BACKEND:-direct}"
export HPS_API_PRECISION="${HPS_API_PRECISION:-fp16}"
export HPS_API_DEVICE_ID="${HPS_API_DEVICE_ID:-0}"
export HPS_API_STARTUP_TIMEOUT="${HPS_API_STARTUP_TIMEOUT:-300}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/tmp/paddlex-host/.paddlex}"
export PADDLEX_HOME="${PADDLEX_HOME:-/tmp/paddlex-host}"
export HPS_API_ENGINE_DIR="${HPS_API_ENGINE_DIR:-/tmp/paddlex-engines}"

export PYTHONPATH="${REPO_ROOT}:${HPS_DIR}:${PYTHONPATH:-}"

SITE_PKGS="${VENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/cu13/lib:${SITE_PKGS}/nvidia/cudnn/lib:${SITE_PKGS}/nvidia/nccl/lib:${SITE_PKGS}/nvidia/cusparselt/lib:${SITE_PKGS}/paddle/libs:${SITE_PKGS}/tensorrt_libs:${LD_LIBRARY_PATH:-}"

echo ">>> HPS_API_BACKEND=${HPS_API_BACKEND} precision=${HPS_API_PRECISION} port=${PORT}"
echo ">>> Launching granian (ASGI) on 0.0.0.0:${PORT} ..."
exec "${VENV}/bin/granian" --interface asgi --host 0.0.0.0 --port "${PORT}" --workers 1 "api_compat.docling_api.app:app"