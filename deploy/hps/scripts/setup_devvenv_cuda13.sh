#!/usr/bin/env bash
#
# Create the CUDA 13 development virtualenv for the HPS API compat layer.
#
# Why this approach:
#   The Paddle CDN (paddle-whl.bj.bcebos.com) is ISP-blocked on this host, so
#   we cannot `pip install paddlepaddle-gpu`. Instead we transplant the paddle
#   runtime from the locally-cached paddlepaddle/paddle:3.3.1-gpu-cuda13.0 base
#   image (Python 3.10 ABI — matches the image). tensorrt_cu13 installs fine
#   from PyPI.
#
# Usage:
#   ./scripts/setup_devvenv_cuda13.sh
#
# Creates: deploy/hps/.venv-cuda13-py310  (Python 3.10 + tensorrt_cu13 + paddle)
#
# Note: paddlex is installed from the WORKSPACE source (repo root) as an
# editable, no-deps package so the custom TensorRT engine patches in the
# workspace repo are used (the Dockerfile applies these too). This also makes
# `importlib.metadata.version("paddlex")` resolve so deps.py checks pass.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HPS_DIR="${REPO_ROOT}/deploy/hps"
VEN_DIR="${HPS_DIR}/.venv-cuda13-py310"
PADDLE_IMG="docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13"
IN_IMG="/usr/local/lib/python3.10/dist-packages"
SITE_PKGS="${VEN_DIR}/lib/python3.10/site-packages"

# ── 0. Python 3.10 (uv fetches it without the blocked CDN) ────────────────
if ! command -v python3.10 >/dev/null 2>&1; then
    echo ">>> Installing Python 3.10 via uv..."
    uv python install 3.10
fi

echo ">>> Creating venv at ${VEN_DIR}"
uv venv --python python3.10 "${VEN_DIR}"
mkdir -p "${SITE_PKGS}"

# ── 1. tensorrt_cu13 from PyPI ─────────────────────────────────────────────
echo ">>> Installing tensorrt_cu13==10.16.1.11"
"${VEN_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${VEN_DIR}/bin/python" -m pip install --quiet -U pip setuptools wheel
"${VEN_DIR}/bin/python" -m pip install "tensorrt_cu13==10.16.1.11"

# ── 2. Extract paddle + nvidia libs from the local cuda13 base image ────────
echo ">>> Creating container from ${PADDLE_IMG} to extract paddle"
CID="$(podman create "${PADDLE_IMG}" sleep 1 2>/dev/null)"
trap 'podman rm -f "${CID}" >/dev/null 2>&1 || true' EXIT

echo ">>> Copying paddle / numpy / nvidia from image"
for pkg in paddle numpy numpy.libs nvidia; do
    if [[ ! -e "${SITE_PKGS}/${pkg}" ]]; then
        podman cp "${CID}:${IN_IMG}/${pkg}" "${SITE_PKGS}/"
    fi
done

echo ">>> Copying .dist-info files"
for d in \
    paddlepaddle_gpu-3.3.1.dist-info numpy-2.2.6.dist-info \
    nvidia_cublas-13.0.2.14.dist-info nvidia_cudnn_cu13-9.13.0.50.dist-info \
    nvidia_cufft-12.0.0.61.dist-info nvidia_curand-10.4.0.35.dist-info \
    nvidia_cusolver-12.0.4.66.dist-info nvidia_cusparse-12.6.3.3.dist-info \
    nvidia_cusparselt_cu13-0.8.1.dist-info nvidia_nccl_cu13-2.28.3.dist-info \
    nvidia_nvjitlink-13.0.88.dist-info nvidia_nvtx-13.0.85.dist-info \
    nvidia_cufile-1.15.1.6.dist-info nvidia_cuda_cccl-13.0.85.dist-info \
    nvidia_cuda_cupti-13.0.85.dist-info cuda_python-13.0.3.dist-info \
    ; do
    podman cp "${CID}:${IN_IMG}/${d}" "${SITE_PKGS}/" 2>/dev/null \
        || echo "    (skip) ${d}"
done

podman rm -f "${CID}" >/dev/null 2>&1 || true
trap - EXIT

# ── 3. Pure-python deps (PyPI is reachable) ────────────────────────────────
echo ">>> Installing pure Python deps"
"${VEN_DIR}/bin/python" -m pip install \
    httpx "opt_einsum==3.3.0" networkx pillow typing_extensions \
    safetensors protobuf decorator astor setuptools wheel \
    scipy==1.15.2 "scikit-image==0.24.0" "scikit-learn==1.6.1" \
    "faiss-cpu>=1.8" pycocotools "pypdfium2>=4.30.0" \
    colorlog aiohttp packaging prettytable py-cpuinfo \
    "PyYAML==6.0.2" ruamel.yaml pandas pyclipper shapely tqdm \
    filelock huggingface-hub chardet einops ftfy GPUtil jieba \
    imagesize joblib lxml matplotlib openpyxl premailer pypinyin \
    python-bidi regex requests sentencepiece soundfile ujson openai \
    tiktoken tokenizers modelscope aistudio-sdk langchain \
    langchain-community langchain-core langchain-text-splitters \
    beautifulsoup4 python-docx chinese-calendar OpenCC latex2mathml \
    prettytable premailer pypdfium2

# ── 3b. opencv-contrib (NOT headless) — paddlex deps.py checks metadata ────
# paddlex.utils.deps.is_dep_available("opencv-contrib-python") requires the
# package installed under that exact name (importlib.metadata lookup), so the
# headless wheel is insufficient.
echo ">>> Installing opencv-contrib-python"
"${VEN_DIR}/bin/python" -m pip install "opencv-contrib-python==4.10.0.84"

# ── 3c. Workspace paddlex (editable, no-deps) + pycuda ──────────────────────
echo ">>> Installing workspace paddlex (editable, no deps)"
"${VEN_DIR}/bin/python" -m pip install --no-deps -e "${REPO_ROOT}"
echo ">>> Installing pycuda"
"${VEN_DIR}/bin/python" -m pip install "pycuda>=2024.1"

# ── 4. API compat app deps ──────────────────────────────────────────────────
echo ">>> Installing HPS API deps"
"${VEN_DIR}/bin/python" -m pip install \
    -r "${HPS_DIR}/requirements.base.txt" \
    -r "${HPS_DIR}/requirements.direct.txt"

cat <<'EOF'

✅ Dev venv ready. Activate it with:

    source deploy/hps/.venv-cuda13-py310/bin/activate
    export VENV="$(pwd)/deploy/hps/.venv-cuda13-py310"
    export LD_LIBRARY_PATH="$VENV/lib/python3.10/site-packages/nvidia/cu13/lib:$VENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.10/site-packages/nvidia/nccl/lib:$VENV/lib/python3.10/site-packages/nvidia/cusparselt/lib:$VENV/lib/python3.10/site-packages/paddle/libs:$VENV/lib/python3.10/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}"
EOF