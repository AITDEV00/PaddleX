# Custom PaddleX HPS Image — NGC Triton Build Guide

This document describes how to build a custom PaddleX HPS (High Stability Serving)
Docker image using NGC Triton as the base instead of the official Baidu-hosted
image. This is needed when `bcebos.com` / `baidubce.com` is blocked (e.g. Etisalat
networks in the UAE) or when you need GPU architecture support not available in
the stock image.

## Background

The stock PaddleX HPS build pipeline (`scripts/build_deployment_image.sh`) pulls
base images and paddlepaddle-gpu wheels from Baidu's registry
(`ccr-2vdh3abv-pub.cnc.bj.baidubce.com`), which is **blocked** on some networks.
The custom Dockerfiles in this directory bypass the Baidu dependency entirely by:

1. Extracting `paddlepaddle-gpu` from the official Docker Hub image
   (`docker.io/paddlepaddle/paddle`)
2. Using NGC Triton 24.10-py3 as the serving base (Python 3.10, CUDA 12.6,
   TensorRT 10.5)
3. Installing PaddleX + HPS server from the local repo (no Baidu downloads)

## Two Variants

| File | PaddlePaddle | CUDA | cuDNN | Target | sm_ |
|------|-------------|------|-------|------------|-----|
| `docker/cuda13/base/Dockerfile` | 3.3.1-cuda13.0-cudnn9.13 | 13.0 | 9.13 | **RTX 50-series** (needs driver ≥ CUDA 13) | sm_120 |
| `docker/cuda12/base/Dockerfile` | 3.3.1-cuda12.6-cudnn9.5 | 12.6 | 9.5 | **H200/H100**, A100, L40S, RTX 40-series | sm_90/80/89 |

### Why two variants?

- **CUDA 13 (RTX 50-series / Blackwell, sm_120)** requires CUDA 13.0 — PaddlePaddle 3.0.0 (the
  previous version) lacks sm_120 kernels, causing silent inference failures
  (CUDA error 209, 0 results returned). PaddlePaddle 3.3.1 + CUDA 13.0 adds
  sm_120 kernels.
- **CUDA 12 (H200, sm_90)** works with CUDA 12.6, which is the **exact same**
  CUDA/cuDNN version as Triton 24.10. This means no extra runtime libraries need
  to be copied, making the build simpler and the image ~1.5 GB smaller.

> **Why separate by CUDA version, not GPU:** the container's CUDA runtime libs
> must be ≤ the maximum CUDA the **host driver** supports. An H200 host whose
> driver tops out at CUDA 12.8 cannot run a CUDA 13.0 image, even though the
> engine is architecture-agnostic. Hence the `cuda13/` and `cuda12/` folders.
> The lean images build the TensorRT engine at runtime, so each is universal
> across GPU archs within its CUDA version.

> **Note:** PaddlePaddle does not publish a CUDA 12.8 image. The closest options
> are 12.6 and 12.9. We chose 12.6 because it matches Triton 24.10 exactly,
> avoiding any library conflicts. CUDA 12.6 supports Hopper (sm_90) fully.

## Prerequisites

### Network requirements

Accessible hosts (not blocked):
- `docker.io` — PaddlePaddle + base images
- `nvcr.io` — NGC Triton image
- `huggingface.co` — model weights (7 models, ~2 GB total)
- `pypi.org` — Python dependencies

Blocked (not needed by this build):
- `*.bcebos.com`, `*.baidubce.com` — Baidu model/file hosting

### Host requirements

- **Podman** ≥ 4.0 or **Docker** ≥ 20.10 with BuildKit
- **NVIDIA Container Toolkit** (or CDI for Podman) for GPU access
- **NVIDIA driver** ≥ 550 (for CUDA 12.6) or ≥ 580 (for CUDA 13.0)
  - RTX 5090 requires driver ≥ 580 + CUDA 13.0
  - H200 requires driver ≥ 545 + CUDA 12.6

### GPU driver note (CUDA 13 / RTX 5090 only)

The RTX 5090 (Blackwell, sm_120) host must run driver ≥ 591. The PaddlePaddle
3.3.1 CUDA 13.0 image includes a **compat `libcuda.so`** at
`/usr/local/cuda-13.0/compat/` that only supports drivers ≤ 580. The
`docker/cuda13/base/Dockerfile` handles this by:

- **NOT copying** the compat directory
- Setting `LD_LIBRARY_PATH` to exclude it, so the host driver's `libcuda.so`
  (provided via CDI / NVIDIA Container Toolkit) is used instead

Without this fix, `paddle.device.cuda.device_count()` returns 0.

## Build

### 1. Clone and prepare

```bash
cd /path/to/PaddleX
```

### 2. Build the image

**For CUDA 13 / RTX 5090 (needs driver ≥ CUDA 13):**
```bash
podman build -t paddlex-hps-ngc \
  -f deploy/hps/docker/cuda13/base/Dockerfile .
```

**For CUDA 12.6 (H200 / H100 / A100 / L40S / RTX 40-series):**
```bash
podman build -t paddlex-hps-ngc-cuda12 \
  -f deploy/hps/docker/cuda12/base/Dockerfile .
```

Build takes ~10 minutes on first run (pulling base images), ~2 minutes on
subsequent runs (cached layers).

### 3. Verify the image

```bash
# Check PaddlePaddle GPU is detected inside the image
podman run --rm --device nvidia.com/gpu=all -e NVIDIA_DISABLE_REQUIRE=1 \
  paddlex-hps-ngc \
  python3 -c "import paddle; print('GPU count:', paddle.device.cuda.device_count()); print('GPU:', paddle.device.cuda.get_device_name(0))"
```

Expected output (RTX 5090):
```
GPU count: 1
GPU: NVIDIA GeForce RTX 5090
```

## Deploy

### Standard deployment

```bash
podman run --rm -d \
  --name paddlex-hps \
  --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  --device nvidia.com/gpu=all \
  paddlex-hps-ngc:latest
```

### H200 MIG deployment

When using H200 MIG partitions, specify the MIG device instead of `gpu=all`:

```bash
# Example: use MIG device 1g.20gb on GPU 0
podman run --rm -d \
  --name paddlex-hps-mig \
  --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  -e CUDA_VISIBLE_DEVICES=MIG-<uuid> \
  --device nvidia.com/gpu=all \
  paddlex-hps-ngc-cuda12:latest
```

To find the MIG device UUID:
```bash
nvidia-smi --query-gpu=mig.uuid --format=csv
# or for a specific instance:
nvidia-smi -i 0 --query-gpu=mig.uuid --format=csv
```

Alternatively, use CDI with a MIG-specific CDI spec:
```bash
--device nvidia.com/gpu=<mig-cdi-name>
```

### Health check

Wait ~60 seconds for models to load, then check:

```bash
# Server ready?
curl -s http://localhost:8000/v2/health/readycheck
# Model loaded?
curl -s http://localhost:8000/v2/models/layout-parsing/ready
```

## Inference test (hello world)

```bash
python3 -c "
import json, base64, requests

with open('sample.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

inner = json.dumps({'file': img_b64, 'fileType': 1, 'visualize': False})
payload = {
    'inputs': [{
        'name': 'input',
        'datatype': 'BYTES',
        'shape': [1, 1],
        'data': [inner]
    }]
}

resp = requests.post('http://localhost:8000/v2/models/layout-parsing/infer',
                     json=payload, timeout=300)
result = resp.json()
output = json.loads(result['outputs'][0]['data'][0])
print('errorCode:', output['errorCode'])

blocks = output['result']['layoutParsingResults'][0]['prunedResult']['parsing_res_list']
print(f'Blocks detected: {len(blocks)}')
for b in blocks[:5]:
    print(f\"  {b['block_label']:16s} | {b['block_content'][:80]}...\")
"
```

Expected output:
```
errorCode: 0
Blocks detected: 6
  doc_title        | LayoutParser:A Unified Toolkit for Deep Learning Based Document Image Analysis...
  text             | Zejiang Shen1 (), Ruochen Zhang², Melissa Dell², Benjamin Charles Germain...
  ...
```

## Architecture

### Dockerfile structure (5 stages)

```
Stage 1 (paddle-source)  ─── Extract paddlepaddle-gpu from Docker Hub image
                             [+ CUDA 13.0 runtime libs — CUDA 13 variant only]
Stage 2 (base)           ─── NGC Triton 24.10-py3 + system deps (fonts, libs)
Stage 3 (deps)           ─── COPY paddle libs + pip install Python dependencies
Stage 4 (app)            ─── Install PaddleX + paddlex-hps-server from local repo
Stage 5 (runtime)        ─── Copy server.sh, pipeline_config, model_repo + ENV
```

### CUDA 13.0 variant — library coexistence

The CUDA 13.0 variant (`docker/cuda13/base/Dockerfile`) copies CUDA 13.0 runtime `.so` files
into the Triton image. These coexist with Triton's CUDA 12.6 libraries because
they use **different sonames**:

| Library | CUDA 12.6 (Triton) | CUDA 13.0 (Paddle) |
|---------|--------------------|--------------------|
| cudart  | `libcudart.so.12`  | `libcudart.so.13`  |
| cublas  | `libcublas.so.12`  | `libcublas.so.13`  |
| cufft   | `libcufft.so.12`   | `libcufft.so.12`   |
| etc.    | `.so.12`           | `.so.13`           |

cuDNN 9.13 libs are copied to `/usr/local/cudnn-9.13/` (separate directory) to
avoid clobbering Triton's cuDNN 9.5 files (both use soname `.so.9`).

The CUDA 12.6 variant (`docker/cuda12/base/Dockerfile`) does **not** need any of this —
PaddlePaddle's CUDA 12.6 + cuDNN 9.5 exactly matches Triton 24.10.

### Model weights

Models are downloaded from HuggingFace on first startup (not Baidu):
- `PADDLE_PDX_MODEL_SOURCE=huggingface` (set by default in PaddleX 3.x)
- 7 models for the `layout_parsing` pipeline (~2 GB total):
  - `PP-LCNet_x1_0_doc_ori` — document orientation
  - `UVDoc` — image unwarping
  - `RT-DETR-H_layout_17cls` — layout detection
  - `PP-OCRv4_server_det` — text detection
  - `PP-OCRv4_server_rec` — text recognition
  - `PP-OCRv4_server_seal_det` — seal detection
  - `SLANet_plus` — table recognition

## Known issues

### Font download blocked (bcebos.com)

When `visualize: true` is set in the inference request, PaddleX tries to download
a font from `paddle-model-ecology.bj.bcebos.com` (blocked). Workarounds:

1. **Set `visualize: false`** in the request payload (recommended for API usage)
2. **Bake a font into the image** — set the `PADDLE_PDX_LOCAL_FONT_FILE_PATH`
   environment variable to a local `.ttf` file:
   ```bash
   -e PADDLE_PDX_LOCAL_FONT_FILE_PATH=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
   ```

### Triton Python backend version lock

Triton's Python backend embeds Python **in-process** (not a subprocess). The
Triton image's Python version **must** match the PaddlePaddle image's Python
version. Both PaddlePaddle 3.3.1 images and Triton 24.10-py3 use **Python 3.10**.

Triton 24.11+ switched to Python 3.12 — do not use those images with this
Dockerfile.

### `CudaDriverHelper has not been initialized` warning

Triton logs `E ... server.cc:241] "CudaDriverHelper has not been initialized."`
on startup. This is a **benign warning** — it does not prevent GPU inference.
PaddlePaddle initializes CUDA through its own runtime, not Triton's CudaDriverHelper.
