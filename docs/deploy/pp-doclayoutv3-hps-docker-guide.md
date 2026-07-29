# Serving PP-DocLayoutV3 via PaddleX HPS (Triton) — Podman Deployment Guide

> **Goal:** Deploy a Podman container that serves **only PP-DocLayoutV3** (layout detection) using PaddleX's High-Stability Serving (HPS) with Triton Inference Server, including dynamic batching and concurrent GPU instances.
>
> **Note:** The official PaddleX docs use `docker`. This guide substitutes `podman` for all container commands. The pre-built image is OCI-compatible and runs identically under Podman.

> **Official doc reference:** [PaddleX Serving Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html) → Section 2: High-Stability Serving
>
> **Source files:** `deploy/hps/` in `AITDEV00/PaddleX` fork, branch `jya0-v3.7.2` (release `v3.7.2`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Podman Container (paddlex/hps:paddlex3.7-gpu)             │
│                                                             │
│  ┌─────────────┐   ┌──────────────────────────────────────┐ │
│  │  Triton     │   │  Python Backend (model.py)            │ │
│  │  Inference  │──▶│  loads PipelineWrapper                │ │
│  │  Server     │   │  runs layout_parsing pipeline         │ │
│  │  :8000 HTTP │   │  with only LayoutDetection            │ │
│  │  :8001 gRPC │   │  (PP-DocLayoutV3) enabled             │ │
│  │  :8002 Metrics│ └──────────────────────────────────────┘ │
│  └─────────────┘                                           │
│         ▲                                                   │
│         │ instance_group (count: N, KIND_GPU)               │
│         │ dynamic_batching (max_batch_size: 8)              │
└─────────┼───────────────────────────────────────────────────┘
          │
     Client (gRPC/HTTP)
```

**Key difference from basic serving:**
- Basic serving (`paddlex --serve`): Uvicorn single-worker + Queue/Thread serialization = **1 request at a time, no cross-request batching**
- HPS (Triton): dynamic batching + concurrent model instances = **8× throughput by default, scalable to N× with instance count**

> ⚠️ **PaddleX serves pipelines, not modules.** PP-DocLayoutV3 is a *module* (layout_detection). To serve it, you deploy the `layout_parsing` *pipeline* with all other submodules disabled, as documented in the official serving guide.

---

## Prerequisites

### Host machine requirements (from official doc)

| Requirement | Version |
|---|---|
| OS | **Linux only** (HPS does not support Windows/macOS) |
| Podman | ≥ 4.0 (tested with 4.9.3) |
| NVIDIA GPU | Drivers supporting CUDA 11.8 |
| NVIDIA Container Toolkit | Latest (for `--gpus all` via CDI) |
| Disk space | ~15 GB (image + model weights) |

> **Podman GPU setup:** Podman uses CDI (Container Device Interface) for GPU access. The NVIDIA Container Toolkit generates the CDI spec at `/etc/cdi/nvidia.yaml`. Verify with:
> ```bash
> podman run --rm --gpus all nvcr.io/nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
> ```
> If `--gpus all` fails, use CDI directly: `--device nvidia.com/gpu=all`

### Two paths to deployment

The official PaddleX serving doc (`docs/pipeline_deploy/serving.en.md`) documents **two paths**:

| | Path A: Pre-built (recommended) | Path B: Build from source |
|---|---|---|
| **Container image** | `podman pull` from Baidu registry | Build with cuDNN + TensorRT tarballs |
| **Pipeline SDK** | Download pre-built tarball from Baidu | Run `assemble.sh` from repo source |
| **When to use** | Standard deployment — fastest path | Custom image modifications, air-gapped environments, or Baidu registry unreachable |

---

## Path A: Pre-built (Official Recommended Flow)

This follows the exact steps documented in `docs/pipeline_deploy/serving.en.md` § 2.

### Step A1: Download the pre-built HPS SDK

From the official serving doc (§ 2.1), download the `layout_parsing` SDK:

```bash
mkdir -p ~/paddlex-layout-server && cd ~/paddlex-layout-server

# Download the pre-built layout_parsing HPS SDK (v3.7)
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.7/paddlex_hps_layout_parsing_sdk.tar.gz

# Extract
tar xzf paddlex_hps_layout_parsing_sdk.tar.gz --strip-components=1
ls
# Expected: server/  client/  version.txt
```

### Step A2: Edit `server/pipeline_config.yaml` for layout-only PP-DocLayoutV3

The downloaded SDK's `server/pipeline_config.yaml` defaults to `RT-DETR-H_layout_17cls` with OCR, table, seal, and formula submodules enabled. Per the official doc (§ 2.2): *"Users can modify this file to set the model directory to use, etc."*

```bash
cd ~/paddlex-layout-server/server
nano pipeline_config.yaml
```

Replace the contents with this layout-only config:

```yaml
pipeline_name: layout_parsing

use_doc_preprocessor: False
use_seal_recognition: False
use_table_recognition: False
use_formula_recognition: False

SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    model_dir: null
    batch_size: 8
    threshold: 0.3
    layout_nms: True
    layout_unclip_ratio: [1.0, 1.0]
    layout_merge_bboxes_mode:
      0: "union"    # abstract
      1: "union"    # algorithm
      2: "union"    # aside_text
      3: "large"    # chart
      4: "union"    # content
      5: "large"    # display_formula
      6: "large"    # doc_title
      7: "union"    # figure_title
      8: "union"    # footer
      9: "union"    # footer_image
      10: "union"   # footnote
      11: "union"   # formula_number
      12: "union"   # header
      13: "union"   # header_image
      14: "union"   # image
      15: "large"   # inline_formula
      16: "union"   # number
      17: "large"   # paragraph_title
      18: "union"   # reference
      19: "union"   # reference_content
      20: "union"   # seal
      21: "union"   # table
      22: "union"   # text
      23: "union"   # text
      24: "union"   # vision_footnote
```

> **Why the 25-class `layout_merge_bboxes_mode`:** PP-DocLayoutV3 outputs 25 layout classes. The default config has 17 (for RT-DETR-H_layout_17cls). The 25-class mapping is sourced from `PaddleOCR-VL-1.6/server/pipeline_config.yaml` in the repo.

### Step A3: Adjust Triton config for concurrency (§ 2.2 of official doc)

Per the official doc: *"In the `server/model_repo/{endpoint name}` directory ... you can find one or more `config*.pbtxt` files. If a `config_{device type}.pbtxt` file exists ... modify the configuration file corresponding to the desired device type. Otherwise, please modify `config.pbtxt`."*

Check what's in the downloaded SDK:
```bash
ls server/model_repo/layout-parsing/
```

If `config_gpu.pbtxt` exists, edit it. If not, create it. The official doc's recommended tuning is adjusting `instance_group`:

```text
# server/model_repo/layout-parsing/config_gpu.pbtxt
backend: "python"
max_batch_size: 8
input [
  { name: "input" data_type: TYPE_STRING dims: [ -1 ] }
]
output [
  { name: "output" data_type: TYPE_STRING dims: [ -1 ] }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching { }
```

**Official doc example — placing 4 instances on GPU 0:**
```text
instance_group [
{
    count: 4
    kind: KIND_GPU
    gpus: [ 0 ]
}
]
```

**Official doc example — 2 instances on GPU 1, 1 each on GPUs 2 and 3:**
```text
instance_group [
{
    count: 2
    kind: KIND_GPU
    gpus: [ 1 ]
},
{
    count: 1
    kind: KIND_GPU
    gpus: [ 2, 3 ]
}
]
```

> For more configuration details, the official doc directs to the [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

### Step A4: Pull the pre-built container image (§ 2.3 of official doc)

```bash
# GPU image (requires NVIDIA drivers supporting CUDA 11.8)
podman pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu

# CPU-only image (alternative)
# podman pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-cpu
```

### Step A5: Run the server (§ 2.3 of official doc — adapted for Podman)

```bash
cd ~/paddlex-layout-server/server

podman run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE=gpu \
    -v "$(pwd)":/app \
    -w /app \
    --rm \
    --gpus all \
    --init \
    --network host \
    --shm-size 8g \
    --userns=keep-id \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu \
    /bin/bash server.sh
```

**Official doc notes (adapted for Podman):**
- Replace `-it` with `-d` for background mode; then `podman logs -f {container ID}`
- Replace `/bin/bash server.sh` with `/bin/bash` to enter container for debugging, then run `server.sh` manually
- Add `-e PADDLEX_HPS_USE_HPIP=1` to enable the PaddleX high-performance inference plugin (HPI) for accelerated inference
- For CPU deployment, omit `--gpus` and use the CPU image
- `--userns=keep-id` maps your host UID into the container so mounted volume files are writable — required because Podman runs rootless by default (Docker runs as root)
- If `--gpus all` fails, use `--device nvidia.com/gpu=all` (CDI syntax) instead

Expected output:
```text
I1216 11:37:21.601943 35 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I1216 11:37:21.602333 35 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I1216 11:37:21.643494 35 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

> **Note on `--network host`:** The official doc uses host networking (ports 8000/8001/8002 are directly on the host). If you need port isolation, replace `--network host` with `-p 8000:8000 -p 8001:8001 -p 8002:8002`.

### Step A6: Invoke the service (§ 2.4 of official doc)

#### Option 1: Python client (official method)

```bash
cd ~/paddlex-layout-server/client

# Install dependencies (Python 3.8–3.12 supported)
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl

# Run inference
python client.py \
  --file /path/to/test_document.jpg \
  --file-type 1 \
  --url localhost:8001 \
  --no-visualization
```

#### Option 2: HTTP request (official method — any language)

Per the official doc, the request body format (endpoint: `http://{host}:8000/v2/models/layout-parsing/infer`):

```bash
curl -s -X POST http://localhost:8000/v2/models/layout-parsing/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [
          "{\"file\":\"https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png\",\"visualize\":false}"
        ]
      }
    ],
    "outputs": [
      { "name": "output" }
    ]
  }' | python -m json.tool
```

> The `file` field accepts a **URL** or **base64-encoded string**. The JSON inside `data` follows the same format as basic serving.

---

## Path B: Build from Source (Alternative)

Use this if you need to customize the container image itself, or if the Baidu registry/SDK URLs are unreachable. This follows `deploy/hps/README_en.md`.

### Step B1: Clone the fork and checkout the release

```bash
git clone https://github.com/AITDEV00/PaddleX.git
cd PaddleX
git checkout jya0-v3.7.2
git describe --tags  # should show v3.7.2
```

### Step B2: Download cuDNN and TensorRT

Place these in `deploy/hps/server_env/` (GPU build only). **Filenames must match exactly** — the Containerfile uses `--mount=type=bind,source=...` with hardcoded paths.

| Package | Exact filename | Download URL |
|---|---|---|
| cuDNN 8.9.7 | `cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz` | [NVIDIA cuDNN Archive](https://developer.nvidia.cn/rdp/cudnn-archive) → "Local Installer for Linux x86_64 (Tar)" for cuDNN v8.9.7 for CUDA 11.x |
| TensorRT 8.6.1.6 GA | `TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz` | [TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download) → "TensorRT 8.6 GA for Linux x86_64 and CUDA 11.x TAR package" |

Triton Inference Server (v2.15.0) is downloaded automatically during build.

```bash
cd deploy/hps/server_env/
ls -lh cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz \
      TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
```

### Step B3: Build the container image

```bash
cd deploy/hps/server_env/
./scripts/build_deployment_image.sh -k gpu -t paddlex3.7-gpu
```

> **Note:** The build script calls `docker build` internally. To use Podman, either:
> - Set `alias docker=podman` before running the script, **or**
> - Edit the script to replace `docker build` with `podman build`
> - Podman is Docker-compatible, so the `--mount=type=bind` and multi-stage syntax work as-is

**What the multi-stage Containerfile does:**
1. Base: `nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04` + Python 3.10 venv
2. Extracts cuDNN 8.9.7 → `/paddlex/libs/`
3. Extracts TensorRT 8.6.1.6 → `/paddlex/tensorrt/`
4. Downloads Triton 2.15.0 → `/opt/tritonserver/`
5. Installs PaddleX + HPI-GPU + `paddlex-hps-server` from repo source
6. Sets `LD_LIBRARY_PATH` for TensorRT/cuDNN
7. Creates non-root `paddlex` user, installs fonts (for OCR visualization)

Build time: ~15-30 min. Verify:
```bash
podman images | grep paddlex
```

### Step B4: Assemble the pipeline SDK from source

```bash
cd deploy/hps/sdk

# Edit the pipeline config (same as Step A2 above)
nano pipelines/layout_parsing/server/pipeline_config.yaml

# Add GPU Triton config with batching (same as Step A3 above)
# Create: pipelines/layout_parsing/server/model_repo/layout-parsing/config_gpu.pbtxt

# Package the SDK
./scripts/assemble.sh layout_parsing

# Extract
mkdir -p ~/paddlex-layout-server && cd ~/paddlex-layout-server
tar xzf /home/jyao/ADEO/OCR/PaddleX/deploy/hps/sdk/output/paddlex_hps_layout_parsing_sdk.tar.gz --strip-components=1
```

### Step B5: Run the server

Same as Step A5, but use the locally built image tag:
```bash
cd ~/paddlex-layout-server/server
podman run -it -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  -v "$(pwd)":/app -w /app --rm --gpus all --init \
  --network host --shm-size 8g --userns=keep-id \
  paddlex/hps:paddlex3.7-gpu \
  /bin/bash server.sh
```

---

## Serving Multiple Pipelines

### Approach A: Multiple containers (recommended)

Each pipeline gets its own container. With `--network host`, use different ports via `--env PADDLEX_HPS_PORT` or use port mapping with `-p`:

```bash
# Pipeline 1: Layout-only (default ports 8000-8002)
podman run -d --name paddlex-layout --gpus all \
  -v ~/paddlex-layout-server/server:/app -w /app \
  --network host --shm-size 8g --userns=keep-id \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu \
  /bin/bash server.sh

# Pipeline 2: Full OCR (map to different ports)
podman run -d --name paddlex-ocr --gpus all \
  -v ~/paddlex-ocr-server/server:/app -w /app \
  -p 8003:8000 -p 8004:8001 -p 8005:8002 --shm-size 8g --userns=keep-id \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu \
  /bin/bash server.sh
```

Use `--gpus '"device=0"'` / `'"device=1"'` to pin containers to specific GPUs.

### Approach B: Single Triton with merged model repos

Merge multiple `model_repo/` directories from different SDKs:
```
merged_model_repo/
  layout-parsing/    # from layout_parsing SDK
  ocr/               # from OCR SDK
```

Launch with a custom script (not `server.sh`):
```bash
podman run -d --gpus all --network host --shm-size 8g --userns=keep-id \
  -v ~/merged_model_repo:/models \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu \
  tritonserver --model-repository=/models \
    --backend-config=python,shm-default-byte-size=104857600
```

> ⚠️ This requires manually setting `PADDLEX_HPS_PIPELINE_CONFIG_PATH` for each model, which is non-trivial. Approach A is strongly recommended.

---

## Concurrency Tuning Reference

All tuning is done via `config_gpu.pbtxt` in the SDK's `server/model_repo/layout-parsing/` directory, as documented in § 2.2 of the official serving guide.

| Config | Default | Effect |
|---|---|---|
| `max_batch_size` | 1 (common) or 8 (custom) | Max requests merged per GPU forward pass |
| `instance_group count` | 1 | Concurrent model instances (official doc: "adjust the number of execution instances") |
| `instance_group gpus` | [0] | Which GPU IDs to use |
| `dynamic_batching` | absent or `{}` | Enables Triton's request batching |
| `preferred_batch_size` | — | Preferred batch sizes Triton aims for |
| `max_queue_delay_microseconds` | — | Max wait before flushing a partial batch |

**Example: high-throughput config for A100**

```text
backend: "python"
max_batch_size: 16
input [ { name: "input" data_type: TYPE_STRING dims: [ -1 ] } ]
output [ { name: "output" data_type: TYPE_STRING dims: [ -1 ] } ]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}
```

Estimated throughput on A100 with PP-DocLayoutV3 (~24ms/inference):
- `count: 1, max_batch_size: 8` → ~320 req/s
- `count: 2, max_batch_size: 16` → ~1,200+ req/s

---

## Troubleshooting

### Model download fails on first startup

PP-DocLayoutV3 weights are auto-downloaded from `https://paddle-model-ecology.bj.bcebos.com/`. If the container has no internet, pre-download and mount:
```bash
pip install paddlex
paddlex --install layout_detection
podman run -v ~/.paddlex:/home/paddlex/.paddlex --userns=keep-id ...
```

### Triton model not ready

```bash
curl http://localhost:8000/v2/models/
curl http://localhost:8000/v2/models/layout-parsing/config
podman logs {container_id} 2>&1 | grep -i error
```

### Out of memory (OOM) with multiple instances

Reduce `instance_group count` or `max_batch_size`. Monitor:
```bash
nvidia-smi -l 1
```

### cuDNN/TensorRT file not found (Path B only)

Ensure exact filenames in `deploy/hps/server_env/`:
```bash
ls deploy/hps/server_env/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
ls deploy/hps/server_env/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
```

---

## Official Documentation References

| Document | Location | Covers |
|---|---|---|
| **PaddleX Serving Guide** (primary) | `docs/pipeline_deploy/serving.en.md` | § 2: HPS SDK download, config, container run, client invocation — **the authoritative source** (uses `docker`; substitute `podman`) |
| HPS Project README | `deploy/hps/README_en.md` | Image building from source, SDK packaging from source |
| HPS Server Dockerfile | `deploy/hps/server_env/Dockerfile` | Multi-stage build internals (Path B) |
| HPS SDK source | `deploy/hps/sdk/` | Pipeline configs, model.py backends, client code, assemble scripts |
| Triton config docs | [NVIDIA Triton docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) | `instance_group`, `dynamic_batching`, etc. |
| HPI guide | `docs/pipeline_deploy/high_performance_inference.en.md` | `PADDLEX_HPS_USE_HPIP=1` for GPU acceleration |

---

## Quick-Reference: Path A (Recommended)

```bash
# 1. Download pre-built SDK
mkdir -p ~/paddlex-layout-server && cd ~/paddlex-layout-server
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.7/paddlex_hps_layout_parsing_sdk.tar.gz
tar xzf paddlex_hps_layout_parsing_sdk.tar.gz --strip-components=1

# 2. Edit server/pipeline_config.yaml → set model_name: PP-DocLayoutV3, disable all submodules, add 25-class merge mode
# 3. Edit server/model_repo/layout-parsing/config_gpu.pbtxt → set max_batch_size: 8, dynamic_batching {}, instance_group count

# 4. Pull pre-built image
podman pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu

# 5. Run server
cd server
podman run -d -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  -v "$(pwd)":/app -w /app --rm --gpus all --init \
  --network host --shm-size 8g --userns=keep-id \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.7-gpu \
  /bin/bash server.sh

# 6. Verify
podman logs -f {container_id}  # wait for "Started GRPCInferenceService"
curl http://localhost:8000/v2/models/
```
