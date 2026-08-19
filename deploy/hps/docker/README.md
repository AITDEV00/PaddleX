# PaddleX HPS Dockerfiles — Directory Index

Custom `direct`/TensorRT Docling API images, organized by **CUDA version**,
then by **variant**. The images are separated by CUDA version (NOT GPU
architecture) because the container's CUDA runtime libs must be <= the maximum
CUDA the **host driver** supports. To add support for a new CUDA version,
create a new folder (e.g. `cuda11/`) with `base/`, `full/`, `lean/` subfolders.

```
docker/
├── cuda13/                    # CUDA 13.0 — needs host driver >= CUDA 13
│   ├── base/  Dockerfile        #     NGC Triton base (paddle 3.3.1-cuda13.0)
│   ├── full/  Dockerfile        #     full API image on the base
│   │          Dockerfile.multistage
│   └── lean/  Dockerfile        #   ★ WINNER — 7.02 GB, universal build-at-runtime
│             Dockerfile.lean    #   historical 25.4 GB attempt
│
├── cuda12/                    # CUDA 12.6 — needs host driver >= CUDA 12.6
│   ├── base/  Dockerfile        #   Triton base (paddle 3.3.1-cuda12.6)
│   ├── full/  Dockerfile        #   full API image on the base
│   └── lean/  Dockerfile        #   ★ lean — CUDA 12.6, FP16, engine on first boot
│
└── legacy/
    └── Dockerfile             # Triton-server deployment variant (not CUDA-specific)
```

Each lean image builds the TensorRT engine **from ONNX at runtime (first boot)**
in FP16, so within a given CUDA version it works across GPU architectures
(RTX 5090 sm_120, RTX 4090 sm_89, H200/H100 sm_90, A100 sm_80, L40S sm_89).
The CUDA version is the only hard boundary.

## Building

All builds run from the **PaddleX repo root** (the build context must be the
repo so `deploy/hps/...`, `models/...`, and `paddlex/...` paths resolve).

### CUDA 13 (RTX 50-series / Blackwell; needs driver >= CUDA 13)

# base
podman build -t paddlex-hps:layout-cu13-base \
  -f deploy/hps/docker/cuda13/base/Dockerfile .

# full API
podman build -t paddlex-hps:layout-cu13-full \
  -f deploy/hps/docker/cuda13/full/Dockerfile .

# lean (7.02 GB winner)
# Bakes in the sm_120 (RTX 5090) FP8 engine -> instant load on Blackwell.
# Falls back to building the TRT engine from ONNX on first boot (FP8) on any
# other GPU within CUDA 13.
podman build -t paddlex-hps:layout-cu13-lean-sm120 \
  -f deploy/hps/docker/cuda13/lean/Dockerfile .

# run (engine):
podman run -d --name paddlex-hps-lean \
  --device nvidia.com/gpu=all \
  --env HPS_API_BACKEND=direct \
  --env HPS_API_PRECISION=fp8 \
  --env HPS_API_STARTUP_TIMEOUT=300 \
  -p 8080:8080 paddlex-hps:layout-cu13-lean-sm120
```

### CUDA 12 (H200 / H100 / A100 / L40S, RTX 40-series; needs driver >= CUDA 12.6)

```bash
# base
podman build -t paddlex-hps:layout-cu12-base \
  -f deploy/hps/docker/cuda12/base/Dockerfile .

# full
podman build -t paddlex-hps:layout-cu12-full \
  -f deploy/hps/docker/cuda12/full/Dockerfile .

# lean (CUDA 12.6)
# Bakes in the sm_90a (H200/H100) FP16 engine -> instant load on Hopper.
# Falls back to building the TRT engine from ONNX on first boot (FP16) on
# other CUDA 12.6 GPUs (A100 sm_80, L40S/4090 sm_89).
podman build -t paddlex-hps:layout-cu12-lean-sm90a \
  -f deploy/hps/docker/cuda12/lean/Dockerfile .
```

### Legacy (Triton-server deployment variant)

```bash
podman build -t paddlex-hps:layout-triton \
  -f deploy/hps/docker/legacy/Dockerfile .
```

---

## Image naming

All images are named `paddlex-hps` with tags `layout-cu<CUDA>-<variant>[-<arch>]`:

| Tag | Meaning |
|-----|---------|
| `paddlex-hps:layout-cu13-base` | CUDA 13 NGC base |
| `paddlex-hps:layout-cu13-full` | CUDA 13 full API image |
| `paddlex-hps:layout-cu13-lean-sm120` | CUDA 13 lean, **sm_120 (RTX 5090) engine baked in** |
| `paddlex-hps:layout-cu12-base` | CUDA 12 NGC base |
| `paddlex-hps:layout-cu12-full` | CUDA 12 full API image |
| `paddlex-hps:layout-cu12-lean-sm90a` | CUDA 12 lean, **sm_90a (H200/H100) engine baked in** |
| `paddlex-hps:layout-triton` | Triton-server variant |

The optional `<arch>` segment (sm90a/sm120) marks that a pre-built TRT engine
for that GPU architecture is baked into the image. When it is absent, the image
builds its engine from ONNX at first boot.

## Which one do I want?

| Host / GPU | Use |
|------------|-----|
| Driver >= CUDA 13 (RTX 50-series etc.) | `paddlex-hps:layout-cu13-lean-sm120` (engine baked in) |
| Driver >= CUDA 12.6 (H200/H100/A100/L40S, RTX 40-series) | `paddlex-hps:layout-cu12-lean-sm90a` |
| Full image (all formats / debug), CUDA 13 | `paddlex-hps:layout-cu13-full` |
| Full image for CUDA 12.6 / H200 | `paddlex-hps:layout-cu12-full` |
| Triton-server deployment mode | `paddlex-hps:layout-triton` |

## Adding a new CUDA version

1. Create `docker/<cuda>/base/Dockerfile` (CUDA base).
2. Create `docker/<cuda>/full/Dockerfile` (full API image).
3. Create `docker/<cuda>/lean/Dockerfile` (minimal runtime; follow
   `cuda13/lean` or `cuda12/lean` as a template).
4. Update `BUILD_NGC.md` and this index.
