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

```bash
# base
podman build -t paddlex-hps-ngc \
  -f deploy/hps/docker/cuda13/base/Dockerfile .

# full API
podman build -t paddlex-hps-api \
  -f deploy/hps/docker/cuda13/full/Dockerfile .

# lean (7.02 GB winner, universal)
# Builds the TRT engine from ONNX on first boot (FP16) -> works on any GPU
# within CUDA 13. ~3 min first boot.
podman build -t paddlex-hps-api-lean2 \
  -f deploy/hps/docker/cuda13/lean/Dockerfile .

# run (engine):
podman run -d --name paddlex-hps-lean \
  --device nvidia.com/gpu=all \
  --env HPS_API_BACKEND=direct \
  --env HPS_API_PRECISION=fp16 \
  --env HPS_API_STARTUP_TIMEOUT=300 \
  -p 8080:8080 paddlex-hps-api-lean2
```

### CUDA 12 (H200 / H100 / A100 / L40S, RTX 40-series; needs driver >= CUDA 12.6)

```bash
# base
podman build -t paddlex-hps-ngc-cuda12 \
  -f deploy/hps/docker/cuda12/base/Dockerfile .

# full
podman build -t paddlex-hps-api-h200 \
  -f deploy/hps/docker/cuda12/full/Dockerfile .

# lean (CUDA 12.6, FP16; engine built from ONNX on first boot)
podman build -t paddlex-hps-api-h200-lean \
  -f deploy/hps/docker/cuda12/lean/Dockerfile .
```

### Legacy (Triton-server deployment variant)

```bash
podman build -t paddlex-hps-api-triton \
  -f deploy/hps/docker/legacy/Dockerfile .
```

---

## Which one do I want?

| Host / GPU | Use |
|------------|-----|
| Driver >= CUDA 13 (RTX 50-series etc.) | `cuda13/lean/Dockerfile` (7.02 GB, engine on first boot) |
| Driver >= CUDA 12.6 (H200/H100/A100/L40S, RTX 40-series) | `cuda12/lean/Dockerfile` |
| Full image (all formats / debug), CUDA 13 | `cuda13/full/Dockerfile` |
| Full image for CUDA 12.6 / H200 | `cuda12/full/Dockerfile` |
| Triton-server deployment mode | `legacy/Dockerfile` |

## Adding a new CUDA version

1. Create `docker/<cuda>/base/Dockerfile` (CUDA base).
2. Create `docker/<cuda>/full/Dockerfile` (full API image).
3. Create `docker/<cuda>/lean/Dockerfile` (minimal runtime; follow
   `cuda13/lean` or `cuda12/lean` as a template).
4. Update `BUILD_NGC.md` and this index.
