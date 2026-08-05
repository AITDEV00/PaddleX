# PaddleX HPS Dockerfiles — Directory Index

Custom `direct`/TensorRT Docling API images, organized by **GPU architecture**,
then by **variant**. This makes it easy to add support for a new GPU: create a
new architecture folder (e.g. `ampere/`) with `base/`, `full/`, `lean/`
subfolders.

```
docker/
├── blackwell/                 # RTX 5090 — sm_120, CUDA 13.0
│   ├── base/  Dockerfile      #     NGC Triton base (paddle 3.3.1-cuda13.0)
│   ├── full/  Dockerfile      #     full API image on the base
│   │          Dockerfile.multistage
│   └── lean/  Dockerfile      #   ★ WINNER — 7.02 GB, Docling-validated (canonical)
│             Dockerfile.lean #     historical 25.4 GB attempt
│
├── hopper/                    # H200 / H100 / A100 / L40S — sm_90/80/89, CUDA 12.6
│   ├── base/  Dockerfile      #     NGC base (paddle 3.3.1-cuda12.6)
│   ├── full/  Dockerfile      #     full API image on the base
│   └── lean/  Dockerfile      #   ★ lean — CUDA 12.6, FP16, engine built on first boot
│
└── legacy/
    └── Dockerfile             # Triton-server deployment variant (not HW-specific)
```

## Building

All builds run from the **PaddleX repo root** (the build context must be the
repo so `deploy/hps/...`, `models/...`, and `paddlex/...` paths resolve).

### Blackwell (RTX 5090, sm_120, CUDA 13.0)

```bash
# base
podman build -t paddlex-hps-ngc \
  -f deploy/hps/docker/blackwell/base/Dockerfile .

# full API
podman build -t paddlex-hps-api \
  -f deploy/hps/docker/blackwell/full/Dockerfile .

# lean (7.02 GB winner, canonical)
podman build -t paddlex-hps-api-lean2 \
  -f deploy/hps/docker/blackwell/lean/Dockerfile .
```

### Hopper (H200 / H100 / A100 / L40S, sm_90/80/89, CUDA 12.6)

```bash
# base
podman build -t paddlex-hps-ngc-cuda12 \
  -f deploy/hps/docker/hopper/base/Dockerfile .

# full
podman build -t paddlex-hps-api-h200 \
  -f deploy/hps/docker/hopper/full/Dockerfile .

# lean (CUDA 12.6, FP16; engine built from ONNX on first boot)
podman build -t paddlex-hps-api-h200-lean \
  -f deploy/hps/docker/hopper/lean/Dockerfile .
```

### Legacy (Triton-server deployment variant)

```bash
podman build -t paddlex-hps-api-triton \
  -f deploy/hps/docker/legacy/Dockerfile .
```

---

## Which one do I want?

| Need | Use |
|------|-----|
| Smallest possible image, validated, RTX 5090 | `blackwell/lean/Dockerfile` (7.02 GB) |
| Small image for H200/H100 (build engine on target) | `hopper/lean/Dockerfile` |
| Full image (all formats / debug), RTX 5090 | `blackwell/full/Dockerfile` |
| Full image for H200 / MIG | `hopper/full/Dockerfile` |
| Triton-server deployment mode | `legacy/Dockerfile` |

## Adding a new GPU architecture

1. Create `docker/<arch>/base/Dockerfile` (NGC/CUDA base).
2. Create `docker/<arch>/full/Dockerfile` (full API image).
3. Create `docker/<arch>/lean/Dockerfile` (minimal runtime; follow
   `blackwell/lean` or `hopper/lean` as a template).
4. Update `BUILD_NGC.md` and this index.