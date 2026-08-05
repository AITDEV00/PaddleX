# Making the HPS API Image Truly Slim — Design Plan (BUILT & VALIDATED)

> **Status:** ✅ **BUILT & VALIDATED — FINAL (Blackwell): `docker/blackwell/lean/Dockerfile` → `paddlex-hps-api-lean2` = 7.02 GB** (2026-08-05).
>
> The **Blackwell (RTX 5090)** lean is the validated winner over the full
> image (25.4 GB). The transplant approach (tiny `nvidia/cuda:13.0.1-runtime-ubuntu22.04`
> base + paddle runtime copied from the official base) plus aggressive
> kernel-stripping achieved an **18.4 GB / 72% reduction** (25.4 → 7.02 GB),
> and **validates end-to-end with the Docling client**.
>
> An **H200 / Hopper** lean variant exists at
> `docker/hopper/lean/Dockerfile` (CUDA 12.6, tensorrt_cu12, FP16, engine
> built from ONNX on first boot — no sm_120 engine baked in). It is not yet
> built/validated (no H200 on this host).
>
> Summary of the progression (single-stage → final):
> | Image | Size | Note |
> |-------|------|------|
> | single-stage | 37.1 GB | baseline |
> | multi-stage | 32.2 GB | split build/runtime |
> | `Dockerfile.lean` → `docker/blackwell/lean/Dockerfile.lean` | 25.4 GB | paddle base, drop Triton |
> | `Dockerfile.lean2` (LD fix) | 15.1 GB | transplant on nvidia runtime base |
> | + TRT win/arch strip | 13.2 → 11.1 → 9.72 GB | drop Windows DLLs, other-arch builder resources, redundant nvidia/cu13 |
> | **`Dockerfile.lean2` (kernel strip)** | **7.02 GB** | **strip paddle compute kernels + compat, single-layer copy** |

> **Note (2026-08-05):** the validated winner is now the canonical
> `docker/blackwell/lean/Dockerfile` (renamed from `Dockerfile.lean2`); the old
> 25.4 GB single-stage attempt is kept at `docker/blackwell/lean/Dockerfile.lean`.
>
> The Triton-server deployment mode is intentionally NOT supported by the lean
> images (they require the full Triton base).
>
> **Date:** 2026-08-05

---

## 0. FINAL APPROACH — `Dockerfile` (canonical, 7.02 GB, validated)

**Key insight:** the official paddle base is `-devel` and huge (19.1 GB). We do
NOT need paddle installed fresh (bcebos CDN is blocked from this host anyway),
and we do NOT need CUDA dev headers at runtime. So:

- **Stage 0 `paddle-base`** — import-only `docker.io/paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13`. Used purely as a source to copy the already-built paddle runtime.
- **Stage 1 `builder`** — on paddle-base. Builds artifacts that need full CUDA headers + toolchain: `pycuda` (from source), plus installs `paddle2onnx/onnx/onnxsim/polygraphy/nvidia-modelopt`. These are copied into runtime.
- **Stage 2 `runtime`** — on `nvidia/cuda:13.0.1-runtime-ubuntu22.04` (**2.4 GB**). Steps:
  1. `apt install` python3/pip/git/curl/ca-certificates/libgomp1/libquadmath0/libgfortran5/libgl1/libglib2.0-0
  2. copy paddle + nvidia + numpy + dist-info from `paddle-base`
  3. `pip install -U "pip>=26"` then `pip install tensorrt_cu13==10.16.1.11`; `COPY --from=builder` pycuda
  4. `pip install paddlex==3.7.2` + opencv/scikit + `requirements.api_compat.txt` + orjson
  5. patch_paddlex_trt.py, app code, models, runner user

### ⚠️ The biggest win: strip paddle compute kernels (9.72 → 7.02 GB)
The api_compat **`direct` backend deserializes a pre-built TensorRT engine** and
runs inference entirely in-process via pycuda + TensorRT. It NEVER calls
paddle's phi / flash-attention / MKL / CINN kernels. Verified live (fresh server
restart + Docling SUCCESS after EACH removal):

| Removed | Size | What it is |
|---------|------|------------|
| `libflashattn.so` | 855M | flash-attention kernels |
| `libphi_core.so` | 580M | phi compute kernels |
| `libphi_gpu.so` | 462M | phi GPU kernels |
| `libflashattnv3.so` | 203M | flash-attention v3 |
| `libmklml_intel.so` | 125M | oneDNN/MKL (CPU) |
| `libcinnapi.so` | 105M | CINN compiler |
| `libflashmaskv2.so` | 81M | flash-mask |
| `libdnnl.so.3` | 61M | oneDNN |
| `paddle/include` | 47M | C++ headers |
| `paddle/distributed` | 13M | training-only |
| `cuda-13.0/compat` | 307M | stub libcuda (build-time only) |
| **Total** | **~2.8 GB** | |

### ⚠️ CRITICAL Docker layer-semantics gotcha
`COPY paddle/` creates a **3.08 GB layer**; a **separate** `RUN rm` only adds a
~20 kB **whiteout** layer — the deleted data **stays baked into the image**. To
actually reclaim space, copy + strip must happen in **ONE RUN layer**:

```dockerfile
RUN --mount=type=bind,from=paddle-base,source=.../paddle,target=/paddle-src \
    mkdir -p .../paddle && cp -a /paddle-src/. .../paddle/ \
    && cd .../paddle/libs && rm -f libphi_core.so ... \
    && rm -rf .../paddle/include .../paddle/distributed \
    && du -sh .../paddle
```

`--mount=type=bind,from=...` is **not committed to a layer**, so only the
final stripped files land in the image. (First attempt with `COPY`+`RUN rm` left
the image at 9.72 GB despite the strip layer; switching to the bind-mount RUN
dropped it to 7.02 GB.)

### ⚠️ CRITICAL LD_LIBRARY_PATH order (cuInit failure)
The `nvidia` runtime base ships `/usr/local/cuda-13.0/compat/libcuda.so` — a
**STUB** (only useful at build time for linking). At runtime the **real** driver
`libcuda.so.1` is injected into `/usr/local/nvidia/lib64` by the NVIDIA
container runtime. If `compat/` is listed **before** `/usr/local/nvidia/lib64`
in `LD_LIBRARY_PATH`, pycuda/TensorRT load the stub and fail:

```
pycuda._driver.RuntimeError: cuInit failed: no CUDA-capable device is detected
```

**Correct `LD_LIBRARY_PATH`** (driver first):
```
/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/local/lib/python3.10/dist-packages/paddle/libs:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64
```

### Validation
`DoclingServiceClient(url="http://localhost:8083", ws_fallback_to_poll=True, job_timeout=120)` with a generated PNG → **`ConversionStatus.SUCCESS`**, 4 elements extracted. Note newer docling uses `from docling_core.types.io import DocumentStream` (not `docling_core.types`).

---

## 1. Why the current slim image is still 32.2 GB

The multi-stage `docker/blackwell/full/Dockerfile.multistage` split the build toolchain
(modelopt/onnx/polygraphy/paddle2onnx) out of the runtime, but **both stages
inherit the same 26.6 GB base `paddlex-hps-ngc`**. The base is a Triton
Inference Server image that bundles far more than the direct/TensorRT path
needs:

| Base component | Size | Needed by direct backend? |
|----------------|------|:---:|
| `/opt/tritonserver` (binary + backends) | **5.6 GB** | ❌ No (direct mode) |
| `paddle` (`dist-packages/paddle`) | **2.9 GB** | ✅ Yes |
| `paddlex` | 19 MB | ✅ Yes |
| CUDA 13.0 libs (several COPY layers) | ~5 GB | ✅ partially |
| CUDA 12.6 + cuDNN + TRT 10.5 + NCCL | ~6.5 GB | ✅ partially |
| `pip tensorrt_cu13==10.16.1.11` + pycuda | **4.5 GB** | ✅ yes (runtime) |

**Key Docker fact:** `RUN rm -rf /opt/tritonserver` in a later layer does
**NOT** shrink the image — Docker layers are additive, the base layer's bytes
remain. Only a fresh base (that never copied Triton) removes them.

---

## 2. Goal

A runtime image for the **`direct`** backend built from a lean base, dropping
Triton entirely:

- Base: `nvidia/cuda:13.0.1-devel-ubuntu22.04` (or 12.6 if H200) instead of the
  Triton NGC image.
- Install **`paddlepaddle-gpu==3.3.1`** + **`paddlex`** fresh (no Triton).
- Install `tensorrt_cu13==10.16.1.11` + `pycuda` + docling API deps.
- Bake in pre-built FP8 `.trt` engine + ONNX fallback + Paddle params.

Target: **~10-14 GB** (CUDA base ~5 GB + paddle 2.9 GB + TRT 4.5 GB + deps).

---

## 3. Approach: keep multi-stage, swap the base

Reuse the already-validated structure of `docker/blackwell/full/Dockerfile.multistage`,
but replace both `FROM` lines with a lean CUDA base:

```dockerfile
# STAGE 1 (builder)  — from lean base too; build/quantize toolchain
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04 AS builder
# ...same toolchain install...

# STAGE 2 (runtime, final)
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04 AS runtime
# ...install paddlepaddle-gpu + paddlex + tensorrt_cu13 + pycuda + API deps...
```

### Requirements verified against the source (already confirmed)
- **Direct backend needs** (from `inference.py` / `engine.py`):
  `paddlex` (create_model), `paddlepaddle-gpu`, `tensorrt` (deserialize),
  `pycuda` (direct CUDA backend), plus docling/FastAPI deps
  (`requirements.api_compat.txt`), `torch` (only if `HPS_GPU_PRE=1`,
  optional), `websockets` (docling client websocket watcher).
- **Not needed:** `tritonserver`, Triton model repo, `tritonclient[grpc]`,
  `nvidia-modelopt`, `onnx`, `onnxsim`, `polygraphy`, `paddle2onnx` (all
  build-time only).

---

## 4. Concrete steps (when executed)

### 4.1 New base (`nvidia/cuda:13.0-devel-ubuntu22.04`)
- Verify the CUDA 13 base exists on docker hub: `nvidia/cuda:13.0.1-devel-ubuntu22.04`.
- For H200 use `nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04` (Hopper, CUDA 12.6).

### 4.2 Runtime stage install list
```dockerfile
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==3.3.1 \
    paddlex \
    "tensorrt_cu13==10.16.1.11" \
    pycuda \
    -r requirements.api_compat.txt
```
Add `torch==2.13.0+cu130` **only** if you need `HPS_GPU_PRE=1` GPU pre.

### 4.3 Model + engine
- Bake in `models/PP-DocLayoutV3/` (FP8 `.trt`, FP32 ONNX, Paddle params).
- The pre-built `inference_fp8.trt` is sm_120/5090; rebuild per target GPU.

### 4.4 Cleanup after pip (to strip lib bloat)
- `pip cache purge` + `rm -rf ~/.cache/pip`.
- Remove unused TRT plugin/dispatch/lean libs if not needed (keep
  `libnvinfer.so.10`, `libnvinfer_plugin.so.10`, `libnvonnxparser.so.10`).
- Remove CUDA libs unused by paddle/TRT (e.g., Nsight, nsys, cusparseLt if
  unused) — optional, must verify runtime links with `ldd`.

### 4.5 Validation checklist
1. Build: `docker build -t paddlex-hps-api-lean -f deploy/hps/docker/blackwell/lean/Dockerfile .`
2. `docker images paddlex-hps-api-lean` → expect ~10-14 GB.
3. Run: `--env HPS_API_BACKEND=direct`, port 8080.
4. Health: `/health-check` returns 200.
5. Docling client (in-container): `DoclingServiceClient(url=...:8080)` →
   `ConversionStatus.SUCCESS`, valid markdown.
6. `ldd` on the TRT + paddle libs to confirm no missing runtime deps.

---

## 5. Risks / Caveats

- **Compatibility:** the lean base installs paddle/paddlex fresh; paddle 3.3.1
  is the same version pinned in the Triton base, so risk is low, but CUDA lib
  versions (cuBLAS/cuDNN/NCCL) must be present. Verify with `ldd` and a smoke
  run.
- **Loss of Triton mode:** the lean image is **direct-only**. If you later
  need the Triton backend deployment, keep the `paddlex-hps-ngc`-based images.
- **torch optional:** dropping torch saves ~3 GB but disables `HPS_GPU_PRE=1`.
- **Engine GPU-specific:** the FP8 engine must be rebuilt on the target GPU.

---

## 6. Result (implemented)

**Created:**
- `deploy/hps/docker/blackwell/lean/Dockerfile` — new lean multi-stage Dockerfile (built on the
  official `paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13` base, not the
  Triton NGC image).
- Built & tagged `paddlex-hps-api-lean:latest` → **25.4 GB**.
- Validated: `/health-check` 200, FP8 direct backend, `paddle 3.3.1`,
  `tensorrt 10.16.1.11`, `cv2 4.10.0`, and the **official DoclingServiceClient**
  returns `ConversionStatus.SUCCESS` with valid markdown (4 layout elements).

**Extra runtime deps added vs. original design** (discovered via smoke tests;
the NGC base bundled these but the official paddle base does not):
- `opencv-contrib-python==4.10.0.84` (cv2 — api_compat hard import)
- `scikit-image==0.24.0`, `scikit-learn==1.6.1`, `scipy==1.15.2`,
  `faiss-cpu>=1.8`, `pycocotools>=2` (paddlex runtime)
- `pypdfium2>=4.30.0` (PDFReaderBackend — required by `ImageBatchSampler` at
  model init; missing → `DependencyError` at startup)
- `orjson>=3.10.0` (ORJSONResponse; without it FastAPI returns 500)
- `decorator`, `astor` (paddlex runtime)
- `torch` NOT installed (saves ~3 GB; disables `HPS_GPU_PRE=1`)

**Note on size:** design target was ~10-14 GB via a bare `nvidia/cuda` base.
The official `paddlepaddle/paddle` CUDA 13 base (19.1 GB) already includes
paddle 3.3.1 + CUDA 13 + cuDNN and a full compiler toolchain (gcc-13), so it
is larger than a bare CUDA base but avoids a separate paddle install layer. To
approach ~10-14 GB, start from `nvidia/cuda:13.0.1-devel-ubuntu22.04` and
`pip install paddlepaddle-gpu==3.3.1` (see §2), but note the TRT + pycuda
layers (~4.5 GB) and OpenCV/scipy are still required at runtime.

**Current validated state:**
- `docker/blackwell/full/Dockerfile` — original single-stage (37.1 GB).
- `docker/blackwell/full/Dockerfile.multistage` — slim, direct-validated (32.2 GB).
- `docker/blackwell/lean/Dockerfile.lean` — historical lean (25.4 GB, kept for reference).
- `docker/blackwell/lean/Dockerfile` — canonical lean, direct-validated (7.02 GB).
- `docker/hopper/lean/Dockerfile` — H200/Hopper lean (CUDA 12.6, FP16, ~7 GB target, not yet built — no H200 host).
- `deploy/hps/scripts/slim_loop.sh` — build → measure → run → Docling-client loop for lean images.