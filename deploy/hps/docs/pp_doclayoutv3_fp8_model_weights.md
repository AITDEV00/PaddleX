# PP-DocLayoutV3 FP8 — Model Weight Investigation & Solution

> **Date:** 2026-08-05
> **Context:** Making PP-DocLayoutV3 run at FP8 precision on NVIDIA H200 (sm_90a / Hopper). FP16 works; FP8 is desired for latency.
> **Target deployment:** `localhost/paddlex-hps-api-h200:latest` (CUDA 12.6, TensorRT 10.16.1.16).

---

## 1. Summary

- **There are no usable pre-quantized (FP8/INT8) PP-DocLayoutV3 weights anywhere on HuggingFace.**
- The repo's own `inference_fp8.onnx` is a **defective NVIDIA modelopt export** and cannot be rebuilt into a correct FP8 TensorRT engine on H200.
- The only viable path is to **re-quantize the clean FP32 model on the H200 with the correct modelopt config**, then build the engine on the target GPU.
- TensorRT engines are **hardware-specific** — they must be built on the GPU they will run on. An engine built for sm_89 (RTX 4090) will not load on sm_90a (H200).

---

## 2. HuggingFace Survey (no quantized weights exist)

### Official PaddlePaddle repos (all FP32)
| Repo | Contents | Quantized? |
|------|----------|-----------|
| `PaddlePaddle/PP-DocLayoutV3` | `inference.pdiparams` (130MB) | ❌ FP32 |
| `PaddlePaddle/PP-DocLayoutV3_safetensors` | `model.safetensors` (133MB) | ❌ FP32 |
| `PaddlePaddle/PP-DocLayoutV3_onnx` | `inference.onnx` (130MB) | ❌ FP32 |

### Community repos (all FP32 / FP16, wrong format or wrong GPU)
| Repo | Contents | Precision / Target |
|------|----------|-----------|
| `bndos/pp-doclayout-v3-trt` | TRT engine + ONNX | **FP16, sm_89 (4090)** — cannot run on H200 |
| `alex-dinh/PP-DocLayoutV3-ONNX` | ONNX (130MB) | FP32 |
| `Bei0001/PP-DocLayoutV3-ONNX` | ONNX + `.onnx.data` | FP32 |
| `ningpp/PP-DocLayoutV3-ONNX` | ONNX (133MB) | FP32 |
| `Echo9Zulu/PP-DocLayoutV3-FP16-OpenVINO` | OpenVINO `.bin`/`.xml` | FP16, wrong format |
| `Fiona1019/PP-DocLayoutV3-ov` | OpenVINO | FP32/FP16, wrong format |

**Empty searches:** `doclayout fp8`, `rt-detr fp8`, `paddlex fp8`, `rtdetr quantized`.

### Conclusion
No community or official FP8/INT8/FP4 weights exist for PP-DocLayoutV3 (or its RT-DETR base architecture). The only actionable path is local re-quantization on the H200.

---

## 3. Compute Capability (why engines don't cross GPUs)

| GPU | Architecture | Compute Capability |
|-----|-------------|-------------------|
| RTX 4090 | Ada Lovelace | **sm_89** |
| RTX 5090 | Blackwell | **sm_120** |
| H100 / H200 | Hopper | **sm_90 / sm_90a** |

TensorRT engines are compiled to **SASS/PTX for a specific compute capability**. Deserializing on a different sm_ fails, e.g.:

```
[E] Engine built for a different platform: expected compute capability 8.9 (sm_89),
    detected 9.0 (sm_90)
```

- `HARDWARE_COMPATIBILITY_LEVEL` (PTX/NVRTC-JIT fallback) can produce cross-arch engines, but loses native kernel optimizations — pointless for a performance-driven FP8 build.
- **Rule:** always build the engine on the exact target GPU. Pre-built engines from a 5090 (sm_120) also won't run on H200 (sm_90a), and vice-versa.

---

## 4. Root Cause of the Defective Repo `inference_fp8.onnx`

The FP8 ONNX exported by modelopt for this repo:
- Conv kernels stored as **FP16 initializers**, but Q/DQ scale tensors as **FP32**.
- Per ONNX spec, `DQ` output type = scale type → TRT strongly-typed resolves DQ outputs as FP32, kernels stay FP16 → parse fails with `input Float but kernel Half`.
- The export is **missing** the standard NVIDIA FP8 config / graph modifications:
  - `trt_high_precision_dtype: "Half"` on weight/input quantizers
  - `convert_zp_fp8`, `cast_resize_io`, `convert_fp16_io`, `cast_fp8_mha_io`
- As a result TRT cannot fuse native FP8 kernels and falls into Myelin / FP8 bugs.

### Build attempts that failed on H200 (from repo's `inference_fp8.onnx`)
1. Patched FP16-scale + strongly-typed → parse OK, **build fails**: `MyelinCheckException: utils.cpp:2951: CHECK(perm.size()==t.size()) failed` + `Could not find any implementation for node {ForeignNode[...]}`
2. Removing boundary `Cast` nodes → same Myelin error
3. Wrapping the 8 `Resize` in FP32 casts → same Myelin error
4. Weakly-typed + FP16-scale → parse OK, build fails (same Myelin)
5. Weakly-typed + FP32-scale + kFP16 + kFP8 → build succeeds but **NaN/dead output** (numerically broken)

### Validated reference (proves FP8 is real & faster)
The known-good pre-built `inference_fp8.trt` (55MB, sm_120/5090) works correctly:
300 valid detections, max_conf 0.917, **avg 3.82 ms vs FP16 5.56 ms**.
I/O: FP32 in/out; `fetch_name_0` [-1,7] = box_id, conf, x1, y1, x2, y2, class; `fetch_name_1` INT32 [-1]; `fetch_name_2` INT32 [-1,200,200].

---

## 5. Correct Recipe (Re-quantize the clean FP32 model on H200)

Start from the clean FP32 `inference.onnx` (already at `models/PP-DocLayoutV3/inference.onnx`, 130MB) — **not** the defective `inference_fp8.onnx`.

```python
import modelopt.onnx.quantization as moq

# 1. Quantize with the CORRECT config (Half high-precision dtype)
quant_cfg = {
    "*weight_quantizer": {"num_bits": (4, 3), "trt_high_precision_dtype": "Half"},
    "*input_quantizer":  {"num_bits": (4, 3), "trt_high_precision_dtype": "Half"},
    "*bias_quantizer":   {"num_bits": (4, 3)},
}
moq.quantize(onnx_model, qformat="FP8_DQ", algorithm="MAX", quant_cfg=quant_cfg)
moq.convert(onnx_model, "FP8")

# 2. Apply the graph modifications TRT needs for native FP8 fusion
modify_fp8_graph(onnx)   # convert_zp_fp8, cast_resize_io, convert_fp16_io, cast_fp8_mha_io

# 3. Save the corrected FP8 ONNX, then build engine on the H200
```

Notes:
- Requires `modelopt` (nvidia-modelopt) in the build environment, plus a CUDA-capable build (the H200 container).
- Must run the quantization **on the target GPU** so calibration data/precision is representative.
- Build the resulting ONNX into a `.trt` engine **on the H200 itself** (sm_90a).

---

## 6. Pre-existing H200 Image

Current Docker images (2026-08-05):

| Image | Built | Size | Notes |
|-------|-------|------|-------|
| `localhost/paddlex-hps-api-h200:latest` | 4 days ago | 36.1 GB | **H200 target image** (CUDA 12.6, TRT 10.16.1.16, precision=FP16) |
| `localhost/paddlex-hps-api:latest` | 5 days ago | 37.1 GB | generic (5090) image |
| `localhost/paddlex-hps-api-triton:latest` | 7 days ago | 37.2 GB | Triton variant |
| `localhost/paddlex-hps-ngc-cuda12:latest` | 2 weeks | 22 GB | NGC base |
| `localhost/paddlex-hps-ngc:latest` | 2 weeks | 26.6 GB | NGC base |

The H200 image is currently present.