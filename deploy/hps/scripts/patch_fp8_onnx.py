#!/usr/bin/env python3
"""
Patch the FP8-quantized ONNX model so TensorRT 10.16's OnnxParser accepts it.

Root cause
----------
nvidia-modelopt produces a Q/DQ (quantize/dequantize) ONNX graph where:
  • Conv kernels are stored as FLOAT16 initializers
  • DequantizeLinear / QuantizeLinear *scale* tensors are FLOAT32

Per the ONNX spec, the output type of a DequantizeLinear node equals the type
of its *scale* input.  TRT's strongly-typed parser honours this rule, so every
DQ output (and every Concat/ElementWise that consumes it) is resolved as FP32.
The Conv kernels, however, remain FP16 → ``input Float but kernel Half``.

The value_info annotations in the ONNX say the DQ outputs are FP16, but TRT
computes types from the graph flow, not from value_info — so the annotations
are ignored.

Fix
---
1. Convert every Q/DQ scale initializer from FLOAT32 → FLOAT16.
   This makes DQ outputs FP16 (matching the Conv kernels and the annotations).
2. Remove the 935 Paddle-export Identity no-op nodes so the graph is clean
   and TRT's type-inference is unambiguous.

Usage
-----
    python3 patch_fp8_onnx.py <input.onnx> <output.onnx>

Both onnx and onnx-graphsurgeon must be importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto, numpy_helper


def patch_fp8_onnx(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)

    print(f"[patch_fp8_onnx] Loading {src} ({src.stat().st_size / 1e6:.1f} MB)...")
    model = onnx.load(str(src))
    g = model.graph

    # ── Step 1: collect Q/DQ scale names ──────────────────────────────────
    scale_names: set[str] = set()
    for node in g.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 1:
            scale_names.add(node.input[1])

    # ── Step 2: convert FP32 scales → FP16 ─────────────────────────────────
    converted = 0
    for init in g.initializer:
        if init.name in scale_names and init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
            converted += 1

    print(f"[patch_fp8_onnx] Converted {converted} Q/DQ scales FLOAT32 → FLOAT16")

    # ── Step 3: remove Identity no-op nodes ────────────────────────────────
    graph = gs.import_onnx(model)
    removed = 0
    for node in list(graph.nodes):
        if node.op == "Identity":
            (inp,) = node.inputs
            for out in node.outputs:
                # Rewire consumers: replace Identity output with its input
                for consumer in list(graph.nodes):
                    for i, ci in enumerate(consumer.inputs):
                        if ci == out:
                            consumer.inputs[i] = inp
                # Rewire graph outputs if the Identity feeds a graph output
                for i, go in enumerate(graph.outputs):
                    if go == out:
                        graph.outputs[i] = inp
            node.inputs.clear()
            node.outputs.clear()
            graph.nodes.remove(node)
            removed += 1

    print(f"[patch_fp8_onnx] Removed {removed} Identity no-op nodes")

    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), str(dst))
    print(f"[patch_fp8_onnx] Saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def _verify_parse(onnx_path: str) -> bool:
    """Try parsing the patched ONNX with TRT STRONGLY_TYPED to confirm it works."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        data = f.read()

    success = parser.parse(data)
    if success:
        print(f"[patch_fp8_onnx] TRT parse: SUCCESS ({network.num_inputs} inputs, "
              f"{network.num_outputs} outputs)")
    else:
        print(f"[patch_fp8_onnx] TRT parse: FAILED ({parser.num_errors} errors)")
        for i in range(min(3, parser.num_errors)):
            err = parser.get_error(i)
            print(f"  Error {i}: {err.desc}")
    return success


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_fp8.onnx> <output_fp8.onnx>", file=sys.stderr)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    patch_fp8_onnx(src, dst)

    # Verify if tensorrt is available
    try:
        _verify_parse(dst)
    except ImportError:
        print("[patch_fp8_onnx] tensorrt not available — skipping parse verification")
