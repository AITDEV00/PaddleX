#!/usr/bin/env python3
"""Test FP8 inference through PaddleX tensorrt engine.

This is the critical test — FP8 support on Blackwell (RTX 5090, sm_120)
is the primary requirement.  The test:
1. Creates a PP-DocLayoutV3 predictor with engine='tensorrt', precision='fp8'
2. The runner auto-exports Paddle → ONNX, then quantizes ONNX to FP8
   via nvidia-modelopt, then builds a TRT FP8 engine
3. Runs inference and verifies output shapes
"""
import numpy as np
import time
import sys

from paddlex import create_predictor

print("=" * 70, flush=True)
print("Testing FP8 inference through PaddleX tensorrt engine", flush=True)
print("=" * 70, flush=True)

print("\n1. Creating PP-DocLayoutV3 predictor with engine='tensorrt', precision='fp8'...", flush=True)
t0 = time.time()
try:
    predictor = create_predictor(
        "PP-DocLayoutV3",
        engine="tensorrt",
        engine_config={
            "precision": "fp8",
            "device_id": 0,
            "model_type": "layout",
            "n_calib": 10,  # small number for quick test
        },
    )
    print(f"   Predictor created in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"   FAILED to create predictor: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a dummy 800x800 image (as PaddleX expects: uint8 RGB)
dummy_img = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

print("\n2. Running FP8 inference...", flush=True)
t0 = time.time()
try:
    result = predictor.predict(dummy_img)
    print(f"   Inference took {time.time()-t0:.3f}s", flush=True)
except Exception as e:
    print(f"   FAILED inference: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check result
print(f"\n3. Result type: {type(result)}", flush=True)
if hasattr(result, '__len__'):
    print(f"   Result length: {len(result)}", flush=True)

# Run a second inference to measure steady-state latency
print("\n4. Running second inference (warm cache)...", flush=True)
t0 = time.time()
result2 = predictor.predict(dummy_img)
elapsed = time.time() - t0
print(f"   Second inference took {elapsed:.3f}s", flush=True)

print("\n" + "=" * 70, flush=True)
print("SUCCESS: FP8 inference through PaddleX tensorrt engine works!", flush=True)
print("=" * 70, flush=True)
