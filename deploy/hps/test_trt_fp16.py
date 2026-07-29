#!/usr/bin/env python3
"""Test FP16 inference through PaddleX tensorrt engine."""
import numpy as np
import time
import sys

# Test FP16 inference through PaddleX create_predictor
from paddlex import create_predictor

print("Creating PP-DocLayoutV3 predictor with engine='tensorrt', precision='fp16'...", flush=True)
t0 = time.time()
try:
    predictor = create_predictor(
        "PP-DocLayoutV3",
        engine="tensorrt",
        engine_config={
            "precision": "fp16",
            "device_id": 0,
        },
    )
    print(f"  Predictor created in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED to create predictor: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a dummy 800x800 image (as PaddleX expects: uint8 RGB)
dummy_img = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

print("Running inference...", flush=True)
t0 = time.time()
try:
    result = predictor.predict(dummy_img)
    print(f"  Inference took {time.time()-t0:.3f}s", flush=True)
except Exception as e:
    print(f"  FAILED inference: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check result
print(f"  Result type: {type(result)}", flush=True)
if hasattr(result, '__len__'):
    print(f"  Result length: {len(result)}", flush=True)

print("\nSUCCESS: FP16 inference through PaddleX tensorrt engine works!", flush=True)
