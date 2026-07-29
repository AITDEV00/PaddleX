#!/usr/bin/env python3
"""Patch the installed PaddleX package to register the TensorRT engine.

The PyPI wheel of PaddleX (v3.7.2) does not include the custom
TensorRTEngine module that was added to the jya0-v3.7.2 branch.
This script copies the new files and patches the existing __init__.py
/ deps.py so PaddleX registers the "tensorrt" inference engine.

Usage:
    python3 patch_paddlex_trt.py <patch_dir> <paddlex_site_dir>

    patch_dir       — directory containing tensorrt.py and tensorrt_runner.py
    paddlex_site_dir — installed paddlex package root
                      (e.g. /usr/local/lib/python3.10/dist-packages/paddlex)
"""

import pathlib
import shutil
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: patch_paddlex_trt.py <patch_dir> <paddlex_site_dir>")
        sys.exit(1)

    patch_dir = pathlib.Path(sys.argv[1])
    pdx = pathlib.Path(sys.argv[2])

    # ─── 1. Copy new files ──────────────────────────────────────────────
    src_engine = patch_dir / "tensorrt.py"
    src_runner = patch_dir / "tensorrt_runner.py"
    dst_engine = pdx / "inference" / "models" / "engines" / "tensorrt.py"
    dst_runner = pdx / "inference" / "models" / "runners" / "tensorrt_runner.py"

    shutil.copy2(src_engine, dst_engine)
    shutil.copy2(src_runner, dst_runner)
    print(f"Copied {src_engine.name} → {dst_engine}")
    print(f"Copied {src_runner.name} → {dst_runner}")

    # ─── 1b. Copy optimized processors ─────────────────────────────────
    # Vectorized post-processing (NMS, containment, filter_large_image,
    # restructured_boxes, unclip_boxes, threshold) and Normalize.
    src_obj_det = patch_dir / "obj_det_processors.py"
    src_vision = patch_dir / "vision_processors.py"
    src_predictor = patch_dir / "obj_det_predictor.py"
    src_layout_pred = patch_dir / "layout_predictor.py"
    dst_obj_det = (
        pdx / "inference" / "models" / "object_detection" / "processors.py"
    )
    dst_vision = (
        pdx / "inference" / "models" / "common" / "vision" / "processors.py"
    )
    dst_predictor = (
        pdx / "inference" / "models" / "object_detection" / "predictor.py"
    )
    dst_layout_pred = (
        pdx / "inference" / "models" / "layout_analysis" / "predictor.py"
    )
    if src_obj_det.exists():
        shutil.copy2(src_obj_det, dst_obj_det)
        print(f"Copied obj_det_processors.py → {dst_obj_det}")
    if src_vision.exists():
        shutil.copy2(src_vision, dst_vision)
        print(f"Copied vision_processors.py → {dst_vision}")
    if src_predictor.exists():
        shutil.copy2(src_predictor, dst_predictor)
        print(f"Copied obj_det_predictor.py → {dst_predictor}")
    if src_layout_pred.exists():
        shutil.copy2(src_layout_pred, dst_layout_pred)
        print(f"Copied layout_predictor.py → {dst_layout_pred}")

    # ─── 2. Patch engines/__init__.py ───────────────────────────────────
    p = pdx / "inference" / "models" / "engines" / "__init__.py"
    s = p.read_text()
    if "tensorrt" not in s:
        s = s.replace(
            "    paddle,\n    transformers,",
            "    paddle,\n    tensorrt,\n    transformers,",
        )
        p.write_text(s)
        print("Patched engines/__init__.py: added tensorrt import")
    else:
        print("engines/__init__.py already has tensorrt")

    # ─── 3. Patch runners/__init__.py ───────────────────────────────────
    p = pdx / "inference" / "models" / "runners" / "__init__.py"
    s = p.read_text()
    if "TensorRTRunner" not in s:
        s = s.replace(
            "from .onnxruntime_runner import ONNXRuntimeRunner, ONNXRuntimeRunnerConfig",
            "from .onnxruntime_runner import ONNXRuntimeRunner, ONNXRuntimeRunnerConfig\n"
            "from .tensorrt_runner import TensorRTRunner, TensorRTRunnerConfig",
        )
        s = s.replace(
            '"PaddleStaticRunnerConfig",\n]',
            '"PaddleStaticRunnerConfig",\n    "TensorRTRunner",\n    "TensorRTRunnerConfig",\n]',
        )
        p.write_text(s)
        print("Patched runners/__init__.py: added TensorRTRunner import")
    else:
        print("runners/__init__.py already has TensorRTRunner")

    # ─── 4. Patch utils/deps.py ─────────────────────────────────────────
    p = pdx / "utils" / "deps.py"
    s = p.read_text()
    if 'dep == "tensorrt"' not in s:
        s = s.replace(
            'elif dep == "onnxruntime":\n'
            '        return importlib.util.find_spec("onnxruntime") is not None',
            'elif dep == "onnxruntime":\n'
            '        return importlib.util.find_spec("onnxruntime") is not None\n'
            '    elif dep == "tensorrt":\n'
            '        return importlib.util.find_spec("tensorrt") is not None\n'
            '    elif dep == "pycuda":\n'
            '        return importlib.util.find_spec("pycuda") is not None',
        )
        p.write_text(s)
        print("Patched utils/deps.py: added tensorrt/pycuda dep checks")
    else:
        print("utils/deps.py already has tensorrt")

    # ─── 5. Patch layout_analysis/__init__.py ───────────────────────────
    p = pdx / "inference" / "models" / "layout_analysis" / "__init__.py"
    s = p.read_text()
    if '"tensorrt"' not in s:
        s = s.replace(
            '"onnxruntime": LAYOUTANALYSIS_MODELS,',
            '"onnxruntime": LAYOUTANALYSIS_MODELS,\n        "tensorrt": LAYOUTANALYSIS_MODELS,',
        )
        p.write_text(s)
        print("Patched layout_analysis/__init__.py: registered tensorrt binding")
    else:
        print("layout_analysis/__init__.py already has tensorrt")

    # ─── 6. Clear bytecode cache ────────────────────────────────────────
    import subprocess
    subprocess.run(
        ["find", str(pdx), "-name", "__pycache__", "-type", "d",
         "-exec", "rm", "-rf", "{}", "+"],
        capture_output=True,
    )
    print("Cleared __pycache__ directories")

    # ─── 7. Verify ──────────────────────────────────────────────────────
    import importlib
    # Force reimport
    for mod_name in list(sys.modules):
        if "paddlex" in mod_name:
            del sys.modules[mod_name]
    from paddlex.inference.models.engines import InferenceEngine
    engines = InferenceEngine.all()
    assert "tensorrt" in engines, (
        f"tensorrt engine not registered! engines={list(engines)}"
    )
    print(f"Registered engines: {list(engines)}")
    print("✓ tensorrt engine patch verified OK")


if __name__ == "__main__":
    main()
