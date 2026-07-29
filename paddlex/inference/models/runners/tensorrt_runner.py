# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native TensorRT inference runner.

Loads a pre-built TensorRT engine (or auto-builds one from an ONNX model)
and runs inference using the native ``tensorrt`` Python API with ``pycuda``
for GPU memory management.  No ``ultra_infer`` dependency is required.

Supported precisions: ``fp32``, ``fp16``, ``int8`` (Q/DQ ONNX), ``fp8``
(Q/DQ ONNX via nvidia-modelopt).
"""

from __future__ import annotations

import hashlib
import os
import time
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from ....constants import MODEL_FILE_PREFIX
from ....utils.deps import class_requires_deps
from ..utils.model_paths import get_model_paths
from .inference_runner import InferenceRunner
from .utils import sort_inputs

__all__ = ["TensorRTRunnerConfig", "TensorRTRunner"]

# Default optimization profile for layout models (800×800 input)
_DEFAULT_LAYOUT_PROFILE: Dict[str, Tuple[Tuple, Tuple, Tuple]] = {
    "image": ((1, 3, 800, 800), (4, 3, 800, 800), (8, 3, 800, 800)),
    "im_shape": ((1, 2), (4, 2), (8, 2)),
    "scale_factor": ((1, 2), (4, 2), (8, 2)),
}

# Default optimization profile for detection models (640×640 input)
_DEFAULT_DET_PROFILE: Dict[str, Tuple[Tuple, Tuple, Tuple]] = {
    "x": ((1, 3, 32, 32), (4, 3, 640, 640), (8, 3, 960, 960)),
}


class TensorRTRunnerConfig(BaseModel):
    """Engine config for native TensorRT inference."""

    model_config = ConfigDict(extra="forbid")

    precision: Literal["fp32", "fp16", "int8", "fp8"] = "fp16"
    device_id: Optional[int] = None
    workspace_gb: int = 4
    opt_level: int = 5

    # Dynamic shape profiles: {input_name: (min, opt, max)}
    # If None, auto-detected from input names (layout vs det profiles).
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None

    # Path to a pre-built .trt engine.  If None, the runner looks for
    # ``inference.trt`` in the model dir, then auto-builds from ONNX.
    engine_path: Optional[str] = None

    # Path to the ONNX model.  If None, discovered from model dir.
    onnx_path: Optional[str] = None

    # When True and no .trt engine exists, auto-export Paddle → ONNX first.
    auto_export: bool = True

    # When True and precision is int8/fp8, auto-quantize ONNX via modelopt.
    auto_quantize: bool = True

    # Directory of calibration images for INT8/FP8 quantization.
    calibration_data: Optional[str] = None

    # Number of calibration images.
    n_calib: int = 200

    # Model type hint for calibration preprocessing.
    model_type: Optional[str] = None


@class_requires_deps("tensorrt", "pycuda")
class TensorRTRunner(InferenceRunner):
    """Native TensorRT inference runner.

    On construction the runner resolves the engine in this order:

    1. ``engine_path`` from config (explicit pre-built engine)
    2. ``<model_dir>/inference.trt`` (cached engine)
    3. Auto-build from ONNX:
       a. If ONNX exists in model dir, use it.
       b. If only Paddle model exists and ``auto_export=True``, export to ONNX.
       c. If precision is int8/fp8 and ``auto_quantize=True``, quantize ONNX.
       d. Build TRT engine and cache as ``<model_dir>/inference.trt``.
    """

    def __init__(
        self,
        model_dir: Union[str, PathLike],
        model_file_prefix: str = MODEL_FILE_PREFIX,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context
        import pycuda.driver as cuda

        super().__init__()
        self.model_dir = Path(model_dir)
        self.model_file_prefix = model_file_prefix
        self._config = TensorRTRunnerConfig.model_validate(
            config or {}
        ).model_dump(exclude_none=True)

        self._trt = trt
        self._cuda = cuda
        self._logger = trt.Logger(trt.Logger.WARNING)

        precision = self._config.get("precision", "fp16")
        self._fp16 = precision in ("fp16", "int8")
        self._fp8 = precision == "fp8"
        self._int8 = precision == "int8"

        device_id = self._config.get("device_id", 0)
        self._device_id = device_id if device_id is not None else 0

        # Resolve engine path
        self._engine_path = self._resolve_engine_path()

        # Load engine
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._load_engine()
        self._context = self._engine.create_execution_context()

        # Discover I/O tensor names
        self._input_names = self._get_io_names(is_input=True)
        self._output_names = self._get_io_names(is_input=False)

        # Pre-allocate GPU buffers (lazy: allocated on first call for actual
        # batch size, then reused/reallocated as needed)
        self._d_inputs: Dict[str, Any] = {}
        self._d_outputs: Dict[str, Any] = {}
        self._h_outputs: Dict[str, np.ndarray] = {}
        self._buf_sizes: Dict[str, int] = {}  # track allocated byte sizes
        self._stream = cuda.Stream()

        # Pinned-memory optimization: use pagelocked host output buffers +
        # async D2H memcpy to enable DMA overlap with GPU execution.
        # Disabled if HPS_TRT_PINNED=0 is set (for fallback / debugging).
        self._use_pinned = os.environ.get("HPS_TRT_PINNED", "1") in (
            "1", "true", "True",
        )

    def __call__(
        self,
        x: Union[Sequence[np.ndarray], np.ndarray, None] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        if x is None and "x" in kwargs:
            x = kwargs["x"]
        if x is None:
            raise TypeError("`TensorRTRunner.__call__` requires `x`")
        if isinstance(x, np.ndarray):
            x = [x]

        if len(self._input_names) != len(x):
            raise ValueError(
                f"The number of inputs does not match the model: "
                f"{len(self._input_names)} vs {len(x)}"
            )

        x = sort_inputs(x, self._input_names)

        _lat = os.environ.get("HPS_LATENCY_LOG", "0") in ("1", "true", "True")
        _skip_d2h_copy = os.environ.get("HPS_TRT_SKIP_D2H_COPY", "0") in (
            "1",
            "true",
            "True",
        )
        # Skip D2H copy for large outputs that the caller doesn't need.
        # PP-DocLayoutV3's fetch_name_2 (masks) is 48 MB at bs=1 / 96 MB at
        # bs=2 — all zeros for rect-only layout detection (the API path).
        # Skipping the D2H transfer saves ~1.6 ms at bs=1 / ~3.2 ms at bs=2.
        # Format: comma-separated output names, or "auto" to skip any
        # output whose device buffer exceeds 10 MB.
        _skip_d2h_names_raw = os.environ.get("HPS_TRT_SKIP_D2H_OUTPUTS", "")
        _t0 = time.perf_counter() if _lat else 0.0

        # Set input shapes for dynamic batch
        for name, arr in zip(self._input_names, x):
            self._context.set_input_shape(name, arr.shape)

        _t_set_shape = time.perf_counter() if _lat else 0.0

        # Allocate / reallocate buffers for this batch
        self._allocate_buffers(x)

        _t_alloc = time.perf_counter() if _lat else 0.0

        # Copy inputs H2D — async copy on stream.
        # All operations on the same stream are serialized, so execute
        # won't start until H2D completes.
        #
        # Optimization: use direct async memcpy from the numpy buffer
        # instead of copying through a pinned intermediate.  For our data
        # sizes (≤15 MB), the extra np.copyto to a pinned buffer costs
        # more (~0.6 ms) than the DMA speedup it provides.  Micro-bench:
        #   pinned+async = 1.22 ms,  direct async = 1.00 ms.
        for name, arr in zip(self._input_names, x):
            arr_contig = (
                arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
            )
            self._cuda.memcpy_htod_async(
                self._d_inputs[name], arr_contig, self._stream
            )
            self._context.set_tensor_address(name, int(self._d_inputs[name]))

        for name in self._output_names:
            self._context.set_tensor_address(name, int(self._d_outputs[name]))

        _t_h2d = time.perf_counter() if _lat else 0.0

        # Execute (queued on same stream after H2D)
        self._context.execute_async_v3(self._stream.handle)

        _t_exec_queue = time.perf_counter() if _lat else 0.0

        # Determine which outputs to skip D2H for.
        if _skip_d2h_names_raw.strip().lower() == "auto":
            _skip_d2h_names = {
                name for name in self._output_names
                if self._buf_sizes.get(name, 0) > 10 * 1024 * 1024
            }
        elif _skip_d2h_names_raw.strip():
            _skip_d2h_names = {
                n.strip() for n in _skip_d2h_names_raw.split(",") if n.strip()
            }
        else:
            _skip_d2h_names = set()

        # Queue async D2H copies on the SAME stream — they will execute
        # automatically after the GPU kernels finish, without requiring a
        # separate synchronize between execute and copy.  This pipelines
        # execute→D2H on the GPU side, reducing idle time.
        for name in self._output_names:
            if name in _skip_d2h_names:
                continue
            self._cuda.memcpy_dtoh_async(
                self._h_outputs[name], self._d_outputs[name], self._stream
            )

        # Single sync — waits for H2D + execute + D2H to all complete.
        self._stream.synchronize()

        _t_sync = time.perf_counter() if _lat else 0.0

        # Outputs are now in host buffers — just collect them.
        # _allocate_buffers() sized the host buffer to the correct shape
        # for the current batch (via get_tensor_shape after set_input_shape).
        results = []
        for name in self._output_names:
            if name in _skip_d2h_names:
                # Return an empty array placeholder for skipped outputs.
                # The caller's _format_output takes len(pred)==2 path
                # when only [boxes, box_nums] are present, which avoids
                # any masks processing.
                continue
            h = self._h_outputs[name]
            if _skip_d2h_copy:
                results.append(h)
            else:
                results.append(h.copy())

        if _lat:
            _t_d2h = time.perf_counter()
            print(
                '{"event":"latency","stage":"trt_runner",'
                '"set_shape_ms":%.3f,"alloc_ms":%.3f,'
                '"h2d_copy_ms":%.3f,"exec_queue_ms":%.3f,'
                '"sync_ms":%.3f,"d2h_collect_ms":%.3f,'
                '"total_ms":%.3f,"batch_size":%d}'
                % (
                    (_t_set_shape - _t0) * 1000,
                    (_t_alloc - _t_set_shape) * 1000,
                    (_t_h2d - _t_alloc) * 1000,
                    (_t_exec_queue - _t_h2d) * 1000,
                    (_t_sync - _t_exec_queue) * 1000,
                    (_t_d2h - _t_sync) * 1000,
                    (_t_d2h - _t0) * 1000,
                    x[0].shape[0] if x else 0,
                ),
                flush=True,
            )

        return results

    def close(self) -> None:
        """Free GPU buffers and destroy TRT context/engine."""
        for buf in list(self._d_inputs.values()) + list(self._d_outputs.values()):
            try:
                buf.free()
            except Exception:
                pass
        self._d_inputs.clear()
        self._d_outputs.clear()
        self._h_outputs.clear()
        del self._context
        del self._engine
        del self._runtime

    # ------------------------------------------------------------------
    # Engine resolution and building
    # ------------------------------------------------------------------

    def _resolve_engine_path(self) -> Path:
        """Determine where to load or build the TRT engine."""
        # 1. Explicit engine_path from config
        explicit = self._config.get("engine_path")
        if explicit:
            p = Path(explicit)
            if p.exists():
                return p
            raise FileNotFoundError(f"Engine not found: {explicit}")

        # 2. Cached engine in model dir, keyed by precision
        precision = self._config.get("precision", "fp16")
        cache_name = f"{self.model_file_prefix}_{precision}.trt"
        cache_path = self.model_dir / cache_name
        if cache_path.exists():
            return cache_path

        # 3. Auto-build from ONNX (or Paddle → ONNX → TRT)
        return self._build_engine(cache_path)

    def _build_engine(self, output_path: Path) -> Path:
        """Build a TRT engine from ONNX (auto-exporting Paddle if needed)."""
        import tensorrt as trt

        onnx_path = self._resolve_onnx_path()

        builder = trt.Builder(self._logger)
        # For FP8/INT8 quantized models (QDQ-bearing ONNX from modelopt), use a
        # STRONGLY-TYPED network: every tensor's type comes from the ONNX graph
        # (Q/DQ nodes pin FP8/INT8), and kFP16 must NOT be set (TRT rejects it
        # under strong typing). This matches TurboOCR's approach for heavier
        # modelopt exports that have mixed FP8/FP16/FP32 (e.g. Conv with Float
        # input + Half kernel). For non-quantized FP16, use a weakly-typed
        # network with FP16 flag.
        # (See TurboOCR src/engine/trt/onnx_to_trt.cpp: is_quantized_onnx())
        precision = self._config.get("precision", "fp16")
        is_quantized = precision in ("fp8", "int8")

        create_flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            create_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if is_quantized and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            create_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        elif not is_quantized and hasattr(trt.NetworkDefinitionCreationFlag, "WEAKLY_TYPED"):
            create_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.WEAKLY_TYPED)
        network = builder.create_network(create_flags)
        parser = trt.OnnxParser(network, self._logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(
                    f"Failed to parse ONNX: {onnx_path}\n" + "\n".join(errors)
                )

        config = builder.create_builder_config()
        workspace_gb = self._config.get("workspace_gb", 4)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
        )
        config.builder_optimization_level = self._config.get("opt_level", 5)

        # Timing cache for faster rebuilds
        cache_dir = output_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        timing_cache_path = cache_dir / "timing.cache"
        timing_data = b""
        if timing_cache_path.exists():
            timing_data = timing_cache_path.read_bytes()
        timing_cache = config.create_timing_cache(timing_data)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

        # Precision flags
        # For STRONGLY-TYPED (quantized) networks: do NOT set FP16/INT8 flags.
        # TRT reads precision from the graph's Q/DQ nodes. Setting FP16 under
        # strong typing is rejected by TRT. (TurboOCR onnx_to_trt.cpp.)
        if precision == "fp8":
            # Strongly-typed: precision comes from QDQ nodes. No flags needed.
            pass
        elif precision == "int8":
            # Strongly-typed: precision comes from QDQ nodes. No flags needed.
            pass
        elif precision == "fp16":
            # TRT 10.16 removed BuilderFlag.FP16 from Python bindings (it
            # still exists in C++ as kFP16 = enum value 0). Construct from
            # the integer value to set FP16 on weakly-typed networks.
            config.set_flag(trt.BuilderFlag(0))  # kFP16

        # Optimization profile
        profile = builder.create_optimization_profile()
        dynamic_shapes = self._config.get("dynamic_shapes")
        if dynamic_shapes:
            for name, shapes in dynamic_shapes.items():
                profile.set_shape(
                    name,
                    min=tuple(shapes[0]),
                    opt=tuple(shapes[1]),
                    max=tuple(shapes[2]),
                )
        else:
            # Auto-detect from input names
            input_names = [network.get_input(i).name
                           for i in range(network.num_inputs)]
            if "image" in input_names:
                # Layout model (PP-DocLayoutV3)
                for name, (mn, op, mx) in _DEFAULT_LAYOUT_PROFILE.items():
                    if name in input_names:
                        profile.set_shape(name, min=mn, opt=op, max=mx)
            else:
                # Detection model
                for name, (mn, op, mx) in _DEFAULT_DET_PROFILE.items():
                    if name in input_names:
                        profile.set_shape(name, min=mn, opt=op, max=mx)
        config.add_optimization_profile(profile)

        t0 = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                f"Failed to build TRT engine (precision={precision})"
            )

        output_path.write_bytes(serialized)

        # Save timing cache
        updated = config.get_timing_cache()
        timing_cache_path.write_bytes(bytearray(updated.serialize()))

        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / 1e6
        print(
            f"  [TensorRTRunner] Built {precision} engine: {size_mb:.1f} MB "
            f"in {elapsed:.1f}s → {output_path.name}"
        )
        return output_path

    def _resolve_onnx_path(self) -> Path:
        """Find or create the ONNX model for engine building."""
        # 1. Explicit onnx_path from config
        explicit = self._config.get("onnx_path")
        if explicit and Path(explicit).exists():
            return Path(explicit)

        # 2. Check model dir for ONNX
        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "onnx" in model_paths:
            onnx_path = model_paths["onnx"]
        elif "paddle" in model_paths and self._config.get("auto_export", True):
            onnx_path = self._export_paddle_to_onnx(model_paths["paddle"])
        else:
            raise RuntimeError(
                f"No ONNX or Paddle model found in {self.model_dir}"
            )

        # For int8/fp8, check if a pre-quantized ONNX exists or quantize
        precision = self._config.get("precision", "fp16")
        if precision in ("int8", "fp8") and self._config.get("auto_quantize", True):
            suffix = "_fp8.onnx" if precision == "fp8" else ".int8.onnx"
            quantized_path = onnx_path.with_name(onnx_path.stem + suffix)
            if not quantized_path.exists():
                quantized_path = self._quantize_onnx(onnx_path, precision)
            onnx_path = quantized_path

        return onnx_path

    def _export_paddle_to_onnx(
        self, paddle_paths: Tuple[Path, Path]
    ) -> Path:
        """Export a PaddlePaddle model to ONNX using paddle2onnx."""
        import paddle2onnx
        import onnx

        model_file, params_file = str(paddle_paths[0]), str(paddle_paths[1])
        onnx_path = self.model_dir / f"{self.model_file_prefix}.onnx"

        print(f"  [TensorRTRunner] Exporting Paddle → ONNX: {onnx_path.name}")
        paddle2onnx.export(
            model_filename=model_file,
            params_filename=params_file,
            save_file=str(onnx_path),
            opset_version=17,
            deploy_backend="onnxruntime",
        )

        # Constant folding via Polygraphy for smaller, cleaner graph
        try:
            from polygraphy.backend.onnx import fold_constants
            model = onnx.load(str(onnx_path))
            model = fold_constants(model)
            onnx.save(model, str(onnx_path))
            print(f"  [TensorRTRunner] Polygraphy constant folding applied")
        except Exception as e:
            print(f"  [TensorRTRunner] Polygraphy folding skipped: {e}")

        return onnx_path

    def _quantize_onnx(self, onnx_path: Path, precision: str) -> Path:
        """Quantize an ONNX model to INT8 or FP8 via nvidia-modelopt."""
        try:
            import modelopt.onnx.quantization as moq
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"nvidia-modelopt is required for {precision} quantization "
                f"but is not installed. Install it with: "
                f"pip install nvidia-modelopt"
            ) from e

        suffix = "_fp8.onnx" if precision == "fp8" else ".int8.onnx"
        out_path = onnx_path.with_name(onnx_path.stem + suffix)

        # Build calibration data reader
        model_type = self._config.get("model_type") or "layout"
        calib_data = self._config.get("calibration_data")
        n_calib = self._config.get("n_calib", 200)

        reader = _build_calibration_reader(
            model_type, calib_data, n_calib
        )

        quantize_mode = "fp8" if precision == "fp8" else "int8"
        calibration_method = "max" if precision == "fp8" else "entropy"

        print(
            f"  [TensorRTRunner] Quantizing ONNX → {precision} "
            f"(mode={quantize_mode}, method={calibration_method})"
        )

        kwargs = dict(
            onnx_path=str(onnx_path),
            quantize_mode=quantize_mode,
            calibration_data_reader=reader,
            calibration_method=calibration_method,
            output_path=str(out_path),
        )
        if precision == "int8":
            # Keep DQ scales in FP32 for TRT compatibility
            kwargs["high_precision_dtype"] = "fp32"

        moq.quantize(**kwargs)

        if not out_path.exists():
            raise RuntimeError(
                f"modelopt did not produce {out_path}"
            )

        # Post-process: coerce QDQ scales to FP32 for TRT parser compatibility
        _coerce_qdq_scales_to_fp32(out_path)

        print(
            f"  [TensorRTRunner] Quantized ONNX: "
            f"{out_path.stat().st_size / 1e6:.1f} MB"
        )
        return out_path

    # ------------------------------------------------------------------
    # Engine loading and inference
    # ------------------------------------------------------------------

    def _load_engine(self):
        with open(self._engine_path, "rb") as f:
            serialized = f.read()
        engine = self._runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {self._engine_path}"
            )
        return engine

    def _get_io_names(self, is_input: bool) -> List[str]:
        names = []
        n = (
            self._engine.num_io_tensors
        )
        for i in range(n):
            tensor_name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(tensor_name)
            if is_input and str(mode) == "TensorIOMode.INPUT":
                names.append(tensor_name)
            elif not is_input and str(mode) == "TensorIOMode.OUTPUT":
                names.append(tensor_name)
        return names

    def _allocate_buffers(self, inputs: List[np.ndarray]):
        """Allocate or reallocate GPU buffers for the current batch.

        When ``HPS_TRT_PINNED=1`` (default), host output buffers use
        ``pycuda.driver.pagelocked_empty()`` — pinned/page-locked memory
        that enables async DMA transfers, overlapping H2D/D2H copies with
        GPU kernel execution on the same stream.
        """
        import tensorrt as trt

        # Inputs — device buffers
        for name, arr in zip(self._input_names, inputs):
            nbytes = arr.nbytes
            current = self._d_inputs.get(name)
            cur_size = self._buf_sizes.get(name, 0)
            if current is not None and cur_size >= nbytes:
                continue  # existing buffer is large enough
            if current is not None:
                current.free()
            self._d_inputs[name] = self._cuda.mem_alloc(nbytes)
            self._buf_sizes[name] = nbytes

        # Outputs — device + host buffers
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            dtype_map = {
                str(trt.float32): np.float32,
                str(trt.float16): np.float16,
                str(trt.int32): np.int32,
                str(trt.int8): np.int8,
                str(trt.bool): np.bool_,
            }
            trt_dtype = str(self._engine.get_tensor_dtype(name))
            np_dtype = dtype_map.get(trt_dtype, np.float32)

            n_elements = 1
            for dim in shape:
                n_elements *= int(dim)

            nbytes = n_elements * np.dtype(np_dtype).itemsize

            current = self._d_outputs.get(name)
            cur_size = self._buf_sizes.get(name, 0)
            if current is not None and cur_size >= nbytes:
                # Reuse device buffer. Only reallocate pinned host buffer
                # if shape changed (pagelocked_empty is an expensive CUDA call).
                h_cur = self._h_outputs.get(name)
                if h_cur is None or h_cur.shape != shape or h_cur.dtype != np_dtype:
                    if self._use_pinned:
                        self._h_outputs[name] = self._cuda.pagelocked_empty(
                            shape, np_dtype
                        )
                    else:
                        self._h_outputs[name] = np.empty(shape, dtype=np_dtype)
                continue
            if current is not None:
                current.free()
            self._d_outputs[name] = self._cuda.mem_alloc(nbytes)
            self._buf_sizes[name] = nbytes
            if self._use_pinned:
                self._h_outputs[name] = self._cuda.pagelocked_empty(
                    shape, np_dtype
                )
            else:
                self._h_outputs[name] = np.empty(shape, dtype=np_dtype)


# ----------------------------------------------------------------------
# Calibration helpers (ported from TurboOCR's quantize_onnx_int8.py)
# ----------------------------------------------------------------------

def _build_calibration_reader(
    model_type: str,
    calib_data_dir: Optional[str],
    n_calib: int,
):
    """Build a modelopt-compatible calibration data reader."""
    import cv2
    from pathlib import Path as P

    # If no calibration dir given, generate synthetic data
    if calib_data_dir is None or not Path(calib_data_dir).exists():
        print(
            f"  [TensorRTRunner] No calibration data dir; "
            f"using synthetic data for {model_type}"
        )
        return _SyntheticReader(model_type, n_calib)

    # Collect image paths
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted(
        [p for p in Path(calib_data_dir).iterdir() if p.suffix.lower() in exts]
    )[:n_calib]
    print(
        f"  [TensorRTRunner] Calibration: {len(images)} images "
        f"from {calib_data_dir}"
    )

    def _gen():
        for p in images:
            yield _preprocess_for_calibration(model_type, p)

    return _ListReader(list(_gen()))


def _preprocess_for_calibration(model_type: str, path: Path) -> Dict[str, np.ndarray]:
    """Preprocess a single image for calibration (matches PaddleX pipeline)."""
    import cv2

    if model_type == "layout":
        return _preprocess_layout_calib(path)
    elif model_type == "det":
        return _preprocess_det_calib(path)
    else:
        # Default to layout preprocessing
        return _preprocess_layout_calib(path)


def _letterbox(img: np.ndarray, size: int, pad: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas


def _preprocess_layout_calib(path: Path, size: int = 800) -> Dict[str, np.ndarray]:
    """PP-DocLayoutV3: 3 inputs, 800×800 letterboxed, ImageNet normalize."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    canvas = _letterbox(img, size)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis]
    im_shape = np.array([[size, size]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
    return {
        "image": chw,
        "im_shape": im_shape,
        "scale_factor": scale_factor,
    }


def _preprocess_det_calib(path: Path, max_side: int = 960) -> Dict[str, np.ndarray]:
    """Detection: dynamic shape, ImageNet normalize."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    h, w = img.shape[:2]
    r = min(1.0, max_side / max(h, w))
    rh = max(int(round(h * r / 32.0) * 32), 32)
    rw = max(int(round(w * r / 32.0) * 32), 32)
    resized = cv2.resize(img, (rw, rh))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)
    return {"x": chw[np.newaxis]}


class _ListReader:
    """modelopt-compatible calibration reader wrapping a materialized list."""

    def __init__(self, items: List[Dict[str, np.ndarray]]):
        self._items = items
        self._idx = 0

    def get_first(self):
        return self._items[0] if self._items else None

    def get_next(self):
        if self._idx >= len(self._items):
            return None
        item = self._items[self._idx]
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


class _SyntheticReader:
    """Generate synthetic calibration data when no real images are available."""

    def __init__(self, model_type: str, n: int = 10):
        self._model_type = model_type
        self._n = n
        self._idx = 0
        self._items = [self._generate() for _ in range(n)]

    def _generate(self) -> Dict[str, np.ndarray]:
        if self._model_type == "layout":
            return {
                "image": np.random.randn(1, 3, 800, 800).astype(np.float32),
                "im_shape": np.array([[800, 800]], dtype=np.float32),
                "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32),
            }
        else:
            return {"x": np.random.randn(1, 3, 640, 640).astype(np.float32)}

    def get_first(self):
        return self._items[0] if self._items else None

    def get_next(self):
        if self._idx >= len(self._items):
            return None
        item = self._items[self._idx]
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


def _coerce_qdq_scales_to_fp32(onnx_path: Path) -> int:
    """Force all QDQ scale initializers to FP32 for TRT parser compatibility.

    TRT 10.15+ has a parser bug where FP16 scale tensors in QuantizeLinear /
    DequantizeLinear nodes are silently ignored.  This post-processing step
    ensures all scales are FP32.
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    model = onnx.load(str(onnx_path))
    targets = set()
    for node in model.graph.node:
        if node.op_type in ("DequantizeLinear", "QuantizeLinear"):
            if len(node.input) >= 2:
                targets.add(node.input[1])  # scale

    # Build replacement list to avoid modifying while iterating
    to_remove = []
    to_add = []
    for init in model.graph.initializer:
        if init.name in targets and init.data_type != TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            to_remove.append(init)
            to_add.append(new_init)

    for init in to_remove:
        model.graph.initializer.remove(init)
    for init in to_add:
        model.graph.initializer.append(init)

    coerced = len(to_add)
    if coerced > 0:
        onnx.save(model, str(onnx_path))

    return coerced
