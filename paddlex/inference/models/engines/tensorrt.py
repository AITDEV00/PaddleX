#!/usr/bin/env python3
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

"""TensorRT engine.

Provides native TensorRT inference through a custom ``TensorRTRunner``
that uses the ``tensorrt`` Python API directly with ``pycuda`` for GPU
memory management — no ``ultra_infer`` dependency required.

Supported precisions: ``fp32``, ``fp16``, ``int8``, ``fp8``.
The engine can auto-export Paddle models to ONNX and auto-quantize
to INT8/FP8 via nvidia-modelopt when the corresponding precision is
requested.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from ....constants import MODEL_FILE_PREFIX
from ....utils.deps import is_dep_available
from ..runners import TensorRTRunner
from ..runners.inference_runner import InferenceRunner
from ..runners.tensorrt_runner import TensorRTRunnerConfig
from ..utils.model_paths import LocalModelFormat
from ._base import RunnerBuilder, RunnerEngine

__all__ = ["TensorRTEngine"]


class TensorRTEngine(RunnerEngine):
    """Engine for native TensorRT inference.

    This engine builds a ``TensorRTRunner`` that manages TRT engine
    lifecycle (build, load, execute) using the ``tensorrt`` and
    ``pycuda`` Python packages.  It supports FP32, FP16, INT8 (Q/DQ),
    and FP8 (Q/DQ) precisions.
    """

    entities = "tensorrt"

    @property
    def name(self) -> str:
        return "tensorrt"

    @property
    def engine_config_model(self) -> Type[TensorRTRunnerConfig]:
        return TensorRTRunnerConfig

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        # TRT engine can be built from either ONNX or Paddle models.
        # The runner auto-exports Paddle → ONNX when needed.
        return ("paddle", "onnx")

    def prepare_config_dict(
        self,
        raw: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name
        self._apply_device(raw, device)
        # TensorRT only runs on GPU; coerce device_type to "gpu"
        device_type = raw.pop("device_type", None)
        if device_type is not None and device_type != "gpu":
            raise ValueError(
                f"Engine 'tensorrt' requires GPU, but device type "
                f"'{device_type}' was specified."
            )
        # device_id is already set by _apply_device if a device string
        # like "gpu:0" was passed.
        return raw

    def ensure_environment(self) -> None:
        missing = []
        if not is_dep_available("tensorrt"):
            missing.append("tensorrt")
        if not is_dep_available("pycuda"):
            missing.append("pycuda")
        if missing:
            raise RuntimeError(
                "Engine 'tensorrt' is unavailable because the following "
                f"dependencies are not installed: {', '.join(missing)}."
            )

    def _check_device_support(self, engine_config: Dict[str, Any]) -> None:
        # TensorRT always runs on GPU — nothing to check beyond what
        # ensure_environment already validates.
        pass

    def get_default_runner_builder(self) -> RunnerBuilder:
        def runner_builder(
            *,
            model_name: str,
            model_dir: Optional[Path],
            model_config: Optional[Dict[str, Any]],
            engine_config: Dict[str, Any],
            default_builder: Optional[RunnerBuilder] = None,
        ) -> InferenceRunner:
            del model_name, model_config, default_builder
            if model_dir is None:
                raise ValueError("`model_dir` is required for engine='tensorrt'.")
            self._check_device_support(engine_config)
            return TensorRTRunner(
                model_dir=model_dir,
                model_file_prefix=MODEL_FILE_PREFIX,
                config=engine_config,
            )

        return runner_builder

    def validate_runner(self, runner: InferenceRunner) -> None:
        if not isinstance(runner, TensorRTRunner):
            raise TypeError(
                "Engine 'tensorrt' must build a TensorRTRunner, "
                f"but got {type(runner).__name__}."
            )
