# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from packaging import version
from .deprecation_utils import deprecate
from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    OptionalDependencyNotAvailable,
    is_accelerate_available,
    is_accelerate_version,
    is_flax_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_librosa_available,
    is_omegaconf_available,
    is_onnx_available,
    is_safetensors_available,
    is_scipy_available,
    is_tensorboard_available,
    is_tf_available,
    is_torch_available,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    is_wandb_available,
    is_xformers_available,
    requires_backends,
)
from .logging import get_logger


logger = get_logger(__name__)
