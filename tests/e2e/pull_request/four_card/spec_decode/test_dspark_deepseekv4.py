#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig
from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory
from tests.e2e.spec_decode_utils import assert_spec_decode_acceptance

MODELS = ["UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Confidence-based dynamic verify-length; keep in sync with
# tests/e2e/pull_request/one_card/spec_decode/test_dynamic.py (dspark).
DSPARK_DYNAMIC_SPEC_CONFIG = {
    "method": "dspark",
    "method_params": {
        "initial_verify_budget_per_req": 3,
        "budget_update_interval": 1,
        "budget_threshold": 0.7,
    },
}


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("golden", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(
            [0.88, 0.74, 0.58, 0.49, 0.40],
            5,
            {"enable_dsa_cp": False},
            id="dspark",
        ),
        pytest.param(
            [0.88, 0.74, 0.58, 0.49, 0.40, 0.30, 0.18],
            7,
            {"enable_dsa_cp": True},
            id="dsa-cp-dspark",
        ),
        pytest.param(
            [0.65, 0.55, 0.45, 0.35, 0.30],
            5,
            {
                "enable_flashcomm1": False,
                "enable_dsa_cp": False,
                "dynamic_spec_config": DSPARK_DYNAMIC_SPEC_CONFIG,
            },
            id="dspark-dynamic",
        ),
    ],
)
@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_deepseek_v4_dspark_acceptance_tp4(
    model_name,
    golden,
    num_speculative_tokens,
    additional_config,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 1024

    with VllmRunner(
        model_name,
        tensor_parallel_size=4,
        max_model_len=4096,
        enable_expert_parallel=True,
        disable_log_stats=False,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            "enforce_eager": True,
        },
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[6, 8, 16, 18],
        ),
        additional_config=additional_config,
    ) as spec_vllm_model:
        _ = spec_vllm_model.generate_greedy(example_prompts, max_tokens)
        metrics = spec_vllm_model.model.get_metrics()

    assert_spec_decode_acceptance(
        metrics,
        num_speculative_tokens,
        minimum_per_pos=golden,
        tolerance=0.03,
    )
    cleanup_dist_env_and_memory()
