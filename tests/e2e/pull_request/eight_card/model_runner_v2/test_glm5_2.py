# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.spec_decode_utils import assert_spec_decode_acceptance

MODEL = "Eco-Tech/GLM-5.2-w4a8"
DSPARK_MODEL = "RedHatAI/GLM-5.2-speculator.dspark"
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _run_glm5_2_spec_decode(method: str, num_speculative_tokens: int, model: str | None = None) -> None:
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
    speculative_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if model is not None:
        speculative_config["model"] = model
        speculative_config["enforce_eager"] = True

    with VllmRunner(
        MODEL,
        quantization="ascend",
        tensor_parallel_size=8,
        max_model_len=8192,
        max_num_seqs=16,
        enable_expert_parallel=True,
        disable_log_stats=False,
        enforce_eager=True,
        speculative_config=speculative_config,
    ) as runner:
        outputs = runner.model.generate(PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()

    assert len(outputs) == len(PROMPTS)
    assert all(output.outputs[0].token_ids for output in outputs)
    assert_spec_decode_acceptance(metrics, num_speculative_tokens)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_mtp_eager() -> None:
    _run_glm5_2_spec_decode("mtp", num_speculative_tokens=3)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_model(DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_dspark_eager() -> None:
    _run_glm5_2_spec_decode("dspark", num_speculative_tokens=7, model=DSPARK_MODEL)
