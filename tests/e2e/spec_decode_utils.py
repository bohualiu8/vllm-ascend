# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, Sequence

from vllm.v1.metrics.reader import Counter, Vector


def assert_spec_decode_acceptance(
    metrics: list[Any],
    num_speculative_tokens: int,
    *,
    minimum_per_pos: Sequence[float] | None = None,
    tolerance: float = 0.0,
) -> list[float]:
    """Validate speculative-decoding metrics and return acceptance by position."""
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    found_drafts = False
    found_accepted = False

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            found_drafts = True
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            found_accepted = True
            assert len(metric.values) == num_speculative_tokens
            for pos, value in enumerate(metric.values):
                num_accepted_tokens_per_pos[pos] += value

    assert found_drafts, "Missing vllm:spec_decode_num_drafts metric"
    assert found_accepted, "Missing vllm:spec_decode_num_accepted_tokens_per_pos metric"
    assert num_drafts > 0, "Speculative decoding did not generate any draft tokens"

    acceptance_per_pos = [accepted / num_drafts for accepted in num_accepted_tokens_per_pos]
    assert any(acceptance_per_pos), "Speculative decoding did not accept any draft tokens"
    assert all(0 <= acceptance <= 1 for acceptance in acceptance_per_pos)

    if minimum_per_pos is not None:
        assert len(minimum_per_pos) == num_speculative_tokens
        assert all(
            actual >= minimum - tolerance
            for actual, minimum in zip(acceptance_per_pos, minimum_per_pos, strict=True)
        ), (
            f"acceptance_per_pos {acceptance_per_pos} is below minimum "
            f"{list(minimum_per_pos)} with tolerance {tolerance}"
        )

    print(f"Speculative decoding acceptance_per_pos={acceptance_per_pos} (num_drafts={num_drafts})")
    return acceptance_per_pos
