# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import os
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.utils import communicator_switch


class EagleAclGraphManager(SpeculatorCudaGraphManager):
    """AclGraphManager for Eagle speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )

        # set speculator attribute, so we can access attributes speculator
        # when call `run_fullgraph` method in CudaGraphManager,
        # then we don't need to # copy `propose` method in `AscendEagleSpeculator` class.
        self.speculator: Any = None
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # vllm-ascend need to update draft graph params of attention backend.
        # so we need to set draft graph params before capture full graph.
        # `prefill` graph and `decodes` graph are different, `decode_query_len` can be used to distinguish them
        self.is_draft_model_prefill = decode_query_len > 1
        if super().needs_capture():
            if self.is_draft_model_prefill:
                set_draft_graph_prefill_params(self.capture_sizes)
            else:
                set_draft_graph_params(self.capture_sizes)

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for Eagle."""

        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
            if self.is_draft_model_prefill:
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc=progress_bar_desc,
                )
                return

            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                if vllm_version_is("0.26.0"):
                    prepare_inputs_to_capture(
                        num_reqs,
                        num_tokens,
                        model_state,
                        input_buffers,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                        skip_attn=(desc.cg_mode != CUDAGraphMode.PIECEWISE),
                    )
                else:
                    prepare_inputs_to_capture(
                        num_reqs,
                        num_tokens,
                        model_state,
                        input_buffers,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                        full_cudagraph=(desc.cg_mode == CUDAGraphMode.FULL),
                    )
                seq_lens_cpu_upper_bound = input_buffers.seq_lens_cpu[:num_reqs]
                if vllm_version_is("0.26.0"):
                    return lambda cg_mode: forward_fn(
                        num_reqs,
                        cg_mode == CUDAGraphMode.PIECEWISE,
                        BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                        num_tokens_across_dp,
                    )
                else:
                    return lambda cg_mode: forward_fn(
                        num_reqs,
                        cg_mode == CUDAGraphMode.PIECEWISE,
                        BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                        num_tokens_across_dp,
                        seq_lens_cpu_upper_bound,
                    )

            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc=progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Update runtime attention parameters and replay the Eagle graph."""
        num_tokens = desc.num_tokens
        if self.is_draft_model_prefill:
            logger.info_once("PrefillEagleAclGraphManager: draft prefill run_fullgraph with num_tokens=%s", num_tokens)
        else:
            logger.info_once("DecodeEagleAclGraphManager: draft run_fullgraph with num_tokens=%s", num_tokens)

        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(desc.num_reqs, self.is_draft_model_prefill)
        debug_graph = os.getenv("VLLM_ASCEND_EAGLE3_DEBUG", "0") == "1"

        if debug_graph:
            metadata_summary = []
            for step, per_step_metadata in enumerate(draft_attn_metadatas or []):
                if not per_step_metadata:
                    continue
                key, metadata = next(iter(per_step_metadata.items()))
                metadata_summary.append(
                    {
                        "step": step,
                        "key": key,
                        "seq_lens": list(metadata.seq_lens_list[: min(desc.num_reqs or 0, 4)]),
                        "actual_q": list(metadata.actual_seq_lengths_q[: min(desc.num_reqs or 0, 4)]),
                    }
                )
            graph_params = self._get_debug_graph_params()
            captured_param_count = (
                len(graph_params.attn_params.get(num_tokens, [])) if graph_params is not None else -1
            )
            # logger.warning(
            #     "[eagle3/dfx] replay: prefill=%s, num_reqs=%s, "
            #     "num_tokens=%s, draft_steps=%s, captured_attn_params=%s, metadata=%s",
            #     self.is_draft_model_prefill,
            #     desc.num_reqs,
            #     num_tokens,
            #     len(draft_attn_metadatas or []),
            #     captured_param_count,
            #     metadata_summary,
            # )

        def update_runtime_graph_params() -> None:
            num_tokens_across_dp = torch.full(
                [self.speculator.dp_size],
                num_tokens,
                dtype=torch.int32,
                device="cpu",
            )
            with set_forward_context(
                self.speculator.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,
                slot_mapping=None,
            ):
                _EXTRA_CTX.is_draft_model = True
                _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill
                update_full_graph_params(
                    # FIXME(Ronald1995): support hybrid attn backend
                    list(self.speculator.attn_backends.values())[0],
                    self.update_stream,
                    get_forward_context(),
                    num_tokens,
                    self.vllm_config,
                    self.speculator.speculative_config,
                    draft_attn_metadatas=draft_attn_metadatas,
                )

        update_runtime_graph_params()
        torch.npu.current_stream().wait_stream(self.update_stream)
        return super().run_fullgraph(desc)

    def _get_debug_graph_params(self):
        if self.is_draft_model_prefill:
            from vllm_ascend.compilation.acl_graph import get_draft_graph_prefill_params

            return get_draft_graph_prefill_params()

        from vllm_ascend.compilation.acl_graph import get_draft_graph_params

        return get_draft_graph_params()
