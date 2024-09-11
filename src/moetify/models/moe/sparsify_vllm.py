# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Mixtral model."""

from typing import Iterable, Dict, Any, List, Optional, Tuple
import functools
import json
import os

import torch
from torch import nn

import triton
import triton.language as tl

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear
)

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import (
    #FusedMoeWeightScaleSupported,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_config_dtype_str,
    invoke_fused_moe_kernel,
    try_get_optimal_moe_config,
    moe_align_block_size
)

logger = init_logger(__name__)

from enum import Enum
# unable to import in old vllm
class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"

class FineGrainedMixtureOfMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = ""
    ):
        
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate_gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate_gate"
        )

        self.up_gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.up_gate"
        )

        self.down_gate = ReplicatedLinear(
            intermediate_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.down_gate"
        )

        expert_kwargs = dict(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            tp_size=tp_size,
        )

        self.experts = FusedFineGranedMLPMoe(
            **expert_kwargs,
            prefix=f"{prefix}.experts"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits_up, _ = self.up_gate(hidden_states)
        router_logits_gate, _ = self.gate_gate(hidden_states)
        router_logits_down, _ = self.down_gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits_up,
            router_logits_gate,
            router_logits_down
        )
        return final_hidden_states.view(orig_shape)


class FusedFineGranedMLPMoe(torch.nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedFineGranedMLPMoeMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)


    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int
    )-> List[Tuple[str, str, int, str]]:

        mapping = {
            "up_proj": "experts.weight_up",
            "gate_proj": "experts.weight_gate",
            "down_proj": "experts.weight_down"
        }
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                mapping[weight_name],
                f"{weight_name}.experts.{expert_id}.weight",
                expert_id,
                shard_id
            ) for expert_id in range(num_experts) for shard_id, weight_name in [
                ("up", ckpt_up_proj_name),
                ("gate", ckpt_gate_proj_name),
                ("down", ckpt_down_proj_name),
            ]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits_up: torch.Tensor,
        router_logits_gate: torch.Tensor,
        router_logits_down: torch.Tensor
    ):

        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits_up=router_logits_up,
            router_logits_gate=router_logits_gate,
            router_logits_down=router_logits_down,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group)

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int
    ):
        raise NotImplementedError
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(
        self,
        expert_data:torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def load_weight(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int
    ):
        shard_size = expert_data.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim,
            shard_size * tp_rank,
            shard_size
            )
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int
    ):
        raise NotImplementedError

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int
    ) -> None:
        
        raise NotImplementedError

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int
    ):

        raise NotImplementedError

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,\
        weight_name: str,
        shard_id: str,
        expert_id: int
    ) -> None:

        if shard_id not in ("up", "gate", "down"):
            raise ValueError(f"shard_id must be ['up','gate','down'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"up": 0, "down": 1, "gate": 0}

        expert_data = param.data[expert_id]
        tp_rank = get_tensor_model_parallel_rank()

        # is_transposed: whether or not the parameter is transposed on disk
        # If transposed, the loaded weight will be transposed and the dim
        # to shard the loaded weight will be flipped.
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            loaded_weight = loaded_weight.t().contiguous()
            shard_dim = ~shard_dim

        # Case weight_scales
        if "weight_scale" in weight_name:
            # load the weight scaling based on the quantization scheme
            # supported weight scales can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank
                )
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                self.load_weight(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        if "weight_shape" in weight_name:
            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=expert_id
            )
            return

        # Case input scale
        if "input_scale" in weight_name:
            # Note: input_scale loading is only supported for fp8
            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self.load_weight(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank)
            return



class UnquantizedFusedFineGranedMLPMoeMethod(FusedMoEMethodBase, CustomOp):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):

        # Fused gate_up_proj (column parallel)
        weight_up = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size,
                dtype=params_dtype
                ),
            requires_grad=False
        )
        layer.register_parameter("weight_up", weight_up)
        set_weight_attrs(weight_up, extra_weight_attrs)

        weight_gate = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size,
                dtype=params_dtype
                ),
            requires_grad=False
        )
        layer.register_parameter("weight_gate", weight_gate)
        set_weight_attrs(weight_gate, extra_weight_attrs)

        # down_proj (row parallel)
        weight_down = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=params_dtype
                ),
            requires_grad=False
        )
        layer.register_parameter("weight_down", weight_down)
        set_weight_attrs(weight_down, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits_up: torch.Tensor,
        router_logits_gate: torch.Tensor,
        router_logits_down: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None
    ) -> torch.Tensor:

        return self.forward(
            x=x,
            layer=layer,
            router_logits_up=router_logits_up,
            router_logits_gate=router_logits_gate,
            router_logits_down=router_logits_down,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits_up: torch.Tensor,
        router_logits_gate: torch.Tensor,
        router_logits_down: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None
    ) -> torch.Tensor:

        topk_weights_up, topk_ids_up = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits_up,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group
        )

        topk_weights_gate, topk_ids_gate = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits_up,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group
        )

        topk_weights_down, topk_ids_down = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits_up,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group
        )

        return apply_fg_mlp_moe(
            hidden_states=x,
            w_up=layer.weight_up,
            w_gate=layer.weight_gate,
            w_down=layer.weight_down,
            topk_weights_up=topk_weights_up,
            topk_ids_up=topk_ids_up,
            topk_weights_gate=topk_weights_gate,
            topk_ids_gate=topk_ids_gate,
            topk_weights_down=topk_weights_down,
            topk_ids_down=topk_ids_down,
            inplace=True
        )

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError(
            "The CPU backend currently does not support MoE.")

    def forward_tpu(self, *args, **kwargs):
        raise NotImplementedError(
            "The CPU backend currently does not support MoE.")



def apply_fg_mlp_moe(
    hidden_states: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights_up: torch.Tensor,
    topk_ids_up: torch.Tensor,
    topk_weights_gate: torch.Tensor,
    topk_ids_gate: torch.Tensor,
    topk_weights_down: torch.Tensor,
    topk_ids_down: torch.Tensor,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w_gate_scale: Optional[torch.Tensor] = None,
    w_up_scale: Optional[torch.Tensor] = None,
    w_down_scale: Optional[torch.Tensor] = None,
    a_gate_scale: Optional[torch.Tensor] = None,
    a_up_scale: Optional[torch.Tensor] = None,
    a_down_scale: Optional[torch.Tensor] = None
    ):
    
    # Check constraints.
    #assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    #assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    #assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    #assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    #assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape
    num_experts, _intermediate_size, _hidden_size = w_up.shape # could be TP!
    _intermediate_size_x2 = _intermediate_size * 2

    top_k = topk_ids_up.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        dtype=hidden_states.dtype
    )

    """
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        top_k,
        config_dtype,
        override_config=override_config,
    )"""
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w_gate.shape,
        w_down.shape,
        top_k,
        config_dtype,
        override_config=override_config,
    )

    config = get_config_func(M)

    intermediate_cache1 = torch.empty(
        (M, top_k, _intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )

    intermediate_cache2 = torch.empty(
        (M, top_k, _intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )

    intermediate_cache3 = torch.empty(
        (M * top_k, _intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )

    intermediate_cache4 = torch.empty(
        (M, top_k, _hidden_size), # NOTE: maybe w_down.shape[1] safer ?
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids_up[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights_up[begin_chunk_idx:end_chunk_idx]
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], num_experts)
        )
        invoke_fused_moe_kernel(
            curr_hidden_states,
            w_up,
            intermediate_cache1,
            a_up_scale,
            w_up_scale,
            topk_weights=curr_topk_weights,
            topk_ids=curr_topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=top_k,
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16
        )

        curr_topk_ids = topk_ids_gate[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights_gate[begin_chunk_idx:end_chunk_idx]
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], num_experts)
        )
        invoke_fused_moe_kernel(
            curr_hidden_states,
            w_gate,
            intermediate_cache2,
            a_gate_scale,
            w_gate_scale,
            topk_weights=curr_topk_weights,
            topk_ids=curr_topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=top_k,
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16
        )

        ops.silu_and_mul(
            intermediate_cache3,
            torch.cat((intermediate_cache2, intermediate_cache1), dim=2)
            #intermediate_cache1.view(-1, _intermediate_size_x2)
        )

        invoke_fused_moe_kernel(
            intermediate_cache3,
            w_down,
            intermediate_cache4,
            a_down_scale,
            w_down_scale,
            topk_weights=curr_topk_weights,
            topk_ids=curr_topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,    #NOTE: not sure why
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16
        )

        torch.sum(
            intermediate_cache4.view(*intermediate_cache4.shape),
            dim=1,
            out=out_hidden_states[begin_chunk_idx:end_chunk_idx]
        )

    return out_hidden_states

