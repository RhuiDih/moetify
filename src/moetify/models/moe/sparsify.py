from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
try:
    from .sparsify_vllm import fused_experts_linear
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def naive_select_experts(
    router_logits,
    top_k,
    dtype
    ):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(dtype)

    return routing_weights, selected_experts

def vllm_fused_experts_linear_kernel(
    hidden_states,
    router_logits,
    weights,
    top_k,
    renormalize=True,
    use_grouped_topk=False,
    topk_group=None,
    num_expert_group=None
    ):

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group)

    return fused_experts_linear(
        hidden_states=hidden_states,
        w=weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids
    )

def vllm_fused_experts_mlp_kernel(
    hidden_states,
    router_logits,
    up_weights,
    gate_weights,
    down_weights,
    top_k,
    renormalize=True,
    use_grouped_topk=False,
    topk_group=None,
    num_expert_group=None
    ):

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group)

    return fused_experts(
        hidden_states=hidden_states,
        w1=torch.cat((up_weights, gate_weights), dim=1),
        w2=down_weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True
        )

def naive_cuda_experts(
    hidden_states,
    routing_weights,
    selected_experts,
    experts_modules,
    num_experts,
    batch_size,
    sequence_length,
    out_features,
    *args,
    **kwargs,
):

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, out_features),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # selected_experts #bseq, (dim), topk
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts) # bseq, (dim), topk, exp
    #expert_mask = expert_mask.permute(2,1,0) # exp, topk, bseq
    in_features = hidden_states.shape[-1]

    for expert_idx in range(num_experts):

        expert_layer = experts_modules[expert_idx]
        #topk_idx, bseq_idx = torch.where(expert_mask[..., expert_idx])
        bseq_idx, topk_idx = torch.where(expert_mask[..., expert_idx])
        if bseq_idx.shape[0] == 0:
            continue
        expert_weight = routing_weights[bseq_idx, topk_idx, None]

        current_state = hidden_states[None, bseq_idx].reshape(-1, in_features)
        current_hidden_states = expert_layer(
            current_state, *args, **kwargs
        ) * expert_weight

        final_hidden_states.index_add_(0, bseq_idx, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, out_features)

    return final_hidden_states

class MoeForward(nn.Module):

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        selected_experts: torch.Tensor = None,
        *args,
        **kwargs) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.config.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.config.jitter_noise, 1.0 + self.config.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)

        if not self.config.global_router:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.router(hidden_states)
            if self.config.output_router_logits:
                self.router_logits = router_logits
        
        if False: # self.fused_experts and not self.training and VLLM_AVAILABLE:
            w_up_all_experts = torch.stack([x.up_proj.weight for x in self.experts], dim=0)
            w_down_all_experts = torch.stack([x.down_proj.weight for x in self.experts], dim=0)
            w_gate_all_experts = torch.stack([x.gate_proj.weight for x in self.experts], dim=0)
            return vllm_fused_experts_mlp_kernel(
                hidden_states,
                router_logits,
                up_weights=w_up_all_experts,
                gate_weights=w_gate_all_experts,
                down_weights=w_down_all_experts,
                top_k = self.top_k
            ).view(batch_size, sequence_length, hidden_dim)
        elif False and self.fused_experts and not self.training and VLLM_AVAILABLE:
            if getattr(self, "fused_experts_weights", False) is False:
                self.fused_experts_weights = torch.stack([x.weight for x in self.experts], dim=0)
                for exp in self.experts:
                    delattr(exp, "weight")
            return vllm_fused_experts_linear_kernel(
                hidden_states,
                router_logits,
                weights=self.fused_experts_weights,
                top_k = self.top_k
            ).view(batch_size, sequence_length, self.out_features)
        else:
            if routing_weights is None or selected_experts is None:
                routing_weights, selected_experts = naive_select_experts(
                    router_logits, self.top_k, hidden_states.dtype)
            return naive_cuda_experts(
                hidden_states=hidden_states,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                experts_modules=self.experts,
                num_experts=self.num_experts,
                batch_size=batch_size,
                sequence_length=sequence_length,
                out_features=self.out_features,
                *args, **kwargs
            )

class DeepRouterMoeForward(nn.Module):

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.config.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.config.jitter_noise, 1.0 + self.config.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        router_logits = router_logits.view(-1, self.out_features, self.num_experts)

        if self.config.output_router_logits:
            self.router_logits = router_logits
        
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # selected_experts #bseq, (dim), topk
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # bseq, (dim), topk, exp
        expert_mask = expert_mask.permute(3,2,0,1) # exp, topk, bseq, out_feat
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            topk_idx, bseq_idx, feat_idx = torch.where(expert_mask[expert_idx])
            if feat_idx.shape[0] == 0:
                continue
            expert_weight = routing_weights[bseq_idx, feat_idx, topk_idx, None]

            current_state = hidden_states[None, bseq_idx].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, *args, **kwargs
            ) * expert_weight

            final_hidden_states.index_add_(0, bseq_idx, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)

        return final_hidden_states


class AlwaysOnMoeOnForward(nn.Module):

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.config.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.config.jitter_noise, 1.0 + self.config.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        if self.config.output_router_logits:
            self.router_logits = router_logits
        
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # selected_experts #bseq, (dim), topk
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts-1) # bseq, (dim), topk, exp
        expert_mask = expert_mask.permute(2,1,0) # exp, topk, bseq
        
        always_on_out = self.experts[0](hidden_states, *args, **kwargs)
        for expert_idx in range(1, self.num_experts):

            expert_layer = self.experts[expert_idx]
            topk_idx, bseq_idx = torch.where(expert_mask[expert_idx-1])
            if bseq_idx.shape[0] == 0:
                continue
            expert_weight = routing_weights[bseq_idx, topk_idx, None]

            current_state = hidden_states[None, bseq_idx].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, *args, **kwargs
            ) * expert_weight

            final_hidden_states.index_add_(0, bseq_idx, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states += always_on_out
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)

        return final_hidden_states


class GatelessMoeForward(nn.Module):

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        expert_out = self.experts[0](hidden_states, *args, **kwargs) * self.router[0]
        for idx in range(1,self.num_experts):
            expert_out += self.experts[idx](hidden_states, *args, **kwargs) * self.router[idx]
        return expert_out


def make_moe_modules(
    moe_class_partial,
    config_class,
    deep_router=False,
    gateless=False,
    always_on=False,
    fused_experts=False
    ):

    if gateless:
        abstract = GatelessMoeForward
    elif deep_router:
        abstract = DeepRouterMoeForward
    elif always_on:
        abstract = AlwaysOnMoeOnForward
    else:
        abstract = MoeForward

    class MoeModules(abstract):

        def __init__(
            self,
            config: config_class,
            in_features: int,
            out_features: int
        ):

            super().__init__()
            self.config = config
            self.top_k = config.num_experts_per_tok
            self.num_experts = config.num_local_experts

            assert sum([self.global_router, self.deep_router, self.gateless, self.always_on]) <= 1, "Only 1 or None!"
            
            self.experts = nn.ModuleList([moe_class_partial() for _ in range(self.num_experts)])
            self.out_features = out_features

            if not self.global_router:
                if not self.gateless:
                    if self.deep_router:
                        inter_dim = in_features // 4
                        self.router = nn.Sequential(
                            nn.Linear(in_features, inter_dim, bias=False),
                            nn.Hardswish(inplace=True),
                            nn.Linear(inter_dim, out_features*self.num_experts, bias=False)
                        )
                    else:
                        router_out = self.num_experts if not self.always_on else self.num_experts-1
                        self.router = nn.Linear(in_features, router_out, bias=False)
                else:
                    self.register_buffer("router", torch.ones(self.num_experts)/self.num_experts)

            # storing purpose, not to disrupt original forward
            self.router_logits = None

            self.fused_experts = fused_experts

        @property
        def deep_router(self):
            return self.config.deep_router
        
        @property
        def global_router(self):
            return self.config.global_router
        
        @property
        def gateless(self):
            return self.config.gateless
        
        @property
        def always_on(self):
            return self.config.always_on

    return MoeModules


def make_moa_modules(attention_class, config_class):

    class MixtureOfAttention(attention_class):

        def __init__(self, config: config_class, layer_idx:int):
            
            super().__init__(config, layer_idx)

            # replacing
            if self.config.moe_query:
                LINEAR_CLS = make_moe_modules(partial(
                        nn.Linear,
                        in_features=self.q_proj.in_features,
                        out_features=self.q_proj.out_features,
                        bias=self.q_proj.bias
                    ),
                    config_class,
                    deep_router=config.deep_router,
                    gateless=config.gateless)
                out_features = self.q_proj.out_features
                in_features = self.q_proj.in_features
                delattr(self, "q_proj")
                self.q_proj = LINEAR_CLS(config=config, in_features=in_features, out_features=out_features)

            if self.config.moe_key:
                LINEAR_CLS = make_moe_modules(partial(
                        nn.Linear,
                        in_features=self.k_proj.in_features,
                        out_features=self.k_proj.out_features,
                        bias=self.k_proj.bias
                    ),
                    config_class,
                    deep_router=config.deep_router,
                    gateless=config.gateless)
                out_features = self.k_proj.out_features
                in_features = self.k_proj.in_features
                delattr(self, "k_proj")
                self.k_proj = LINEAR_CLS(config=config, in_features=in_features, out_features=out_features)
            if self.config.moe_value:
                LINEAR_CLS = make_moe_modules(partial(
                    nn.Linear,
                        in_features=self.v_proj.in_features,
                        out_features=self.v_proj.out_features,
                        bias=self.v_proj.bias
                    ),
                    config_class,
                    deep_router=config.deep_router,
                    gateless=config.gateless)
                out_features = self.v_proj.out_features
                in_features = self.v_proj.in_features
                delattr(self, "v_proj")
                self.v_proj = LINEAR_CLS(config=config, in_features=in_features, out_features=out_features)
        
        @property
        def router_logits(self):
            router_logits = ()
            if self.moe_query:
                router_logits += (self.q_proj.router_logits,)
            if self.moe_key:
                router_logits += (self.k_proj.router_logits,)
            if self.moe_value:
                router_logits += (self.v_proj.router_logits,)
            return router_logits

    return MixtureOfAttention


