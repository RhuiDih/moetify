from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


class MoeForward(nn.Module):

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
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # bseq, (dim), topk, exp
        expert_mask = expert_mask.permute(2,1,0) # exp, topk, bseq
        
        for expert_idx in range(self.num_experts):

            expert_layer = self.experts[expert_idx]
            topk_idx, bseq_idx = torch.where(expert_mask[expert_idx])
            if bseq_idx.shape[0] == 0:
                continue
            expert_weight = routing_weights[bseq_idx, topk_idx, None]

            current_state = hidden_states[None, bseq_idx].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, *args, **kwargs
            ) * expert_weight

            final_hidden_states.index_add_(0, bseq_idx, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)

        return final_hidden_states

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
    
def make_moe_modules(moe_class_partial, config_class, deep_router=False, gateless=False, always_on=False):

    if gateless:
        abstract = GatelessMoeForward
    elif deep_router:
        abstract = DeepRouterMoeForward
    elif always_on:
        abstract = AlwaysOnMoeOnForward
    else:
        abstract = MoeForward

    class MoeModules(abstract):

        def __init__(self, config: config_class, in_features: int, out_features: int):

            super().__init__()
            self.config = config
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.deep_router = config.deep_router
            self.gateless = config.gateless
            self.always_on = config.always_on

            assert sum([self.deep_router, self.gateless, self.always_on]) <= 1, "Only 1 or None!"
            
            self.experts = nn.ModuleList([moe_class_partial() for _ in range(self.num_experts)])
            self.out_features = out_features

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

    return MoeModules

def make_moa_modules(attention_class, config_class):

    class MixtureOfAttention(attention_class):

        def __init__(self, config: config_class, layer_idx:int):
            
            super().__init__(config, layer_idx)
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.always_on = config.always_on
            self.moe_query = config.moe_query
            self.moe_key = config.moe_key
            self.moe_value = config.moe_value

            # replacing
            if self.moe_query:
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
            if self.moe_key:
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
            if self.moe_value:
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
