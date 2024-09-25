from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from moetify.models import EXPERT_NAME_TEMPLATE

class MoetifyLoraModel(LoraModel):
        
    @staticmethod
    def _create_new_module(config, adapter_name, target, **kwargs):
        
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            # kwargs shud be avoided if possible
            new_module = MixtureOfLinear(
                target,
                adapter_name = adapter_name,
                num_experts = config.num_experts,
                gate_dimension = config.gate_dimension,
                r = config.r,
                lora_alpha = config.lora_alpha,
                init_lora_weights = config.init_lora_weights,
                global_router = config.global_router,
                deep_router = config.deep_router,
                store_router_logits = config.router_aux_loss_coef > 0.,
                num_experts_per_tok = config.num_experts_per_tok,
                router_norm = config.router_norm,
                )
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`"
            )

        return new_module
    
    # deal with newly added adapter is not trainable
    # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py#L177-L179
    def _replace_module(self, parent, child_name, new_module, child):
        super()._replace_module(parent, child_name, new_module, child)
        new_module.requires_grad_(True)

class MixtureOfLinear(nn.Module, LoraLayer):

    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "num_experts", "gate_dimension")

    def __init__(
        self,
        base_layer,
        adapter_name:str,
        gate_dimension:int,
        num_experts:int,
        r: int,
        lora_alpha: int,
        num_experts_per_tok: int,
        global_router: bool,
        deep_router: bool,
        router_norm: str,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        use_rslora: bool = False,
        use_dora: bool = False,
        store_router_logits: bool = False,
    ):
        
        assert not (fan_in_fan_out and is_target_conv_1d_layer and use_rslora and use_dora)

        # TODO: support universal gate and output gate
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self.num_experts = num_experts
        self.gate_dimension = gate_dimension
        self.global_router = global_router
        self.deep_router = deep_router
        self.store_router_logits = store_router_logits

        if not self.global_router:
            self.lora_router = nn.ModuleDict({
                adapter_name : nn.Linear(self.in_features, (self.num_experts), bias=False)
            })
            self.adapter_layer_names += ("lora_router",)
        
        self.num_experts_per_tok = num_experts_per_tok

        if router_norm == "sum":
            self.router_norm = lambda x  : x / (x.sum(-1, keepdim=True) + 1e-6)
        elif router_norm == "softmax":
            self.router_norm = nn.Softmax(dim=-1)
        elif router_norm == "sigmoid":
            self.router_norm = nn.Sigmoid()
        else:
            raise NotImplementedError
        
        for expert_name in self.active_expert_adapters(adapter_name):
            self.update_layer(
                adapter_name = expert_name,
                r = r,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                init_lora_weights = init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                )
        # only last adapter will be set/active inside self.update_layer
        # hence only 1 set of adapter with requires_grad=True
        #self.set_adapter([adapter_name] + self.active_expert_adapters(adapter_name))
        self._active_adapter = adapter_name

    def active_expert_adapters(self, adapter_name):
        return [EXPERT_NAME_TEMPLATE.format(adapter_name, idx) for idx in range(self.num_experts)]

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError
    
    def unmerge(self) -> None:
        raise NotImplementedError
    
    def lora_forward(self, adapter_name:str, x:torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        dropout = self.lora_dropout[adapter_name]
        scaling = self.scaling[adapter_name]
        x = x.to(lora_A.weight.dtype)
        return lora_B(lora_A(dropout(x))) * scaling
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        self._check_forward_args(x, *args, **kwargs)

        base_out = self.base_layer(x, *args, **kwargs)      # bs, seq, layer_dim
        bs, seq, out_dim = base_out.size()
        torch_result_dtype = base_out.dtype

        _x = x.view(-1, x.size(-1))

        for active_adapter in self.active_adapters:

            if not self.global_router:
                router = self.lora_router[active_adapter]
                router_out = router(_x.to(router.weight.dtype))
                if self.deep_router:
                    router_out = router_out.view(-1, out_dim, self.num_experts)
                    if self.store_router_logits:
                        self.router_logits = router_out.view(-1, out_dim, self.num_experts)
                elif self.store_router_logits:
                    self.router_logits = router_out.view(-1, self.num_experts)
                routing_weights, selected_experts = torch.topk(router_out, self.num_experts_per_tok, dim=-1)
                routing_weights =  self.router_norm(routing_weights)
            else:
                routing_weights = kwargs["routing_weights"]
                selected_experts = kwargs["selected_experts"]
            
            expert_adapter_names = self.active_expert_adapters(active_adapter)
            #_x = x.view(-1, x.size(-1)).to(self.lora_A[expert_adapter_names[0]].weight.dtype)

            experts_outputs = torch.zeros(bs*seq, out_dim, dtype=_x.dtype, device=_x.device)
            """
            if self.num_experts_per_tok == self.num_experts:
                router_out_norm =  self.router_norm(router_out)
                for idx, expert_adapter in enumerate(expert_adapter_names):
                    expert_weight = router_out_norm[...,[idx]]
                    if self.deep_router:
                        expert_weight = expert_weight.squeeze(-1)
                    experts_outputs += self.lora_forward(expert_adapter, _x) * expert_weight
            else:
            """
            
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # bseq, (dim), topk, exp
            if self.deep_router:
                expert_mask = expert_mask.permute(3,2,0,1) # exp, topk, bseq, out_feat
            else:
                expert_mask = expert_mask.permute(2,1,0) # exp, topk, bseq
            for idx, expert_adapter in enumerate(expert_adapter_names):
                if not self.deep_router:
                    topk_idx, bseq_idx = torch.where(expert_mask[idx])
                    if bseq_idx.shape[0] == 0:
                        continue
                    topk_idx_list = topk_idx.tolist()
                    bseq_idx_list = bseq_idx.tolist()
                    expert_weight = routing_weights[bseq_idx_list, topk_idx_list, None]
                else:
                    topk_idx, bseq_idx, feat_idx = torch.where(expert_mask[idx])
                    if feat_idx.shape[0] == 0:
                        continue
                    topk_idx_list = topk_idx.tolist()
                    bseq_idx_list = bseq_idx.tolist()
                    feat_idx_list = feat_idx.tolist()
                    expert_weight = routing_weights[bseq_idx_list, feat_idx_list, topk_idx_list, None]

                _x_filtered = _x[None, bseq_idx_list].reshape(-1, _x.size(-1))
                _x_filtered = self.lora_forward(expert_adapter, _x_filtered) * expert_weight
                experts_outputs.index_add_(0, bseq_idx, _x_filtered.to(experts_outputs.dtype))
                
            #experts_outputs = torch.stack(experts_outputs, dim=2)  # bs, seq, num_experts, out_features
            #experts_outputs *= experts_weights.unsqueeze(-1)
            #experts_outputs = experts_outputs.sum(2)

            #print(experts_weights[...,-1])
            #print(router_out[...,-1])
            #input()
            layer_out = (base_out + experts_outputs.view(bs, seq, out_dim)).to(torch_result_dtype)

        return layer_out