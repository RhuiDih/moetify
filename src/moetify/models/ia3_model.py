import torch
import torch.nn as nn

from peft.tuners.ia3.model import IA3Model
from peft.tuners.ia3.layer import IA3Layer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists

from moetify.models import EXPERT_NAME_TEMPLATE

class MoetifyIA3Model(IA3Model):
        
    def __init__(self, model, peft_config, adapter_name):
        # NOTE: custom adapter name is not supported! PEFT implementation is very confusing.
        if isinstance(peft_config, dict):
            _config = peft_config.pop(adapter_name)
        else:
            _config = peft_config
        _adapter_rename = EXPERT_NAME_TEMPLATE.format(str(0))
        super().__init__(
            model, {_adapter_rename:_config}, adapter_name=_adapter_rename
            )
        self.active_adapter = [EXPERT_NAME_TEMPLATE.format(str(x)) for x in range(_config.num_experts)]

    @staticmethod
    def _create_new_module(ia3_config, adapter_name, target, **kwargs):
        
        is_feedforward = kwargs['is_feedforward']

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            # kwargs shud be avoided if possible
            new_module = MixtureOfVector(
                target,
                num_experts = ia3_config.num_experts,
                gate_dimension = ia3_config.gate_dimension,
                is_feedforward= is_feedforward,
                init_ia3_weights = ia3_config.init_ia3_weights,
                )
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`"
            )

        return new_module
    
class MixtureOfVector(nn.Module, IA3Layer):

    adapter_layer_names = ("ia3_l", "ia3_gate_in", "ia3_gate_out")
    other_param_names = ("num_experts", "gate_dimension")

    def __init__(
        self,
        base_layer,
        gate_dimension:int,
        num_experts:int,
        is_feedforward: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_ia3_weights: bool = True,
    ):

        # TODO: support universal gate and output gate
        super().__init__()
        IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)

        self.num_experts = num_experts
        self.gate_dimension = gate_dimension
        self.ia3_gate_in = nn.ModuleDict({
            EXPERT_NAME_TEMPLATE.format(str(0)):nn.Linear(self.in_features, self.gate_dimension)
        })
        self.ia3_gate_out = nn.ModuleDict({
            EXPERT_NAME_TEMPLATE.format(str(0)):nn.Linear(self.gate_dimension, self.num_experts)
        })
        
        self._active_adapter = []    # used in property method of `peft.tuners.tuners_utils.BaseTunerLayer.active_adapters`
        for idx in range(self.num_experts):
            expert_name = EXPERT_NAME_TEMPLATE.format(str(idx))
            self.update_layer(
                adapter_name = expert_name,
                init_ia3_weights = init_ia3_weights
                )
            self._active_adapter.append(expert_name)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        previous_dtype = x.dtype

        # TODO: implement merging
        if self.disable_adapters:
            raise NotImplementedError
        elif self.merged:
            raise NotImplementedError

        gate_in = self.ia3_gate_in[self.active_adapters[0]]
        gate_out = self.ia3_gate_out[self.active_adapters[0]]
        _x = x.to(gate_in.weight.dtype)
        experts_weights = nn.functional.softmax(gate_out(gate_in(_x)), dim=-1)  # bs, seq, num_experts

        experts_vectors = [self.ia3_l[active_adapter].flatten() for active_adapter in self.active_adapters]
        experts_vectors = torch.stack(experts_vectors, dim=0)   # num_experts, in/out features
        experts_vectors = experts_vectors.unsqueeze(0).unsqueeze(0) * experts_weights.unsqueeze(-1)
        experts_vectors = experts_vectors.sum(2)

        dtype = experts_vectors.dtype
        if self.is_feedforward:
            # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
            # e.g. bf16 vs fp32. Is that okay?
            interm = (x.to(dtype) * experts_vectors).to(self.get_base_layer().weight.dtype)
            layer_out = self.base_layer(interm, *args, **kwargs)
        else:
            base_out = self.base_layer(x, *args, **kwargs)
            layer_out = base_out.to(dtype) * experts_vectors

        layer_out = layer_out.to(previous_dtype)

        return layer_out