from dataclasses import dataclass, field

from peft import LoraConfig, IA3Config
from peft import get_peft_config
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import \
    PEFT_TYPE_TO_CONFIG_MAPPING, \
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING, \
    PEFT_TYPE_TO_TUNER_MAPPING

from .models.base_model import moetify_METHODS, moetify_LORA, moetify_IA3
from .models.base_model import MoetifyModelForCausalLM
from .models.lora_model import MoetifyLoraModel
from .models.ia3_model import MoetifyIA3Model

from .models.moe import *

@dataclass
class MoetifyLoraConfig(LoraConfig):
    num_experts: int = field(default=8)
    gate_dimension: int = field(default=64)
    global_router: bool = field(default=False)
    deep_router: bool = field(default=False)
    router_aux_loss_coef: float = field(default=0.00)
    router_norm: str = field(default="sigmoid")
    num_experts_per_tok: int = field(default=0)
    
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = moetify_LORA
        if not self.num_experts_per_tok:
            self.num_experts_per_tok = self.num_experts
        assert not(self.deep_router and self.router_aux_loss_coef>0)

@dataclass
class MoetifyIA3Config(IA3Config):
    num_experts: int = field(default=8)
    gate_dimension: int = field(default=64)
    deep_router: bool = field(default=False)
    router_aux_loss_coef: float = field(default=0.00)
    router_norm: str = field(default="sum")
    num_experts_per_tok: int = field(default=0)
    
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = moetify_IA3
        if not self.num_experts_per_tok:
            self.num_experts_per_tok = self.num_experts
        assert not(self.deep_router and self.router_aux_loss_coef>0)

PEFT_TYPE_TO_MODEL_MAPPING.update({
    moetify_LORA: MoetifyLoraModel,
    moetify_IA3: MoetifyIA3Model
})

PEFT_TYPE_TO_CONFIG_MAPPING.update({
    moetify_LORA: MoetifyLoraConfig,
    moetify_IA3: MoetifyIA3Config
})

# !!! this is used by PretrainedModel.load_adapter, shud not be used
# use MoetifyModelForCausalLM.load_adapter instead
"""
PEFT_TYPE_TO_TUNER_MAPPING.update({
    moetify_LORA: MoetifyLoraModel,
    moetify_IA3: MoetifyIA3Model
})
"""

MODEL_TYPE_TO_PEFT_MODEL_MAPPING.update({
    "moetify_CAUSAL_LM": MoetifyModelForCausalLM
})



# original implementation doesn't throw error when unsupported `task_type` is used
def get_peft_model(model, peft_config, *args, **kwargs):
    from peft import get_peft_model
    assert peft_config.task_type in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys(), MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys()
    return get_peft_model(model, peft_config, *args, **kwargs)



# vllm
from transformers.utils.import_utils import _is_package_available
if _is_package_available("vllm"):
    from vllm.model_executor.models import ModelRegistry
    from moetify.models.moe.llama.vllm_moellama import VLLMMoeLlamaForCausalLM
    ModelRegistry.register_model("moellama", VLLMMoeLlamaForCausalLM)
    ModelRegistry.register_model("MoeLlamaForCausalLM", VLLMMoeLlamaForCausalLM)