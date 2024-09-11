import transformers
from transformers import LlamaConfig
from transformers.utils import logging

from packaging import version

logger = logging.get_logger(__name__)

class MoeLlamaConfig(LlamaConfig):

    model_type = "moellama"

    def __init__(
        self,
        moe_mlp=True,
        moe_mlp_fg=False,
        moe_query=False,
        moe_key=False,
        moe_value=False,
        moe_layer_idx=[],
        num_local_experts=4,
        num_experts_per_tok=2,
        output_router_logits=False,
        router_aux_loss_coef=0.1,
        jitter_noise=0.,
        always_on=False,
        deep_router=False,
        gateless=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        #assert version.parse(transformers.__version__) >= version.parse('4.40.0'), \
        #    "MoeLLaMa is implemented for transformers>=4.40.0!"
        self.moe_mlp = moe_mlp
        self.moe_mlp_fg = moe_mlp_fg
        self.moe_query = moe_query
        self.moe_key = moe_key
        self.moe_value = moe_value
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.jitter_noise = 0.
        self.always_on = always_on
        self.deep_router = deep_router
        self.gateless = gateless
        if not moe_layer_idx:
            moe_layer_idx = list(range(self.num_hidden_layers))
        self.moe_layer_idx = moe_layer_idx
        if self.gateless:
            self.output_router_logits = False