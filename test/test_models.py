import unittest
import torch

from transformers import LlamaConfig, LlamaForCausalLM
from moetify import MoeLlamaConfig, MoeLlamaForCausalLM
from peft import PeftModel
from moetify import MoetifyLoraConfig, get_peft_model

class TestMoeLlama(unittest.TestCase):

    def get_seq_len(self):
        return 32

    def get_dummy_input_tensor(self):
        return {
            "input_ids": (torch.randn(2,self.get_seq_len()).abs()*1024).long()
        }

    def get_small_lora_config_kwargs(self):
        return dict(
            r=8,
            task_type="moetify_CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    def get_small_arch_kwargs(self):
        return dict(
            num_local_experts=4,
            num_hidden_layers=4,
            hidden_size=256,
            intermediate_size=512
        )

    def get_moetify_model(self, config):
        config._attn_implementation = "sdpa"
        return MoeLlamaForCausalLM(config)

    def get_plain_model(self, config):
        config._attn_implementation = "sdpa"
        return LlamaForCausalLM(config)

    @torch.no_grad()
    def test_mlp_moe(self):
        config = MoeLlamaConfig(
            moe_mlp=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_mlp_moe_with_global_router(self):
        config = MoeLlamaConfig(
            moe_mlp=True,
            global_router=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_all_moe(self):
        config = MoeLlamaConfig(
            moe_mlp=True,
            moe_key=True,
            moe_query=True,
            moe_value=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_all_moe_global_router(self):
        config = MoeLlamaConfig(
            moe_mlp=True,
            moe_key=True,
            moe_query=True,
            moe_value=True,
            global_router=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_fgmlp_moe(self):
        config = MoeLlamaConfig(
            moe_mlp=False,
            moe_mlp_fg=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_fgmlp_moe_global_router(self):
        config = MoeLlamaConfig(
            moe_mlp=False,
            moe_mlp_fg=True,
            global_router=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_plain_model_with_moe_lora(self):
        config = LlamaConfig(
            **self.get_small_arch_kwargs()
        )
        model = self.get_plain_model(config)
        lora_config = MoetifyLoraConfig(
            **self.get_small_lora_config_kwargs()
        )
        model = get_peft_model(model, lora_config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_empty_moe_with_moe_lora(self):
        config = MoeLlamaConfig(
            moe_mlp=False,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        lora_config = MoetifyLoraConfig(
            **self.get_small_lora_config_kwargs()
        )
        model = get_peft_model(model, lora_config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)

    @torch.no_grad()
    def test_empty_moe_global_router_with_moe_lora(self):
        config = MoeLlamaConfig(
            moe_mlp=False,
            global_router=True,
            **self.get_small_arch_kwargs()
        )
        model = self.get_moetify_model(config)
        lora_config = MoetifyLoraConfig(
            global_router=True,
            **self.get_small_lora_config_kwargs()
        )
        model = get_peft_model(model, lora_config)
        inp = self.get_dummy_input_tensor()
        out = model(**inp)
        out = model.generate(**inp, max_length=self.get_seq_len()+2)