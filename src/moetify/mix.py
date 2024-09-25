import os, logging
from typing import List
from tqdm import tqdm
import gc

import torch
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM
)
from transformers.modeling_utils import no_init_weights
from datasets import load_from_disk

from moetify import (
    MoeLlamaConfig,
    MoeLlamaForCausalLM,
    MoeMistralForCausalLM,
    MoeMistralConfig
)

MODELS_MOE_CLASS = {
    LlamaForCausalLM: (MoeLlamaForCausalLM, MoeLlamaConfig),
    MistralForCausalLM: (MoeMistralForCausalLM, MoeMistralConfig),
}

ROUTERS_NAME = ".router"
EXPERTS_NAME = ".experts."


@torch.no_grad()
def mix(
    base_model:str,
    ingredients:List[str],
    modules_to_mix:List[str],
    output_dir:str,
    positive_tokens:List[str]=[],
    num_samples:int=1000,
    num_experts_per_tok:int=2,
    moe_layer_idx:list=[],
    router_base_init:bool=False,
    gateless:bool=False,
    global_router:bool=False,
    deep_router:bool=False,
    always_on:bool=False,
    merge_base:bool=False,
    mlp_fg:bool=False
):
    
    set_seed(1399)

    logging.info("loading base model...")
    config_base = AutoConfig.from_pretrained(base_model)
    model_base = AutoModelForCausalLM.from_pretrained(base_model)
    model_base_type = type(model_base)
    sd_base = model_base.state_dict()
    MOE_MODEL_CLS, MOE_CFG_CLS = MODELS_MOE_CLASS[model_base_type]
    MOE_CFG_CLS.register_for_auto_class()
    MOE_MODEL_CLS.register_for_auto_class("AutoModelForCausalLM")
    
    # SUPPORT CHECK
    assert num_experts_per_tok <= len(ingredients)

    assert len(modules_to_mix)>0, \
        "Modules to mix must have at least 'mlp'!"
    
    assert any([isinstance(model_base, supported_model) for supported_model in MODELS_MOE_CLASS.keys()]), \
        "Model not supported! Only supports {}!".format(MODELS_MOE_CLASS.keys())
    
    if positive_tokens:
        assert len(positive_tokens) == len(ingredients)

    if moe_layer_idx:
        moe_layer_idx = [int(x) for x in moe_layer_idx]
    # /SUPPORT CHECK
    
    logging.info("creating base model...")
    config_base.torch_dtype = torch.float16
    config = MOE_CFG_CLS(
        num_local_experts= len(ingredients),
        moe_mlp ="mlp" in modules_to_mix and not mlp_fg,
        moe_mlp_fg = "mlp" in modules_to_mix and mlp_fg, 
        moe_query="q_proj" in modules_to_mix,
        moe_key = "k_proj" in modules_to_mix,
        moe_value="v_proj" in modules_to_mix,
        num_experts_per_tok=num_experts_per_tok,
        moe_layer_idx=moe_layer_idx,
        global_router=global_router,
        deep_router=deep_router,
        gateless=gateless,
        always_on=always_on,
        **config_base.to_dict()
    )
    
    if "mlp" in modules_to_mix and mlp_fg:
        modules_to_mix.remove("mlp")
        modules_to_mix += ["up_proj", "gate_proj", "down_proj"]

    logging.info(config)
    logging.info("creating moe model...")

    with no_init_weights():
        moe_model = MOE_MODEL_CLS(config) 
    moe_sd = moe_model.state_dict()
    
    experts_keys = [] # to be replaced with ingredients weights later
    routers_keys = [] # to be replaced later, if positive tokens are provided
    base_keys = [] # no use currently

    stem_param = 0
    experts_param = 0
    routers_param = 0

    for key in moe_sd:

        has_key_in_modules_to_mix = any([f"{x}{EXPERTS_NAME}" in key for x in modules_to_mix])

        # stem
        if not has_key_in_modules_to_mix and not ROUTERS_NAME in key:
            logging.info(f"copying {key} from base...")
            moe_sd[key].copy_(sd_base.pop(key))
            base_keys.append(key)
            stem_param += moe_sd[key].numel()

        # router
        elif ROUTERS_NAME in key:
            if not gateless:
                if len(positive_tokens):
                    logging.info(f"zeroing {key}...")
                    torch.nn.init.zeros_(moe_sd[key])
                elif router_base_init:
                    logging.info(f"init {key}...")
                    torch.nn.init.constant_(moe_sd[key], 1e-6)
                    moe_sd[key][0] = 1.
                else:
                    logging.info(f"randomizing {key}...")
                    torch.nn.init.normal_(moe_sd[key], mean=0, std=moe_model.config.initializer_range)
            routers_keys.append(key)
            routers_param += moe_sd[key].numel()
        
        #  experts
        elif has_key_in_modules_to_mix:
            experts_keys.append(key)
            experts_param += moe_sd[key].numel()

        else:
            raise Exception("Something wrong! Current `key`={}".format(key))
    
    if positive_tokens and not gateless:
        for expert_idx in range(len(ingredients)):
            tokens_path = positive_tokens[expert_idx]
            tokens = load_from_disk(tokens_path)
            if isinstance(tokens, dict): tokens = tokens["train"]
            tokens = tokens["input_ids"][:num_samples]
            #ingred_model.cuda().eval()
            model_base.cuda().eval()
            logging.info("Computing hidden states using positive tokens from {}".format(tokens_path))
            for token_idx in tqdm(range(len(tokens))):
                #_hidden_states: List = ingred_model(
                _hidden_states: List = model_base(
                    torch.tensor(tokens[token_idx]).unsqueeze(0).cuda(),
                    output_hidden_states=True,
                    return_dict=True
                ).hidden_states[:-1]
                _hidden_states = torch.stack(_hidden_states, dim=0).mean(-2) # average across sequence
                hidden_states = _hidden_states.clone() if not token_idx else hidden_states + _hidden_states
            hidden_states = hidden_states.mean(1) # average across batch

            # for each specified module
            for module_name in modules_to_mix:

                keyword = f"{module_name}{EXPERTS_NAME}"
                if "." in keyword: keyword = keyword[:keyword.find(".")]
                keyword += ROUTERS_NAME
                matched_keys = [x for x in routers_keys if keyword in x]
                #NOTE: assume `routers_keys` are layer ordered

                for layer_idx, key in enumerate(matched_keys):

                    logging.info("Replacing {}[{}] using hidden states computed.".format(key, expert_idx))
                    router_weight =  hidden_states[layer_idx]
                    router_weight /= router_weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    moe_sd[key][expert_idx] += router_weight.cpu()

                    # for record
                    if expert_idx == len(ingredients)-1:
                        routers_keys.remove(key)

    del model_base
    del sd_base
    gc.collect()
    
    ## loading each ingredient models and and copy the weights to respectivce experts
    # all `experts_keys` should be overwritten with weightsafter this loop
    for expert_idx, path in enumerate(ingredients):

        logging.info("loading expert {} from {}...".format(expert_idx, path))
        ingred_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=moe_model.config.torch_dtype
        )
        ingred_sd = ingred_model.state_dict()

        # for each specified module
        for module_name in modules_to_mix:

            keyword = f"{module_name}{EXPERTS_NAME}{expert_idx}"
            matched_keys = [x for x in experts_keys if keyword in x]
            assert matched_keys, keyword

            # for each matched experts weight
            for key in matched_keys:
                key_cand = key.replace(keyword, module_name)

                logging.info("copying {} from expert {} to MOE {}...".format(key_cand, expert_idx, key))

                if always_on and expert_idx > 1:
                    moe_sd[key].copy_(ingred_sd[key_cand] - base_sd[key_cand])
                else:
                    moe_sd[key].copy_(ingred_sd[key_cand])

                # for record
                experts_keys.remove(key)

            if merge_base:
                merge_base_skip = []
                for key in base_keys:
                    if moe_sd[key].size() == ingred_sd[key].size():
                        logging.info("adding {} from expert {} to MOE {}...".format(key, expert_idx, key))
                        moe_sd[key] += ingred_sd[key]
                    else:
                        merge_base_skip.append(key)
                        continue
        
        if always_on and expert_idx==0:
            base_sd = ingred_sd
        else:
            del ingred_model
            del ingred_sd
            gc.collect()

    if merge_base:
        for key in base_keys:
            if not key in merge_base_skip:
                logging.info("averaging MOE {}...".format(key))
                moe_sd[key] /= len(ingredients)

    # END CHECK
    # ensure no weights are left empty/uncopied
    assert len(experts_keys) == 0, "Cannot match {}".format(experts_keys)

    if len(positive_tokens): assert len(routers_keys) == 0, "Cannot match {}".format(routers_keys)
    # /END CHECK

    # parameters
    model_info = {
        "stem_param": stem_param,
        "experts_param": experts_param,
        "routers_param": routers_param,
        "total_param": stem_param + experts_param + routers_param,
        "active_param": stem_param + routers_param + int(experts_param/len(ingredients)*num_experts_per_tok)
    }
    logging.info("Stem parameters: {}".format(model_info["stem_param"]))
    logging.info("Experts parameters: {}".format(model_info["experts_param"]))
    logging.info("Routers parameters: {}".format(model_info["routers_param"]))
    logging.info("MOE total parameters (numel): {}".format(
        sum(p.numel() for p in moe_model.parameters())))
    logging.info("MOE total parameters : {}".format(model_info["total_param"]))
    logging.info("MOE active parameters: {}".format(model_info["active_param"]))

    logging.info("Saving model...")
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    moe_model.to(torch.bfloat16).save_pretrained(output_dir)

    config_module_path = MOE_CFG_CLS.__module__
    config_file_path = os.path.join(output_dir, config_module_path.split(".")[-1] + ".py")
    model_module_path = MOE_MODEL_CLS.__module__
    model_file_path = os.path.join(output_dir, model_module_path.split(".")[-1] + ".py")
    assert os.path.exists(config_file_path)
    assert os.path.exists(model_file_path)
    with open(config_file_path, "w") as s:
        s.write("from {} import *".format(config_module_path))
    with open(model_file_path, "w") as s:
        s.write("from {} import *".format(model_module_path))
    logging.info("Done!")
    
    return moe_model, model_info

from transformers import DataCollatorForSeq2Seq
if __name__ == "__main__":

    import argparse, os, inspect
    from transformers import AutoTokenizer
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ingredients', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--always_on', action="store_true", default=False)
    parser.add_argument('--modules', nargs='+', default=["mlp"])
    parser.add_argument('--positive_tokens', nargs='+', default=[])
    parser.add_argument('--moe_layer_idx', nargs='+', default=[])
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_experts_per_tok', type=int, default=2)
    parser.add_argument('--global_router', action="store_true", default=False)
    parser.add_argument('--deep_router', action="store_true", default=False)
    parser.add_argument('--mlp_fg', action="store_true", default=False)
    parser.add_argument('--gateless', action="store_true", default=False)
    parser.add_argument('--router_base_init', action="store_true", default=False)
    parser.add_argument('--merge_base', action="store_true", default=False)
    args = parser.parse_args()

    model, model_info = mix(
        args.model_path,
        args.ingredients,
        args.modules,
        args.output_dir,
        args.positive_tokens,
        args.num_samples,
        moe_layer_idx=args.moe_layer_idx,
        global_router=args.global_router,
        deep_router=args.deep_router,
        gateless=args.gateless,
        always_on=args.always_on,
        mlp_fg=args.mlp_fg,
        num_experts_per_tok=args.num_experts_per_tok,
        router_base_init=args.router_base_init,
        merge_base=args.merge_base
    )

    import json
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f)