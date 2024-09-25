import os, logging
from typing import List
import gc

import torch
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForCausalLM
)

from moetify import (
    MoetifyLoraConfig,
    get_peft_model
)

from peft import load_peft_weights

def decompose(weight_0, weight, device, rank=64):
    weight_delta = weight.to(device) - weight_0.to(device)
    u,s,v = torch.linalg.svd(weight_delta.float().to(device), full_matrices=False)
    s = torch.diag(s[:rank])
    B = u[:, :rank] @ s
    A = v[:rank, :]
    return B, A

def moe2expert(key, module_name, decompose_ingredients=False):
    key_cand = key[:key.find(module_name) + len(module_name)]
    if decompose_ingredients:
        key_cand = key_cand.removeprefix("base_model.model.") + ".weight"
    else:
        key_cand += ".lora_A.weight" 
    return key_cand

@torch.no_grad()
def mix(
    model_base_path:str,
    model_0_path:str,
    ingredients:List[str],
    modules_to_mix:List[str],
    decompose_base:bool=False,
    lora_rank:int=64,
    lora_alpha:int=64,
    gate_dimension:int=128,
    global_router:bool=False,
    deep_router:bool=False,
    router_aux_loss_coef:float=0.0,
    num_experts_per_tok:int=0,
    decompose_ingredients:bool=False
):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dummyT = torch.ones((1,8)).int().to(device)

    set_seed(1399)

    logging.info("loading base model...")
    config_base = AutoConfig.from_pretrained(model_base_path)
    model_base = AutoModelForCausalLM.from_pretrained(model_base_path).eval().to(device)
    num_experts = len(ingredients) + 1 if decompose_base else len(ingredients)
    base_out = model_base(dummyT).logits.clone()

    # SUPPORT CHECK
    assert num_experts_per_tok <= len(ingredients)

    assert len(modules_to_mix)>0, \
        "Modules to mix must have at least 'mlp'!"
    
    assert decompose_base == bool(model_0_path)

    #assert lora_alpha <= lora_rank
    # /SUPPORT CHECK
    
    # ZERO
    model_0_sd = AutoModelForCausalLM.from_pretrained(model_0_path).eval().state_dict()if decompose_base else model_base.state_dict()

    logging.info("creating base model...")
    config_base.torch_dtype = torch.float16
    config = MoetifyLoraConfig(
        task_type="moetify_CAUSAL_LM",
        r=lora_rank,
        lora_alpha=lora_alpha,
        gate_dimension=gate_dimension,
        num_experts=num_experts,
        target_modules=modules_to_mix,
        global_router=global_router,
        deep_router=deep_router,
        router_aux_loss_coef=router_aux_loss_coef,
        router_norm = "sum" if decompose_base else "sigmoid",
        num_experts_per_tok=num_experts_per_tok
    )

    moe_model = get_peft_model(model_base, config).eval()
    moe_sd = moe_model.state_dict()

    experts_keys = [] # to be replaced with ingredients weights later
    routers_keys = [] # to be replaced later, if positive tokens are provided
    stem_keys = [] # no use currently

    stem_param = 0
    experts_param = 0
    routers_param = 0

    decompose_multiplier = lora_rank / lora_alpha

    for key in moe_sd:

        has_key_in_modules_to_mix = any([x in key for x in modules_to_mix])

        # router
        if "router" in key:
            routers_keys.append(key)
            routers_param += moe_sd[key].numel()
            if "bias" in key:
                torch.nn.init.zeros_(moe_sd[key])
            elif "lora_router" in key:
                torch.nn.init.normal_(moe_sd[key], mean=0, std=moe_model.config.initializer_range)
                if decompose_base:
                    if deep_router:
                        moe_sd[key].view(-1, num_experts, moe_sd[key].size(1))[:, -1] = 1.
                    else:
                        moe_sd[key][-1] = 1.
        
        #  experts
        elif has_key_in_modules_to_mix and "expert" in key:
            if not decompose_base or not "expert_{}".format(num_experts-1) in key:
                experts_keys.append(key)
            experts_param += moe_sd[key].numel()

        else:
            stem_keys.append(key)
            stem_param += moe_sd[key].numel()

        # decompose
        if decompose_base and "base_layer" in key:
            # decompose base into last expert
            key_cand = moe2expert(key, "base_layer").replace("base_layer.", "")
            assert key_cand in model_0_sd, key_cand
            key_loraA = key.replace("base_layer", "lora_A.default-expert_{}".format(num_experts-1))
            key_loraB = key.replace("base_layer", "lora_B.default-expert_{}".format(num_experts-1))
            logging.info("decomposing {}...".format(key))
            B, A = decompose(
                model_0_sd[key_cand] * decompose_multiplier ,
                moe_sd[key] * decompose_multiplier ,
                device=device,
                rank=lora_rank)
            logging.info("svd error: {}".format(
                torch.dist(
                    moe_sd[key].to(device) * decompose_multiplier,
                    model_0_sd[key_cand].to(device) * decompose_multiplier + B@A
                )
            ))
            moe_sd[key].copy_(model_0_sd[key_cand].cpu())   # replace with W0
            moe_sd[key_loraA].copy_(A.cpu())
            moe_sd[key_loraB].copy_(B.cpu())
    
    ## loading each ingredient models and and copy the weights to respectivce experts
    # all `experts_keys` should be overwritten with weightsafter this loop
    # for each experts
    for expert_idx, path in enumerate(ingredients):

        logging.info("loading expert {} from {}...".format(expert_idx, path))
        ingred_sd = load_peft_weights(path)

        # for each specified module
        for module_name in modules_to_mix:

            keyword = f"expert_{expert_idx}"
            matched_keys = [x for x in experts_keys if keyword in x and module_name in x and not "lora_B" in x]
            assert matched_keys, keyword

            # for each matched moe weight
            for key in matched_keys:

                # convert moe_key to key_cand
                key_cand = moe2expert(key, module_name, decompose_ingredients=decompose_ingredients)

                if decompose_ingredients:
                
                    assert key_cand in ingred_sd, key_cand
                    assert key_cand in model_0_sd, key_cand

                    logging.info("decomposing {} from expert {}...".format(key_cand, expert_idx, key))
                    weight_cand = ingred_sd[key_cand]
                    
                    B, A = decompose(
                        model_0_sd[key_cand] * decompose_multiplier,
                        weight_cand * decompose_multiplier,
                        device=device,
                        rank=lora_rank)
                    logging.info("svd error: {}".format(
                        torch.dist(
                            weight_cand.to(device) * decompose_multiplier,
                            model_0_sd[key_cand].to(device) * decompose_multiplier + B@A
                        )
                    ))
                    logging.info("copying {} from expert {} to MOE {}...".format(key_cand, expert_idx, key))
                    moe_sd[key].copy_(A.cpu())
                    moe_sd[key.replace("lora_A", "lora_B")].copy_(B.cpu())

                else:

                    assert key_cand in ingred_sd, "{} not in {}".format(key_cand, ingred_sd.keys())
                    logging.info("copying {} from expert {} to MOE {}...".format(key_cand, expert_idx, key))
                    moe_sd[key].copy_(ingred_sd[key_cand].cpu())
                    moe_sd[key.replace("lora_A", "lora_B")].copy_(ingred_sd[key_cand.replace("lora_A", "lora_B")].cpu())

                # for record
                experts_keys.remove(key)
                experts_keys.remove(key.replace("lora_A", "lora_B"))
            
        del ingred_sd
        gc.collect()

    # END CHECK
    # ensure no weights are left empty/uncopied
    assert len(experts_keys) == 0, "Cannot match {}".format(experts_keys)

    # check error
    if decompose_ingredients:
        error = torch.dist(base_out, moe_model(dummyT).logits)
        logging.info("ERROR: {}".format(error))
    # /END CHECK

    # parameters
    model_info = {
        "stem_param": stem_param,
        "experts_param": experts_param,
        "routers_param": routers_param,
        "total_param": stem_param + experts_param + routers_param,
    }
    logging.info("Stem parameters: {}".format(model_info["stem_param"]))
    logging.info("Experts parameters: {}".format(model_info["experts_param"]))
    logging.info("Routers parameters: {}".format(model_info["routers_param"]))
    logging.info("MOE total parameters (numel): {}".format(
        sum(p.numel() for p in moe_model.parameters())))
    logging.info("MOE total parameters : {}".format(model_info["total_param"]))

    return moe_model.to(torch.float16), model_info


if __name__ == "__main__":

    import argparse, os
    from transformers import AutoTokenizer
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ingredients', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_0_path', type=str, default ="")
    parser.add_argument('--modules', nargs='+', required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--deep_router', action="store_true", default=False)
    parser.add_argument('--global_router', action="store_true", default=False)
    parser.add_argument('--decompose_base', action="store_true", default=False)
    parser.add_argument('--lora_rank', type=int, default=False)
    parser.add_argument('--lora_alpha', type=int, default=False)
    parser.add_argument('--decompose_ingredients', action="store_true", default=False)

    args = parser.parse_args()
    
    model, model_info = mix(
        args.model_path,
        args.model_0_path,
        args.ingredients,
        args.modules,
        args.decompose_base,
        global_router = args.global_router,
        deep_router = args.deep_router,
        lora_rank = args.lora_rank,
        lora_alpha = args.lora_alpha,
        decompose_ingredients = args.decompose_ingredients
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    import json
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f)