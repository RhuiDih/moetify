# moetify

**[Installation](#installation)**<br>
**[Create MOE](#create-moe)**<br>
**[Load MOE](#load-moe)**<br>
**[Create MOE (LORA ingredients)](#create-moe-lora-ingredients)**<br>
**[Load MOE (LORA ingredients)](#load-moe-lora-ingredients)**<br>

## Installation
```
pip3 install -e .
```

## Create MOE
```cli
python3 -m moetify.mix \
    --output_dir ./llama-3-4x8B-mlp-query \
    --model_path  meta-llama/Meta-Llama-3-8B \
    --modules mlp q_proj \
    --ingredients \
        meta-llama/Meta-Llama-3-8B \
        cognitivecomputations/dolphin-2.9-llama3-8b \
        openchat/openchat-3.6-8b-20240522 \
        aaditya/Llama3-OpenBioLLM-8B
```

| Arguments | Description |
| - | - |
| `--ingredients` | any huggingface loadable. Currently only support Llama and Mistral |
| `--modules` | either all or some from `mlp q_proj k_proj v_proj` |
| `--moe_layer_idx` | if specified, only modules in given layers will be mixed. Eg: `28 29 30 31` |
| `--num_experts_per_tok` | default to `2`, must be less than number of ingredients |
| `--always_on` | if specified, base model will always be active on top of `num_experts_per_tok` experts |
| `--gateless` | if specified, router will be absent and all experts will be active |
| `--mlp_fg` | if specified, three routeres will be used for each layer in MLP instead of one |


## Load MOE
```python
# make sure moetify is installed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "llama-3-4x8B-mlp-query",
    trust_remote_code=True,
)
# ready to proceed with training
```

## Create MOE (LORA ingredients)
```cli
python3 -m moetify.blend \
    --output_dir ./llama-3-3x8B-lora \
    --modules q_proj v_proj \
    --model_path  mistralai/Mistral-7B-v0.1 \
    --lora_rank 8 --lora_alpha 16 \
    --ingredients \
        tgaddair/mistral-7b-magicoder-lora-r8 \
        tgaddair/mistral-7b-gsmk8k-lora-r8 \
        tgaddair/mistral-7b-amazon-reviews-lora-r8
```

*NOTE: `--modules` lora weights must present in the ingredients*

## Load MOE (LORA ingredients)
```python
#  make sure moetify imported before loading model
import moetify
from transfomers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
)

model = PeftModel.from_pretrained(
    model, "llama-3-3x8B-lora"
)
# ready to proceed with training
```