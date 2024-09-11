from typing import Optional, Union, Any
import os, warnings
import collections

import torch

from peft.peft_model import PeftModelForCausalLM
from peft.peft_model import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, id_tensor_storage
from peft.utils.other import EMBEDDING_LAYER_NAMES
from peft.utils import infer_device, load_peft_weights

from safetensors.torch import save_file as safe_save_file


moetify_LORA = "moetify_LORA"
moetify_IA3 = "moetify_IA3"
moetify_METHODS = [moetify_LORA, moetify_IA3]

class MoetifyModelForCausalLM(PeftModelForCausalLM):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_ids=task_ids,
            **kwargs,
        )
        
        if "loss" in ret and self.active_peft_config.router_aux_loss_coef > 0:
            moe_modules = getattr(self, "lora_modules", [])
            if not moe_modules:
                moe_modules = tuple(m for m in self.modules() if hasattr(m, "router_logits"))
                assert len(moe_modules)
                self.moe_modules = moe_modules

            # addition 
            from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
            lb_loss = load_balancing_loss_func(
                tuple(m.router_logits for m in self.moe_modules),
                self.active_peft_config.num_experts,
                top_k = self.active_peft_config.num_experts_per_tok,
                attention_mask=attention_mask
            )
            ret.loss += self.active_peft_config.router_aux_loss_coef * lb_loss

        return ret


    # just to unroll peft/utils/save_and_load.get_peft_model_state_dict and extend
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        convert_pissa_to_lora: Optional[str] = None,
        **kwargs: Any,
    ):
        
        # copy from `peft`, since `get_peft_model_state_dict` is not extendale
        # `selected_adapters` is not used, all experts will be saved

        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        state_dict = kwargs.get("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()

        for adapter_name in self.active_adapters:

            peft_config = self.peft_config[adapter_name]
            assert peft_config.peft_type in moetify_METHODS, "This is moetify model. Supports {} only.".format(moetify_METHODS)
            
            prefix = self.base_model.prefix
            
            if peft_config.peft_type in moetify_LORA:
                bias = peft_config.bias
                if bias == "none":
                    output_state_dict = {k: state_dict[k] for k in state_dict if prefix in k}
                elif bias == "all":
                    output_state_dict = {k: state_dict[k] for k in state_dict if prefix in k or "bias" in k}
                elif bias == "lora_only":
                    output_state_dict = {}
                    for k in state_dict:
                        if prefix in k:
                            output_state_dict[k] = state_dict[k]
                            bias_name = k.split(prefix)[0] + "bias"
                            if bias_name in state_dict:
                                output_state_dict[bias_name] = state_dict[bias_name]
                else:
                    raise NotImplementedError
                output_state_dict = {k: v for k, v in output_state_dict.items() if ((prefix in k and adapter_name in k) or ("bias" in k))}
            
            elif peft_config.peft_type == moetify_IA3:
                output_state_dict = {k: state_dict[k] for k in state_dict if prefix in k}

            # = {k: v for k, v in output_state_dict.items() if ".{}.".format(adapter_name) in k or k.endswith(adapter_name)}

            if getattr(self, "modules_to_save", None) is not None:
                for key, value in state_dict.items():
                    if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in self.modules_to_save):
                        output_state_dict[key.replace("modules_to_save.", "")] = value

            # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
            if (
                save_embedding_layers == "auto"
                and hasattr(peft_config, "target_modules")
                and any(k in peft_config.target_modules for k in EMBEDDING_LAYER_NAMES)
            ):
                raise NotImplementedError
            elif save_embedding_layers == "auto":
                save_embedding_layers = False

            if save_embedding_layers and hasattr(self, "get_input_embeddings"):
                raise NotImplementedError
            elif save_embedding_layers:
                warnings.warn("Could not identify embedding layer(s) because the model is not a 🤗 transformers model.")

            #output_state_dict = {k.replace(f".{adapter_name}", ""): v for k, v in output_state_dict.items()} # why ?
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # Safetensors does not allow tensor aliasing.
                # We're going to remove aliases before saving
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    # Sometimes in the state_dict we have non-tensor objects.
                    # e.g. in bitsandbytes we have some `str` objects in the state_dict
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        # In the non-tensor case, fall back to the pointer of the object itself
                        ptrs[id(tensor)].append(name)

                # These are all the pointers of shared tensors.
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    # Here we just clone the shared tensors to avoid tensor aliasing which is
                    # not supported in safetensors.
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                if convert_pissa_to_lora is not None:
                    raise NotImplementedError
                
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                if convert_pissa_to_lora is not None:
                    raise NotImplementedError
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if convert_pissa_to_lora is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    peft_config.lora_alpha *= 2
                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode
    
    
    # this is to unroll peft.utils.save_and_load.set_peft_model_state_dict
    def load_adapter(
        self,
        model_id: str,
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        **kwargs: Any,
    ):
        
        # expect model has full set of experts/adapters attached
        # does not suppoert hf hub
        assert adapter_name in self.peft_config
        peft_config = self.peft_config[adapter_name]
        if peft_config.is_prompt_learning:
            raise NotImplementedError
        
        #assert os.path.exists(os.path.join(model_id, self.active_adapters[0]))

        if torch_device is None:
            torch_device = infer_device()

        adapters_weights = load_peft_weights(model_id, device=torch_device)

        parameter_prefix = self.base_model.prefix
        state_dict = {}
        if getattr(self, "modules_to_save", None) is not None:
            for key, value in adapters_weights.items():
                if any(module_name in key for module_name in self.modules_to_save):
                    for module_name in self.modules_to_save:
                        if module_name in key:
                            key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                            break
                state_dict[key] = value
        else:
            state_dict = adapters_weights
        
        
        moetify_key = []
        if peft_config.peft_type in moetify_METHODS:
            for k, v in state_dict.items():
                if parameter_prefix in k:
                    moetify_key.append(k)
        else:
            raise NotImplementedError
            
        load_result = self.load_state_dict(state_dict, strict=False)

        assert not load_result.unexpected_keys
        assert not any([x in load_result.missing_keys for x in moetify_key])
        if not is_trainable:
            self.eval()

        return load_result
    