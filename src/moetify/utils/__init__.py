import logging
from copy import deepcopy

def patched_forward(self, *args, routing_weights=None, selected_experts=None, **kwargs):
    return self.old_forward(*args, **kwargs)

def patch_torch_linear_forward_signature():
    logging.warning("Patching torch linear!")
    from torch.nn import Linear
    if not hasattr(Linear, "old_forward"):
        Linear.old_forward = Linear.forward
        Linear.forward = patched_forward