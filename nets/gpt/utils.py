import torch  
import torch as tc
import numpy as np
import torch.nn as nn
import inspect
def init_weights(module):
  if isinstance(module, nn.Linear):
    tc.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
        tc.nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    tc.nn.init.normal_(module.weight, mean=0.0, std=0.02)  

def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
  param_dict = {pn: p for pn, p in self.named_parameters()}# start with all candidate params
  param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}# filter out those that do not require grad
  #create optim groups. Any params that is 2D will be weight decayed, otherwise no.
  # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
  decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
  nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
  optim_groups = [
     {'params': decay_params, 'weight_decay': weight_decay},
     {'params': nodecay_params, 'weight_decay': 0.0}
  ]
  num_decay_params = sum(p.numel() for p in decay_params)
  num_nodecay_params = sum(p.numel() for p in nodecay_params)
  print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
  print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
  # Create AdamW optimizer and use the fused version if it is available
  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and device_type == 'cuda'
  extra_args = dict(fused=True) if use_fused else dict()
  optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
  print(f"using fused AdamW: {use_fused}")
  return optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
