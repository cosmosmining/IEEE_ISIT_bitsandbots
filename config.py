from dataclasses import dataclass
@dataclass
class GPTConfig:
  n_layer:int = 6;  n_head:int = 6;  n_embd:int = 384
  block_size:int = 256 # context of up to 256 previous characters
  bias:bool = False # do we use bias inside LayerNorm and Linear layers?
  #bias True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
  vocab_size:int = None   #will be set by shakespeare's metadata
  #vocab_size: GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
  dropout:float = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
