import math
import torch as tc
import torch.nn as nn
from torch.nn import functional as F   

from config import GPTConfig
from nets.gpt.trans import LayerNorm,Block
from nets.gpt.utils import init_weights
class GPT(nn.Module):
  def __init__(self, c:GPTConfig):
    super().__init__()
    assert c.vocab_size is not None; assert c.block_size is not None
    self.c = c
    self.wte = nn.Embedding(c.vocab_size, c.n_embd)
    self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
    self.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
    self.wpe = nn.Embedding(c.block_size, c.n_embd)
    self.drop = nn.Dropout(c.dropout)
    self.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
    self.ln_f = LayerNorm(c.n_embd, bias=c.bias)
    
    self.apply(init_weights)# init all weights
    for pn, p in self.named_parameters():# apply special scaled init to the residual projections, per GPT-2 paper
      if pn.endswith('c_proj.weight'):
        tc.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * c.n_layer))
  def forward(self, idx, targets=None,terminate_index=None):
    device = idx.device
    b, t = idx.size() #*idx [bs,sentence_len=block_len]#*ex if prompt = "hello",then block len = 5
    assert t <= self.c.block_size, f"cant forward seq of len {t}, block size is only {self.c.block_size}"
    pos = tc.arange(0, t, dtype=tc.long, device=device) # shape (t)  #*pos =[0]
    # pos = [1,2,3,...,sentence_len] for pos embed
    tok_emb = self.wte(idx) # token embed (b,t)->(b, t, n_embd)
    pos_emb = self.wpe(pos) # pos embed     (t)->   (t, n_embd)
    x = self.drop(tok_emb + pos_emb) # x(b,t,n_embd)
    for block in self.h:
      x = block(x)
    x = self.ln_f(x)  
    if targets is not None:  # if we are given some desired targets also calculate the loss
      # if terminate_index[0]*2 < idx.shape[1]:
      #   # breakpoint()
      #   assert idx[0,terminate_index[0]*2] ==0
      logits = self.lm_head(x)#(b,t,n_embd=384)->(b,t,vocab_size=65)
      # breakpoint()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
    else:  #* in self.generate
      # inference-time mini-optimization: only forward lm_head on very last position
      logits = self.lm_head(x[:, [-1], :]) # using list [-1] to preserve the time dim
      #* x(b,t,n_embd) -> (b,1,b_embd)->(b,1,vocab_size)
      loss = None
    # breakpoint()
    return logits, loss
  @tc.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Take condition seq of indices (b,t):tc.Long 
    complete seq max_new_tokens times, 
    feed predictions back into model each time."""
    for _ in range(max_new_tokens):
      # if seq context grow too long we must crop it at block_size
      idx_cond = idx if idx.size(1) <= self.c.block_size else idx[:, -self.c.block_size:]
      #* ex if prompt = "hello"  then block len = 5
      logits, _ = self(idx_cond)#*idx_cond(bs=1,block_len)# forward the model to get logits for index in seq
      # logits = [bs,vocab_size]   (last character)
      logits = logits[:, -1, :] / temperature# pluck logits at final step and scale by desired temperature
      if top_k is not None:# optionally crop the logits to only the top k options
        #* topk=200
        v, _ = tc.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      # apply softmax to convert logits to (normalized) probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = tc.multinomial(probs, num_samples=1)
      # append sampled index to the running sequence and continue
      idx = tc.cat((idx, idx_next), dim=1)  #add the newly gen last word to the end of the sentence 
    return idx
