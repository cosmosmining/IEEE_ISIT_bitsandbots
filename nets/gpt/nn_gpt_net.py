import math
import torch as tc
import torch.nn as nn
from torch.nn import functional as F   
import numpy as np
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
    
    if terminate_index != None:
      term_idx = (terminate_index*3)
      # assert tc.all(term_idx>=3)
      # print(idx[tc.arange(b),term_idx-2])
      # idx[0]
      # idx[0,term_idx[0]]  #nothing
      # idx[0,term_idx[0]-1]  #t=0
      # idx[0,term_idx[0]-2]  #y=971
      # idx[0,term_idx[0]-3]  #x=383
      # print(targets[tc.arange(b),term_idx-2])
      # targets[0]
      # targets[0,term_idx[0]-1]  #nothing
      # targets[0,term_idx[0]-2]  #t=0
      # targets[0,term_idx[0]-3]  #y=971
      term_idx = term_idx.reshape(1,-1)
      visible_end = tc.cat((term_idx-3,term_idx-2,term_idx-1))    
      max_end = visible_end.max()
      assert max_end <= idx.shape[1]
      if max_end == idx.shape[1]:
        visible_end = tc.where(visible_end==max_end,max_end-1,visible_end)
    pos = tc.arange(0, t, dtype=tc.long, device=device) # shape (t)  #*pos =[0]
    # pos = [1,2,3,...,sentence_len] for pos embed
    tok_emb = self.wte(idx) # token embed (b,t)->(b, t, n_embd)
    pos_emb = self.wpe(pos) # pos embed     (t)->   (t, n_embd)
    x = self.drop(tok_emb + pos_emb) # x(b,t,n_embd)
    for block in self.h:
      x = block(x,visible_end)
    x = self.ln_f(x)  
    logits = self.lm_head(x)#(b,t,n_embd=384)->(b,t,vocab_size=65)
    if targets is not None:  # if we are given some desired targets also calculate the loss
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
    else:
      loss = None
    return logits, loss