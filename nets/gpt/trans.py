from config import GPTConfig
import torch
import torch as tc
import torch.nn as nn
from torch.nn import functional as F
import math
class LayerNorm(nn.Module):
  """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
  def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
class CausalSelfAttention(nn.Module):
  def __init__(self, c:GPTConfig):
    super().__init__()
    assert c.n_embd % c.n_head == 0
    self.c_attn = nn.Linear(c.n_embd, 3*c.n_embd,bias=c.bias)# qkv projections for all heads,in a batch
    self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)# output projection
    # regularization
    self.attn_dropout=nn.Dropout(c.dropout);self.resid_dropout=nn.Dropout(c.dropout)
    self.dropout = c.dropout

    self.n_head = c.n_head;  self.n_embd = c.n_embd
    # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
    self.flash = False#hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    # breakpoint()
    if not self.flash:
      print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
      # causal mask to ensure that attn is only applied to left in input seq
      self.register_buffer("bias", tc.tril(tc.ones(c.block_size, c.block_size))
                                  .view(1, 1, c.block_size, c.block_size))
  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if self.flash:
      # efficient attention using Flash Attention CUDA kernels
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
      # manual implementation of attention
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    # output projection
    y = self.resid_dropout(self.c_proj(y))
    return y
class MLP(nn.Module):
  def __init__(self, c:GPTConfig):
    #c(n_layer=6,n_head=6,n_embd=384,block_size=256,bias=F,vocab_size=65,dropout=0.2)
    super().__init__()
    self.c_fc    = nn.Linear(c.n_embd, 4*c.n_embd, bias=c.bias)
    self.gelu    = nn.GELU()
    self.c_proj  = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
    self.dropout = nn.Dropout(c.dropout)
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x
class Block(nn.Module):
  def __init__(self, c:GPTConfig):
    super().__init__()
    self.ln_1 = LayerNorm(c.n_embd, bias=c.bias)
    self.attn = CausalSelfAttention(c)
    self.ln_2 = LayerNorm(c.n_embd, bias=c.bias)
    self.mlp = MLP(c)
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x