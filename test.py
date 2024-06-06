import torch
import torch as tc
class Discretizer:
  #todo   use log
  def __init__(self,cont_max,token_num):
    self.cont_max = cont_max
    self.token_num = token_num
    self.scale = (token_num-1)/cont_max  
    self.interval  = cont_max/(token_num-1)
  def cont_2_token(self,cont_values):
    assert cont_values.max()<=self.cont_max
    token_values_raw=cont_values*self.scale  
    token_values = token_values_raw.to(dtype=tc.int64)
    assert token_values.max()<=self.token_num
    return token_values
  def token_2_cont(self,token_values):
    token_cont = token_values.to(tc.float32)
    cont_values_raw = token_cont/self.scale
    cont_values = cont_values_raw +\
         tc.rand(token_values.shape)*self.interval
    assert cont_values.max()<=self.cont_max
    return cont_values
cont_max = 1400 
vocab_size= 4420
disc = Discretizer(cont_max,vocab_size)   
cont_vals = tc.rand(30,50)*cont_max
token_vals = disc.cont_2_token(cont_vals)  
cont_val2 = disc.token_2_cont(token_vals) 
interval = disc.interval  
err = cont_val2-cont_vals 
assert tc.abs(err).max()<interval
breakpoint()

