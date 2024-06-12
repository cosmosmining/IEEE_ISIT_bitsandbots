import torch
import torch as tc
class Discretizer:
  #todo   use log
  def __init__(self,cont_max,token_num):
    
    self.token_num = token_num
    
    self.min_val = 10**(-6)
    self.min_val_log = tc.log(tc.tensor([self.min_val]))
    self.cont_max = cont_max
    self.max_val_log = tc.log(tc.tensor([cont_max]))
    self.scale = (token_num-1)/(-self.min_val_log+self.max_val_log)  
    self.interval  = tc.abs((-self.min_val_log+self.max_val_log)/(token_num-1))
  def cont_2_token(self,cont_values):
    cont_no_zero = tc.clone(cont_values)
    cont_no_zero = tc.where(cont_no_zero<10**(-6),10**(-6),cont_no_zero)
    cont_v_log = tc.log(cont_no_zero)  
    print(cont_v_log.min(),cont_v_log.max())
    assert cont_v_log.min()>= self.min_val_log -10**(-5)
    cont_v_log += (-self.min_val_log)
    assert cont_values.max()<=self.cont_max
    assert cont_v_log.min()>=0 and cont_v_log.max() <= (-self.min_val_log+self.max_val_log)  
    token_values_raw=cont_v_log*self.scale  
    token_values = token_values_raw.to(dtype=tc.int64)
    assert token_values.max()<=self.token_num
    return token_values
  def token_2_cont(self,token_values):
    token_cont = token_values.to(tc.float32)
    cont_values_raw = token_cont/self.scale
    cont_v_log = cont_values_raw +\
         tc.rand(token_values.shape)*self.interval
    assert cont_v_log.min()>=0-self.interval-10**(-5) and cont_v_log.max() <= (-self.min_val_log+self.max_val_log)+self.interval+10**(-5)
    cont_v_log += self.min_val_log
    assert cont_v_log.min()>=self.min_val_log and cont_v_log.max() <= self.max_val_log
    cont_values = tc.exp(cont_v_log)
    assert 0-tc.exp(self.interval)<=cont_values.max()<=self.cont_max+tc.exp(self.interval)
    return cont_values
cont_max = 1400 
vocab_size= 4420
disc = Discretizer(cont_max,vocab_size)   
cont_vals = tc.rand(30,50)*cont_max
cont_vals[0,:30] = 0
token_vals = disc.cont_2_token(cont_vals)  
cont_val2 = disc.token_2_cont(token_vals) 
interval = disc.interval  
err = cont_val2-cont_vals 
# breakpoint()
# assert tc.abs(err).max()<=tc.exp(interval)
breakpoint()

