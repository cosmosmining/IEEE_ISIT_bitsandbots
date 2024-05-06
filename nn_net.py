import torch
import torch as tc
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

  def forward(self, time_diff,pos_x,pos_y,terminate_idx):
    '''
    time_diff (bs,max_len) = (128,70)
    pos_x (bs,max_len)
    pos_y (bs,max_len)
    terminate_idx (bs)  range 0~max_len
    '''
    bs, max_len = time_diff.shape
    '''
    for i in range(bs):  #the paddings   ex.  [1,2,4,3,0,0,0]  terminate_idx = 4
      if terminate_idx[i] == max_len:
          continue
      print(time_diff[i,terminate_idx[i]:])
    '''
    # print(time_diff[tc.arange(bs),terminate_idx-1])
    breakpoint()
    x = self.flatten(pos_x)
    logits = self.linear_relu_stack(x)
    return logits  #(bs,user_type)  user_type is 0,1,..4