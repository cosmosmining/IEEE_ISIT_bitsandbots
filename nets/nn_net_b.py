import torch
import torch as tc
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
class NeuralNetwork_b(nn.Module):
  def __init__(self,max_len,output_size):
    super().__init__()
    self.max_len = max_len   #70 = 210
    self.flatten = nn.Flatten()
    neural_size = 512
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(max_len*4, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, output_size),
    )

  def forward(self, time_diff,pos_x,pos_y,eventName,terminate_idx):
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
    time_diff2 = time_diff.unsqueeze(-1)
    pos_x2 = pos_x.unsqueeze(-1)
    pos_y2 = pos_y.unsqueeze(-1)
    eventName2 = eventName.unsqueeze(-1)
    xxx = tc.cat((time_diff2,pos_x2,pos_y2,eventName2),dim=2)
    # breakpoint()
    xx = xxx.reshape(bs,-1)
    # print(time_diff[tc.arange(bs),terminate_idx-1])
    logits = self.linear_relu_stack(xx)
    return logits  #(bs,user_type)  user_type is 0,1,..4