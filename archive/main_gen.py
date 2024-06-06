import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset_gen,make_df_from_data_dir
from config import GPTConfig   
from nets.gpt.nn_gpt_net import GPT
from nets.gpt.utils import configure_optimizers,Discretizer   

from contextlib import nullcontext
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
gpt_conf = GPTConfig()
train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
#[183365,6],[146692,6],[36673,6]  
#df_dict_all.keys() #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
#df_dict_all['hlisa_traces']['x'].to_numpy().max() =1195
#df_dict_all['hlisa_traces']['y'].to_numpy().max() =2798
max_yyy = 0;max_xxx =0;max_ttt=0
print(max_xxx,max_yyy)
for k in df_dict_all.keys():
  x_max = df_dict_all[k]['x'].to_numpy().max()
  y_max = df_dict_all[k]['y'].to_numpy().max()
  t_max = df_dict_all[k]['time_diff'].to_numpy().max()
  
  print('x',x_max,k)
  print('y',y_max,k)
  if x_max > max_xxx:
    max_xxx = x_max
  if y_max > max_yyy:
    max_yyy = y_max
  if t_max > max_ttt:
    max_ttt = t_max
vocab_size = max(max_xxx,max_yyy)+1  #todo
gpt_conf.vocab_size = vocab_size
# np.sort(df_dict_all['survey_desktop']['time_diff'].to_numpy()
disc = Discretizer(max_ttt,vocab_size)

# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 1
training_len  = 1000  
eval_len = 10
batch_size = 128
train_dataset = ISITDataset_gen(df_dict_train,training_len=training_len*batch_size,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)   
eval_dataset  = ISITDataset_gen(df_dict_eval,training_len=eval_len*batch_size,
                            min_sample_len=max_sample_len_train,
                            max_sample_len=max_sample_len_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
idx = 0
model = GPT(gpt_conf).to(device)
#** gpt blocksize  
blocksize = 140-1
blocksize = max_sample_len_train*2-1
# learning_rate = 0.01
beta1 = 0.9;  beta2 = 0.99  
learning_rate:float = 1e-3;weight_decay:float = 1e-1;grad_clip:float = 1.0
# optim = tc.optim.Adam(model.parameters(),lr=learning_rate)
optim=configure_optimizers(model,weight_decay, learning_rate, (beta1,beta2), device)
for batch in train_dataloader:
  # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
  _, time_diff, pos_x, pos_y, _,terminate_idx = batch  
  #* pos_x,pos_y(bs=128,n_of_events=70),terminate_idx(bs=128),userType(bs=128)
  # assert (pos_x.max()<=max_xxx) and (pos_x.min()>=0) and (pos_y.max()<=max_yyy) and (pos_y.min()>=0) # assert (pos_x.max()<2841) and (pos_x.min()>=0) and (pos_y.max()<4428) and (pos_y.min()>=0)   
  # assert tc.equal(pos_x, tc.floor(pos_x)) and tc.equal(pos_y, tc.floor(pos_y))  #pos_x and pos_y are integer
  # userId, time_diff, x, y, eventName, userType = batch
  time_diff_token = disc.cont_2_token(time_diff).to(dtype=tc.float32)  #todo use exp
  pos_xy = tc.stack((pos_x,pos_y),dim=-1).reshape(batch_size,-1)  #(bs=128,n_of_events*2 = 140)
  assert  (pos_xy.max()<=vocab_size) and (pos_xy.min()>=0) and (time_diff.max()<= max_ttt)
  pos_xyt = tc.stack((pos_x,pos_y,time_diff_token),dim=-1).reshape(batch_size,-1)
  breakpoint()
  pos_xy = pos_xy.to(device=device,dtype=tc.int64)  #* (bs=128,len=140)
  xx = pos_xy[:,:blocksize]
  yy = pos_xy[:,1:blocksize+1]
  breakpoint()
  assert xx.shape[0]==batch_size and xx.shape[1]==blocksize
  assert yy.shape[0]==batch_size and yy.shape[1]==blocksize
  assert xx.max()<vocab_size and yy.max()<vocab_size
  terminate_idx =terminate_idx.to(device) 
  _,loss = model(xx,yy,terminate_idx)#,terminate_idx)
  optim.zero_grad() 
  loss.backward()
  optim.step()
  idx+=1
  # breakpoint()
  print(f"{idx}/{training_len}, term_idx {terminate_idx[0].item()}, {loss.item()=}")
  assert min_sample_len_train<=terminate_idx[0].item()<=max_sample_len_train

# ---------------------------
num_samples = 10 # number of samples to draw
max_new_tokens = blocksize # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if tc.cuda.is_available() and tc.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32':tc.float32,'bfloat16':tc.bfloat16,'float16':tc.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else tc.amp.autocast(device_type=device, dtype=ptdtype)
# encode the beginning of the prompt
x = 21  
y = 153
start_ids = [x,y]
x = (tc.tensor(start_ids, dtype=tc.long, device=device)[None, ...])
#*** x.shape  [1,2]=  bs, start_sentence_len
# run generation
with tc.no_grad():
  with ctx:
    for k in range(num_samples):
      y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
      print(y)
      breakpoint()
      print('---------------')