import os
import torch as tc
from torch.utils.data import DataLoader
from data_set_.data_loader_ import ISITDataset_gen,make_df_from_data_dir
from config import GPTConfig   
from nets.gpt.nn_gpt_net import GPT
from nets.gpt.utils import configure_optimizers,Discretizer_old   
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F  
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
gpt_conf = GPTConfig()
train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
max_yyy = 0;max_xxx =0;max_ttt=0
print(max_xxx,max_yyy)
for k in df_dict_all.keys():
  x_max = df_dict_all[k]['x'].to_numpy().max()
  y_max = df_dict_all[k]['y'].to_numpy().max()
  t_max = df_dict_all[k]['time_diff'].to_numpy().max()
  print('x',x_max,k)
  print('y',y_max,k)
  if x_max > max_xxx: max_xxx = x_max
  if y_max > max_yyy: max_yyy = y_max
  if t_max > max_ttt: max_ttt = t_max
vocab_size = max(max_xxx,max_yyy)+1  #todo
gpt_conf.vocab_size = vocab_size
# np.sort(df_dict_all['survey_desktop']['time_diff'].to_numpy()
disc = Discretizer_old(max_ttt,vocab_size)
# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 3
training_len  =1000#600#* 1000  
eval_len = 10
bs = 128
load_from_checkpoint = True

ckpt_path = 'gpt.ckpt'
if not os.path.isfile(ckpt_path):
  print("cant file ckpt path, train from scratch")
  load_from_checkpoint = False 
train_dataset = ISITDataset_gen(df_dict_train,training_len=training_len*bs,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)   
eval_dataset  = ISITDataset_gen(df_dict_eval,training_len=eval_len*bs,
                            min_sample_len=max_sample_len_train,
                            max_sample_len=max_sample_len_train)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True)
idx = 0
model = GPT(gpt_conf).to(device)
#** gpt blocksize  
blocksize = max_sample_len_train*3-1
beta1 = 0.9;  beta2 = 0.99  
learning_rate:float = 1e-3;weight_decay:float = 1e-1;grad_clip:float = 1.0
# optim = tc.optim.Adam(model.parameters(),lr=learning_rate)
optim=configure_optimizers(model,weight_decay, learning_rate, (beta1,beta2), device)
if load_from_checkpoint == False:
  for batch in train_dataloader:
    # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
    _, time_diff, pos_x, pos_y, _,terminate_idx = batch  
    #* pos_x,pos_y(bs=128,n_of_events=70),terminate_idx(bs=128),userType(bs=128)
    time_diff_token = disc.cont_2_token(time_diff).to(dtype=tc.float32)  #todo use log for small numbers
    pos_xyt = tc.stack((pos_x,pos_y,time_diff_token),dim=-1).reshape(bs,-1)  #(bs=128,n_of_events*2 = 140)
    if tc.any(time_diff_token>0):
      print(time_diff_token[time_diff_token>0])  #todo  use log at discretizer
    assert  (pos_xyt.max()<=vocab_size) and (pos_xyt.min()>=0) and (time_diff.max()<= max_ttt)
    pos_xyt = pos_xyt.to(device=device,dtype=tc.int64)  #* (bs=128,len=140)
    xx = pos_xyt[:,:blocksize]
    yy = pos_xyt[:,1:blocksize+1]
    assert xx.shape[0]==bs and xx.shape[1]==blocksize and yy.shape[0]==bs and yy.shape[1]==blocksize
    assert xx.max()<vocab_size and yy.max()<vocab_size
    _,loss = model(xx,yy,terminate_idx)
    optim.zero_grad() 
    loss.backward()
    optim.step()
    idx+=1
    # breakpoint()
    print(f"{idx}/{training_len}, term_idx {terminate_idx[0].item()}, {loss.item()=}")
    assert min_sample_len_train<=terminate_idx[0].item()<=max_sample_len_train
  tc.save({'state_dict':model.state_dict()},ckpt_path)
else:
  checkpoint = tc.load(ckpt_path)
  model.load_state_dict(checkpoint['state_dict'])

# ---------------------------
num_samples = 3 # number of samples to draw
max_new_tokens = blocksize+1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if tc.cuda.is_available() and tc.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32':tc.float32,'bfloat16':tc.bfloat16,'float16':tc.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else tc.amp.autocast(device_type=device, dtype=ptdtype)
# encode the beginning of the prompt
x = 21;y = 153;t = 0
end_idx = 3*50
start_xyt = (tc.tensor([x,y,t], dtype=tc.long, device=device)[None, ...])
ini_xyt = tc.zeros((1,max_new_tokens),dtype=tc.long, device=device)
ini_xyt[0,0] = 21   #x 
ini_xyt[0,1] = 153  #y
ini_xyt[0,2] = 0    #t
ini_xyt[0,end_idx] = 56  #x  
ini_xyt[0,end_idx+1] = 321  #y
ini_xyt[0,end_idx+2] = 0    #t
#*** x.shape  [1,3]=  bs, start_sentence_len
# run generation
@tc.no_grad()
def generate(model_, idx, end_idx_, temperature=1.0, top_k=None):
  """Take condition seq of indices (b,t):tc.Long 
  complete seq max_new_tokens times, 
  feed predictions back into model each time."""
  assert idx.size(1) <= model_.c.block_size
  for ii in range(2,end_idx_-1):
    # if seq context grow too long we must crop it at block_size
    idx_cond = idx 
    logits, _ = model_(idx_cond,None,terminate_index=tc.tensor([(end_idx_+3)//3],dtype=tc.int64)) 
    logits = logits[:,[ii],:]
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
    # idx = tc.cat((idx, idx_next), dim=1)  #add the newly gen last word to the end of the sentence 
    idx[0,ii+1]= idx_next[0,0]
  return idx
with tc.no_grad():
  with ctx:
    for k in range(num_samples):
      gen_xyt = generate(model,ini_xyt,end_idx, temperature=temperature, top_k=top_k)
      print(gen_xyt)
      bs,bl = gen_xyt.shape 
      gen_xyt = gen_xyt.reshape(bs,-1,3)  
      print('---------------')

from nets.nn_net import NeuralNetwork
from data_set_.data_loader_ import ISITDataset
max_sample_len_eval = 70  #*70
bs_eval = 1
eval_len = 100
eval_dataset_mlp  = ISITDataset(df_dict_eval,training_len=eval_len*bs_eval,
                            min_sample_len=max_sample_len_eval,
                            max_sample_len=max_sample_len_eval)
eval_dataset  = ISITDataset_gen(df_dict_eval,training_len=eval_len*bs_eval,
                            min_sample_len=max_sample_len_eval,
                            max_sample_len=max_sample_len_eval)

eval_dataloader = DataLoader(eval_dataset, batch_size=bs_eval, 
                             shuffle=True)

category = len(eval_dataset_mlp.idx_2_name)  #5 #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
ckpt_path = 'model.ckpt'
model_mlp = NeuralNetwork(max_len=max_sample_len_train,
                      output_size=category).to(device)
if not os.path.isfile(ckpt_path):
  raise Exception('you dont have defensive task ckpt, please run "python main.py" first') 
checkpoint = tc.load(ckpt_path)
model_mlp.load_state_dict(checkpoint['state_dict'])
thres_num = 10
conf_thres_list = np.linspace(0,1,thres_num+1)[1:-1]
#*  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tot_num_correct = 0; tot_num_samples = 0
test_i = 0
conf_n_correct_list      = np.zeros(len(conf_thres_list))
conf_tot_n_samples_list  = np.zeros(len(conf_thres_list))
n_to_detect_list       = np.zeros(len(conf_thres_list))
for i, name in enumerate(eval_dataset_mlp.idx_2_name):
  if name == 'survey_desktop':
    human_idx = i
assert eval_dataset_mlp.idx_2_name[human_idx] == 'survey_desktop'
blocksize_mlp = 70
for batch in eval_dataloader:
  _, time_diff, pos_x, pos_y, _,terminate_idx = batch
  time_diff=time_diff.to(device);  
  pos_x=pos_x.to(device);          pos_y=pos_y.to(device);
  end_idx_raw = np.random.randint(10,max_sample_len_eval)
  assert 10<= end_idx_raw<max_sample_len_eval

  assert terminate_idx[0]==max_sample_len_eval
  x2 = int(pos_x[0,0].cpu().item())
  y2 = int(pos_y[0,0].cpu().item())
  t2 = int(time_diff[0,0].cpu().item())  #todo   disc 2 token
  start_xyt = (tc.tensor([x2,y2,t2], dtype=tc.long, device=device)[None, ...])
  
  ini_xyt = tc.zeros((1,max_new_tokens),dtype=tc.long, device=device)
  end_idx = end_idx_raw*3
  ini_xyt[0,0] = pos_x[0,0]   #x 
  ini_xyt[0,1] = pos_y[0,0]   #y
  ini_xyt[0,2] = time_diff[0,0]    #t
  ini_xyt[0,end_idx] = pos_x[0,end_idx_raw] #x  
  ini_xyt[0,end_idx+1] = pos_y[0,end_idx_raw] #y
  ini_xyt[0,end_idx+2] = time_diff[0,end_idx_raw]    #t
  # gen_xyt = model.generate(start_xyt, max_new_tokens, temperature=temperature, top_k=top_k)
  gen_xyt = generate(model,tc.clone(ini_xyt),end_idx, temperature=temperature, top_k=top_k)
  bs,bl = gen_xyt.shape 
  gen_xyt = gen_xyt.reshape(bs,-1,3)   
  pos_x_gen = gen_xyt[:,:,0].to(tc.float32)[:,:blocksize_mlp]
  pos_y_gen = gen_xyt[:,:,1].to(tc.float32)[:,:blocksize_mlp]
  time_diff_gen_token = gen_xyt[:,:,2]
  time_diff_gen = disc.token_2_cont(time_diff_gen_token.cpu())[:,:blocksize_mlp].to(device)  #todo use log  or just use continous prediction
  y_target = tc.tensor([human_idx]).to(device)
  for conf_i,conf_thres in enumerate(conf_thres_list):
    max_conf = 0;
    n_to_detect = 0;   
    pos_x2=tc.zeros_like(pos_x_gen);
    pos_y2=tc.zeros_like(pos_y_gen);time_diff2=tc.zeros_like(time_diff_gen)  
    while n_to_detect<max_sample_len_eval and max_conf<conf_thres:
      #**  request more data if conf is lower then threshold
      pos_x2[0,n_to_detect] = pos_x_gen[0,n_to_detect]
      pos_y2[0,n_to_detect] = pos_y_gen[0,n_to_detect]
      time_diff2[0,n_to_detect] = time_diff_gen[0,n_to_detect] 
      y_hat = model_mlp(time_diff2,pos_x2,pos_y2,terminate_idx)
      conf_y_hat = tc.softmax(y_hat,dim=1)
      max_conf = conf_y_hat.max(dim=1)[0]
      n_to_detect += 1   
      # breakpoint()
      # print(ii,"max_conf",max_conf)
    _, predictions = y_hat.max(1)
    n_correct = (predictions == y_target).sum();
    n_sample  = predictions.size(0)  #*1
    n_to_detect_list[conf_i] += n_to_detect
    conf_n_correct_list[conf_i] += n_correct
    conf_tot_n_samples_list[conf_i] += n_sample
    tot_num_correct += n_correct
    tot_num_samples += n_sample
  test_i+=1
  print(f"{test_i}/{eval_len}___")

n_to_detect_list /= eval_len
assert tot_num_samples == (test_i * bs_eval *len(conf_thres_list)) == eval_len*len(conf_thres_list)
print(f"Accuracy on test set: {tot_num_correct/tot_num_samples*100:.2f}")
accu_list = conf_n_correct_list/conf_tot_n_samples_list
# breakpoint()
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.suptitle('OffenseTask_unimodal', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title,fontsize=23)
ax.set_ylabel('probability of correct classification',fontsize =18)
ax.set_xlabel('number of events to detection',fontsize =18)
ax.set_ylim([0, 1])
plt.yticks(np.arange(0, 1+0.05, 0.05))
plt.grid(True)
ax.scatter(n_to_detect_list,accu_list,label="accuracy")
plt.tight_layout() 
plt.savefig(f"./img/offenseTask_{training_len=}.png")
plt.show()

breakpoint()



