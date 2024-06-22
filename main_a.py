import os
import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset,make_df_from_data_dir
from nets.nn_net import NeuralNetwork
from utils import to_devices,mul_zeros_likes

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
# Use Training set for ISIT2024
max_sample_len_train = 70; min_sample_len_train = 1
max_sample_len_eval = 70  #*70
training_len  = 1000 #* 1000
eval_len = 10*128
bs = 128
bs_eval = 1
load_from_checkpoint = True
ckpt_path = 'model.ckpt'
if not os.path.isfile(ckpt_path):
  print("cant file ckpt path, train from scratch")
  load_from_checkpoint = False 
train_dataset = ISITDataset(df_dict_train,training_len=training_len*bs,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)   
eval_dataset  = ISITDataset(df_dict_eval,training_len=eval_len*bs_eval,
                            min_sample_len=max_sample_len_eval,
                            max_sample_len=max_sample_len_eval)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=bs_eval, shuffle=True)

category = len(train_dataset.idx_2_name)  #5 #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
model = NeuralNetwork(max_len=max_sample_len_train,
                      output_size=category).to(device)
criterion = nn.CrossEntropyLoss()  # sigma [ y log(y_hat) ]
learning_rate = 0.01
optimizer = tc.optim.Adam(model.parameters(),lr=learning_rate)
train_i = 0
if load_from_checkpoint == False:
  for batch in train_dataloader:
    # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
    _, td, px, py, _,_, userType = batch  #px:pos_x,py:pos_y,td:time_diff
    #* pos_x.shape  [128,70] = (bs,n_of_events)
    td,px,py,userType = to_devices([td,px,py,userType],device)
    y_target = userType   
    y_hat = model(td,px,py)
    loss = criterion(y_hat, y_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    train_i+=1
    if train_i%10 == 0:
      print(f"{train_i}/{training_len}, {loss.item()=}")
  tc.save({'state_dict':model.state_dict()},ckpt_path)
else:
  checkpoint = tc.load(ckpt_path)
  model.load_state_dict(checkpoint['state_dict'])

thres_num = 10
conf_thres_list = np.linspace(0,1,thres_num+1)[1:-1]
#*  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tot_num_correct = 0; tot_num_samples = 0
test_i = 0
conf_n_correct_list      = np.zeros(len(conf_thres_list))
conf_tot_n_samples_list  = np.zeros(len(conf_thres_list))
n_to_detect_list       = np.zeros(len(conf_thres_list))

for batch in eval_dataloader:
  _, td, px, py, _,_, userType = batch
  td,px,py,userType = to_devices([td,px,py,userType],device)  
  y_target = userType
  for conf_i,conf_thres in enumerate(conf_thres_list):
    max_conf = 0; n_to_detect = 0;   
    px2,py2,td2 = mul_zeros_likes([px,py,td])
    # breakpoint()
    while n_to_detect<max_sample_len_eval and max_conf<conf_thres:
      #**  request more data if conf is lower then threshold
      px2[0,n_to_detect] = px[0,n_to_detect]
      py2[0,n_to_detect] = py[0,n_to_detect]
      td2[0,n_to_detect] = td[0,n_to_detect] 
      y_hat = model(td2,px2,py2)
      conf_y_hat = tc.softmax(y_hat,dim=1)
      max_conf = conf_y_hat.max(dim=1)[0]
      n_to_detect += 1   
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
  print(f"\r{test_i}/{eval_len}___",end='')
print("")
n_to_detect_list /= eval_len
assert tot_num_samples == (test_i * bs_eval *len(conf_thres_list)) == eval_len*len(conf_thres_list)
print(f"Accuracy on test set: {tot_num_correct/tot_num_samples*100:.2f}")
accu_list = conf_n_correct_list/conf_tot_n_samples_list
# breakpoint()
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.suptitle('Bits and bots', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title,fontsize=23)
ax.set_ylabel('probability of correct classification',fontsize =18)
ax.set_xlabel('number of events to detection',fontsize =18)
ax.set_ylim([0, 1])
plt.yticks(np.arange(0, 1+0.05, 0.05))
plt.grid(True)
ax.scatter(n_to_detect_list,accu_list,label="accuracy")
plt.tight_layout() 
plt.savefig(f"./img/{title}_{training_len=}.png")
plt.show()
#from utils import plot_threshold;   plot_threshold(title,conf_thres_list,n_to_detect_list,accu_list)
avg_length = np.mean(n_to_detect_list)
print("avg_length",avg_length)




