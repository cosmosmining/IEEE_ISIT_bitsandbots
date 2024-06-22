import os
import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset,make_df_from_data_dir
from nets.nn_net_b import NeuralNetwork_b
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 1

max_sample_len_eval = 70  #*70
training_len  = 1000 #* 1000
eval_len = 10*128
bs = 128
bs_eval = 1
load_from_checkpoint = True
ckpt_path = 'model_b.ckpt'
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
model = NeuralNetwork_b(max_len=max_sample_len_train,
                      output_size=category).to(device)
criterion = nn.CrossEntropyLoss()  # sigma [ y log(y_hat) ]
learning_rate = 0.01
optimizer = tc.optim.Adam(model.parameters(),lr=learning_rate)
train_i = 0
if load_from_checkpoint == False:
  for batch in train_dataloader:
    # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
    _, time_diff, pos_x, pos_y, eventName,terminate_idx, userType = batch
    # userId, time_diff, x, y, eventName, userType = batch
    #* pos_x.shape  [128,70] = (bs,n_of_events)
    
    time_diff = time_diff.to(device);pos_x = pos_x.to(device);pos_y = pos_y.to(device)
    eventName = eventName.to(device);terminate_idx =terminate_idx.to(device)
    userType = userType.to(device)
    y_target = userType   
    y_hat = model(time_diff,pos_x,pos_y,eventName,terminate_idx)
    # breakpoint()
    loss = criterion(y_hat, y_target)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    train_i+=1
    print(f"{train_i}/{training_len}, term_idx {terminate_idx[0].item()}, {loss.item()=}")
    assert min_sample_len_train<=terminate_idx[0].item()<=max_sample_len_train
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
  _, time_diff, pos_x, pos_y, eventName,terminate_idx, userType = batch
  time_diff=time_diff.to(device);  y_target = userType.to(device)
  pos_x=pos_x.to(device);          pos_y=pos_y.to(device);
  eventName = eventName.to(device)
  assert terminate_idx[0]==max_sample_len_eval
  for conf_i,conf_thres in enumerate(conf_thres_list):
    max_conf = 0;
    n_to_detect = 0;   
    pos_x2=tc.zeros_like(pos_x);pos_y2=tc.zeros_like(pos_y);
    time_diff2=tc.zeros_like(time_diff); eventName2 = tc.zeros_like(eventName)  
    while n_to_detect<max_sample_len_eval and max_conf<conf_thres:
      #**  request more data if conf is lower then threshold
      pos_x2[0,n_to_detect] = pos_x[0,n_to_detect]
      pos_y2[0,n_to_detect] = pos_y[0,n_to_detect]
      time_diff2[0,n_to_detect] = time_diff[0,n_to_detect] 
      eventName2[0,n_to_detect] = eventName[0,n_to_detect]
      y_hat = model(time_diff2,pos_x2,pos_y2,eventName2,terminate_idx)
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
fig.suptitle('Bits and bots', fontsize=20)
title = "Multimodal_Classification"
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
avg_length = np.mean(n_to_detect_list)
print("avg_length",avg_length)



