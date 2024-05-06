import torch as tc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from data_loader_ import ISITDataset,make_df_from_data_dir
from nn_net import NeuralNetwork
train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
# Use Training set for ISIT2024
max_sample_len = 70
training_len  = 100
batch_size = 128
train_dataset = ISITDataset(df_dict_train,training_len=training_len*batch_size,max_sample_len=max_sample_len)
# Use Testing set for ISIT2024
# eval_dataset = ISITDataset(df_dict_eval,max_sample_len=max_sample_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
idx = 0
category = len(train_dataset.idx_2_name)  #5 #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
net = NeuralNetwork(max_len=max_sample_len,output_size=category)
for batch in train_dataloader:
  print(f"{idx}/{training_len}")
  # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
  _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
  # userId, time_diff, x, y, eventName, userType = batch
  y_target = userType   
  y_hat = net(time_diff,pos_x,pos_y,terminate_idx)
  idx+=1
breakpoint()