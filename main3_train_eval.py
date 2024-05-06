import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_loader_ import ISITDataset,make_df_from_data_dir
from nn_net import NeuralNetwork
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 1
training_len  = 100
eval_len = 10
batch_size = 128
train_dataset = ISITDataset(df_dict_train,training_len=training_len*batch_size,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)   
eval_dataset  = ISITDataset(df_dict_eval,training_len=eval_len*batch_size,
                            min_sample_len=max_sample_len_train,
                            max_sample_len=max_sample_len_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
idx = 0
category = len(train_dataset.idx_2_name)  #5 #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
model = NeuralNetwork(max_len=max_sample_len_train,output_size=category).to(device)
criterion = nn.CrossEntropyLoss()  # sigma [ y log(y_hat) ]
learning_rate = 0.01
optimizer = tc.optim.Adam(model.parameters(),lr=learning_rate)
for batch in train_dataloader:
  # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
  _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
  # userId, time_diff, x, y, eventName, userType = batch
  time_diff = time_diff.to(device)
  pos_x = pos_x.to(device)
  pos_y = pos_y.to(device)
  terminate_idx =terminate_idx.to(device)
  userType = userType.to(device)
  y_target = userType   
  y_hat = model(time_diff,pos_x,pos_y,terminate_idx)
  loss = criterion(y_hat, y_target)
  optimizer.zero_grad() 
  loss.backward()
  optimizer.step()
  idx+=1
  print(f"{idx}/{training_len}, term_idx {terminate_idx[0].item()}, {loss.item()=}")
  assert min_sample_len_train<=terminate_idx[0].item()<=max_sample_len_train
num_correct = 0
num_samples = 0
for batch in eval_dataloader:
  _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
  time_diff = time_diff.to(device)
  pos_x = pos_x.to(device)
  pos_y = pos_y.to(device)
  terminate_idx =terminate_idx.to(device)
  userType = userType.to(device)
  y_target = userType   
  y_hat = model(time_diff,pos_x,pos_y,terminate_idx)
  _, predictions = y_hat.max(1)
  num_correct += (predictions == y_target).sum()
  num_samples += predictions.size(0)
  print(f"{idx}/{eval_len}")
print(f"Accuracy on test set: {num_correct/num_samples*100:.2f}")
breakpoint()