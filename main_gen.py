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
#[183365,6],[146692,6],[36673,6]  
#df_dict_all.keys() #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
#df_dict_all['hlisa_traces']['x'].to_numpy().max() =1195
#df_dict_all['hlisa_traces']['y'].to_numpy().max() =2798
max_yyy = 0;max_xxx =0
print(max_xxx,max_yyy)
for k in df_dict_all.keys():
  x_max = df_dict_all[k]['x'].to_numpy().max()
  y_max = df_dict_all[k]['y'].to_numpy().max()
  print('x',x_max,k)
  print('y',y_max,k)
  if x_max > max_xxx:
    max_xxx = x_max
  if y_max > max_yyy:
    max_yyy = y_max
vocab_size = max(max_xxx,max_yyy)  #todo
# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 1
training_len  = 1000
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
  #* pos_x,pos_y(bs=128,n_of_events=70),terminate_idx(bs=128),userType(bs=128)
  # assert (pos_x.max()<=max_xxx) and (pos_x.min()>=0) and (pos_y.max()<=max_yyy) and (pos_y.min()>=0) # assert (pos_x.max()<2841) and (pos_x.min()>=0) and (pos_y.max()<4428) and (pos_y.min()>=0)   
  # assert tc.equal(pos_x, tc.floor(pos_x)) and tc.equal(pos_y, tc.floor(pos_y))  #pos_x and pos_y are integer
  # userId, time_diff, x, y, eventName, userType = batch
  pos_xy = tc.stack((pos_x,pos_y),dim=-1).reshape(128,-1)  #(bs=128,n_of_events*2 = 140)
  assert  (pos_xy.max()<=vocab_size) and (pos_xy.min()>=0) 
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

x_axis_len = 5
n_to_detect_list = np.linspace(1,max_sample_len_train,x_axis_len).astype(int)  #[1,18,35,52,70]
y_correct_list = np.zeros(x_axis_len)
test_i = 0
for batch in eval_dataloader:
  _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
  ##time_diff() pos_x (128,70)=(bs,n_of_events)
  time_diff = time_diff.to(device)
  pos_x = pos_x.to(device)
  pos_y = pos_y.to(device)
  terminate_idx =terminate_idx.to(device)
  userType = userType.to(device)
  y_target = userType   
  assert tc.all(terminate_idx == max_sample_len_train).cpu()
  for i in range(x_axis_len):
    pos_x2 = pos_x.clone()
    pos_y2 = pos_y.clone()
    time_diff2 =time_diff.clone()
    stop_idx = n_to_detect_list[i]
    if i != (x_axis_len-1):
      pos_x2[:,stop_idx:] = 0 
      pos_y2[:,stop_idx:] = 0 
      time_diff2[:,stop_idx:] = 0 
    y_hat = model(time_diff2,pos_x2,pos_y2,terminate_idx)
    _, predictions = y_hat.max(1)
    y_correct_list[i] += (predictions == y_target).sum()
  num_correct += (predictions == y_target).sum()
  num_samples += predictions.size(0)
  test_i+=1
  print(f"{test_i}/{eval_len}___")
assert num_samples == test_i * batch_size
y_correct_list /= test_i * batch_size
print(f"Accuracy on test set: {num_correct/num_samples*100:.2f}")
print(f"{y_correct_list*100}")

fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.suptitle('Bits and bots', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title,fontsize=23)
ax.set_ylabel('probability of correct classification',fontsize =18)
ax.set_xlabel('number of events to detection',fontsize =18)
ax.set_ylim([0, 1])
plt.yticks(np.arange(0, 1+0.05, 0.05))
plt.grid(True)
ax.plot(n_to_detect_list,y_correct_list,'r',label="relu")


plt.tight_layout() 
plt.savefig(f"result_{title}_{training_len=}.png")
plt.show()
breakpoint()

