import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset,make_df_from_data_dir
from nets.nn_net import NeuralNetwork
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
# Use Training set for ISIT2024
max_sample_len_train = 70
min_sample_len_train = 1

max_sample_len_eval = 70  #*70
training_len  = 100 #* 1000
eval_len = 10*128
bs = 128
bs_eval = 1
load_from_checkpoint = False
ckpt_path = 'model.ckpt'
train_dataset = ISITDataset(df_dict_train,training_len=training_len*bs,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)   
eval_dataset  = ISITDataset(df_dict_eval,training_len=eval_len*bs_eval,
                            min_sample_len=max_sample_len_eval,
                            max_sample_len=max_sample_len_eval)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=bs_eval, shuffle=True)
idx = 0
category = len(train_dataset.idx_2_name)  #5 #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
model = NeuralNetwork(max_len=max_sample_len_train,
                      output_size=category).to(device)
criterion = nn.CrossEntropyLoss()  # sigma [ y log(y_hat) ]
learning_rate = 0.01
optimizer = tc.optim.Adam(model.parameters(),lr=learning_rate)
if load_from_checkpoint == False:
  for batch in train_dataloader:
    # for task A. Defense Task,  only the time_diff and position [x,y] elements recorded for each event can be used as input to classifier  
    _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
    # userId, time_diff, x, y, eventName, userType = batch
    #* pos_x.shape  [128,70] = (bs,n_of_events)
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
  tc.save({'state_dict':model.state_dict()},ckpt_path)
else:
  checkpoint = tc.load(ckpt_path)
  model.load_state_dict(checkpoint['state_dict'])
num_correct = 0; num_samples = 0
test_i = 0
conf_thres_hold = 0.5
n_correct_dict = {}
n_sample_dict = {}
#todo   the official example is using avg event with difference threshold   
for batch in eval_dataloader:
  _, time_diff, pos_x, pos_y, _,terminate_idx, userType = batch
  ##time_diff() pos_x (128,70)=(bs,n_of_events)
  time_diff = time_diff.to(device)
  pos_x = pos_x.to(device); pos_y = pos_y.to(device);
  assert terminate_idx[0]==max_sample_len_eval
  userType = userType.to(device)
  y_target = userType   
  # assert tc.all(terminate_idx == max_sample_len_eval).cpu()
  pos_x2 = tc.zeros_like(pos_x)  
  pos_y2 = tc.zeros_like(pos_y)  
  time_diff2 = tc.zeros_like(time_diff)  
  ii = 0   
  max_conf = 0
  while ii<max_sample_len_eval and max_conf<conf_thres_hold:
    #**  request more data if the confi
    pos_x2[0,ii] = pos_x[0,ii]
    pos_y2[0,ii] = pos_y[0,ii]
    time_diff2[0,ii] = time_diff[0,ii] 
    y_hat = model(time_diff2,pos_x2,pos_y2,terminate_idx)
    conf_y_hat = tc.softmax(y_hat,dim=1)
    max_conf = conf_y_hat.max(dim=1)[0]
    ii += 1   
    # breakpoint()
    # print(ii,"max_conf",max_conf)
  # breakpoint()
  _, predictions = y_hat.max(1)
  ii = ii-1
  if not (ii in n_correct_dict):
    n_correct_dict[ii] = 0.
  if not (ii in n_sample_dict):
    n_sample_dict[ii] = 0.
  n_correct_dict[ii] +=  (predictions == y_target).sum().cpu().item() 
  n_sample_dict[ii] +=  predictions.size(0)
  num_correct += (predictions == y_target).sum();
  num_samples += predictions.size(0)
  test_i+=1
  print(f"{test_i}/{eval_len}___")
assert num_samples == test_i * bs_eval
n_correct_dict = dict(sorted(n_correct_dict.items()))
n_sample_dict = dict(sorted(n_sample_dict.items()))
assert np.all([*n_correct_dict]==[*n_sample_dict])
for k in n_correct_dict.keys():
  n_correct_dict[k]/=n_sample_dict[k] * bs_eval
print(f"Accuracy on test set: {num_correct/num_samples*100:.2f}")
aaa = [*n_sample_dict.values()]

least_sample_num = 20
n_sample_dict_eval = n_sample_dict.copy()
n_correct_dict_eval = n_correct_dict.copy()
for k in n_correct_dict.keys():
  if n_sample_dict[k] < least_sample_num: 
     del n_sample_dict_eval[k]
     del n_correct_dict_eval[k]
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.suptitle('Bits and bots', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title,fontsize=23)
ax.set_ylabel('probability of correct classification',fontsize =18)
ax.set_xlabel('number of events to detection',fontsize =18)
ax.set_ylim([0, 1])
plt.yticks(np.arange(0, 1+0.05, 0.05))
plt.grid(True)
ax.plot( [*n_correct_dict_eval],[*n_correct_dict_eval.values()],'r',label="accuracy")
plt.tight_layout() 
plt.savefig(f"result_{title}_{training_len=}.png")
plt.show()
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig.suptitle('Bits and bots', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title,fontsize=23)
ax.set_ylabel('length for classification',fontsize =18)
ax.set_xlabel('number of events to detection',fontsize =18)
ax.plot( [*n_sample_dict],[*n_sample_dict.values()])
plt.tight_layout() 
plt.savefig(f"result_{title}_{training_len=}_sample_len.png")
plt.show()
detect_lengths = np.array([*n_sample_dict])   
detect_length_num = np.array([*n_sample_dict.values()])
detect_length_rate = detect_length_num/np.sum(detect_length_num)  
avg_length = detect_lengths@detect_length_rate.T
print("avg_length",avg_length)
breakpoint()



