import os
import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset, make_df_from_data_dir
from nets.nn_net_b import NeuralNetwork_b
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import to_devices,mul_zeros_likes,plot_accu
# Device configuration
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
# Load and preprocess data
train_data_ratio = 0.8
df_dict_all, df_dict_train, df_dict_eval = make_df_from_data_dir(train_data_ratio=train_data_ratio)
# Hyperparameters
max_sample_len_train = 70
min_sample_len_train = 1
max_sample_len_eval = 70
training_len = 100  #* 500  
eval_len = 600
bs = 128
bs_eval = 1
learning_rate = 0.002
# Load from checkpoint if available
ckpt_path = 'model_a2.ckpt'
load_from_checkpoint = os.path.isfile(ckpt_path)
if not os.path.isfile(ckpt_path):
  print("cant find file ckpt path, train from scratch")
  load_from_checkpoint = False
# Prepare datasets and dataloaders
train_dataset = ISITDataset(df_dict_train, training_len=training_len*bs,
                            min_sample_len=min_sample_len_train,
                            max_sample_len=max_sample_len_train)
eval_dataset = ISITDataset(df_dict_eval, training_len=eval_len*bs_eval,
                           min_sample_len=max_sample_len_eval,
                           max_sample_len=max_sample_len_eval)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=12, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=bs_eval, shuffle=True, num_workers=12, pin_memory=True)

# Model, loss function, optimizer, and scheduler
category = len(train_dataset.idx_2_name)  # 5 categories
model = NeuralNetwork_b(max_len=max_sample_len_train, output_size=category).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-3)
# Load model from checkpoint if available
if load_from_checkpoint:
  checkpoint = tc.load(ckpt_path)
  model.load_state_dict(checkpoint['state_dict'])
else:
  for epoch in range(training_len):
    model.train()
    for batch in train_dataloader:
      _, td, px, py, eventName, _, userType = batch  #* td: time_diff, px: pos_x, py: pos_y
      td,px,py,userType,eventName = to_devices([td,px,py,userType,eventName],device)
      optimizer.zero_grad()
      y_hat = model(td,px,py,eventName)
      loss = criterion(y_hat, userType)
      loss.backward()
      optimizer.step()
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{training_len}], Loss: {loss.item()}")

  tc.save({'state_dict': model.state_dict()}, ckpt_path)

# Evaluation
model.eval()
thres_num = 10
conf_thres_list = np.linspace(0, 1, thres_num + 1)[1:-1]
tot_num_correct = 0
tot_num_samples = 0
tot_num_false_positives = 0
tot_num_false_negatives = 0
tot_num_HLISA_correct = 0
tot_num_HLISA_samples = 0
n_to_detect_list = np.zeros(len(conf_thres_list))
conf_n_correct_list = np.zeros(len(conf_thres_list))
conf_tot_n_samples_list = np.zeros(len(conf_thres_list))
conf_false_positives_list = np.zeros(len(conf_thres_list))
conf_false_negatives_list = np.zeros(len(conf_thres_list))
conf_HLISA_correct_list = np.zeros(len(conf_thres_list))
conf_HLISA_samples_list = np.zeros(len(conf_thres_list))

with tc.no_grad():
  for batch in eval_dataloader:
    _, td, px, py, eventName,_, userType = batch
    td,px,py,userType,eventName = to_devices([td,px,py,userType,eventName],device)
    for conf_i, conf_thres in enumerate(conf_thres_list):
      max_conf = 0
      n_to_detect = 0
      px2,py2,td2,eventName2 = mul_zeros_likes([px,py,td,eventName])
      while n_to_detect < max_sample_len_eval and max_conf < conf_thres:
        px2[0, n_to_detect] = px[0, n_to_detect]
        py2[0, n_to_detect] = py[0, n_to_detect]
        td2[0, n_to_detect] = td[0, n_to_detect]
        eventName2[0,n_to_detect] = eventName[0,n_to_detect]
        y_hat = model(td2,px2,py2,eventName2)
        conf_y_hat = tc.softmax(y_hat, dim=1)
        max_conf = conf_y_hat.max(dim=1)[0]
        n_to_detect += 1

      _, predictions = y_hat.max(1)
      n_correct = (predictions == userType).sum().item()
      n_sample = predictions.size(0)

      # Calculate false positives (human misclassified as bot)
      false_positives = ((predictions != userType) & (userType == 3)).sum().item()  # userType 3 corresponds to human
      # Calculate false negatives (bot misclassified as human)
      false_negatives = ((predictions != userType) & (userType != 3)).sum().item()  # userType 3 corresponds to human
      # Calculate HLISA detection rate
      HLISA_correct = ((predictions == userType) & (userType == 0)).sum().item()  # userType 0 corresponds to HLISA
      HLISA_samples = (userType == 0).sum().item()  # userType 0 corresponds to HLISA

      n_to_detect_list[conf_i] += n_to_detect
      conf_n_correct_list[conf_i] += n_correct
      conf_tot_n_samples_list[conf_i] += n_sample
      conf_false_positives_list[conf_i] += false_positives
      conf_false_negatives_list[conf_i] += false_negatives
      conf_HLISA_correct_list[conf_i] += HLISA_correct
      conf_HLISA_samples_list[conf_i] += HLISA_samples

      tot_num_correct += n_correct
      tot_num_samples += n_sample
      tot_num_false_positives += false_positives
      tot_num_false_negatives += false_negatives
      tot_num_HLISA_correct += HLISA_correct
      tot_num_HLISA_samples += HLISA_samples
print(f"Accuracy on test set: {tot_num_correct / tot_num_samples * 100:.2f}%")
# Calculate metrics
accuracy_list = conf_n_correct_list / conf_tot_n_samples_list
false_positive_rate_list = conf_false_positives_list / conf_tot_n_samples_list
false_negative_rate_list = conf_false_negatives_list / conf_tot_n_samples_list
HLISA_detection_rate_list = conf_HLISA_correct_list / conf_HLISA_samples_list

avg_n_to_detect_list = n_to_detect_list / eval_len
#from utils import plot_threshold;   plot_threshold(title,conf_thres_list,n_to_detect_list,accu_list)
avg_length = np.mean(n_to_detect_list)
print("Average number of events to detection:", avg_length)
# Visualization
suptitle = "Multi-modal Classification"
title = "Accuracy vs Number of Events to Detection"
save_path = "./img/a2_accuracy.png"
plot_accu(avg_n_to_detect_list , accuracy_list, title,save_path,
          suptitile=suptitle,
          xlabel='Number of Events to Detection',
          ylabel='Probability of Correct Classification')

title = "False Positive Rate vs Number of Events to Detection"
save_path = "./img/a2_false_positive_rate.png"
plot_accu(avg_n_to_detect_list , false_positive_rate_list, title,save_path,
          suptitile=suptitle,
          xlabel='Number of Events to Detection',
          ylabel='False Positive Rate')
# 
title = "False Negative Rate vs Number of Events to Detection"
save_path = "./img/a2_false_negative_rate.png"
plot_accu(avg_n_to_detect_list , false_negative_rate_list, title,save_path,
          suptitile=suptitle,
          xlabel='Number of Events to Detection',
          ylabel='False Negative Rate')

# HLISA Detection Rate vs Number of Events to Detection
title = "HLISA Detection Rate vs Number of Events to Detection"
save_path = "./img/a2_HLISA_Detection_Rate.png"
plot_accu(avg_n_to_detect_list , HLISA_detection_rate_list,  title,save_path,
          suptitile=suptitle,
          xlabel='Number of Events to Detection',
          ylabel='HLISA Detection Rate')
