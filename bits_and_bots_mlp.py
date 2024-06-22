import torch as tc
import os
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset,make_df_from_data_dir
from nets.nn_net import NeuralNetwork
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
print(device)
train_data_ratio = 0.8
df_dict_all,df_dict_train,df_dict_eval = make_df_from_data_dir(train_data_ratio = train_data_ratio)
max_sample_len_train = 70
min_sample_len_train = 1
training_len  = 10
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
learning_rate = 1e-4
# optimizer = tc.optim.Adam(model.parameters(),lr=learning_rate)
optimizer = tc.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
# calculate confusion metrics
def cal_metrics(pred, ans):
    # Convert tensors to numpy arrays
    # pred_np = pred.numpy()
    # ans_np = ans.numpy()
    if pred.get_device() != 'cpu':
        pred_np = pred.detach().cpu().numpy()
    if ans.get_device() != 'cpu':
        ans_np = ans.detach().cpu().numpy()
    # Calculate metrics
    accuracy = accuracy_score(pred_np, ans_np)
    f1 = f1_score(pred_np, ans_np, average='weighted')
    recall = recall_score(pred_np, ans_np, average='weighted')
    precision = precision_score(pred_np, ans_np, average='weighted')

    return accuracy, f1, recall, precision

scheduler_type = 'ReduceLROnPlateau'
scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='deprecated')

load_from_checkpoint = True
ckpt_path = 'model2.ckpt'
if not os.path.isfile(ckpt_path):
  print("cant file ckpt path, train from scratch")
  load_from_checkpoint = False 
# Training loop
# Use Training set for ISIT2024
if load_from_checkpoint == False:
  idx = 0
  num_epochs = 10
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_dataloader:
      _, time_diff, pos_x, pos_y, _, terminate_idx, userType = batch
      time_diff = time_diff.to(device)
      pos_x = pos_x.to(device)
      pos_y = pos_y.to(device)
      terminate_idx = terminate_idx.to(device)
      userType = userType.to(device)
      y_target = userType
      y_hat = model(time_diff, pos_x, pos_y, terminate_idx)
      loss = criterion(y_hat, y_target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      idx += 1
      print(f"{idx}/{training_len * num_epochs}, term_idx {terminate_idx[0].item()}, {loss.item()=}")
      assert min_sample_len_train <= terminate_idx[0].item() <= max_sample_len_train
      running_loss += loss.item()
    # Step the scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler.step(running_loss)
    else:
        scheduler.step()
  tc.save({'state_dict':model.state_dict()},ckpt_path)
else:
  print("load from ckpt")
  checkpoint = tc.load(ckpt_path)
  model.load_state_dict(checkpoint['state_dict'])
# Assuming you have a DataLoader for evaluation
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
num_correct = 0; num_samples = 0
x_axis_len = 5
n_to_detect_list = np.linspace(1, max_sample_len_train, x_axis_len).astype(int)  #[1,18,35,52,70]
y_correct_list = np.zeros(x_axis_len)
test_i = 0

all_preds = []
all_labels = []

for batch in eval_dataloader:
    _, time_diff, pos_x, pos_y, _, terminate_idx, userType = batch
    time_diff = time_diff.to(device)
    pos_x = pos_x.to(device)
    pos_y = pos_y.to(device)
    terminate_idx = terminate_idx.to(device)
    userType = userType.to(device)
    y_target = userType   
    assert tc.all(terminate_idx == max_sample_len_train).cpu()
    
    for i in range(x_axis_len):
        pos_x2 = pos_x.clone()
        pos_y2 = pos_y.clone()
        time_diff2 = time_diff.clone()
        stop_idx = n_to_detect_list[i]
        if i != (x_axis_len - 1):
            pos_x2[:, stop_idx:] = 0 
            pos_y2[:, stop_idx:] = 0 
            time_diff2[:, stop_idx:] = 0 
        y_hat = model(time_diff2, pos_x2, pos_y2, terminate_idx)
        _, predictions = y_hat.max(1)
        y_correct_list[i] += (predictions == y_target).sum().item()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y_target.cpu().numpy())
        
    num_correct += (predictions == y_target).sum().item()
    num_samples += predictions.size(0)
    test_i += 1
    print(f"{test_i}/{len(eval_dataloader)}___")

assert num_samples == test_i * eval_dataloader.batch_size

# Calculate overall accuracy
overall_accuracy = num_correct / num_samples * 100
print(f"Accuracy on test set: {overall_accuracy:.2f}%")

# Calculate detailed metrics using cal_metrics
accuracy, f1, recall, precision = cal_metrics(tc.tensor(all_preds), tc.tensor(all_labels))
print(f"Evaluation - Accuracy: {accuracy}, F1 Score: {f1}, Recall: {recall}, Precision: {precision}")

# Calculate accuracy per detection length
y_correct_list /= num_samples
print(f"Accuracy per detection length: {y_correct_list * 100}")

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
# Calculate false positive rate
def cal_false_positive_rate(pred, ans, positive_label=1):
    # Convert tensors to numpy arrays if necessary
    if isinstance(pred, tc.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(ans, tc.Tensor):
        ans = ans.cpu().numpy()
    # Calculate false positives and total negatives
    false_positives = np.sum((pred == positive_label) & (ans != positive_label))
    total_negatives = np.sum(ans != positive_label)
    # False positive rate
    fpr = false_positives / total_negatives if total_negatives > 0 else 0
    return fpr

# Calculate false positive rate for each detection length
false_positive_rates = np.zeros(x_axis_len)
for i in range(x_axis_len):
    pos_x2 = pos_x.clone()
    pos_y2 = pos_y.clone()
    time_diff2 = time_diff.clone()
    stop_idx = n_to_detect_list[i]
    if i != (x_axis_len - 1):
        pos_x2[:, stop_idx:] = 0 
        pos_y2[:, stop_idx:] = 0 
        time_diff2[:, stop_idx:] = 0 
    y_hat = model(time_diff2, pos_x2, pos_y2, terminate_idx)
    _, predictions = y_hat.max(1)
    false_positive_rates[i] = cal_false_positive_rate(predictions, y_target, positive_label=1) # Assuming class '1' is the automated agent class

breakpoint()
# Average number of events to detection
avg_times = np.mean(n_to_detect_list)

# Plotting the false positive rate
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and bots', fontsize=20)
title = "Unimodal_Classification"
ax.set_title(title, fontsize=23)
ax.set_ylabel('False Positive Rate', fontsize=18)
ax.set_xlabel('Average Number of Events to Detection', fontsize=18)
ax.set_ylim([0, 1])
plt.yticks(np.arange(0, 1 + 0.05, 0.05))
plt.grid(True)
ax.scatter(avg_times, false_positive_rates, color='blue', label="False Positive Rate")

plt.tight_layout()
plt.savefig(f"fpr_result_{title}_{training_len=}.png")
plt.show()