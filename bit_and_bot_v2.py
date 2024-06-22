import os
import torch as tc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_set_.data_loader_ import ISITDataset, make_df_from_data_dir
from nets.nn_net import NeuralNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
# Device configuration
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
# Load and preprocess data
train_data_ratio = 0.8
df_dict_all, df_dict_train, df_dict_eval = make_df_from_data_dir(train_data_ratio=train_data_ratio)

# Hyperparameters
max_sample_len_train = 70
min_sample_len_train = 1
max_sample_len_eval = 70
training_len = 10  #*500
eval_len = 600
bs = 128
bs_eval = 1
learning_rate = 0.002

# Load from checkpoint if available
ckpt_path = 'model.ckpt'
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
model = NeuralNetwork(max_len=max_sample_len_train, output_size=category).to(device)
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
            _, time_diff, pos_x, pos_y, _, terminate_idx, userType = batch
            time_diff, pos_x, pos_y, terminate_idx, userType = time_diff.to(device), pos_x.to(device), pos_y.to(device), terminate_idx.to(device), userType.to(device)
            optimizer.zero_grad()
            y_hat = model(time_diff, pos_x, pos_y)
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
        _, time_diff, pos_x, pos_y, _, terminate_idx, userType = batch
        time_diff, pos_x, pos_y, terminate_idx, userType = time_diff.to(device), pos_x.to(device), pos_y.to(device), terminate_idx.to(device), userType.to(device)
        assert terminate_idx[0] == max_sample_len_eval

        for conf_i, conf_thres in enumerate(conf_thres_list):
            max_conf = 0
            n_to_detect = 0
            pos_x2, pos_y2, time_diff2 = tc.zeros_like(pos_x), tc.zeros_like(pos_y), tc.zeros_like(time_diff)
            while n_to_detect < max_sample_len_eval and max_conf < conf_thres:
                pos_x2[0, n_to_detect] = pos_x[0, n_to_detect]
                pos_y2[0, n_to_detect] = pos_y[0, n_to_detect]
                time_diff2[0, n_to_detect] = time_diff[0, n_to_detect]
                y_hat = model(time_diff2, pos_x2, pos_y2)
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

# Visualization
# Accuracy vs Number of Events to Detection
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("Unimodal Classification", fontsize=23)
ax.set_ylabel('Probability of Correct Classification', fontsize=18)
ax.set_xlabel('Number of Events to Detection', fontsize=18)
ax.set_ylim([0, 1])
ax.scatter(n_to_detect_list / eval_len, accuracy_list, label="accuracy")
plt.yticks(np.arange(0, 1 + 0.05, 0.05))
plt.grid(True)
plt.tight_layout()
plt.savefig("Unimodal_Classification_num_events.png")
plt.show()

# False Positive Rate vs Number of Events to Detection
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("Unimodal Classification", fontsize=23)
ax.set_ylabel('False Positive Rate', fontsize=18)
ax.set_xlabel('Number of Events to Detection', fontsize=18)
ax.set_ylim([0, 1])
ax.scatter(n_to_detect_list / eval_len, false_positive_rate_list, label="false positive rate")
plt.yticks(np.arange(0, 1 + 0.05, 0.05))
plt.grid(True)
plt.tight_layout()
plt.savefig("Unimodal_Classification_num_events_false_positive_rate.png")
plt.show()

# False Negative Rate vs Number of Events to Detection
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("Unimodal Classification", fontsize=23)
ax.set_ylabel('False Negative Rate', fontsize=18)
ax.set_xlabel('Number of Events to Detection', fontsize=18)
ax.set_ylim([0, 1])
ax.scatter(n_to_detect_list / eval_len, false_negative_rate_list, label="false negative rate")
plt.yticks(np.arange(0, 1 + 0.05, 0.05))
plt.grid(True)
plt.tight_layout()
plt.savefig("Unimodal_Classification_num_events_false_negative_rate.png")
plt.show()

# HLISA Detection Rate vs Number of Events to Detection
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("HLISA Detection Rate", fontsize=23)
ax.set_ylabel('HLISA Detection Rate', fontsize=18)
ax.set_xlabel('Number of Events to Detection', fontsize=18)
ax.set_ylim([0, 1])
ax.scatter(n_to_detect_list / eval_len, HLISA_detection_rate_list, label="HLISA detection rate")
plt.yticks(np.arange(0, 1 + 0.05, 0.05))
plt.grid(True)
plt.tight_layout()
plt.savefig("HLISA_Detection_Rate.png")
plt.show()

# Accuracy vs Threshold
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("Unimodal Classification", fontsize=23)
ax.set_ylabel('Probability of Correct Classification', fontsize=18)
ax.set_xlabel('Threshold', fontsize=18)
ax.plot(conf_thres_list, accuracy_list)
plt.tight_layout()
plt.savefig("Unimodal_Classification_threshold_accuracy.png")
plt.show()

# Number of Events to Detection vs Threshold
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle('Bits and Bots', fontsize=20)
ax.set_title("Unimodal Classification", fontsize=23)
ax.set_ylabel('Number of Events to Detection', fontsize=18)
ax.set_xlabel('Threshold', fontsize=18)
ax.plot(conf_thres_list, n_to_detect_list / eval_len)
plt.tight_layout()
plt.savefig("Unimodal_Classification_threshold_events.png")
plt.show()

avg_length = np.mean(n_to_detect_list / eval_len)
print("Average number of events to detection:", avg_length)
# 我有重新訓練一個model.ckpt