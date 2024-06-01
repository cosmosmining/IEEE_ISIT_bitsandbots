### data loader script from https://www.kaggle.com/code/tebs89/bit-and-bots-dataloader/notebook
import os
import json
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
#cleaning datasets and combine all the JSON files together

# Define the list of directories to check. These are the directories which contain your data files.
PATH= './data/isit-2024-bits-and-bots/ISIT_Dataset/'
directories = ['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop']

# Define the list of events to be searched for in the data files.
_docEvents = 'mousedown mouseup mousemove mouseover mouseout mousewheel wheel'
_docEvents += ' touchstart touchend touchmove deviceorientation keydown keyup keypress'
_docEvents += ' click dblclick scroll change select submit reset contextmenu cut copy paste'
_winEvents = 'load unload beforeunload blur focus resize error abort online offline'
_winEvents += ' storage popstate hashchange pagehide pageshow message beforeprint afterprint'


events = _docEvents.split() + _winEvents.split()

num_events=len(events)

df_names=directories
# Initialize an empty dictionary to hold data. One list per directory.
data = {dir: [] for dir in directories}

# Function to convert string to int (used for timestamps)
def str2int(s):
    return int(s.replace(',', ''))

def convert_to_unix_timestamp(dt_object):
    return int(dt_object.timestamp() * 1000)

# Iterate over each directory
for directory in directories:
    print(PATH+directory)
    # Get the list of files in the directory
    files = os.listdir(PATH+directory)
    files_list = [file for file in files[:len(files)-4]]

    # Iterate over each file
    for file in files_list:
        # Make sure we only read .json files
        if file.endswith('.json'):
            # Construct the full file path by joining the directory and file name
            filepath = PATH+os.path.join(directory, file)
            # Open the file
            with open(filepath, 'r') as f:
                # Load the json data
                json_data = json.load(f)

                # Append the data to our data list for the current directory
                data[directory].append(json_data)
for key in data.keys():
    print(key,end=' ')  # #'hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop'
    print(len(data[key]))  #json data len
breakpoint()
# Initialize an empty dictionary to hold the pandas dataframes for each directory
df_dict_init = {}

# Create an empty DataFrame for each name
for name in df_names:
    df_dict_init[name] = pd.DataFrame()


for name in df_names:
    all_matches = []  # List to collect all match data

    for i in range(len(data[name])):
        trace = data[name][i]['trace']

        for j in range(len(trace)):
            # Convert timestamp to seconds and then to datetime object
            timestamp_in_seconds = str2int(trace[j]['timestamp'])/1000
            dt_object = datetime.fromtimestamp(timestamp_in_seconds)
            tp = convert_to_unix_timestamp(dt_object)

            all_matches.append((j, tp,trace[j]['position']['x'], trace[j]['position']['y'], trace[j]['event_name'], name))

    # Convert all matches into a DataFrame outside of loop
    df_dict_init[name] = pd.DataFrame(all_matches, columns=['userId', 'timestamp', 'x', 'y', 'eventName','userType'])
    df_dict_init[name] = df_dict_init[name].sort_values(by='timestamp')
    
def replace_zero_with_next_non_zero(arr):
    next_non_zero = None
    for i in reversed(range(len(arr))):
        if arr[i] != 0 and arr[i] != '0':
            next_non_zero = arr[i]
        elif next_non_zero is not None:
            arr[i] = next_non_zero
    return arr
# Iterate over each DataFrame
for df_name in df_names:
    print(df_name)
    df = df_dict_init[df_name]
    df = df.sort_values(by='timestamp')
    df_x_temp = df['x'].to_numpy()
    df_y_temp = df['y'].to_numpy()
    # Apply the replacement function to the numpy arrays
    df_x_temp = replace_zero_with_next_non_zero(df_x_temp)
    df_y_temp = replace_zero_with_next_non_zero(df_y_temp)
    # Assign the numpy arrays back to the DataFrame
    df['x'] = df_x_temp
    df['y'] = df_y_temp
    # Save the modified DataFrame back into the dictionary
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() * 1000
    #remove 'timestamp' from df
    df = df.drop(columns=['timestamp'])
    df_dict_init[df_name] = df


# Define your custom dataset class
class ISITDataset(Dataset):
    def __init__(self, data):
        self.hlisa = data['hlisa_traces']
        self.gremlin = data['gremlins']
        self.za = data['za_proxy']
        self.human = data['survey_desktop']
        self.data = pd.concat([self.human, self.hlisa, self.gremlin, self.za])
        self.event_names = sorted(self.data['eventName'].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # Convert 'x' to PyTorch tensor
        x = torch.tensor(sample['x'], dtype=torch.float32)
        # Convert 'y' to PyTorch tensor
        y = torch.tensor(sample['y'], dtype=torch.float32)
        # Convert 'userId' to PyTorch tensor:
        userId = torch.tensor(sample['userId'], dtype=torch.int64)
        # Convert 'timestamp' to PyTorch tensor:
        time_diff = torch.tensor(sample['time_diff'], dtype=torch.float32)
        # eventName is categorical, performing one-hot encoding
        eventName_idx = self.event_names.index(sample['eventName'])
        eventName = torch.zeros(len(self.event_names), dtype=torch.int32)
        eventName[eventName_idx] = 1
        # userType is categorical, performing one-hot encoding
        userType = torch.tensor([sample['userType'] == 'hlisa_traces',
                                 sample['userType'] == 'gremlins',
                                 sample['userType'] == 'za_proxy',
                                 sample['userType'] == 'survey_desktop'], dtype=torch.int32)

        return userId, time_diff, x, y, eventName, userType
# Use Training set for ISIT2024
'''
It should be an instance of ISITDataset
'''
train_dataset = ISITDataset(df_dict_init)

# Use Testing set for ISIT2024
'''
It should be an instance of ISITDataset
'''
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
for batch in loader:
    print(batch)
    break
    