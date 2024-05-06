### data loader script from https://www.kaggle.com/code/tebs89/bit-and-bots-dataloader/notebook
import pandas as pd
import torch
import torch as tc
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing import fill_dir_dict_with_json_data,get_df_dict_init,replace_timestamp_to_timediff
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
df_names=directories  #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop']
data = fill_dir_dict_with_json_data(directories,PATH)# Initialize an empty dictionary to hold data. One list per directory.
# data.values(); data['hlisa_traces']:list -> all the json data  #data['hlisa_traces'][0]
for key in data.keys():
    print(key,end=' ')  # #'hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop'
    print(len(data[key]),end='  ')  #json data len
    print(data[key][0].keys())  #['user_id', 'device', 'trace']
'''
hlisa_traces 421 [{user_id:...,device:...,trace:...},...{}]
gremlins 421
za_proxy 84
survey_desktop 35
'''
df_dict_init = get_df_dict_init(df_names,data)
for name in df_dict_init.keys():
  print(name,end=' ')
  a = df_dict_init[name]  #df with [userId,timestamp,x,y,eventName,userType]
  print(a.columns)
df_dict_init = replace_timestamp_to_timediff(df_names,df_dict_init)
for name in df_dict_init.keys():
  print(name,end=' ')
  a = df_dict_init[name]  #df with [userId,timestamp,x,y,eventName,userType]
  print(a.columns)#Index(['userId', 'x', 'y', 'eventName', 'userType', 'time_diff'], dtype='object')
  #df_dict_init['hlisa_traces']['x']
  #df_dict_init['hlisa_traces']['time_diff']
# Define your custom dataset class
class ISITDataset(Dataset):
  def __init__(self, data):
    self.hlisa = data['hlisa_traces']    #399023,6   #bot1
    self.gremlin = data['gremlins']      #185306,6   #bot2
    self.za = data['za_proxy']           #46340,6    #bot3
    self.human = data['survey_desktop']  #597524,6  #399023+185306+46340+597524 = 1228193
    self.data = pd.concat([self.human, self.hlisa, self.gremlin, self.za]) #1228193x6
    # self.data.columns = ['userId', 'x', 'y', 'eventName', 'userType', 'time_diff']
    self.event_names = sorted(self.data['eventName'].unique())
    '''
    ['beforeunload', 'blur', 'change', 'click', 'contextmenu', 'copy', 
    'dblclick', 'error', 'focus', 'hashchange', 'keydown', 'keypress', 
    'keyup', 'load', 'message', 'mousedown', 'mousemove', 'mouseout', 
    'mouseover', 'mouseup', 'offline', 'online', 'pagehide', 'pageshow', 
    'paste', 'popstate', 'resize', 'scroll', 'select', 'storage', 'unload', 'wheel']
    '''
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
    breakpoint()
    eventName = torch.zeros(len(self.event_names), dtype=torch.int32)
    eventName[eventName_idx] = 1
    # userType is categorical, performing one-hot encoding
    userType = torch.tensor([sample['userType'] == 'hlisa_traces',
                             sample['userType'] == 'gremlins',
                             sample['userType'] == 'za_proxy',
                             sample['userType'] == 'survey_desktop'], dtype=torch.int32)
    return userId, time_diff, x, y, eventName, userType
  #           [x0,        x1, x2,x3, x4]          y
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
    userId, time_diff, x, y, eventName, userType = batch
    break
breakpoint()