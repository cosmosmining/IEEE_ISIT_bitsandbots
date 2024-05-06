import torch as tc
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from preprocessing import fill_dir_dict_with_json_data,get_df_dict_init,replace_timestamp_to_timediff
### data loader script from https://www.kaggle.com/code/tebs89/bit-and-bots-dataloader/notebook

def make_df_from_data_dir(train_data_ratio = 0.8):
  #cleaning datasets and combine all the JSON files together
  # Define the list of directories to check. These are the directories which contain your data files.
  PATH= './data/isit-2024-bits-and-bots/ISIT_Dataset/'
  directories = ['hlisa_traces', 'gremlins', 'za_proxy', 
                 'survey_desktop','random_mouse_with_sleep_bot']
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
  '''
  hlisa_traces 421 [{user_id:...,device:...,trace:...},...{}]
  gremlins 421
  za_proxy 84
  survey_desktop 35
  '''
  df_dict_all = get_df_dict_init(df_names,data)
  df_dict_all = replace_timestamp_to_timediff(df_names,df_dict_all)
  #len(df_dict_init['survey_desktop']['userId']) #597524
  #np.unique(df_dict_init['survey_desktop']['userId'].to_numpy()).shape  #64743,
  
  df_dict_train= {}
  df_dict_eval = {}
  for name in df_dict_all.keys():
    print(name,end=' ')   
    all_df = df_dict_all[name] 
    print(all_df.columns,end=' ')
    user_ids = all_df['userId'].to_numpy()
    print(len(all_df))
    train_data_len  = int(len(all_df)*train_data_ratio)
    eval_data_len   = len(all_df) - train_data_len
    train_df = all_df[:train_data_len]   
    eval_df = all_df[train_data_len:]
    assert len(all_df) == len(train_df) + len(eval_df)
    print('unique user',len(np.unique(user_ids)),'/',len(user_ids))
    df_dict_train[name] = train_df
    df_dict_eval[name] = eval_df   
  return df_dict_all,df_dict_train,df_dict_eval
# Define your custom dataset class
class ISITDataset(Dataset):
  def __init__(self, data,training_len = 1000,max_sample_len = 10 ):
    self.hlisa = data['hlisa_traces']    #399023,6   #bot1
    self.gremlin = data['gremlins']      #185306,6   #bot2
    self.za = data['za_proxy']           #46340,6    #bot3
    self.sleep = data['random_mouse_with_sleep_bot']
    self.human = data['survey_desktop']  #597524,6  #399023+185306+46340+597524 = 1228193
    self.actors = data
    self.training_len = training_len
    self.max_sample_len = max_sample_len  
    self.idx_2_name = [key for key in self.actors.keys()]#['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop', 'random_mouse_with_sleep_bot']
    assert len(self.actors) == len(self.idx_2_name)
    self.data = pd.concat([self.human, self.hlisa, self.gremlin, self.za]) #1228193x6
    # self.data.columns = ['userId', 'x', 'y', 'eventName', 'userType', 'time_diff']
    self.event_names = sorted(self.data['eventName'].unique())
    '''['beforeunload', 'blur', 'change', 'click', 'contextmenu', 'copy', 'dblclick', 'error', 'focus', 'hashchange', 
    'keydown', 'keypress', 'keyup', 'load', 'message', 'mousedown', 'mousemove', 'mouseout', 'mouseover', 'mouseup', 'offline', 'online', 'pagehide', 'pageshow', 
    'paste', 'popstate', 'resize', 'scroll', 'select', 'storage', 'unload', 'wheel']
    '''
  def __len__(self):  return self.training_len
  def __getitem__(self, idx):
    data_type_idx = np.random.choice(len(self.actors),1)[0]   #[0,1,2,3,4]  hlisa gremlin za sleep human   
    name = self.idx_2_name[data_type_idx] #['userId', 'x', 'y', 'eventName', 'userType', 'time_diff']
    data_df = self.actors[name]
    len_data = len(data_df)  
    sample_len = np.random.choice(self.max_sample_len,1)[0]+1
    rnd_idx = np.random.choice(len_data-self.max_sample_len,1)[0]  #least len 10
    sample = data_df.iloc[rnd_idx:rnd_idx+sample_len] 
    
    # Convert 'x' to PyTorch tensor
    x_raw = tc.tensor(sample['x'].to_numpy(),dtype=tc.float32)
    x = tc.zeros(self.max_sample_len, dtype=tc.float32)
    x[:len(x_raw)] = x_raw

    terminate_idx = len(x_raw)
    # Convert 'y' to PyTorch tensor
    y_raw = tc.tensor(sample['y'].to_numpy(), dtype=tc.float32)
    y = tc.zeros(self.max_sample_len, dtype=tc.float32)
    y[:len(y_raw)] = y_raw
    # Convert 'userId' to PyTorch tensor:
    userId_raw = tc.tensor(sample['userId'].to_numpy(), dtype=tc.int64)
    userId = tc.zeros(self.max_sample_len,dtype=tc.int64)
    userId[:len(userId_raw)] = userId_raw
    # Convert 'timestamp' to PyTorch tensor:
    time_diff_raw = tc.tensor(sample['time_diff'].to_numpy(), dtype=tc.float32)  
    time_diff = tc.zeros(self.max_sample_len,dtype=tc.float32)  
    time_diff[:len(time_diff_raw)] = time_diff_raw
    # eventName is categorical, performing one-hot encoding
    eventName_idx = tc.zeros(self.max_sample_len,dtype=tc.int64)
    eventName_raw = sample['eventName']
    for idx, event in enumerate(eventName_raw):
      eventName_idx[idx] = self.event_names.index(event)  #todo accelerate
    assert self.max_sample_len == x.shape[0] == y.shape[0] == userId.shape[0] == time_diff.shape[0] #bs
    assert terminate_idx == len(x_raw) == len(y_raw) == len(userId_raw) == len(time_diff_raw) == len(sample['eventName'])
    eventName = tc.zeros((self.max_sample_len,len(self.event_names)), dtype=tc.int32)
    eventName[tc.arange(self.max_sample_len),eventName_idx] = 1
    for user_t in sample['userType']: assert  user_t == name
    # userType is categorical, performing one-hot encoding
    userType = tc.zeros(len(self.idx_2_name),dtype=tc.int32)
    userType[data_type_idx] = 1
    # breakpoint()
    return userId, time_diff, x, y, eventName,terminate_idx, userType
  #           [x0,        x1, x2,x3, x4]          y