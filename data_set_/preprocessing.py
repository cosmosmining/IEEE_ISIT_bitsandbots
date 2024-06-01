import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
# Function to convert string to int (used for timestamps)
def str2int(s):
    return int(s.replace(',', ''))
def convert_to_unix_timestamp(dt_object):
    return int(dt_object.timestamp() * 1000)
def fill_dir_dict_with_json_data(directories,PATH,):
  data = {dir: [] for dir in directories}
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
  return data

def get_df_dict_init(df_names,data):
  df_dict_init = {}# Initialize an empty dictionary to hold the pandas dataframes for each directory
  # Create an empty DataFrame for each name
  for name in df_names:   #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop']
    df_dict_init[name] = pd.DataFrame()
  for name in df_names:  ##['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop']
    all_matches = []  # List to collect all match data
    for i in range(len(data[name])):  #421 for hlisa_traces
      trace = data[name][i]['trace'] #len(trace) = 377 [{'event_name': 'load', 'timestamp': '1694041007773',
                                                        # 'position': {'x': 0, 'y': 0}},...,]
      for j in range(len(trace)):
        # Convert timestamp to seconds and then to datetime object
        timestamp_in_seconds = str2int(trace[j]['timestamp'])/1000
        dt_object = datetime.fromtimestamp(timestamp_in_seconds)
        tp = convert_to_unix_timestamp(dt_object)
        all_matches.append((j, tp,trace[j]['position']['x'], 
                            trace[j]['position']['y'], trace[j]['event_name'], name))
        #all_matches: [(0, 1694041007773, 0, 0, 'load', 'hlisa_traces'),....]
    # Convert all matches into a DataFrame outside of loop
    df_dict_init[name] = pd.DataFrame(all_matches, columns=['userId', 'timestamp', 'x', 'y', 'eventName','userType'])
    df_dict_init[name] = df_dict_init[name].sort_values(by='timestamp')   
  return df_dict_init
def replace_zero_with_next_non_zero(arr):
    next_non_zero = None
    arr2 = np.copy(arr)
    for i in reversed(range(len(arr))):
        if arr[i] != 0 and arr[i] != '0':
            next_non_zero = arr[i]
        elif next_non_zero is not None:
            arr2[i] = next_non_zero
    return arr2
def replace_timestamp_to_timediff(df_names,df_dict_init):
  # Iterate over each DataFrame
  for df_name in df_names:  #['hlisa_traces', 'gremlins', 'za_proxy', 'survey_desktop']
    print(df_name)
    df = df_dict_init[df_name]  #df with [userId,timestamp,x,y,eventName,userType]
    df = df.sort_values(by='timestamp')
    df_x_temp2 = df['x'].to_numpy()
    df_y_temp2 = df['y'].to_numpy()
    # Apply the replacement function to the numpy arrays
    df_x_temp = replace_zero_with_next_non_zero(df_x_temp2)
    df_y_temp = replace_zero_with_next_non_zero(df_y_temp2)  #basically the same  
    # df_x_temp2[:10]: array([ 0,  0,  0,  0,  0, 11, 11,  0,  0, 27])
    # df_x_temp[:10] : array([11, 11, 11, 11, 11, 11, 11, 27, 27, 27])
    # Assign the numpy arrays back to the DataFrame
    df['x'] = df_x_temp
    df['y'] = df_y_temp
    # Save the modified DataFrame back into the dictionary
    df['timestamp'] = pd.to_datetime(df['timestamp'])  
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() * 1000  
    '''
    df['timestamp'][:5]
      25353   1970-01-01 00:28:14.034820921
      25352   1970-01-01 00:28:14.034820921
      88015   1970-01-01 00:28:14.034821169
      88016   1970-01-01 00:28:14.034821169
      88017   1970-01-01 00:28:14.034821246
    df['time_diff'][:5]
      25353         NaN
      25352    0.000000
      88015    0.000248
      88016    0.000000
      88017    0.000077
    '''
    df['time_diff'] = df['time_diff'].fillna(0)  #drop nan to 0
    #remove 'timestamp' from df
    df = df.drop(columns=['timestamp'])
    df_dict_init[df_name] = df
  #   print(df.columns)  #Index(['userId', 'x', 'y', 'eventName', 'userType', 'time_diff'], dtype='object')  
  return df_dict_init