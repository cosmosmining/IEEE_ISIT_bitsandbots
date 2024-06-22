import matplotlib.pyplot as plt
import torch as tc
import numpy as np
def to_devices(varlist,device):
  ret_list = []
  for  vari in varlist:
    ret_list.append(vari.to(device))
  return ret_list
def mul_zeros_likes(varlist):
  ret_list = []
  for  vari in varlist:
    ret_list.append(tc.zeros_like(vari))
  return ret_list
def mul_copy_idx(v1list,v2list,idx):
  for i,v2 in enumerate(v2list):
    v1list[i][0,idx] = v2[0,idx]
  return v1list

def plot_accu(x,y,title,save_path):
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  fig.suptitle('Bits and bots', fontsize=20)
  ax.set_title(title,fontsize=23)
  ax.set_ylabel('probability of correct classification',fontsize =18)
  ax.set_xlabel('number of events to detection',fontsize =18)
  ax.set_ylim([0, 1])
  plt.yticks(np.arange(0, 1+0.05, 0.05))
  plt.grid(True)
  ax.scatter(x,y,label="accuracy")
  plt.tight_layout() 
  plt.savefig(save_path)
  plt.show()
  return

def plot_threshold(title,conf_thres_list,n_to_detect_list,accu_list):
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  fig.suptitle('Bits and bots', fontsize=20)
  ax.set_title(title,fontsize=23)
  ax.set_ylabel('probability of correct classification',fontsize =18)
  ax.set_xlabel('threshold',fontsize =18)
  ax.plot( conf_thres_list,accu_list)
  plt.tight_layout() 
  # plt.savefig(f"{title}_{training_len=}_thres_correct.png")
  plt.show()
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  fig.suptitle('Bits and bots', fontsize=20)
  ax.set_title(title,fontsize=23)
  ax.set_ylabel('number of events to detection',fontsize =18)
  ax.set_xlabel('threshold',fontsize =18)
  ax.plot( conf_thres_list,n_to_detect_list)
  plt.tight_layout() 
  # plt.savefig(f"{title}_{training_len=}_thres_n_to_detect.png")
  plt.show()
