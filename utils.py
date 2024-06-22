def to_devices(varlist,device):
  ret_list = []
  for  vari in varlist:
    ret_list.append(vari.to(device))
  return ret_list