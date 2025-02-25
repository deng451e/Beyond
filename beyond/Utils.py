
import torch  
import pandas as pd
 
  
# Define a function to parse the file
def parse_file(filename):
    data = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Remove whitespace and split key-value pairs
            entries = line.strip().split(',')

            # Create a dictionary for each line
            line_data = {}
            for entry in entries:
                key, value = entry.split(':')
                try:
                    # Attempt to convert numeric values (int or float)
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string if not numeric
                    pass
                line_data[key] = value

            # Add dictionary to list
            data.append(line_data)

    return pd.DataFrame(data)
 

def print_args_info(args):
    log = "====================================================\n"
    cnt = 1 
    for arg, value in vars(args).items():
        log += f"{arg}:{value}, "
        if cnt%4==0: log += "\n"
        cnt +=1 
    log += "\n===================================================="
    print(log)
    return   
 

def check_mem(x,name):
    print(f"{name} is pinned: {x.is_pinned()}")
    print(f"{name} is contiguous: {x.is_contiguous()}")
    print(f"{name} dtype: {x.dtype}")
    print(f"{name} device: {x.device}")
    print('=================')


def check_eq(x,y):
    return (torch.isclose(x.cpu(), y.cpu(),  atol=5e-4, rtol=1e-2 ).sum()/torch.numel(x)).cpu().numpy()
  



def check_tensor_device(x,type_):

    return x.device.type==type_
 


 
def get_device_map(model):
    device_map = {}
    for name, module in model.named_parameters():
        if "attn" in name.lower() or "attention" in name.lower():
            device = module.device
            device_map[name.split(".")[2]] = str(device)
    
    return device_map 