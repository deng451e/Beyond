
import torch  
 

def add_info(args,log):
    log += f"arch name:{args.arch_name}, "
    log += f"seq len:{args.seq_len}, "
    log += f"q len:{args.q_len}, "
    log += f"cpu ratio:{args.ratio}, "
    return log 

def check_workspace(ratio):
    if 0<ratio<1:
        print("mix attention...")
    elif ratio==0:
        print("gpu attention...")
    else:
        print("cpu attention...")

def check_memory(x,name):
    print(f"{name} is pinned: {x.is_pinned()}")
    print(f"{name} is contiguous: {x.is_contiguous()}")
    print(f"{name} dtype: {x.dtype}")
    print(f"{name} device: {x.device}")
    print('=================')

def check_eq(x,y):
    return (torch.isclose(x.cpu(), y.cpu(), rtol=1e-3, atol=1e-3).sum()/torch.numel(x)).cpu().numpy()
    
def check_dtype(x,type_):
    return type(x.dtype)==type(type_)



def check_tensor_device(x,type_):

    return x.device.type==type_
 




def slice0d(x, start, end):
    return x[start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

 

DIM_TO_SLICE = {
    0: slice0d,
    1: slice1d,
    2: slice2d,
    3: slice3d,
}
