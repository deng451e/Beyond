
import torch  
 
import argparse
 
import os.path as osp
import ssl
import urllib.request
import os
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

 
def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(file_path,):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict



def add_info(args,log):
    for arg, value in vars(args).items():
        log += f"{arg}:{value}, "
    
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
