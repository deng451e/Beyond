import torch
import time 
from beyond.loading import *
file_path = "/home/c3/code/Beyond/run-models/kv_manager_InitConfig/llama-vicuna-13b-v1.3.json"
file_path = "/home/c3/code/Beyond/run-models/data/mt_bench.jsonl"
config = load_json(file_path )
print(config)

# with open(file_path, 'r') as file:
#     data = json.load(file)
# print(data)
#   "start_sizes":64,
#     "recent_sizes":1024,
#     "cpu_attn_size":100,
#     "gpu_cache_max":2000,
#     "block_size":10,
   
# Define student_details dictionary
# config = dict()
# config["gpu_cache_device"] = 'cuda'
# for i in range(40):
#     layer = {}
#     layer["start_sizes"] =10
#     layer["recent_sizes"] = 1000
#     layer["cpu_attn_size"] = 100
#     layer["gpu_cache_max"] = 2000
#     layer["block_size"] = 10
#     config[f"layer {i}"] = layer

# # Convert and write JSON object to file
# with open(file_path, "w") as outfile: 
#     json.dump(config, outfile, indent = 4)
