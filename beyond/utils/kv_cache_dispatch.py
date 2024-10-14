import torch
import argparse
from utils import * 


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

 

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class KV_Cache_Dispatch:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        block_size=10,
        k_seq_dim=2,
        v_seq_dim=2,
         
    ):
        print(f"CPU KVCache: {start_size}~{recent_size}, default block size:{block_size}")
        print(f"GPU KVCache: 0~{start_size} and {recent_size}~...")
        self.start_size = start_size
        self.recent_size = recent_size
        self.gpu_cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.blk_size  = block_size
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.cpu_cache = None 
        self.gpu_cache = None
        self.cpu_kv_cache_tracker = []
        self.blk_num   = 0
        self.blk_dim_min = None 
        self.blk_dim_max = None 
    def __call__(self,device):
        if device =='cpu': return self.gpu_cache
        if device =='gpu': return self.cpu_cache
              
    
    def init_kv_cache(self, past_key_values):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.gpu_cache_size:
            self.gpu_cache.append(past_key_values)
        self.gpu_cache = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size , seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size , seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
    def add_kv_cache(self, cache2add):
        add_len = cache2add[0][0].size(self.k_seq_dim)
        if add_len>self.recent_size:
                 self.gpu_cache = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size , seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size , seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


       
    async def process_cpu_cache(cache2add=None):
        if not cache2add:
            cached_len =  self.cpu_cache[0][0].size(self.k_seq_dim)
                
        else:
            self.cpu_cache = [
                [
                    torch.cat([k,k2add], dim=self.k_seq_dim,),
                    torch.cat([v,v2add], dim=self.v_seq_dim,)
                
                ]
                for k,v,k2add, v2add in zip(self.cpu_cache,cache2add)
            ]
        self.blk_num = (cached_len+self.blk_size-1)//self.blk_size



 
 









def test_correctness(args,log):
    log = add_info(args,log)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_name", type=str, default="opt-13b") 
    parser.add_argument("--seq_len", type=int, default=100000)
    parser.add_argument("--q_len", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    test_correctness(args,"")
     