import torch
import torch.nn.functional as F
import argparse
from beyond.utils import * 
import asyncio
import time 
 

class KV_Cache_Dispatch_:
    def __init__(
        self,
        start_size=40,
        recent_size=1024,
        block_size=100,
        k_seq_dim=1,
        v_seq_dim=1,
        num_heads=32,
        head_dim=128,
         
    ):
        print(f"CPU KVCache: {start_size}~{-recent_size}") 
        print(f"default CPU block size:{block_size}")
        print(f"GPU KVCache: 0~{start_size} and -{recent_size}~...")
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        # k_gpu,v_gpu,k_cpu,v_cpu
        self.kv_cache = None 
        # statics  for tracking gpu kv blocks   
        self.start_size = start_size
        self.recent_size = recent_size
        self.gpu_cache_size = start_size + recent_size
        self.blk_size  = block_size
        self.gpu_cache_device = "cpu" 
        # statics  for tracking cpu kv blocks   
        self.cpu_kv_flag = False 
        self.blk_tracker = None 
        self.blk_num   = 0
        self.blk_min_max = None 
         
        self.copy_steam  =  torch.cuda.Stream()
    def __call__(self,):
        return self.kv_cache
              
    
    def init_kv_cache(self, kv_cache_2add):
        seq_len = kv_cache_2add[0][0].size(self.k_seq_dim)
        if not kv_cache_2add: return  
        if seq_len <= self.gpu_cache_size: self.kv_cache = [[k,v,None,None] for k,v in kv_cache_2add]
        if seq_len - self.recent_size> self.start_size: self.cpu_kv_flag=True 

        self.kv_cache = [
            [
                #####################
                #   GPU kv cache    #
                #####################
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
                #####################
                #   CPU kv cache    #
                #####################
                self.k_slice(k, self.start_size , seq_len - self.recent_size) if  self.cpu_kv_flag  else  None ,
                self.v_slice(v, self.start_size , seq_len - self.recent_size) if  self.cpu_kv_flag  else  None
            
            
            ]
            for k, v in kv_cache_2add
        ]
        
      
        # self.update_blk_tracker()
        return 
    

    def add_kv_cache(self, kv_cache_2add):
        gpu_cache_len = self.kv_cache[0][0].size(self.k_seq_dim)
        add_len   = kv_cache_2add[0][0].size(self.k_seq_dim)
       
        evict_len = add_len - (self.gpu_cache_size-gpu_cache_len) 

        if add_len<self.recent_size:
            with torch.cuda.stream(self.copy_steam):
                self.kv_cache = [
                [   
                    #####################
                    #   GPU kv cache    #
                    #####################
                    torch.cat(
                        [
                            self.k_slice(k_gpu, 0, self.start_size),
                            self.k_slice(k_gpu, gpu_cache_len + evict_len - self.recent_size, gpu_cache_len),
                            k2add,
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v_gpu, 0, self.start_size),
                            self.v_slice(v_gpu, gpu_cache_len + evict_len - self.recent_size, gpu_cache_len),
                            v2add
                        ],
                        dim=self.v_seq_dim,
                    ),
                    #####################
                    #   CPU kv cache    #
                    #####################
                    torch.cat(
                        [
                            k_cpu,
                            self.k_slice(k_gpu, self.start_size ,  gpu_cache_len + evict_len - self.recent_size).to('cpu', non_blocking=True),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            v_cpu,
                            self.k_slice(k_gpu,  self.start_size ,  gpu_cache_len + evict_len - self.recent_size).to('cpu', non_blocking=True),
                            
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for (k_gpu,v_gpu,k_cpu,v_cpu),(k2add, v2add) in zip(self.kv_cache,kv_cache_2add)
                ]
        self.update_blk_tracker()
        
        return 
    
    def print_kv_info(self,):
        print(f"GPU kv cache size:{ self.kv_cache[0][0].size(self.k_seq_dim)}")
        if  self.cpu_kv_flag: print(f"CPU kv cache size:{ self.kv_cache[0][2].size(self.k_seq_dim)}")
        return 
    def print_block_info(self,):
        print(f"number of blocks under tracking:{self.blk_num}")
        return 
    

    async def update_blk_tracker(self,):
        # b s h d 
        cpu_cache_len = self.kv_cache[0][2].size(self.k_seq_dim)
        self.blk_num = (cpu_cache_len+self.blk_size-1)//self.blk_size
        pad_len =  self.blk_size - cpu_cache_len% self.blk_size
        if self.k_seq_dim==1:
            k_cache_cpu = [F.pad(k,(0,0,0,0,0,pad_len)).reshape) for _,_,k,_ in self.kv_cache]

        

 
 









def test_correctness(args,log):
    log = add_info(args,log)
    seq_len = 10000;num_heads=32;head_dim=128; batch_size=10
    k_cache = torch.randn(batch_size, seq_len, num_heads,head_dim, device='cpu').half()
    v_cache = torch.randn(batch_size, seq_len, num_heads,head_dim, device='cpu').half()
     

    KV_Cache_Dispatch =  KV_Cache_Dispatch_()
    past_key_values =  [[k_cache,v_cache] for _ in range(10)]
    KV_Cache_Dispatch.init_kv_cache(past_key_values)

    for add_len in [1,2,5,10]:
        k_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device='cpu').half()
        v_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device='cpu').half()

        past_key_values2add =  [[k_cache2add,v_cache2add] for _ in range(10)]
         
        KV_Cache_Dispatch.add_kv_cache(past_key_values2add)
        KV_Cache_Dispatch.print_kv_info()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_name", type=str, default="opt-13b") 
    parser.add_argument("--seq_len", type=int, default=100000)
    parser.add_argument("--q_len", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    test_correctness(args,"")
     