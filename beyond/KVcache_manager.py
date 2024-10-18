import torch
import torch.nn.functional as F
import argparse
from beyond.utils import * 
import asyncio
import time 
 

class KVCache_manager_:
    def __init__(
        self,
        start_size=40,
        recent_size=1024,
        block_size=100,
        k_seq_dim=1,
        v_seq_dim=1,
        head_dim=128,
        num_heads=32,
        num_layers=32,
        gpu_cache_max=2000,
        batched=True,
        gpu_cache_device="cpu",
    ):
        print(f"GPU attn device:{gpu_cache_device}")
        print(f"Default CPU block size:{block_size}")
       
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.batched = batched
        self.num_layers = num_layers
        # k_gpu,v_gpu,k_cpu,v_cpu
        self.kv_cache = [[None,None,None,None] for _ in range(num_layers)] 
        # statics  for tracking gpu kv blocks   
        self.start_sizes  = [start_size  for _ in range(num_layers)]  
        self.recent_sizes = [recent_size  for _ in range(num_layers)] 
        self.gpu_cache_max = gpu_cache_max
        self.blk_size  = block_size
        self.gpu_cache_device = gpu_cache_device 
        # statics  for tracking cpu kv blocks   
        self.cpu_kv_flags = [False for _ in range(num_layers)]   
        self.blk_tracker = None 
        self.blk_num   = [0 for _ in range(num_layers)] 
        self.blk_min_max = None 
         
        self.copy_steam  =  torch.cuda.Stream()

    

    def __call__(self,idx):
        assert idx<self.num_layers, f"invalid layer index {idx}"
        return self.kv_cache[idx]

    def usecase_warn(self,f,log):
        if f: print(f"warning: [{log}]")

    def modify_recent_size_by_layer(self,idx,recent_size):     
        self.recent_sizes[idx] = recent_size 

    
    def modify_start_size_by_layer(self,idx,recent_size ):     
        self.start_sizes[idx]  = start_size 
     


    def add_kv_cache_by_layer(self, idx , kv_cache_2add):
         
        
         

        k_gpu,v_gpu,k_cpu,v_cpu = self.kv_cache[idx]
        k2add, v2add = kv_cache_2add
        if k2add is None: return 
        add_len   = k2add.size(self.k_seq_dim)
        gpu_cache_len = k_gpu.size(self.k_seq_dim) if k_gpu is not None else 0 
        # self.usecase_warn(add_len>self.start_sizes[idx],f"query length > recent window at layer {idx}")
        thresh = min(self.recent_sizes[idx]+self.start_sizes[idx],self.gpu_cache_max)
        with torch.cuda.stream(self.copy_steam):


            if gpu_cache_len==0:
        
            
                ## case 1: no previous cache, added cache smaller than allowed
                if add_len <= thresh: 
                    self.kv_cache[idx] = [k2add, v2add,None,None]  
                    return
                
                ## case 2:  no prevrious and added cache bigger than allowed, evict part to CPU
                self.kv_cache[idx] = [

                    #####################
                    #   GPU kv cache    #
                    #####################
                    torch.cat(
                        [
                            self.k_slice(k_gpu, 0, self.start_sizes[idx]),
                            self.k_slice(v_gpu, add_len - self.recent_sizes[idx] , add_len),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(k_cpu, 0, self.start_sizes[idx]),
                            self.v_slice(v_cpu, add_len - self.recent_sizes[idx] , add_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                    #####################
                    #   CPU kv cache    #
                    #####################
                    self.k_slice(k2add, self.start_sizes[idx] , add_len - self.recent_sizes[idx]).to('cpu', non_blocking=True),
                    self.v_slice(v2add, self.start_sizes[idx] , add_len - self.recent_sizes[idx]).to('cpu', non_blocking=True)
                ]
                
                
            else:
                ## case 3: previous cache + added cache smaller than allowed
                if add_len+gpu_cache_len<= thresh:
                    
                    
                    self.kv_cache[idx] = [
                    
                        #####################
                        #   GPU kv cache    #
                        #####################
                        torch.cat(
                            [
                                k_gpu,
                                k2add,
                            ],
                            dim=self.k_seq_dim,
                        )  ,
                        torch.cat(
                            [
                                v_gpu,
                                v2add
                            ],
                            dim=self.v_seq_dim,
                        ) ,
                        #####################
                        #   CPU kv cache    #
                        #####################
                        k_cpu,
                        v_cpu,
                
                    ]
                ## case 4: previous cache + added cache bigger than allowed, added cache smaller than recent window size
                elif add_len<  self.recent_sizes[idx] :
                    
                    evict_len = add_len - (thresh-gpu_cache_len) 
                    self.cpu_kv_flags[idx] = True
                
                    self.kv_cache[idx] = [
                        
                        #####################
                        #   GPU kv cache    #
                        #####################
                        torch.cat(
                            [
                                self.k_slice(k_gpu, 0, self.start_sizes[idx]),
                                self.k_slice(k_gpu, self.start_sizes[idx] + evict_len, gpu_cache_len),
                                k2add,
                            ],
                            dim=self.k_seq_dim,
                        ) ,
                        torch.cat(
                            [
                                self.v_slice(v_gpu, 0, self.start_sizes[idx]),
                                self.v_slice(v_gpu, self.start_sizes[idx] + evict_len, gpu_cache_len),
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
                                self.k_slice(k_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                            ],
                            dim=self.k_seq_dim,
                        ) if k_cpu is not None else self.k_slice(k_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                        torch.cat(
                            [
                                v_cpu,
                                self.k_slice(k_gpu,  self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                                
                            ],
                            dim=self.v_seq_dim,
                        ) if v_cpu is not None else self.v_slice(v_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                    
                    
                    ]

        self.update_blk_tracker_by_layer(idx)
        return 



    def add_kv_cache(self, kv_cache_2adds):
       
       
        for idx,kv_cache_2add in enumerate(kv_cache_2adds):
            self.add_kv_cache_by_layer(idx,kv_cache_2add)  
        return 
     
    
    
    def update_blk_tracker_by_layer(self,idx):
        
        
        # to be optimized ...
        if self.cpu_kv_flags[idx]:
            cpu_cache_len = self.kv_cache[idx][2].size(self.k_seq_dim)
            pad_len = self.blk_size-cpu_cache_len%self.blk_size
            self.blk_num[idx] = (cpu_cache_len+self.blk_size-1)//self.blk_size
            if self.batched: batch_size  = self.kv_cache[idx][2].size(0)
            if self.k_seq_dim==1:
                blk = F.pad( self.kv_cache[idx][2], pad=(0,0,0,0,0, pad_len), value=0).reshape(batch_size,-1,self.blk_size,self.num_heads,self.head_dim)
                    
                
                self.blk_dim_min_max = [
                    blk.min(dim=self.k_seq_dim+1)[0],
                    blk.max(dim=self.k_seq_dim+1)[0]
                    ] 
                     
        return 
  
      
 
    def print_gpu_coverage(self,idx=None):
        def print_(idx):
            k_gpu,v_gpu,k_cpu,v_cpu = self.kv_cache[idx]
            gpu_len = k_gpu.size(self.k_seq_dim) if k_gpu is not None else 0 
            cpu_len = k_cpu.size(self.k_seq_dim) if k_cpu is not None else 0 
            print(f"layer:{idx}, GPU cache size:{gpu_len}, CPU cache size:{cpu_len}")
        if idx is not None:  
            print_(idx)
        else:
            for idx in range( self.num_layers):
                 print_(idx)
 
        return 

    def print_block_info(self,idx=None):
        if idx is not None:
            print(f"CPU blocks at Layer {idx}:{self.blk_num[idx]}")
        else:
            for idx in range( self.num_layers):
                print(f"CPU blocks at Layer {idx}:{self.blk_num[idx]}")
         
        
        return 









def test_correctness(args,log):
    gpu_cache_device = "cuda"
    num_layers=32; seq_len = args.seq_len;num_heads=32;head_dim=128; batch_size=args.batch_size
    k_cache = torch.randn(batch_size, seq_len, num_heads,head_dim, device=gpu_cache_device).half()
    v_cache = torch.randn(batch_size, seq_len, num_heads,head_dim, device=gpu_cache_device).half()
    batch_size = args.batch_size; block_size=args.block_size

    KVCache_manager =  KVCache_manager_( block_size=block_size,gpu_cache_device=gpu_cache_device,num_layers=num_layers)
    past_key_values =  [[k_cache,v_cache] for _ in range(num_layers)]


    # test case 1: init kv cache 
    KVCache_manager.add_kv_cache(past_key_values)
    st = time.time()
    
    
    # test case 2: add kv cache
    for add_len in [1,10,50,60,100,1000]:
        k_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device=gpu_cache_device).half()
        v_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device=gpu_cache_device).half()

        past_key_values2add =  [[k_cache2add,v_cache2add] for _ in range(num_layers)]
         
        KVCache_manager.add_kv_cache(past_key_values2add)
    KVCache_manager.print_gpu_coverage()
       
    
    print("====================================")
    # print(KVCache_manager(0)[2].device)
    # test case 3: add kv cache by layer
    for add_len in [1,10,50,60,100,1000]:
        for idx in range(num_layers):
            k_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device=gpu_cache_device).half()
            v_cache2add = torch.randn(batch_size, add_len, num_heads,head_dim, device=gpu_cache_device).half()

            past_key_values2add =  [ k_cache2add,v_cache2add]
            
            KVCache_manager.add_kv_cache_by_layer(idx,past_key_values2add)
        KVCache_manager.print_gpu_coverage(0)
        KVCache_manager.print_block_info(0)
    print("====================================")
    # print(KVCache_manager(0)[2].device)
    print(f"time taken:{time.time()-st}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--q_len", type=int, default=10)
    args = parser.parse_args()
    test_correctness(args,"")
     