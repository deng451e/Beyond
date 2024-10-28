import torch
import torch.nn.functional as F
import argparse
from beyond.utils import * 
import asyncio
import time 
 

class KVCache_manager_:
    def __init__(
        self,
        copy_stream=None,
        start_size=40,
        recent_size=1024,
        block_size=100,
        k_seq_dim=2,
        v_seq_dim=2,
        head_dim=128,
        num_heads=32,
        num_layers=32,
        gpu_cache_max=5000,
        cpu_attn_size=1000,
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
        self.num_layers = num_layers
        self.copy_stream  =  copy_stream

        #( k_gpu,v_gpu,k_cpu,v_cpu)
        self.kv_cache = [[None,None,None,None] for _ in range(num_layers)] 
        
        # statics  for tracking gpu kv blocks   
        self.gpu_cache_max = gpu_cache_max
        self.blk_size  = block_size
        self.gpu_cache_device = gpu_cache_device # device for offload gpu attn KV cache
        self.start_sizes   = [start_size  for _ in range(num_layers)]  
        self.recent_sizes  = [recent_size  for _ in range(num_layers)] 
        
        
        # statics  for tracking cpu kv blocks
        self.cpu_kv_flags  = [False for _ in range(num_layers)]   
        self.cpu_attn_sizes = [cpu_attn_size for _ in range(num_layers)] 
        self.blk_num       = [0 for _ in range(num_layers)] 
        self.blk_tracker = None 
        self.blk_min_max = None 
         
         
 
    def __call__(self,idx):
        assert 0<=idx<self.num_layers, f"invalid layer index {idx}"
        return self.kv_cache[idx]

    def usecase_warn(self,f,log):
        if f: print(f"warning: [{log}]")

    def modify_recent_size_by_layer(self,idx,recent_size):     
        self.recent_sizes[idx] = recent_size 

    
    def modify_start_size_by_layer(self,idx,start_size ):     
        self.start_sizes[idx]  = start_size 
     


    def add_kv_cache_by_layer(self, idx , kv_cache_2add):
         
        k2add, v2add = kv_cache_2add
        if k2add is None: return  

        k_gpu,v_gpu,k_cpu,v_cpu = self.kv_cache[idx]
        
        add_len   = k2add.size(self.k_seq_dim)
        gpu_cache_len = k_gpu.size(self.k_seq_dim) if k_gpu is not None else 0 
         
        bound = min(self.recent_sizes[idx]+self.start_sizes[idx],self.gpu_cache_max)
        with torch.cuda.stream(self.copy_stream):

            if gpu_cache_len==0:
        
            
                ## case 1: no previous cache, added cache smaller than allowed on GPU
                if add_len <= bound: 
                   
                    self.kv_cache[idx] = [k2add, v2add,None,None]  
                      
                
                ## case 2:  no prevrious and added cache bigger than allowed on GPU, evict part to CPU
                else:
                    self.kv_cache[idx] = [

                        #####################
                        #   GPU kv cache    #
                        #####################
                        torch.cat(
                            [
                                self.k_slice(k2add, 0, self.start_sizes[idx]),
                                self.k_slice(k2add, add_len - self.recent_sizes[idx] , add_len),
                            ],
                            dim=self.k_seq_dim,
                        ),

                        torch.cat(
                            [
                                self.v_slice(v2add, 0, self.start_sizes[idx]),
                                self.v_slice(v2add, add_len - self.recent_sizes[idx] , add_len),
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
                
                ## case 3: previous cache + added cache smaller than allowed on GPU
                if add_len+gpu_cache_len<= bound:
                    
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
                 
                elif add_len<  self.recent_sizes[idx] :
                    
                     
                    evict_len = add_len - (bound-gpu_cache_len) 
                  
                     
                    ## case 4: previous cache + added cache bigger than allowed on GPU, added cache smaller than recent window size
                    #####################
                    #   GPU kv cache    #
                    #####################
                    self.kv_cache[idx][:2] = [
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
                        )    ]
                    #####################
                    #   CPU kv cache    #
                    #####################
                    ## case 4.1: + have previous cpu cache
                    if self.cpu_kv_flags[idx]:
                        self.kv_cache[idx][2:] =[
                            torch.cat(
                                [
                                    k_cpu,
                                    self.k_slice(k_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                                ], dim=self.k_seq_dim),
                            torch.cat(
                                [
                                    v_cpu,
                                    self.v_slice(v_gpu,  self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),  
                                ], dim=self.v_seq_dim) ]
                    ## case 4.2: + no previous cpu cache
                    else:
                        self.kv_cache[idx][2:] =[
                            self.k_slice(k_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True),
                            self.v_slice(v_gpu, self.start_sizes[idx] , self.start_sizes[idx] + evict_len).to('cpu', non_blocking=True)]
                        self.cpu_kv_flags[idx] = True
                else:
                        print("case 5")

        # if idx==self.num_layers-1:
        #     torch.cuda.synchronize()

        # self.update_blk_tracker_by_layer(idx)
        return 



    def add_kv_cache(self, kv_cache_2adds):
       
       
        for idx,kv_cache_2add in enumerate(kv_cache_2adds):
            self.add_kv_cache_by_layer(idx,kv_cache_2add)  
        return 
     
    
    
    def update_blk_tracker_by_layer(self,idx):
        
        
        # to be optimized ...
        if self.cpu_kv_flags[idx]:
            k_cpu  = self.kv_cache[idx][2]
            k_cpu_shape = k_cpu.size()
            cache_len = k_cpu_shape[self.k_seq_dim]
            pad_len = self.blk_size-cache_len%self.blk_size
            self.blk_num[idx] = (cache_len+self.blk_size-1)//self.blk_size
           
           
            pad_ = ([0 for _ in range((k_cpu.dim()-self.k_seq_dim-1)*2)] + [0,pad_len])
            reshape_ = [k_cpu_shape[i] for i in range(k_cpu.dim())]
            reshape_[self.k_seq_dim] =  self.blk_size
            reshape_.insert(self.k_seq_dim,-1 )   
            reshape_ = tuple(reshape_)
            
            blk = F.pad( k_cpu, pad=pad_, value=0).reshape(reshape_)
                
            
            self.blk_dim_min_max = [
                blk.min(dim=self.k_seq_dim+1)[0],
                blk.max(dim=self.k_seq_dim+1)[0]
                ] 
                     
        return 
  
      
 
    def print_kv_static(self,idx=None):
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
    

    def print_coverage(self,idx=None):
        if idx is not None:
            print(f"GPU coverage at Layer {idx}:{self.start_sizes[idx]}~{self.recent_sizes[idx]}")
        else:
            for idx in range( self.num_layers):
                print(f"GPU coverage at Layer {idx}:{self.start_sizes[idx]}~{self.recent_sizes[idx]}")

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
   
    batch_size = args.batch_size; block_size=args.block_size
    copy_stream = torch.cuda.Stream()
    KVCache_manager =  KVCache_manager_( 
                copy_stream=copy_stream,
                block_size=block_size,
                gpu_cache_device=gpu_cache_device,
                num_layers=num_layers)
    
    st = time.time()
    # test case 1: init kv cache 
    k_cache = torch.randn(batch_size,num_heads, seq_len,head_dim, device=gpu_cache_device).half()
    v_cache = torch.randn(batch_size,num_heads, seq_len,head_dim, device=gpu_cache_device).half()
    past_key_values =  [[k_cache,v_cache] for _ in range(num_layers)]
    KVCache_manager.add_kv_cache(past_key_values)
     
    
    
    # test case 2: add kv cache
    for add_len in [1,10,50,60,100,1000]:
        k_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()
        v_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()

        past_key_values2add =  [[k_cache2add,v_cache2add] for _ in range(num_layers)]
         
        KVCache_manager.add_kv_cache(past_key_values2add)
        KVCache_manager.print_gpu_coverage(0)
       
    
    print("====================================")
    
    # test case 3: add kv cache by layer
    for add_len in [1,10,50,60,100,1000]:
        for idx in range(num_layers):
            k_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()
            v_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()

            past_key_values2add =  [ k_cache2add,v_cache2add]
            
            KVCache_manager.add_kv_cache_by_layer(idx,past_key_values2add)
        KVCache_manager.print_gpu_coverage(0)
        KVCache_manager.print_block_info(0)
    print("====================================")
   
    print(f"time taken:{time.time()-st}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--q_len", type=int, default=10)
    args = parser.parse_args()
    test_correctness(args,"")
     