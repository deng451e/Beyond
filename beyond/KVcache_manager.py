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
        ######################
        self.blk_idx = [[] for _ in range(num_layers)] 
        self.blk_dim_min_max = [[] for _ in range(num_layers)] 
        
    def clear_kv_cache(self,):
        self.blk_num       = [0 for _ in range( self.num_layers )] 
        self.cpu_kv_flags  = [False for _ in range( self.num_layers )]   
        self.kv_cache = [[None,None,None,None] for _ in range( self.num_layers )] 
        print('all kv cacche has been cleared')


    def __call__(self,idx):
        assert 0<=idx<self.num_layers, f"invalid layer index {idx}"
        return self.kv_cache[idx]
 
    def modify_start_size(self,start_size):     
        for idx in range(self.num_layers):
            self.modify_start_size_by_layer(idx,start_size)
        return     
 
    def modify_recent_size(self,recent_size):     
        for idx in range(self.num_layers):
            self.modify_recent_size_by_layer(idx,recent_size)
        return     

    def modify_recent_size_by_layer(self,idx,recent_size):     
        self.recent_sizes[idx] = recent_size 
        return
    
    def modify_start_size_by_layer(self,idx,start_size ):     
        self.start_sizes[idx]  = start_size 
        return 

    def get_past_key_values_length(self,):
        k_gpu,_,k_cpu,_ =  self.kv_cache[0] 
        hold = k_gpu.size(self.k_seq_dim) if k_gpu is not None else 0
        hold +=  k_cpu.size(self.k_seq_dim) if k_cpu is not None else 0
        return hold

    def add_kv_cache_by_layer(self, idx , kv_cache_2add):
         
        k2add, v2add = kv_cache_2add
        if k2add is None: return  

        k_gpu,v_gpu,k_cpu,v_cpu = self.kv_cache[idx]
         
         
       
        add_len   = k2add.size(self.k_seq_dim)
        gpu_cache_len = k_gpu.size(self.k_seq_dim) if k_gpu is not None else 0 
         
        bound = min(self.recent_sizes[idx]+self.start_sizes[idx],self.gpu_cache_max)
        with torch.cuda.stream(self.copy_stream):
            ##################################################################################################
            # load next layer asynchronously to device 
            if self.gpu_cache_device=='cpu':
                if k_gpu is not None:
                    k_gpu,v_gpu = k_gpu.to(self.gpu_cache_device, non_blocking=True),v_gpu.to(self.gpu_cache_device, non_blocking=True) 


                self.preload_layer_kv(idx,k2add.device)
                k2add,v2add = k2add.to(self.gpu_cache_device, non_blocking=True),v2add.to(self.gpu_cache_device, non_blocking=True) 
                
            ##################################################################################################
                 

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
                    self.cpu_kv_flags[idx] = True
                 
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
                     ## case 5: Appended cache larger than recent window 
                    self.kv_cache[idx] = [
                    #####################
                    #   GPU kv cache    #
                    #####################
                    torch.cat(
                        [
                            self.k_slice(k_gpu, 0, self.start_sizes[idx]),
                            self.k_slice(k2add, add_len - self.recent_sizes[idx] , add_len),
                        ],
                        dim=self.k_seq_dim,
                    ),

                    torch.cat(
                        [
                            self.v_slice(k_gpu, 0, self.start_sizes[idx]),
                            self.v_slice(v2add, add_len - self.recent_sizes[idx] , add_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                    #####################
                    #   CPU kv cache    #
                    #####################
                    torch.cat(
                        [
                            k_cpu,
                            self.k_slice(k2add, 0 , add_len - self.recent_sizes[idx]).to('cpu', non_blocking=True),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            v_cpu,
                            self.v_slice(v2add, 0 , add_len - self.recent_sizes[idx]).to('cpu', non_blocking=True),
                        ],
                        dim=self.v_seq_dim,
                    ),
                    ]
                
 
        return 



    def add_kv_cache(self, kv_cache_2adds):
       
       
        for idx,kv_cache_2add in enumerate(kv_cache_2adds):
            self.add_kv_cache_by_layer(idx,kv_cache_2add)  
        return 
     
    
     
  
    def preload_layer_kv(self,idx,device='cuda'):
         
        idx = (idx+1)%(self.num_layers )
        k_gpu,v_gpu,k_cpu,v_cpu = self.kv_cache[idx]
        
        if k_gpu is not None:
            self.kv_cache[idx] = [k_gpu.to(device, non_blocking=True),v_gpu.to(device, non_blocking=True),k_cpu,v_cpu]
        return 
        
    
    

    def print_coverage(self,idx=None):
        if idx is not None:
            print(f"Layer {idx}: start size:{self.start_sizes[idx]}, recent_size: {self.recent_sizes[idx]}")
        else:
            for idx in range( self.num_layers):
                print(f"Layer {idx}: start size:{self.start_sizes[idx]}, recent_size: {self.recent_sizes[idx]}")

        return 


    def print_blk_info(self,idx=None):
        if idx is not None:
            print(f"CPU blocks at Layer {idx}:{self.blk_num[idx]}")
        else:
            for idx in range( self.num_layers):
                print(f"CPU blocks at Layer {idx}:{self.blk_num[idx]}")
         
        
        return 


    def update_blk(self,):
        for idx  in range(self.num_layers):
            self.update_blk_by_layer(idx)  
        return 
     
    def get_blk_info(self,idx):
        return  self.blk_idx[idx],self.blk_dim_min_max[idx] 


    def update_blk_by_layer(self,idx):
        '''
        tracking the coarse grained K cache info
        '''
         
    
        if self.cpu_kv_flags[idx]:
            with torch.cuda.stream(self.copy_stream):
                if  not self.blk_idx[idx]:
                    
                    k_cpu  = self.kv_cache[idx][2]
                    k_cpu_shape = k_cpu.size()
                    cache_len = k_cpu_shape[self.k_seq_dim]
                    pad_len = self.blk_size-cache_len%self.blk_size

                    self.blk_num[idx] = (cache_len+self.blk_size-1)//self.blk_size
                    self.blk_idx[idx] = [(start,min(cache_len,start+self.blk_size)) for start in range(0,cache_len,self.blk_size)]
                
                    pad_ = ([0 for _ in range((k_cpu.dim()-self.k_seq_dim-1)*2)] + [0,pad_len])
                    reshape_ = [k_cpu_shape[Dim] for Dim in range(k_cpu.dim())]
                    reshape_[self.k_seq_dim] =  self.blk_size
                    reshape_.insert(self.k_seq_dim,-1 )   
                    reshape_ = tuple(reshape_)
                    
                    #b,h,s//blks,blks,d
                    blk = F.pad( k_cpu, pad=pad_, value=0).reshape(reshape_)
                        
                    
                    #b,h,s//blks,2,d
                    self.blk_dim_min_max[idx] = [
                        blk.min(dim=self.k_seq_dim+1)[0],
                        blk.max(dim=self.k_seq_dim+1)[0]
                        ] 
                else:
                    
                    pre_start = self.blk_idx[idx][-1][0]
                   
                    k_cpu = self.k_slice(self.kv_cache[idx][2],pre_start,None) 
                
                    k_cpu_shape = k_cpu.size()
                    
                    cache_len = k_cpu_shape[self.k_seq_dim]
                    pad_len = self.blk_size-cache_len%self.blk_size
                    self.blk_num[idx]      += (cache_len+self.blk_size-1)//self.blk_size-1
                    self.blk_idx[idx] = self.blk_idx[idx][:-1]
                    self.blk_idx[idx] += [(start,min(pre_start+cache_len,start+self.blk_size)) for start in range(pre_start,pre_start+cache_len,self.blk_size)]
            

                    pad_ = ([0 for _ in range((k_cpu.dim()-self.k_seq_dim-1)*2)] + [0,pad_len])
                    reshape_ = [k_cpu_shape[Dim] for Dim in range(k_cpu.dim())]
                    reshape_[self.k_seq_dim] =  self.blk_size
                    reshape_.insert(self.k_seq_dim,-1 )   
                    reshape_ = tuple(reshape_)
                    
                    #b,h,s//blks,blks,d
                    blk = F.pad( k_cpu, pad=pad_, value=0).reshape(reshape_)
                        
                    blk_min, blk_max  = self.blk_dim_min_max[idx]  
                    #b,h,s//blks,2,d
                   
                    self.blk_dim_min_max[idx] = [
                        torch.cat([self.k_slice(blk_min,None,-1),blk.min(dim=self.k_seq_dim+1)[0]],dim=self.k_seq_dim),
                        torch.cat([self.k_slice(blk_max,None,-1),blk.max(dim=self.k_seq_dim+1)[0]],dim=self.k_seq_dim)
                        ] 
        return 
    


 













#####################
#      For Test     # 
#####################

 



def test_correctness(args,log):
    gpu_cache_device = "cuda"
    num_layers=32; seq_len = args.seq_len;num_heads=32;head_dim=128; batch_size=args.batch_size
   
    batch_size = args.batch_size; block_size=args.block_size
     
    KVCache_manager =  KVCache_manager_( 
                copy_stream=torch.cuda.Stream(),
                block_size=block_size,
                gpu_cache_device=gpu_cache_device,
                num_layers=num_layers)
    
    st = time.time()
    # test case 1: init kv cache 
    k_cache = torch.randn(batch_size,num_heads, seq_len,head_dim, device=gpu_cache_device).half()
    v_cache = torch.randn(batch_size,num_heads, seq_len,head_dim, device=gpu_cache_device).half()
    past_key_values =  [[k_cache,v_cache] for _ in range(num_layers)]
    KVCache_manager.add_kv_cache(past_key_values)
     
    KVCache_manager.update_blk()
    
    # test case 2: add kv cache
    for add_len in [1,10,50,60,100,1000]:
        k_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()
        v_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()

        past_key_values2add =  [[k_cache2add,v_cache2add] for _ in range(num_layers)]
         
        KVCache_manager.add_kv_cache(past_key_values2add)
        KVCache_manager.update_blk()
       
        KVCache_manager.print_blk_info(0)
    
    print("====================================")
    
    # test case 3: add kv cache by layer
    for add_len in [1,10,50,60,100,200,1000]:
        for idx in range(num_layers):
            k_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()
            v_cache2add = torch.randn(batch_size, num_heads, add_len,head_dim, device=gpu_cache_device).half()
            past_key_values2add =  [ k_cache2add,v_cache2add]
            
            KVCache_manager.add_kv_cache_by_layer(idx,past_key_values2add)
            # KVCache_manager.update_blk_by_layer(idx)
        
        KVCache_manager.print_blk_info(0)
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
     