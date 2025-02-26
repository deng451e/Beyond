import torch
import threading

   

class KVCacheManager:
    def __init__(
        self,
        config,  
        device_map,
        batch_size,
        enable_pos=False, 
        seq_dim=-2,
        max_blk_gpu=8,
        max_blk2gpu=0,
        block_size=128):

        self.model = config.model_type
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size//self.head_num  
        self.layer_num = config.num_hidden_layers
       
        self.enable_pos = enable_pos
         
        

        self.seq_dim = seq_dim
        self.batch_size = batch_size 
        self.blk_size = block_size
         
        self.max_blk_gpu = max_blk_gpu # per requests 
        self.max_blk2gpu = max_blk2gpu # per requests 
         
        self.seq_max = block_size*max_blk_gpu
        self.seq2gpu = block_size*max_blk2gpu
        
        self.beta  = 0.5 # Append heatmap factor
        self.alpha = 0.5 # Update heatmap factor
        self.tasks = []  
        

        # data structure for cpu cache
        self.cache2gpu = (None,None)
        self.cache_cpu = [(None,None) for _ in range(self.layer_num)] 
        self.req_cache = [([],[]) for _ in range(self.layer_num)] 
        self.req_heatmaps  = [[] for _ in range(self.layer_num)]
       
        self.seq_ptrs_gpu = [0 for _ in range(self.layer_num)] # pointer on (bh,@s,d)
        self.gpu_pos_offsets = [None for _ in range(self.layer_num)] 
        self.cpu_pos_ids = [[] for _ in range(self.layer_num)]  
        # trace per device     
        self.copy_streams = []
        self.cache_gpu = []
        self.cache_map = {}
        self.init_workspace(device_map)
     

    def init_workspace(self,device_map):
      

        device = device_map["0"] 
        device_offset = 0
        layer_offset = 0 
        for layer_idx in range(self.layer_num):
            if device_map[f"{layer_idx}"]!=device:
                k = torch.empty(layer_offset,self.batch_size*self.head_num,self.seq_max,self.head_dim,device=device,dtype=torch.float16)
                v = torch.empty(layer_offset,self.batch_size*self.head_num,self.seq_max,self.head_dim,device=device,dtype=torch.float16)
                self.cache_gpu.append((k,v))
                self.copy_streams.append(torch.cuda.Stream(device=device))
                device = device_map[f"{layer_idx}"] 
                layer_offset = 0
                device_offset += 1 

            self.cache_map[layer_idx] = (device_offset,layer_offset)
            layer_offset += 1 

        k = torch.empty(layer_offset,self.batch_size*self.head_num,self.seq_max,self.head_dim,device=device,dtype=torch.float16)
        v = torch.empty(layer_offset,self.batch_size*self.head_num,self.seq_max,self.head_dim,device=device,dtype=torch.float16)
        self.cache_gpu.append((k,v))
        self.copy_streams.append(torch.cuda.Stream(device=device))
        return  



    def set_beta(self,x):
        self.beta = x 
        return 
    
    def set_alpha(self,x):
        self.alpha = x 
        return 

    def clear_req_cache(self,):
        self.req_cache = [([],[]) for _ in range(self.layer_num)] 
        self.req_heatmaps = [[] for _ in range(self.layer_num)]
        return
     
    def clear_all_cache(self,):
        self.clear_req_cache()    
        self.cache_cpu = [(None,None) for _ in range(self.layer_num)] 
        self.seq_ptrs_gpu = [0 for _ in range(self.layer_num)]
        return 
    
    def preload(self,layer_idx):
        if self.seq2gpu==0: return 

        device_offset,_ = self.cache_map[layer_idx]
        copy_stream = self.copy_streams[device_offset]
        
        next_layer_idx = (layer_idx+1)%(self.layer_num )
        k_cpu,v_cpu = self.cache_cpu[next_layer_idx]
        if k_cpu is None: return 
        ptr = min(self.seq2gpu,k_cpu.size(self.seq_dim))
        with torch.cuda.stream(copy_stream):
            k2gpu = k_cpu[:,-ptr:,:].to(f"cuda:{device_offset}", non_blocking=True)
            v2gpu = v_cpu[:,-ptr:,:].to(f"cuda:{device_offset}", non_blocking=True) 
            self.cache2gpu = (k2gpu,v2gpu)

        return  
    
    def get_pos_ids(self, layer_idx): 
        return self.cpu_pos_ids[layer_idx],self.gpu_pos_offsets[layer_idx]
     
    def __call__(self,q,layer_idx):
        qs = q.size(-2)
        seq_ptr = self.seq_ptrs_gpu[layer_idx]
        k_gpu,v_gpu = None,None
        device_offset,layer_offset = self.cache_map[layer_idx] 
        k_cache_dev,v_cache_dev = self.cache_gpu[device_offset]

        if seq_ptr:
            k_gpu = k_cache_dev[layer_offset,:,:seq_ptr,:]
            v_gpu = v_cache_dev[layer_offset,:,:seq_ptr,:]
            
        k2gpu,v2gpu = self.cache2gpu 
        if k2gpu and k_gpu:
            k_gpu = torch.cat([k2gpu.to(device_offset, non_blocking=True),k_gpu],dim=self.seq_dim)
            v_gpu = torch.cat([v2gpu.to(device_offset, non_blocking=True),v_gpu],dim=self.seq_dim)
        elif k2gpu:
            k_gpu = k2gpu.to(device_offset, non_blocking=True)
            v_gpu = v2gpu.to(device_offset, non_blocking=True)

        if qs==1:
            k_cpu,v_cpu = self.req_cache[layer_idx]
        else:
            k_cpu,v_cpu = self.cache_cpu[layer_idx]

         
        return k_gpu,v_gpu,k_cpu,v_cpu
 
     

    def get_past_key_values_length(self,):
        return 2000 

     
 


    @staticmethod
    def distill(k,v,heaptmap,keep_num):
        ids = heaptmap.topk(keep_num,dim=1).indices  
        dim0_ids = torch.arange(k.size(0))[:, None]
        dim0_ids = dim0_ids.expand_as(ids)
        k = k[dim0_ids, ids]
        v = v[dim0_ids, ids]
        heaptmap = heaptmap[dim0_ids, ids]
        return k,v,heaptmap



    def update_heatmap_decode(self,As,layer_idx):
        heatmaps = self.req_heatmaps[layer_idx]
        for idx,(htp,A)in enumerate(zip(heatmaps,As)):   
            heatmaps[idx] = (1-self.alpha)*htp+self.alpha*A.squeeze(-2)
        self.req_heatmaps[layer_idx] = heatmaps
        return 
    
    
    def update_cpu_cache(self,k2cpu,v2cpu,layer_idx):
        if k2cpu is None: return None,None
        k_cpu,v_cpu = self.cache_cpu[layer_idx] 
        if k_cpu is not None:
            k_cpu,v_cpu = torch.concat([k_cpu,k2cpu],dim=self.seq_dim),torch.concat([v_cpu,v2cpu],dim=self.seq_dim)
             
        else:
            k_cpu,v_cpu = k2cpu,v2cpu


        self.cache_cpu[layer_idx] = k_cpu,v_cpu
        
        len2evict = k_cpu.size(1)-self.seq2gpu
        if len2evict<=0:
            return None,None
        else:
            return k_cpu[:,:len2evict,:],v_cpu[:,:len2evict,:]
         



    def init_req_cache(self,k2cpu,v2cpu,A2evict,layer_idx):
        k2evict,v2evict = self.update_cpu_cache(k2cpu,v2cpu,layer_idx)
        if k2evict is None: 
            return 
        A2evict = A2evict.sum(dim=-2)
    
        bh = self.batch_size*self.head_num
      
        heatmaps = []
        req_k,req_v = [],[]
        dim0 = self.head_num//4 
        for st in range(0,bh,dim0):
            ed = st+dim0 
            heatmaps.append(A2evict[st:ed,:])
            req_k.append(k2evict[st:ed,:,:])
            req_v.append(v2evict[st:ed,:,:])

        
        self.req_cache[layer_idx] =  (req_k,req_v)
        self.req_heatmaps[layer_idx] =  heatmaps  
        if self.enable_pos:
                self.update_pos_ids(layer_idx)
        return  

     
    
    def update_req_cache_decode(self,k2cpu,v2cpu,A2evict,A_cpu,layer_idx):     
        k2evict,v2evict = self.update_cpu_cache(k2cpu,v2cpu,layer_idx)
        if k2evict is None: 
            return 
        req_k,req_v = self.req_cache[layer_idx]
        heatmaps    = self.req_heatmaps[layer_idx]
        
        st,ed = 0,0
        for idx,(htp,A0,A1)in enumerate(zip(heatmaps,A_cpu,A2evict)):  
            A0 = A0.squeeze(-2)
            A1 = A1.squeeze(-2)     
            ed = st+ htp.size(0)
            htp = (1-self.alpha)*htp+self.alpha*A0 
            keep_num = torch.sum(htp >= (1/htp.size(1)*self.beta),dim=-1).max()
            k,v = req_k[idx],req_v[idx]
            k,v,htp = self.distill(k,v,htp,keep_num)
            req_k[idx] = torch.concat([k,k2evict[st:ed,:,:]],dim=0)
            req_v[idx] = torch.concat([v,v2evict[st:ed,:,:]],dim=0)
             
             
            heatmaps[idx] = torch.concat([htp,A1[st:ed,:]],dim=1)
            st = ed 
        
        self.req_cache[layer_idx] =  (req_k,req_v)
        self.req_heatmaps[layer_idx] =  heatmaps  
        if self.enable_pos:
                self.update_pos_ids(layer_idx)
        return  
    
     
        
    def update_pos_ids(self,layer_idx):
        k_cpu,_ = self.cache_cpu[layer_idx]
        if k_cpu.size(0)>=self.seq2gpu:
            return 
        device_offset,_ = self.cache_map[layer_idx]
        copy_stream = self.copy_streams[device_offset]

        with torch.cuda.stream(copy_stream):

            gpu_ids_offset = torch.ones(self.batch_size*self.head_num,1,device=device_offset)
            cpu_ids = []
            htps = self.heatmaps[layer_idx]
            st,ed = 0,0
            for htp in htps:
                ed = st+htp.size(0)
                gpu_ids_offset[:,st:ed] = htp.size(1)
                cpu_ids.append(torch.arange(htp.size(1)))

            self.gpu_pos_offsets[layer_idx] = gpu_ids_offset
            self.cpu_pos_ids[layer_idx] = cpu_ids
    
        return  

    def update_req_cache_append(self,k2cpu,v2cpu,A2evict,A_cpu,layer_idx):
        k2evict,v2evict = self.update_cpu_cache(k2cpu,v2cpu,layer_idx)
     
        bh = self.batch_size*self.head_num
        qs = A_cpu.size(-2)
        A_cpu = A_cpu.sum(dim=-2)/qs  
        if A2evict is not None:
            A2evict = A2evict.sum(dim=-2)/qs  
        keep_nums = torch.sum(A_cpu >= (1/A_cpu.size(1)*self.beta),dim=-1) 
        #round_num = (torch.max(keep_nums)-torch.min(keep_nums))  
       
        max_ = torch.max(keep_nums)
        min_ = torch.min(keep_nums)
        thre = int((max_-min_)*0.95 )+ min_ #A.size(0)# *30
         
      
     
        round_up_keep_nums = torch.where(keep_nums<thre,thre,keep_nums)
        # round_up_keep_nums = (keep_nums+ round_num-1)//round_num*round_num
         
        k_cpu,v_cpu = self.cache_cpu[layer_idx]
        if self.seq2gpu:
            k_cpu,v_cpu = k_cpu[:,:-self.seq2gpu],v_cpu[:,:-self.seq2gpu]
        st,ed,pre = 0,0,round_up_keep_nums[0]
        heatmaps = []
        req_k,req_v = [],[]
        for idx in range(bh):
            if pre!=round_up_keep_nums[idx]:
                keep_num = torch.max(keep_nums[st:ed])
             
                k,v = k_cpu[st:ed,:,:],v_cpu[st:ed,:,:]
                htp = A_cpu[st:ed,:]
                
                k,v,htp = self.distill(k,v,htp,keep_num)
                if k2evict:
                    k = torch.concat([k,k2evict[st:ed,:,:]],dim=0)
                    v = torch.concat([v,v2evict[st:ed,:,:]],dim=0) 
                    htp = torch.concat([htp,A2evict[st:ed,:]],dim=1)
                  
                    
                req_k.append(k)
                req_v.append(v)
                heatmaps.append(htp)
                pre = round_up_keep_nums[idx]
                st = ed
            ed += 1 

        keep_num = torch.max(keep_nums[st:ed])
        k,v = k_cpu[st:ed,:,:],v_cpu[st:ed,:,:]
        htp = A_cpu[st:ed,:]
        k,v,htp = self.distill(k,v,htp,keep_num)
       
        if k2evict:
            k = torch.concat([k,k2evict[st:ed,:,:]],dim=0)
            v = torch.concat([v,v2evict[st:ed,:,:]],dim=0)
            htp = torch.concat([htp,A2evict[st:ed,:]*self.alpha],dim=1)
        req_k.append(k)
        req_v.append(v)
        heatmaps.append(htp)
        self.req_cache[layer_idx] = (req_k,req_v)
        self.req_heatmaps[layer_idx] = heatmaps
        if self.enable_pos:
             self.update_pos_ids(layer_idx)

        return 
    
    def print_size(self,idx):
        k,_ = self.cache_cpu[idx]
        log = "CPU size: 0"
        if k is not None:
            log = f"CPU size: {k.size(-2)}"
        log += f"| GPU size: {self.seq_ptrs_gpu[idx]}"
        print(log)


    def print_heatmap(self,idx):
        heatmaps = self.req_heatmaps[idx]
        print(f"number of BH dim splite: {len(heatmaps)}")
        if heatmaps is not None:
            for htp in heatmaps:
                print(htp.size())
        print('======================================')


    def sync_task(self,):
        for task in self.tasks:
            task.join()
        return 
    
   
    
    def load_cache(self,k_cache,v_cache):
        for layer_idx in range(self.layer_num):
            device_offset,layer_offset = self.cache_map[layer_idx]
            seq_offset = k_cache.size(self.seq_dim)
           
         
            self.cache_gpu[device_offset][0][layer_offset,:,:seq_offset,:] = k_cache 
            self.cache_gpu[device_offset][1][layer_offset,:,:seq_offset,:] = v_cache
            self.seq_ptrs_gpu[layer_idx] = seq_offset
        return 
    

    def add_kvcache(self,layer_idx, k2add, v2add,A_gpu,A_cpu=None):
        # bh,s,d
 
        device_offset,layer_offset = self.cache_map[layer_idx]
        copy_stream = self.copy_streams[device_offset]
 
        with torch.cuda.stream(copy_stream):
             
            len2add = k2add.size(self.seq_dim)
            seq_ptr = self.seq_ptrs_gpu[layer_idx]
            k2cpu,v2cpu,A2evict = None,None,None
            k_cache_dev,v_cache_dev = self.cache_gpu[device_offset]
            if seq_ptr+len2add < self.seq_max:
               
                ptr1 = seq_ptr+len2add
                k_cache_dev[layer_offset,:,seq_ptr:ptr1,:] = k2add
                v_cache_dev[layer_offset,:,seq_ptr:ptr1,:] = v2add
                
            else:
                len2cpu = (seq_ptr+len2add-self.seq_max+self.blk_size)//self.blk_size*self.blk_size
                A2evict = A_gpu[:,:,:len2cpu].to("cpu",non_blocking=True)
                 
                
                if len2cpu<seq_ptr:
                    k2cpu = k_cache_dev[layer_offset,:,:len2cpu,:].to("cpu",non_blocking=True)
                    v2cpu = v_cache_dev[layer_offset,:,:len2cpu,:].to("cpu",non_blocking=True)
                    ptr0 = seq_ptr-len2cpu
                    k_cache_dev[layer_offset,:,:ptr0,:] = k_cache_dev[layer_offset,:,len2cpu:seq_ptr,:]
                    v_cache_dev[layer_offset,:,:ptr0,:] = v_cache_dev[layer_offset,:,len2cpu:seq_ptr,:]
                    ptr1 = ptr0 + len2add
                    k_cache_dev[layer_offset,:,ptr0:ptr1,:] = k2add
                    v_cache_dev[layer_offset,:,ptr0:ptr1,:] = v2add
                    
                else:
                    ptr0 =  len2cpu-seq_ptr 
                    if seq_ptr:
                        k2cpu = k_cache_dev[layer_offset,:,:seq_ptr,:] 
                        v2cpu = v_cache_dev[layer_offset,:,:seq_ptr,:] 
                        k2cpu = torch.cat([k2cpu,k2add[:,:ptr0,:]],dim=self.seq_dim).to("cpu",non_blocking=True)
                        v2cpu = torch.cat([v2cpu,v2add[:,:ptr0,:]],dim=self.seq_dim).to("cpu",non_blocking=True)
                         
                    else:
                        k2cpu = k2add[:,:ptr0,:].to("cpu",non_blocking=True)
                        v2cpu = v2add[:,:ptr0,:].to("cpu",non_blocking=True)

                    ptr1 = len2add-ptr0
                    k_cache_dev[layer_offset,:,:ptr1,:] = k2add[:,ptr0:,:]
                    v_cache_dev[layer_offset,:,:ptr1,:] = v2add[:,ptr0:,:]
            
            self.seq_ptrs_gpu[layer_idx] = ptr1

        # Asynchronous launch cpu kvcache managing task 
        if A_cpu is not None:
         
            if k2add.size(self.seq_dim)==1: # decode 
                if k2cpu is not None:
                   
                    task = threading.Thread(target=self.update_req_cache_decode, args=(k2cpu,v2cpu,A2evict,A_cpu,layer_idx,))
                    task.start()
                    self.tasks.append(task)
                else:
                   
                    task = threading.Thread(target=self.update_heatmap_decode, args=(A_cpu,layer_idx,))
                    task.start()
                    self.tasks.append(task)
            else: # append 
                 
                task = threading.Thread(target=self.update_req_cache_append, args=(k2cpu,v2cpu,A2evict,A_cpu,layer_idx,))
                task.start()
                self.tasks.append(task)
        elif k2cpu is not None:
          
            task = threading.Thread(target=self.init_req_cache, args=(k2cpu,v2cpu,A2evict,layer_idx,))
            task.start()
            self.tasks.append(task)
        return 
            
#####################
#      For Test     # 
#####################

 

def run_test(args):


    config = AutoConfig.from_pretrained(args.model_name)
     
    print_args_info(args)
    

    config = config
    batch_size = args.batch_size
    head_num = config.num_attention_heads
    head_dim = config.hidden_size//head_num 
    block_size=args.block_size
    max_blk_gpu=args.max_blocks
    add_len = args.add_len
    device_id = 0
    
    layer_num = config.num_hidden_layers
    device_map = {}
    for i in range(layer_num):
        device_map[str(i)] = "cuda:1"
     
    KVCache_manager = KVCacheManager(config,device_map,batch_size,False,
                                     block_size=block_size,
                                     max_blk_gpu=max_blk_gpu)
    
    
    add_len = 1024
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    KVCache_manager.load_cache(k_add,v_add)
    KVCache_manager.print_size(0)
    KVCache_manager.print_heatmap(0)
 
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = KVCache_manager(k_add,0)
    print(k_cache_gpu.shape)
    
    i  =0 
    add_len = 10
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    A_gpu = torch.randn(batch_size*head_num,add_len,1034,dtype=torch.float16).to(f"cuda:{device_id}")
   
    KVCache_manager.add_kvcache(i,k_add,v_add,A_gpu)
    KVCache_manager.preload(i)
    KVCache_manager.sync_task()
    KVCache_manager.print_size(0)
    KVCache_manager.print_heatmap(0)
  
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = KVCache_manager(k_add,0)
    print(k_cache_gpu.shape)

    add_len = 1
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    A_gpu = torch.randn(batch_size*head_num,add_len,1024,dtype=torch.float16).to(f"cuda:{device_id}")
    
    KVCache_manager.add_kvcache(i,k_add,v_add,A_gpu)
    KVCache_manager.preload(i)
    KVCache_manager.sync_task()
    KVCache_manager.print_size(0)
    KVCache_manager.print_heatmap(0)
  
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = KVCache_manager(k_add,0)
    print(k_cache_gpu.shape)

    add_len = 128
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    A_gpu = torch.randn(batch_size*head_num,add_len,927,dtype=torch.float16).to(f"cuda:{device_id}")
     


    KVCache_manager.add_kvcache(i,k_add,v_add,A_gpu )
    KVCache_manager.preload(i)
    KVCache_manager.sync_task()
    KVCache_manager.print_size(0)
    KVCache_manager.print_heatmap(0)

    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = KVCache_manager(k_add,0)
    print(k_cache_gpu.shape)

    add_len = 10
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    A_gpu = torch.randn(batch_size*head_num,add_len,927,dtype=torch.float16).to(f"cuda:{device_id}")
    A_cpu = torch.randn(batch_size*head_num,add_len,256,dtype=torch.float16) 
    KVCache_manager.add_kvcache(i,k_add,v_add,A_gpu,A_cpu )
    KVCache_manager.preload(i)
    KVCache_manager.sync_task()
    KVCache_manager.print_size(0)
    KVCache_manager.print_heatmap(0)

    
     
    

    return 
        
    

if __name__ == "__main__":
    import time 
    import logging 
    import argparse        
    import numpy as np
    from beyond.Utils import *
    from transformers import AutoConfig
    # torch.cuda.current_device()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--qs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_blocks", type=int, default=8)
    parser.add_argument("--add_len", type=int, default=100)
    
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default= "facebook/opt-66b")
 
    args = parser.parse_args()

    run_test(args)
     
