import torch
import argparse
import flashinfer
import time
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import os 

# for debug use 
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

class flashAttn(torch.nn.Module):
    def __init__(self,head_dim, num_heads,device):
        super(flashAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.device = device 
    def forward(self, q,k,v,mask=None):
        # k shape: s,h,d
        # q shape: qs,h,d
        s  = k.size(0)
        qs = q.size(0)
         

           

        if self.device=="cpu":
            if mask!=None:
                indices = torch.nonzero(mask).squeeze()
                k = k[indices,:,:]
                v = v[indices,:,:]
            q = q.permute(1, 0, 2 )
            k = k.permute(1, 2, 0) # h,d,s
            v = v.permute(1, 0, 2) # h,s,d
            attn_weights = torch.bmm(q,k) # h,qs,s
            logSum = torch.sum(torch.exp(attn_weights).permute(1,0,2),dim=-1) 
            attn_weights = F.softmax(attn_weights, dim=2)
            value  = torch.bmm(attn_weights, v).permute(1,0,2) 
            return value.half().pin_memory(), logSum.pin_memory()

               
        elif self.device=="gpu":
             
         
            q = q.permute(1, 0, 2 )
            k = k.permute(1, 2, 0) # h,d,s
            v = v.permute(1, 0, 2) # h,s,d
            
            attn_weights = torch.bmm(q, k) # h,qs,s
            
            if mask==None:
                logSum =torch.sum(torch.exp(attn_weights).permute(1,0,2),dim=-1)  #  qs,h
            else:

                mask = mask.view( 1, 1, s) 
                logSum =torch.sum(torch.exp( torch.where(mask, attn_weights, 0)).permute(1,0,2),dim=-1)  #  qs,h

                attn_weights = torch.where(mask, attn_weights, -1e4)
          
            attn_weights = F.softmax(attn_weights, dim=2)
            value = torch.bmm(attn_weights, v).permute(1,0,2)

            return value.half(), logSum
    

def main(args):
    arch_name = args.arch_name
    if arch_name == "opt-1.3b":
    
        max_seq_len=2048;num_hidden_layers=24; num_heads=32
        hidden_size=2048; input_dim=2048; ffn_embed_dim=2048 * 4

    elif arch_name == "opt-2.7b":
    
        max_seq_len=2048; num_hidden_layers=32; num_heads=32
        hidden_size=2560; input_dim=2560; ffn_embed_dim=2560 * 4
        
    elif arch_name == "opt-6.7b":

        max_seq_len=2048; num_hidden_layers=32; num_heads=32
        hidden_size=4096; input_dim=4096; ffn_embed_dim=4096 * 4

    elif arch_name == "opt-13b":

        max_seq_len=2048; num_hidden_layers=40; num_heads=40
        hidden_size=5120; input_dim=5120; ffn_embed_dim=5120 * 4


    seq_len = args.seq_len
    head_dim = hidden_size//num_heads
    ratio = args.ratio
    cpu_len = int( seq_len*ratio )
    gpu_len = seq_len-cpu_len

    k_cache = torch.randn(seq_len, num_heads,head_dim, dtype=torch.float32, device='cpu') 
    v_cache = torch.randn(seq_len, num_heads,head_dim, dtype=torch.float32, device='cpu') 
    q       = torch.randn(args.q_len, num_heads,head_dim, dtype=torch.float32, device='cuda:0') 

    cpu_mha = flashAttn(head_dim,num_heads,"cpu") 
    gpu_mha = flashAttn(head_dim,num_heads,"gpu") 
    cpu_stream =  torch.cuda.Stream()
    gpu_stream =  torch.cuda.Stream()
    
    q_cpu = torch.empty(args.q_len, num_heads,head_dim, device='cpu').pin_memory()

    k_cache_gpu = torch.empty(gpu_len, num_heads,head_dim, device='cuda:0')
    v_cache_gpu = torch.empty(gpu_len, num_heads,head_dim, device='cuda:0')
    v_other = torch.zeros(args.q_len, num_heads,head_dim, device='cuda:0') 
    s_other = torch.zeros(args.q_len, num_heads, device='cuda:0') 
    epoch_times = list()
    mem_usages  = list()
    for _ in range(args.repeat):
        st = time.time()
          
        torch.cuda. reset_peak_memory_stats( )
           
        with torch.cuda.stream(gpu_stream):
            indices_gpu = torch.arange(cpu_len,seq_len) 
            hold_k = k_cache.index_select(0,indices_gpu) 
            hold_v = v_cache.index_select(0,indices_gpu) 
            # hold_k = F.embedding(indices_gpu, k_cache)
            # hold_v = F.embedding(indices_gpu, v_cache)

            # print('gpu working')
            hold_k = hold_k.pin_memory()
            hold_v = hold_v.pin_memory()

            k_cache_gpu.copy_(hold_k, non_blocking=True)
            v_cache_gpu.copy_(hold_v , non_blocking=True)
            v_gpu,s_gpu = gpu_mha(q,k_cache_gpu,v_cache_gpu)
             
        if ratio:
                with torch.cuda.stream(cpu_stream):
                    # print('cpu working')
                    q_cpu.copy_(q.clone().detach(), non_blocking=True)
                    indices_cpu = torch.arange(0,cpu_len)
                    k_cache_cpu = k_cache.index_select(0,indices_cpu)
                    v_cache_cpu = v_cache.index_select(0,indices_cpu)
                        
                    v_cpu,s_cpu = cpu_mha(q_cpu,k_cache_cpu,v_cache_cpu)
        if ratio:
            torch.cuda.synchronize()
            # print(v_cpu.is_pinned(),s_cpu.is_pinned())
            flashinfer.merge_state_in_place( v_cpu,s_cpu,v_gpu,s_gpu)
        
        epoch_times.append(time.time()-st)
        mem_usages.append(torch.cuda.max_memory_allocated( )/1024**3 )
    print(f"time taken: {(np.mean(epoch_times[-50:])):.3} sec")
    print(f"memory usages: {(np.mean(mem_usages[-50:])):.3} GB")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_name", type=str, default="opt-1.3b") 
    #parser.add_argument("--batch-size", type=int, default=32)
    #parser.add_argument("--precise", type=str, default="fp16")
    parser.add_argument("--seq-len", type=int, default=25600)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    main(args)