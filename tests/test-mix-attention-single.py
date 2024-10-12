import torch
import argparse
import flashinfer
import time
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import os 
from utils import *
from flashinfer import single_prefill_with_kv_cache_return_lse as flash_decode  


# for debug use 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
 



def main(args,log):
    log = add_info(args,log)
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

    k_cache = torch.randn(seq_len, num_heads,head_dim, device='cpu').half()
    v_cache = torch.randn(seq_len, num_heads,head_dim, device='cpu').half()
    q       = torch.randn(args.q_len, num_heads,head_dim, device='cuda:0').half()

    k_cache_gpu = torch.empty(gpu_len, num_heads,head_dim, device='cuda:0').half()
    v_cache_gpu = torch.empty(gpu_len, num_heads,head_dim, device='cuda:0').half()
    q_cpu = torch.empty(args.q_len, num_heads,head_dim, device='cpu').pin_memory().half()
    
    
    cpu_mha = flashAttnSingle(head_dim,num_heads,"cpu") 
    gpu_mha = flashAttnSingle(head_dim,num_heads,"gpu") 
    cpu_stream =  torch.cuda.Stream()
    gpu_stream =  torch.cuda.Stream()
    
    


    
     
    
    
    v_reference,_ = gpu_mha(q ,k_cache.cuda(),v_cache.cuda())
    

    v0,s0 = cpu_mha(q.cpu() ,k_cache[:seq_len//2] ,v_cache[:seq_len//2] )
    v1,s1 = gpu_mha(q ,k_cache[seq_len//2:].cuda(),v_cache[seq_len//2:].cuda())
 

    va,sa = cpu_mha(q.cpu(),k_cache  ,v_cache  )
    vb,sb = flash_decode(q.half(),k_cache.half().cuda(),v_cache.half().cuda())
    # print(check_eq(va,vb))
    # print(check_eq(sa,sb))
    
      
    # check_memory(v0,'v0')
    # check_memory(s0,'s0')
    # check_memory(v1,'v1')
    # check_memory(s1,'s1')
    v_out,s_out = flashinfer.merge_state( v0 ,s0 ,v1 ,s1 )
     
    
    
    acc = check_eq(v_out,v_reference)  
    assert  (acc>0.9),  f"accuracy {acc*100:.4}%, merge state fail..."
    
    epoch_times = list()
    mem_usages  = list()
    accuracys   = list()
    for _ in range(args.repeat):
        st = time.time()
        torch.cuda.reset_peak_memory_stats( ) 
         
        if ratio!=1:
            with torch.cuda.stream(gpu_stream):
                     
                indices_gpu = torch.arange(cpu_len,seq_len) 
                hold_k = k_cache.index_select(0,indices_gpu) 
                hold_v = v_cache.index_select(0,indices_gpu) 
                # hold_k = hold_k.pin_memory()
                # hold_v = hold_v.pin_memory()
                 
                k_cache_gpu.copy_(hold_k, non_blocking=True)
                v_cache_gpu.copy_(hold_v , non_blocking=True)
               
                v_out,s_out = gpu_mha(q,k_cache_gpu,v_cache_gpu)
                #v_out,s_out = flash_decode(q ,k_cache_gpu ,v_cache_gpu )
        if ratio:
            with torch.cuda.stream(cpu_stream):
                 
                q_cpu.copy_(q.clone().detach(), non_blocking=True)
                indices_cpu = torch.arange(0,cpu_len)
                k_cache_cpu = k_cache.index_select(0,indices_cpu)
                v_cache_cpu = v_cache.index_select(0,indices_cpu)
                v_cpu,s_cpu = cpu_mha(q_cpu,k_cache_cpu,v_cache_cpu)
                # v_cpu,s_cpu =    v_cpu.cuda() ,s_cpu.cuda() 
            torch.cuda.synchronize()
            if ratio!=1: 
               
                v_out,s_out = flashinfer.merge_state( v_cpu ,s_cpu ,v_out ,s_out )
            else:
                v_out,s_out = v_cpu,s_cpu

        # torch.cuda.synchronize()
        mem_usages.append(torch.cuda.max_memory_allocated( )/1024**3 )
        epoch_times.append(time.time()-st)
        accuracys.append(check_eq(v_out,v_reference))
    
    
    log += f"latency:{(np.mean(epoch_times[-(args.repeat)//2:])):.5} sec, "
    log += f"memory:{(np.mean(mem_usages[-(args.repeat)//2:])):.5} GB, "
    log += f"accuracy:{(np.mean(accuracys)*100):.5}"
    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_name", type=str, default="opt-13b") 
    parser.add_argument("--seq_len", type=int, default=100000)
    parser.add_argument("--q_len", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    # check_workspace(args.ratio)
    main(args,"")