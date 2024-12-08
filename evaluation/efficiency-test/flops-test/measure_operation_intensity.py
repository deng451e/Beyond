import torch
import torch.nn.functional as F
from beyond.utils import * 
from beyond.attention_methods import merge_state_
import numpy as np 
import time 
import timeit

 


def mha_hybrid(q,k_cache,v_cache ):
    attn_weights = torch.matmul(q,k_cache.transpose(-2, -1))  
    exp_scores = torch.exp(attn_weights) 
    sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
    log_sum = torch.log(sum_exp_scores)
    attn_weights = exp_scores / sum_exp_scores 
    
    
    value  = torch.bmm(attn_weights, v_cache).permute(1,0,2).half().contiguous()
    log_sum = log_sum.squeeze(-1).permute(1,0).contiguous().float() 
    if check_tensor_device(q,'cpu'):  return  value.pin_memory(),log_sum.pin_memory()
    if check_tensor_device(q,'cuda'): return  value, log_sum


def mha(q,k_cache,v_cache ):
    # Scaled dot-product attention
    attn_weights = torch.matmul(q,k_cache.transpose(-2, -1))  
    exp_scores = torch.exp(attn_weights) 
    sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
    attn_weights = exp_scores / sum_exp_scores 
   
    value  = torch.bmm(attn_weights, v_cache).permute(1,0,2).half().contiguous()
    
    return  value
 

def calculate_actual_attention_flops(batch_size, q_len,seq_len, num_heads,head_dim):
   
    #scaling_flops = batch_size * num_heads * q_len * (q_len+seq_len) 


    # FLOPs for dot product attention (Q x K^T)
    attn_dot_flops = batch_size * num_heads * q_len * (q_len+seq_len) * head_dim
    # FLOPs for scaling (1 mul per element)
     
    # FLOPs for softmax (exp + sum + division)
    softmax_flops = batch_size * num_heads * q_len * (q_len+seq_len)  * 3
    # FLOPs for attention output (A x V)
    attn_output_flops = batch_size * num_heads * q_len * head_dim * (q_len+seq_len) 
    

    total_flops = (
        
        attn_dot_flops +
        #scaling_flops +
        softmax_flops +
        attn_output_flops  
        
    )
    return total_flops
 


#####################
#      For Test     # 
#####################
 
def test(args,log):
    log = add_info(args,log)
    batch_size  = args.batch_size
    hidden_size = args.hidden_size
    num_heads   = args.num_heads
    repeat      = args.repeat
    seq_len     = args.seq_len
    q_len       = args.q_len
    head_dim    = hidden_size//num_heads
    cuda_device = torch.device(args.device)
        
    # Calculate FLOPs
    theoretical_attention_flops = calculate_actual_attention_flops(batch_size,q_len, seq_len, num_heads,head_dim)

    k_dim   =  1
    k_cache = torch.randn(batch_size*num_heads,seq_len,head_dim, device='cpu').half().pin_memory()
    v_cache = torch.randn(batch_size*num_heads,seq_len,head_dim, device='cpu').half().pin_memory()
    q       = torch.randn(batch_size*num_heads,q_len,head_dim, device=cuda_device).half() 
    k       = torch.randn(batch_size*num_heads,q_len,head_dim, device=cuda_device).half() 
    v       = torch.randn(batch_size*num_heads,q_len,head_dim, device=cuda_device).half() 
    
     
    
    
    out= f"batch_size:{batch_size},q_len:{q_len},seq_len:{seq_len} ,num_heads:{num_heads},head_dim:{head_dim},"
   
   
    num_elements = v_cache.numel()  # Total number of elements
    element_size = v_cache.element_size()  # Size of one element in bytes
   
    memory_size =  num_elements * element_size *2/1024**3 # Total memory size in bytes

    out+=f"kv cache size: {memory_size:.4},"
 
    


          
    #####################
    #   CPU Attention   # 
    #####################
    if args.test_cpu:
       
        q_cpu = q.cpu()
        k_cpu = torch.cat([k_cache,k.cpu()],dim=k_dim) 
        v_cpu = torch.cat([v_cache,v.cpu()],dim=k_dim) 

        cpu_t = []
        for _ in range(args.repeat+5):
            torch.cuda.synchronize()
            st = time.time()
             

            output = mha(q_cpu,k_cpu,v_cpu )

            torch.cuda.synchronize()
            cpu_t.append(time.time()-st)

            del output
            

        out+=f"cpu attention flops:{theoretical_attention_flops/np.mean(cpu_t[5:]):.4}"
 

    #####################
    #   GPU Attention   # 
    #####################
    if args.test_gpu:
         
        q_gpu = q.to(cuda_device)
        k_gpu = torch.cat([k_cache.to(cuda_device),k ],dim=k_dim)
        v_gpu = torch.cat([v_cache.to(cuda_device),v ],dim=k_dim)
        gpu_t = []

        for _ in range(args.repeat+5):
            torch.cuda.synchronize()
            st = time.time() 
            

            output = mha(q_gpu,k_gpu ,v_gpu )
            
            torch.cuda.synchronize()
            gpu_t.append(time.time()-st)
            del output
             
            torch.cuda.empty_cache()

    
        out+=f"gpu attention flops:{theoretical_attention_flops/np.mean(gpu_t[5:]):.4}"
        
    #####################
    # Load & Attenttion# 
    #####################
    to_load_k = torch.cat([k_cache,k.cpu()],dim=k_dim).pin_memory()
    to_load_v = torch.cat([v_cache,v.cpu()],dim=k_dim).pin_memory()
    if args.test_offload:
        
        q_gpu = q.to(cuda_device)
        offload_t = []
        for _ in range(args.repeat+5):
            
            torch.cuda.synchronize()
            st = time.time() 
            

         
            k_load = torch.cat([k_cache.to(cuda_device),k ],dim=k_dim)
            v_load = torch.cat([v_cache.to(cuda_device),v ],dim=k_dim)
            output = mha(q_gpu,k_load ,v_load )
            

            torch.cuda.synchronize()
            offload_t.append(time.time()-st)

            del output
            del k_load
            del v_load
            torch.cuda.empty_cache()

        out+=f"offload attention flops:{theoretical_attention_flops/np.mean(offload_t[5:]):.4}"


    #####################
    # hybrid Attenttion # 
    #####################
    if args.test_hybrid:
        
        merge_state = merge_state_(num_heads)
        cpu_stream  = torch.cuda.Stream()
        gpu_stream  = torch.cuda.Stream()
        cut_thre    = int(args.ratio*seq_len)
        slice = DIM_TO_SLICE[k_dim]
        k_cpu = torch.cat([slice(k_cache, 0,cut_thre),k.cpu()],dim=k_dim)
        v_cpu = torch.cat([slice(v_cache, 0,cut_thre),k.cpu()],dim=k_dim)

        k_gpu = torch.cat([slice(k_cache,cut_thre,seq_len).to(cuda_device),k ] ,dim=k_dim)    
        v_gpu = torch.cat([slice(v_cache,cut_thre,seq_len).to(cuda_device),v ] ,dim=k_dim)

        q_cpu = q.detach().to('cpu',non_blocking=True)
        q_gpu = q.detach().to(cuda_device)
        offload_t = []
 
        for _ in range(args.repeat+5):
            
            torch.cuda.synchronize()
            st = time.time() 
            

            with torch.cuda.stream(cpu_stream):
                o_cpu,s_cpu = mha_hybrid(q_cpu, k_cpu,v_cpu )
            
            o_gpu,s_gpu = mha_hybrid(q_gpu,k_gpu,v_gpu )
            torch.cuda.synchronize()
            output,_ = merge_state(o_cpu,s_cpu,o_gpu,s_gpu)
            

            torch.cuda.synchronize()
            offload_t.append(time.time()-st)

            del output
             
            torch.cuda.empty_cache()
        out+=f"cpu ratio:{args.ratio},"
        out+=f"hybrid attention flops:{theoretical_attention_flops/np.mean(offload_t[5:]):.4}"
    
    print(out)


    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=50000 )
    parser.add_argument("--q_len", type=int, default=1)
    
    # model config 
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
   

    # test config 
    parser.add_argument("--repeat", type=int, default=100)
    

    parser.add_argument("--test_gpu",    action="store_true")
    parser.add_argument("--test_cpu",    action="store_true")
    parser.add_argument("--test_hybrid",    action="store_true")
    parser.add_argument("--test_offload", action="store_true")

    args = parser.parse_args()
    
    test(args,"")
     