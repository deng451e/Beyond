from beyond.attention_methods import *
from beyond.utils import mha_normal
import numpy as np 
import torch.nn.functional as F
 

# def select_kv(q, k_cache, v_cache,topk):
    
    
#     max_ = torch.max(p_attn, dim=-1)[0]
   
#     prefetch_idx = torch.topk(p_attn.permute(2, 1, 0),topk, dim=0)[1]

#     prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
#     ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
#     selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
#     selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
#     return selected_k, selected_v


 

def  get_blk_kv_min_max(k_cache,blk_size=10):
           
        k_cache_shape = k_cache.size()
        cache_len = k_cache_shape[2]
        pad_len = blk_size-cache_len%blk_size

        blk_idx  = [(start,min(cache_len,start+blk_size)) for start in range(0,cache_len,blk_size)]
        pad_ = ([0 for _ in range((k_cache.dim()-3)*2)] + [0,pad_len])
        reshape_ = [k_cache_shape[Dim] for Dim in range(k_cache.dim())]
        reshape_[2] = blk_size
        reshape_.insert(2,-1 )   
        reshape_ = tuple(reshape_)
        
        #b,h,s//blks,blks,d
        blk = F.pad( k_cache, pad=pad_, value=0).reshape(reshape_)
            
        
        return blk.min(dim=3)[0], blk.max(dim=3)[0],blk_idx
             

 




def test(args,out):
    out = add_info(args,out)
     
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    
    seq_len = args.seq_len
    q_len = args.q_len
    head_dim = hidden_size//num_heads
    
    blk_size=args.blk_size
    topk = int(args.ratio*seq_len//blk_size)
     
     
    k_dim   =  2
    k_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    v_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    q       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cpu').half() 
    k       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cpu').half() 
    v       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cpu').half() 
    
        
    # num_elements = v_cache.numel()  # Total number of elements
    # element_size = v_cache.element_size()  # Size of one element in bytes
    # memory_size = 40*num_elements * element_size *2/1024**3 # Total memory size in bytes

    # print(f"Tensor memory size: {memory_size} GB")

    #####################
    # Normal Attention  # 
    #####################

 
    # print(out)
    
     
    k_blk_min,k_blk_max,blk_idx =  get_blk_kv_min_max(k_cache,blk_size)
   
 
    
    kv_cache = (k_cache,v_cache)
    blk_dim_min_max = (k_blk_min,k_blk_max)
    block_selection = block_selection_(blk_size,32)
    ts = []
    for _ in range(args.repeat+5):
        st = time.time()
        
        selectd_k,selected_v = block_selection(q,blk_dim_min_max, blk_idx,kv_cache,0,topk)
        
        ts.append((time.time()-st))
    out = f"blk_size:{blk_size}, ratio:{args.ratio}, selection time: {np.mean(ts[5:]):.4}, "

    
    ts = []
    for _ in range(args.repeat+5):
        st = time.time()
        torch.cuda.synchronize()
        _ = mha_normal(q,selectd_k,selected_v)
        torch.cuda.synchronize()
        ts.append((time.time()-st))
    out += f"sparse length: {selectd_k.size(2) },compute time: {np.mean(ts[5:] ):.4}, "


    ###########################
    #      CPU   Attention    # 
    ###########################
    q_cpu = q.cpu() 
    k_cpu = torch.cat([k_cache,k.cpu()],dim=k_dim)
    v_cpu = torch.cat([v_cache,v.cpu()],dim=k_dim)
    ts = []
    for _ in range(args.repeat+5):
        st = time.time()
        torch.cuda.synchronize()
        _ = mha_normal(q_cpu,k_cpu,v_cpu)
        torch.cuda.synchronize()
        ts.append((time.time()-st))
    out += f"dense length: {k_cpu.size(2)-1 }, compute time: {np.mean(ts[5:] ):.4} "
    
    


    #####################
    #   PCIe Transfer   # 
    #####################
    pcie_t = []
    for _ in range(args.repeat+5):
         
        st = time.time() 
        torch.cuda.synchronize()
        k_gpu = k_cache.cuda()
        v_gpu = v_cache.cuda()
        torch.cuda.synchronize()
        pcie_t.append(time.time()-st)
    
    out+=f"transfer_time:{np.mean(pcie_t[5:])},"


    k_gpu = torch.cat([k_gpu,k.cuda()],dim=k_dim )
    v_gpu = torch.cat([v_gpu,v.cuda()],dim=k_dim )
    q = q.cuda()
    #####################
    #   GPU Attention   # 
    #####################
    gpu_t = []
    for _ in range(args.repeat+5):
       
        st = time.time() 
        torch.cuda.synchronize()
        OUTPUT_gpu = mha_normal(q,k_gpu ,v_gpu)
        torch.cuda.synchronize()
        gpu_t.append(time.time()-st)

    out+=f"gpu_time: {np.mean(gpu_t[5:])}"
    
     
    print(out)
 
     
      
    


    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    
    parser.add_argument("--seq_len", type=int, default=50000)
    parser.add_argument("--q_len", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--blk_size", type=int, default=100)

    # model config 
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--model_type", type=str, default="opt")
    parser.add_argument("--RoPE", type=bool, default=True )
    
    # test config 
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
     
    args = parser.parse_args()
    test(args,"")
     