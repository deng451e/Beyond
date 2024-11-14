from beyond.attention_methods import *
from beyond.utils import mha_normal
import numpy as np 
import torch.nn.functional as F
# bh,qs,s
# s,qs,bh
# prefetch_idx = torch.topk( 
#         p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
#     )[1]

def mha_sparse_selection(q,k_cache,v_cache ,topk):
    # b h s d 
    qs = q.size(-2)
    b,h,s,d = k_cache.size()
    attn_weights = torch.matmul(q.float(), k_cache.transpose(2, 3).float())  # b,h,qs,s
    attn_weights = attn_weights.reshape(b*h,qs,s).permute(2,1,0)
    
    indices = torch.topk(attn_weights , topk,dim=0)[1]
        
    if qs!=1:
        #k,qs,bh
        indices = indices.reshape(-1,b*h) 
       
        indices = torch.unique(indices,dim=0)[:topk]
    
    indices = indices.reshape(b*h,-1).permute(1,0)
     
    ind = indices * b*h + torch.arange(b*h,device=k_cache.device)[None, :]
    
    v_cache  = v_cache.permute(2,0,1,3).reshape(-1,d)
    k_cache  = k_cache.permute(2,0,1,3).reshape(-1,d)

    selected_v = F.embedding(ind, v_cache)
    selected_k = F.embedding(ind, k_cache)

    selected_v = selected_v.permute(1,0,2).reshape(b,h,topk,-1)
    selected_k = selected_k.permute(1,0,2).reshape(b,h,topk,-1)

 
    return  selected_k,selected_v
 


#####################
#      For Test     # 
#####################
 
def test(args,log):
    log = add_info(args,log)
    batch_size = 1
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    repeat = args.repeat
    seq_len = args.seq_len
    q_len = args.q_len
    top_k = args.top_k
    head_dim = hidden_size//num_heads
    

    k_dim   =  2
    k_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    v_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    q       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    k       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    v       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
        
     
    
    
    out= f"batch_size:{batch_size},seq_len:{seq_len},q_len:{q_len},num_heads:{num_heads},head_dim:{head_dim},top_k:{top_k},"
    #####################
    #   CPU Attention   # 
    #####################
    q_cpu = q.cpu() 
    k_cpu = k_cache.cpu()
    v_cpu = v_cache.cpu()
  
    cpu_selec_t = []
    cpu_attn_t = []
    for _ in range(repeat+3):
        st = time.time()
        k_cpu_select,v_cpu_select = mha_sparse_selection(q_cpu,k_cpu,v_cpu,top_k)
        cpu_selec_t.append(time.time()-st)

        k_cpu_ = torch.cat([k_cpu_select,k.cpu()],dim=k_dim)
        v_cpu_ = torch.cat([v_cpu_select,v.cpu()],dim=k_dim)

        st = time.time()
        OUTPUT_cpu = mha_normal(q_cpu,k_cpu_,v_cpu_ )
        cpu_attn_t.append(time.time()-st)
   
    out+=f"cpu select time:{np.mean(cpu_selec_t[3:])},cpu attn time:{np.mean(cpu_attn_t[3:])},"
    
    
     
   
    #####################
    #   GPU Attention   # 
    #####################
    q_gpu = q.cuda() 
    k_gpu = k_cache.cuda()
    v_gpu = v_cache.cuda()
  
    gpu_selec_t = []
    gpu_attn_t = []
    for _ in range(repeat+3):
        st = time.time()
        k_gpu_select,v_gpu_select = mha_sparse_selection(q_gpu,k_gpu,v_gpu,top_k)
        gpu_selec_t.append(time.time()-st)

        k_gpu_ = torch.cat([k_gpu_select,k.cuda()],dim=k_dim)
        v_gpu_ = torch.cat([v_gpu_select,v.cuda()],dim=k_dim)

        st = time.time()
        OUTPUT_gpu = mha_normal(q_gpu,k_gpu_,v_gpu_ )
        gpu_attn_t.append(time.time()-st)
   
    out+=f"gpu select time:{np.mean(gpu_selec_t[3:])},gpu attn time:{np.mean(gpu_attn_t[3:])},"
     
      
    acc = check_eq(OUTPUT_cpu,OUTPUT_gpu )  
    out+=f"accuracy:{acc*100:.4}"
    
    print(out)
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=float, default=100)
    
    parser.add_argument("--seq_len", type=int, default=1000 )
    parser.add_argument("--q_len", type=int, default=10)

    # model config 
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
   

    # test config 
    parser.add_argument("--repeat", type=int, default=10)

    args = parser.parse_args()
    args.top_k = int(args.top_k)
    test(args,"")
     