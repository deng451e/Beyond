import torch  
import flashinfer 
import argparse
import time 
from beyond.utils import *
 
 
import logging 

logger = logging.getLogger(__name__)

 
 
            
 
 

 
# logSum attention methods 
class mha_lse_methods:
    def __init__(self,methods ):
        self.methods = methods
        
    def __call__(self, q,k,v,attention_mask=None):
        
        #shape: b,h,s,d
        batch_size,num_heads,qs,head_dim = q.size()
         
        q = q.reshape(batch_size  * num_heads, -1,  head_dim)   # bh,qs,d
        k = k.permute(0,1,3,2).reshape(batch_size  * num_heads, head_dim, -1) # bh,d,s
        v = v.reshape(batch_size  * num_heads, -1, head_dim) # bh,s,d
        attn_weights = torch.bmm(q,k)   # bh,qs,s 

        
        max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
        attn_weights  = attn_weights - max_scores
        # model attention differs by masking mechanism
        
        if attention_mask is not None:
            match self.methods:

                case "llama":
                    attn_weights[:,:,-qs:] = attn_weights[:,:,-qs:] + attention_mask # bh,qs,s 

                case "opt":
                    
                    attn_weights[:,:,-qs:] = attn_weights[:,:,-qs:] + attention_mask 
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                    attn_weights = attn_weights.to(torch.float32)
                case "gpt-neox":
                    mask_value = torch.finfo(attn_weights.dtype).min
                    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to( attn_weights.device)
                    attn_weights[:,:,-qs:] = torch.where(attention_mask, attn_weights[:,:,-qs:], mask_value)
                case "flexgen-opt":
                    print(attention_mask.shape,attn_weights.shape)
                     
                    attn_weights = torch.where(attention_mask, attn_weights[:,:,-qs:], -1e4)

         
        exp_scores = torch.exp(attn_weights).to(v.dtype)
        sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
        log_sum = (torch.log(sum_exp_scores)  + max_scores) * torch.tensor(1.4427) 
        attn_weights = exp_scores / sum_exp_scores 
        #attn_weights = attn_weights.to(v.dtype)
        value  = torch.bmm(attn_weights, v).permute(1,0,2).half().contiguous()
        log_sum = log_sum.squeeze(-1).permute(1,0).contiguous().float() 
        
       
        if check_tensor_device(q,'cpu'):  return  value.pin_memory(),log_sum.pin_memory()
        if check_tensor_device(q,'cuda'): return  value, log_sum
            
    

 
#wrapper for flashinfer merge state functions 
class merge_state_:
    def __init__(self,num_heads,in_place=False):
        self.num_heads = num_heads
        self.in_place = in_place
         
    def __call__(self, va,sa,vb,sb):
        """
        Input shape: s,h,d or s,bh,d
        """
        assert check_tensor_device(va,'cpu') , f"va should be on CPU"
        assert check_tensor_device(sa,'cpu') , f"sa should be on CPU"
        assert check_tensor_device(vb,'cuda') , f"vb should be on GPU"
        assert check_tensor_device(sb,'cuda') , f"sb should be on GPU"
        assert va.is_pinned() and va.is_contiguous(), f"va should be on pinned and contiguous"
        assert sa.is_pinned() and va.is_contiguous(), f"sa should be on pinned and contiguous"
 
        head_dim = va.size(-1)
        
        
        """
        The maximum compute capability of the NVIDIA GPU is 1024 threads per block. 
        Therefore, we partition the batch&head dimension accordingly.
        """
        partitions_dim = 7680//head_dim
        if partitions_dim<va.size(-2):
            v_out,s_out = [],[]
            num_partition = (va.size(-2)+partitions_dim-1)//partitions_dim
            for i in range(num_partition):
                start,end = i*partitions_dim,min((i+1)*partitions_dim,va.size(-2))
                va_,sa_ = va[:,start:end,:].contiguous().pin_memory(),sa[:,start:end].contiguous().pin_memory()
                vb_,sb_ = vb[:,start:end,:].contiguous(),sb[:,start:end].contiguous()
                if self.in_place:
                    flashinfer.merge_state_in_place(va_,sa_, vb_,sb_)
                    v_out_,s_out_ = vb_.detach(),sb_.detach()
                    v_out.append(v_out_);s_out.append(s_out_) 

                else:
                    v_out_,s_out_ =  flashinfer.merge_state(   va_,sa_, vb_,sb_ )
                    v_out.append(v_out_);s_out.append(s_out_) 

            v_out,s_out = torch.cat(v_out,dim=1),torch.cat(s_out,dim=1)
            
        else:
            if self.in_place:
                flashinfer.merge_state_in_place(va_,sa_, vb_,sb_)
                v_out,s_out = vb,sb
            else:
                v_out,s_out = flashinfer.merge_state(  va,sa,vb,sb)

        
            
        s,d = v_out.size(0),v_out.size(2)
        
        v_out = v_out.reshape(s,-1,self.num_heads,d).permute(1,0,2,3) # b,s,h,d

        # s_out shape: s,bh
        return v_out,s_out
    







#####################
#      For Test     # 
#####################

 




def test_correctness(args,log):
    log = add_info(args,log)
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    start  = args.start_size
    recent = args.recent_size
    seq_len = args.seq_len
    q_len = args.q_len
    head_dim = hidden_size//num_heads
     
   
    assert  (start+recent<=seq_len),  f"start+recent greater than seq_len..."
     
    k_dim   =  2
    k_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    v_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    q       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    k       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    v       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    
     
   
    slice = DIM_TO_SLICE[k_dim]
    mha_lse = mha_lse_methods(args.model_type)
    merge_state = merge_state_(num_heads)
    kv_len =  seq_len + q_len
   
    match args.model_type:
        case 'llama':
            
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,rotate_half 
            rotary_emb = LlamaRotaryEmbedding(head_dim)
        case 'gpt-neox':

            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding,rotate_half
            rotary_emb = GPTNeoXRotaryEmbedding(head_dim,2048)
        case 'opt':
            args.RoPE = False 


    #####################
    # Normal Attention  # 
    #####################

    q_ref = q
    k_ref = torch.cat([k_cache,k.cpu()],dim=k_dim)
    v_ref = torch.cat([v_cache,v.cpu()],dim=k_dim)
    if args.RoPE:
        def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
            
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            x_embed = (x * cos) + (rotate_half(x) * sin)
            return x_embed
 
        cos, sin = rotary_emb(v_cache,kv_len)
       
        q_ref =  apply_rotary_pos_emb_single(q_ref.cpu(), cos, sin, torch.arange(kv_len-q_len,kv_len).unsqueeze(0))
        k_ref = apply_rotary_pos_emb_single(k_ref, cos, sin, torch.arange(kv_len).unsqueeze(0))
    
          
    v_reference_nomal,_ = mha_lse(q_ref.cpu() ,k_ref.cpu(),v_ref.cpu())  # b s h d 
    v_reference_nomal = v_reference_nomal.reshape(-1,batch_size,num_heads,head_dim).permute(1,0,2,3)
     

    #####################
    #   CPU Attention   # 
    #####################
    q_cpu = q.cpu() 
    k_cpu = torch.cat([slice(k_cache,start,seq_len-recent),k.cpu()],dim=k_dim)
    v_cpu = torch.cat([slice(v_cache,start,seq_len-recent),v.cpu()],dim=k_dim)
     
    if args.RoPE: 
        q_cpu = apply_rotary_pos_emb_single(q_cpu, cos, sin, torch.arange(kv_len-q_len,kv_len).unsqueeze(0))
        position_ids = torch.cat([  
                torch.arange(start, seq_len-recent, device='cpu'),
                torch.arange(kv_len-q_len, kv_len, device='cpu')
                ],dim=0).unsqueeze(0)
        k_cpu = apply_rotary_pos_emb_single(k_cpu, cos, sin,position_ids)
     
    print(log)
    st = time.time()
    va,sa = mha_lse(q_cpu,k_cpu,v_cpu)
    print(f"CPU attention length: {seq_len-start-recent}, compute time: {time.time()-st}, ")

    #####################
    #   GPU Attention   # 
    #####################
    q_gpu = q
    k_gpu = torch.cat([slice(k_cache,0,start),slice(k_cache,seq_len-recent,seq_len),k.cpu()],dim=k_dim )
    v_gpu = torch.cat([slice(v_cache,0,start),slice(v_cache,seq_len-recent,seq_len),v.cpu()] ,dim=k_dim )

    if args.RoPE:
        q_gpu = q_cpu.cuda()
        k_gpu = apply_rotary_pos_emb_single(k_gpu, cos, sin, torch.cat([torch.arange(0,start),torch.arange(seq_len-recent,kv_len)],dim=0 ).unsqueeze(0))
        
    st = time.time() 
    vb,sb = mha_lse(q_gpu ,k_gpu.cuda(),v_gpu.cuda())
    print(f"GPU attention length: {start+recent}, compute + transfer time: {time.time()-st}, ")

    
    v_out,_ = merge_state( va,sa,vb,sb )
    
    acc = check_eq(v_out.cpu(),v_reference_nomal)  
    assert  (acc>0.9),  f"accuracy {acc*100:.4}%, merge state fail..."
    print(f"Merge success, accuracy {acc*100:.4}%")


    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=50) 
    parser.add_argument("--seq_len", type=int, default=55)
    parser.add_argument("--q_len", type=int, default=1)

    # model config 
    parser.add_argument("--num_heads", type=int, default=40)
    parser.add_argument("--hidden_size", type=int, default=5120)
    parser.add_argument("--model_type", type=str, default="opt")
    parser.add_argument("--RoPE", type=bool, default=True )
    
    # test config 
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
     
    args = parser.parse_args()
    test_correctness(args,"")
     