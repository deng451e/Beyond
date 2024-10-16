import torch  
import flashinfer 
import argparse
import time 
from utils import *

class attention_methods_(torch.nn.Module):
    def __init__(self,head_dim, num_heads,batched=False):
        super(attention_methods_, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.scaling = head_dim ** -0.5
        self.batched = batched

    
    def mha(self, q,k,v,blk_k=None):
            
        if self.batched:
            s  = k.size(1)
            qs = q.size(1)
            batch_size = k.size(0)
            
            q = q.permute(0, 2, 1, 3).reshape(batch_size  * self.num_heads, qs,  self.head_dim)* self.scaling # bh,qs,d
            k = k.permute(0, 2, 3, 1).reshape(batch_size  * self.num_heads, self.head_dim, s) # bh,d,s
            v = v.permute(0, 2, 1, 3).reshape(batch_size  * self.num_heads, s, self.head_dim) # bh,s,d
        else:
            s  = k.size(0)
            qs = q.size(0)
            q = q.permute(1, 0, 2) * self.scaling # h,qs,d
            k = k.permute(1, 2, 0) # h,d,s
            v = v.permute(1, 0, 2) # h,s,d
            
        attn_weights = torch.bmm(q,k)  
        max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
        exp_scores = torch.exp(attn_weights - max_scores) 
        sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
        log_sum = (torch.log(sum_exp_scores)  + max_scores)*torch.tensor(1.4427) 
        attn_weights = exp_scores / sum_exp_scores 
        value  = torch.bmm(attn_weights, v).permute(1,0,2) 
        if check_dtype(value,torch.float32): value = value.half() 
        if check_dtype(log_sum,torch.float16): log_sum = log_sum.float() 
        
        if check_tensor_device(q,'cpu'):  return  value.contiguous().pin_memory(),log_sum.squeeze(-1).permute(1,0).contiguous().pin_memory()
        if check_tensor_device(q,'cuda'): return value.contiguous(), log_sum.squeeze(-1).permute(1,0).contiguous()
    
    
    def select_topk_kv(self, q,k,topk=None):
        assert check_tensor_device(q,'cpu') , f"q should be on CPU"

        if self.batched:
            qs = q.size(1)
            s  = k.size(1)
            batch_size = k.size(0)
            
            q = q.permute(0, 2, 1, 3).reshape(batch_size  * self.num_heads, qs,  self.head_dim)* self.scaling # bh,qs,d
            k = k.permute(0, 2, 3, 1).reshape(batch_size  * self.num_heads, self.head_dim, s) # bh,d,s
             
        else:
            
            q = q.permute(1, 0, 2) * self.scaling # h,qs,d
            k = k.permute(1, 2, 0) # h,d,s
            
        
        attn_weights = torch.bmm(q,k)  # h,qs,s  or bh,qs,s
        attn_weights = torch.sum(attn_weights,dim=-2)
      
        if topk:
            max_scores, indices = torch.topk(attn_weights, topk,dim=-1) 
        else:
            max_scores, indices = torch.max(attn_weights,dim=-1) 
       
        return indices
 


 
#wrapper for flashinfer merge state functions 
class merge_state_:
    def __init__(self,num_heads,batched=False,in_place=False):
       self.num_heads = num_heads
       self.batched = batched
       self.in_place = in_place

    def __call__(self, va,sa,vb,sb):
       
        assert check_tensor_device(va,'cpu') , f"va should be on CPU"
        assert check_tensor_device(sa,'cpu') , f"sa should be on CPU"
        assert check_tensor_device(vb,'cuda') , f"vb should be on GPU"
        assert check_tensor_device(sb,'cuda') , f"sb should be on GPU"
        assert va.is_pinned() and va.is_contiguous(), f"va should be on pinned and contiguous"
        assert sa.is_pinned() and va.is_contiguous(), f"sa should be on pinned and contiguous"

         
        head_dim = va.size(-1)
        if self.in_place:
            flashinfer.merge_state_in_place(va,sa,vb,sb)
            v_out,s_out = vb,sb
        else:
            """
            The maximum compute capability of the NVIDIA GPU is 1024 threads per block. 
            Therefore, we partition the head and batch dimensions accordingly.
            """
            partitions_dim = 7680//head_dim
            if partitions_dim<va.size(-2):
                v_out,s_out = [],[]
                num_partition = (va.size(-2)+partitions_dim-1)//partitions_dim
                for i in range(num_partition):
                    start,end = i*partitions_dim,min((i+1)*partitions_dim,va.size(-2))
                    va_,sa_ = va[:,start:end,:].contiguous().pin_memory(),sa[:,start:end].contiguous().pin_memory()
                    vb_,sb_ = vb[:,start:end,:].contiguous(),sb[:,start:end].contiguous()
                    v_out_,s_out_ =  flashinfer.merge_state(   va_,sa_, vb_,sb_ )
                    v_out.append(v_out_);s_out.append(s_out_) 

                v_out,s_out = torch.cat(v_out,dim=1),torch.cat(s_out,dim=1)
              
            else:
                 
                v_out,s_out = flashinfer.merge_state(  va,sa,vb,sb)

        if self.batched:
             
            s,d = v_out.size(0),v_out.size(2)
            v_out = v_out.reshape(s,-1,self.num_heads,d).permute(1,0,2,3) # s,b,h,d 

        # s_out shape: s,bh
        return v_out,s_out
    







# def test_correctness(args,log):
#     log = add_info(args,log)
#     arch_name = args.arch_name
#     if arch_name == "opt-1.3b":
    
#         num_heads=32; hidden_size=2048  

#     elif arch_name == "opt-2.7b":
    
#         num_heads=32; hidden_size=2560  
        
#     elif arch_name == "opt-6.7b":

#         num_heads=32; hidden_size=4096  

#     elif arch_name == "opt-13b":

#         num_heads=40; hidden_size=5120
     

#     seq_len = args.seq_len
#     head_dim = hidden_size//num_heads
#     ratio = args.ratio
#     partial_len = int( seq_len*ratio )
#     batch_size = args.batch_size
#     print(f"batch_size:{batch_size}")
     
    
#     assert  (partial_len>0),  f"Partial length can't be 0 ..."
#     if batch_size==1:
#         k_cache = torch.randn(seq_len, num_heads,head_dim, device='cpu').half()
#         v_cache = torch.randn(seq_len, num_heads,head_dim, device='cpu').half()
#         q       = torch.randn(args.q_len, num_heads,head_dim, device='cuda:0').half()
#         merge_state = merge_state_(num_heads )
#         attention_methods = attention_methods_(head_dim, num_heads)
#     else:
#         k_cache = torch.randn(batch_size,seq_len, num_heads,head_dim, device='cpu').half()
#         v_cache = torch.randn(batch_size,seq_len, num_heads,head_dim, device='cpu').half()
#         q       = torch.randn(batch_size,args.q_len, num_heads,head_dim, device='cuda:0').half()
#         merge_state = merge_state_(num_heads,batched=True)
#         attention_methods         = attention_methods_(head_dim, num_heads,batched=True)
#     slice = DIM_TO_SLICE[0 if batch_size==1 else 1]


#     indices = attention_methods.select_topk_kv(q.cpu(), slice(k_cache,0,100),10)
    
#     v_reference,_ = attention_methods.mha(q ,k_cache.cuda(),v_cache.cuda())
#     if batch_size>1:
#         v_reference = v_reference.reshape(args.q_len,-1,num_heads,head_dim).permute(1,0,2,3)
  

#     st = time.time()
#     va,sa = attention_methods.mha(q.cpu() ,slice(k_cache,0,partial_len)  ,slice(v_cache,0,partial_len))
#     print(f"CPU attention length: {partial_len}, attention time: {time.time()-st}, ")

    
#     st = time.time()
#     vb,sb = attention_methods.mha(q ,slice(k_cache,partial_len,seq_len).cuda(),slice(v_cache,partial_len,seq_len).cuda())
#     print(f"GPU attention length: {seq_len-partial_len}, attention time: {time.time()-st}, ")

    
#     v_out,_ = merge_state( va,sa,vb,sb )
    
#     acc = check_eq(v_out,v_reference)  
#     assert  (acc>0.9),  f"accuracy {acc*100:.4}%, merge state fail..."
#     print(f"Merge success, accuracy {acc*100:.4}%")


    
    




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--arch_name", type=str, default="opt-13b") 
#     parser.add_argument("--seq_len", type=int, default=100000)
#     parser.add_argument("--q_len", type=int, default=10)
#     parser.add_argument("--ratio", type=float, default=0.1)
#     parser.add_argument("--repeat", type=int, default=10)
#     parser.add_argument("--batch_size", type=int, default=1)
#     args = parser.parse_args()
#     test_correctness(args,"")
     