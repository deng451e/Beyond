import torch  
import argparse
import time 
from beyond.utils import *


class selection_methods_(torch.nn.Module):
    def __init__(self,head_dim, num_heads,batched=False):
        super(selection_methods_, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        
        self.batched = batched

    
    def select_topk_kv(self, q,k,topk):
        assert check_tensor_device(q,'cpu') , f"q should be on CPU"

        if self.batched:
            qs = q.size(1)
            s  = k.size(1)
            batch_size = k.size(0)
            
            q = q.permute(0, 2, 1, 3).reshape(batch_size  * self.num_heads, qs,  self.head_dim)  # bh,qs,d
            k = k.permute(0, 2, 3, 1).reshape(batch_size  * self.num_heads, self.head_dim, s)    # bh,d,s
             
        else:
            
            q = q.permute(1, 0, 2) # h,qs,d
            k = k.permute(1, 2, 0) # h,d,s
            
        
        attn_weights = torch.bmm(q,k)  # h,qs,s  or bh,qs,s
        attn_weights = torch.sum(attn_weights,dim=-2)
      
        
        max_scores, indices = torch.topk(attn_weights, topk,dim=-1) 
       
       
        return indices
    

    def select_max_kv(self, q,k ):
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
      
         
        max_scores, indices = torch.max(attn_weights,dim=-1) 
       
        return indices


# selection_methods = selection_methods_()
# indices = selection_methods.select_topk_kv(q.cpu(), slice(k_cache,0,100),10)