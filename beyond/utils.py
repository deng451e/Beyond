
import torch  
 
import argparse
import os.path as osp

 
 

 

def add_info(args,log):
    idx = 1 
    for arg, value in vars(args).items():
        log += f"{arg}:{value}, "
        if idx%4==0: log += "\n"
        idx +=1 
    log += "\n=========================="
    return log 

def check_workspace(ratio):
    if 0<ratio<1:
        print("mix attention...")
    elif ratio==0:
        print("gpu attention...")
    else:
        print("cpu attention...")

def check_memory(x,name):
    print(f"{name} is pinned: {x.is_pinned()}")
    print(f"{name} is contiguous: {x.is_contiguous()}")
    print(f"{name} dtype: {x.dtype}")
    print(f"{name} device: {x.device}")
    print('=================')

def check_eq(x,y):
    #	FP16 has a precision of about 3 to 4 decimal digits.
    return (torch.isclose(x.cpu(), y.cpu(), rtol=1e-3, atol=1e-4).sum()/torch.numel(x)).cpu().numpy()
    
def check_dtype(x,type_):
    return type(x.dtype)==type(type_)



def check_tensor_device(x,type_):

    return x.device.type==type_
 




def slice0d(x, start, end):
    return x[start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

 

DIM_TO_SLICE = {
    0: slice0d,
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


def mha_logSum( q,k,v,attention_mask=None):
            
        
        #shape: b,h,s,d
        batch_size = q.size(0)
        num_heads  = q.size(1)
        qs         = q.size(2)
        head_dim   = q.size(3)
        
        scaling = head_dim ** -0.5
         
        
        q = q.reshape(batch_size  * num_heads, -1,  head_dim)* scaling # bh,qs,d
        k = k.permute(0,1,3,2).reshape(batch_size  * num_heads, head_dim, -1) # bh,d,s
        v = v.reshape(batch_size  * num_heads, -1, head_dim) # bh,s,d
        attn_weights = torch.bmm(q,k)   # bh,qs,s 
       
        
        if attention_mask is not None: 
            attn_weights[:,:,-qs:] = attn_weights[:,:,-qs:] + attention_mask # bh,qs,s 
        
        
         

        max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
        exp_scores = torch.exp(attn_weights - max_scores) 
        sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
        log_sum = (torch.log(sum_exp_scores)  + max_scores) * torch.tensor(1.4427) 
        attn_weights = exp_scores / sum_exp_scores 
       
        value  = torch.bmm(attn_weights, v).permute(1,0,2).half().contiguous()
        log_sum = log_sum.squeeze(-1).permute(1,0).contiguous().float() 
        
        
        if check_tensor_device(q,'cpu'):  return  value.pin_memory(),log_sum.pin_memory()
        if check_tensor_device(q,'cuda'): return  value, log_sum
            
            
             
  

def mha_normal(self,q,k,v,attention_mask=None):
    # b h s d 
    qs = q.size(-2)
    attn_weights = torch.matmul(q, k.transpose(2, 3))  * (q.size(-1) ** -0.5)

    if attention_mask is not None: 
            attn_weights[:,:,-qs:,:] = attn_weights[:,:,-qs:,:] + attention_mask.to(attn_weights.device) # bh,qs,s 
        
    max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
    attn_weights = attn_weights - max_scores

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    
    return  attn_output.transpose(1, 2).contiguous() 
 