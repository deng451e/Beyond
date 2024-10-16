
import torch  
import argparse

def add_info(args,log):
    log += f"arch name:{args.arch_name},"
    log += f"seq len:{args.seq_len},"
    log += f"q len:{args.q_len},"
    log += f"cpu ratio:{args.ratio},"
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
    return (torch.isclose(x.cpu(), y.cpu(), rtol=1e-3, atol=1e-3).sum()/torch.numel(x)).cpu().numpy()
    
def check_dtype(x,type_):
    return type(x.dtype)==type(type_)


class flashAttnSingle(torch.nn.Module):
    def __init__(self,head_dim, num_heads,device):
        super(flashAttnSingle, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.scaling = head_dim ** -0.5
        self.device = device 
    def forward(self, q,k,v,mask=None):
        # k,v shape: s,h,d
        # q  shape: qs,h,d
        s  = k.size(0)
        qs = q.size(0)
        q = q.permute(1, 0, 2)* self.scaling # h,qs,d

        if self.device=="cpu":
           
            if mask!=None:
                indices = torch.nonzero(mask).squeeze()
                k = k[indices,:,:]
                v = v[indices,:,:]
             
            k = k.permute(1, 2, 0) # h,d,s
            v = v.permute(1, 0, 2) # h,s,d
            
            attn_weights = torch.bmm(q,k) # h,qs,s
            

            max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
            exp_scores = torch.exp(attn_weights - max_scores) 
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            
            log_sum = (torch.log(sum_exp_scores)  + max_scores)*torch.tensor(1.4427) 
             
            attn_weights = exp_scores / sum_exp_scores 

            value  = torch.bmm(attn_weights, v).permute(1,0,2) 
             
            if check_dtype(value,torch.float32): value = value.half() 
            if check_dtype(log_sum,torch.float16): log_sum = log_sum.float() 

            return  value.contiguous().pin_memory(),log_sum.squeeze(-1).permute(1,0).contiguous().pin_memory()

               
        else:
             
         
           
            k = k.permute(1, 2, 0)  # h,d,s
             
            v = v.permute(1, 0, 2)  # h,s,d
             
            attn_weights = torch.bmm(q, k) # h,qs,s
            
            # if mask==None:
            #     logSum = torch.log( torch.sum(torch.exp(attn_weights),dim=-1) ).permute(1,0) # qs,h

               
            # else:

            #     mask = mask.view( 1, 1, s) 
                
            #     logSum =torch.sum(torch.exp( torch.where(mask, attn_weights, 0)).permute(1,0,2),dim=-1)   

            #     attn_weights = torch.where(mask, attn_weights, -1e4)

            max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
            exp_scores = torch.exp(attn_weights - max_scores) 
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            log_sum = (torch.log(sum_exp_scores)  + max_scores)*torch.tensor(1.4427) 
            attn_weights = exp_scores / sum_exp_scores 
            
             
            value = torch.bmm(attn_weights, v).permute(1,0,2)
           


             
            if check_dtype(value,torch.float32): value = value.half() 
            if check_dtype(log_sum,torch.float16):log_sum = log_sum.float() 
            
            return value.contiguous(), log_sum.squeeze(-1).permute(1,0).contiguous()
    




class flashAttnBatch(torch.nn.Module):
    def __init__(self,batch_size,head_dim, num_heads,device):
        super(flashAttnBatch, self).__init__()
         
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.scaling = head_dim ** -0.5
        self.device = device 
    def forward(self, q,k,v,mask=None):
        # k,v shape: b,s,h,d
        # q   shape: b,qs,h,d
        s  = k.size(1)
        qs = q.size(1)
        q = q.permute(0,2,1,3).reshape(self.batch_size  * self.num_heads, qs,  self.head_dim)* self.scaling # bh,qs,d

        if self.device=="cpu":
           
            if mask!=None:
                indices = torch.nonzero(mask).squeeze()
                k = k[indices,:,:]
                v = v[indices,:,:]
             
            k = k.permute(0, 2, 3, 1).reshape(self.batch_size  * self.num_heads, self.head_dim, s) # bh,d,s
            v = v.permute(0, 2, 1, 3).reshape(self.batch_size  * self.num_heads, s, self.head_dim) # bh,s,d
            
            attn_weights = torch.bmm(q,k) # bh,qs,s
            
            max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
            exp_scores = torch.exp(attn_weights - max_scores) 
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            
            log_sum = (torch.log(sum_exp_scores)  + max_scores)*torch.tensor(1.4427) 
             
            attn_weights = exp_scores / sum_exp_scores 

            value  = torch.bmm(attn_weights, v).permute(1,0,2) 
             
            if check_dtype(value,torch.float32): value = value.half() 
            if check_dtype(log_sum,torch.float16): log_sum = log_sum.float() 

            return  value.contiguous().pin_memory(),log_sum.squeeze(-1).permute(1,0).contiguous().pin_memory()

               
        else:
             
         
            k = k.permute(0, 2, 3, 1).reshape(self.batch_size  * self.num_heads, self.head_dim, s) # bh,d,s
            v = v.permute(0, 2, 1, 3).reshape(self.batch_size  * self.num_heads, s, self.head_dim) # bh,s,d
             
             
            attn_weights = torch.bmm(q, k) # h,qs,s
            
            # if mask==None:
            #     logSum = torch.log( torch.sum(torch.exp(attn_weights),dim=-1) ).permute(1,0) # qs,h

               
            # else:

            #     mask = mask.view( 1, 1, s) 
                
            #     logSum =torch.sum(torch.exp( torch.where(mask, attn_weights, 0)).permute(1,0,2),dim=-1)   

            #     attn_weights = torch.where(mask, attn_weights, -1e4)

            max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
            exp_scores = torch.exp(attn_weights - max_scores) 
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            log_sum = (torch.log(sum_exp_scores)  + max_scores)*torch.tensor(1.4427) 
            attn_weights = exp_scores / sum_exp_scores 
            
             
            value = torch.bmm(attn_weights, v).permute(1,0,2)
           


             
            if check_dtype(value,torch.float32): value = value.half() 
            if check_dtype(log_sum,torch.float16):log_sum = log_sum.float() 
            
            return value.contiguous(), log_sum.squeeze(-1).permute(1,0).contiguous()
    
