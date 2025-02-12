import torch 
import torch.nn as nn
from typing import List,Tuple,Optional,AnyStr
from torch.jit import fork, wait





def masking_(model):
        
    match model:

        case "llama":
            def fn(A,A_mask):
                qs = A.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask  
                return  A

        case "opt":
            def fn(A,A_mask):
                qs = A.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask 
                A = torch.max(A, torch.tensor(torch.finfo(A.dtype).min, device=A.device))
                A = A.to(torch.float32)
                return A

        case "gpt-neox":
            def fn(A,A_mask):
                qs = A.size(1)
                mask_value  = torch.finfo(A.dtype).min
                mask_value  = torch.tensor(mask_value, dtype=A.dtype).to( A.device)
                A[:,:,-qs:] = torch.where(A_mask, A[:,:,-qs:], mask_value)
                return A
        case "flexgen-opt":
            def fn(A,A_mask):
                qs = A.size(1)
                A  = torch.where(A_mask, A[:,:,-qs:], -1e4)
                return A
            
    return fn


def apply_pos_emb_(config):
    cpu_pos_emb_cache = None
    match  config.model_type:

        case "llama":
            from transformers.models.llama.modeling_llama import  rotate_half, repeat_kv

            num_key_value_groups = config.model_type.head_num // config.num_key_value_heads
            def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
                
                cos = cos.squeeze(1).squeeze(0)       # [seq_len, dim]
                sin = sin.squeeze(1).squeeze(0)       # [seq_len, dim]
                cos = cos[position_ids]               # [bs, seq_len, dim]
                sin = sin[position_ids]               # [bs, seq_len, dim]
                x_embed = (x * cos) + (rotate_half(x) * sin)
                return x_embed
            
                

            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            if scaling_type == "linear":
                from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding
                cpu_pos_emb_cache = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=config.rope_theta,
                    device='cpu'
                )
            elif scaling_type == "dynamic":
                from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding
                cpu_pos_emb_cache = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=config.rope_theta,
                    device='cpu'
                )

            def fn(k,v,pos_ids,cos ,sin ):
                k = apply_rotary_pos_emb_single(k,  cos, sin, pos_ids)
                k = repeat_kv(k, self.num_key_value_groups)
                v = repeat_kv(v, self.num_key_value_groups)
                return k,v

        case  "gpt-neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding,rotate_half
            self.rotary_ndims = int(self.head_dim * config.rotary_pct)

            cpu_pos_emb_cache = GPTNeoXRotaryEmbedding(
                int(self.head_dim * config.rotary_pct),
                max_position_embeddings=config.max_position_embeddings,
                device='cpu')
            def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
                gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
                gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
                cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
                sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
                x_embed = (x * cos) + (rotate_half(x) * sin)
                return x_embed
            
            def fn(k,v,pos_ids,cos,sin):
                k_rot  = k[..., : self.rotary_ndims]
                k_pass = k[..., self.rotary_ndims :]
                k_rot  = apply_rotary_pos_emb_single(k_rot,  cos, sin, pos_ids)
                k = torch.cat((k_rot,k_pass), dim=-1)
                return k,v


        case "opt":
            fn = None 

        case "flexgen-opt":
            fn = None 
            
    return fn,cpu_pos_emb_cache
    

    
    


class hybrid_attention((nn.Module)):

    def __init__(self,batch_size,config):
        super().__init__()
        config   = config
        
        self.model   =  config.model_type
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size//self.head_num 
        self.max_position_embeddings = config.max_position_embeddings
         
        self.rotary_ndims = None
        self.num_key_value_groups =  None
        
         
        self.batch_size = batch_size
        
        self.masking   =  masking_(self.model)
        self.apply_pos_emb,self.cpu_pos_emb_cache = apply_pos_emb_(self.model)
 

        self.out_cpu = torch.empty(1,self.batch_size*self.head_num,self.head_dim,dtype=torch.float16).pin_memory().contiguous()
        self.lse_cpu = torch.empty(1,self.batch_size*self.head_num,dtype=torch.float32).pin_memory().contiguous()
        
        self.hybrid_mha = torch.jit.script(self.hybrid_mha_)
        self.gpu_mha = gpu_mha_
        
    
 
     
    def mha_lse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                enable_pos:  bool, cos: torch.Tensor, sin:  torch.Tensor, pos_ids: torch.Tensor,
                enable_mask: bool, mask: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        

        if enable_pos:
            k,v = self.apply_pos_emb(k,v,pos_ids,cos,sin )

        k = k.permute(0,2,1) 
         
        A = q@k  #.to(torch.float32)
        if enable_mask:
            A = self.masking(A,mask)   
        A_max, _ = A.max(dim=-1, keepdim=True) 
        A  = A - A_max
        
        e = torch.exp2(A).to(v.dtype)
        se = e.sum(dim=-1, keepdim=True)
        
        out = ((e / se) @ v).permute(1,0,2) 
        lse = (torch.log2(se) + A_max).squeeze(-1).permute(1,0).float() 
        
        return out,lse


    def hybrid_mha_(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                    k_gpu_cache: torch.Tensor, v_gpu_cache: torch.Tensor,
                    k_cpu_cache: List[torch.Tensor], v_cpu_cache:List[torch.Tensor],
                    enable_pos:  bool, cos: torch.Tensor, sin:  torch.Tensor, 
                    pos_ids_gpu:  torch.Tensor, pos_ids_cpu:  List[torch.Tensor],
                    enable_mask: bool, mask:  torch.Tensor)->torch.Tensor:
        
        
        tasks = torch.jit.annotate(List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor]]], [])

        # launch cpu attn 
        for batch_idx in range(self.batch_size):
            begin,end = batch_idx*self.head_num,batch_idx*self.head_num+self.head_num
            if enable_pos:
                cos_cpu,sin_cpu = cpu_pos_emb_cache(k_cpu_cache,torch.max(pos_ids_cpu[batch_idx]))
                tasks.append(fork(self.mha_lse, q_cpu[begin:end,:,:],k_cpu_cache[batch_idx],v_cpu_cache[batch_idx],
                                    True,cos_cpu,sin_cpu,pos_ids_cpu[batch_idx],False,None))
            else:
                tasks.append(fork(self.mha_lse, q_cpu[begin:end,:,:],k_cpu_cache[batch_idx],v_cpu_cache[batch_idx],
                                    False,None,None,None,False,None))
        # launch gpu attn 
        k_gpu = torch.cat([k_gpu_cache,k],dim=-2)
        v_gpu = torch.cat([v_gpu_cache,v],dim=-2)
        out_gpu,lse_gpu = self.mha_lse(q ,k_gpu,v_gpu,enable_pos,cos,sin,pos_ids_gpu,enable_mask,mask)

        # sync cpu attn 
        for batch_idx,task in enumerate(tasks):
            begin,end = batch_idx*self.head_num,batch_idx*self.head_num+self.head_num
            self.out_cpu[:,begin:end,:],self.lse_cpu[:,begin:end] = wait(task)
                
        # merge results 
         
        
        tasks =  torch.jit.annotate(List[torch.jit.Future[None]], [])
        for batch_idx in range(self.batch_size):
            begin,end = batch_idx*self.head_num,batch_idx*self.head_num+self.head_num
            tasks.append(fork(flashinfer.merge_state_in_place,out_gpu[:,begin:end,:],
                                                             lse_gpu[:,begin:end],
                                                             out_cpu[:,begin:end,:],
                                                             lse_cpu[:,begin:end]))
        for task in tasks:
            wait(task)

      
        
        return out 
    
    
    
    def gpu_mha_(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,
                enable_pos:  bool, cos: torch.Tensor, sin:  torch.Tensor, pos_ids:  torch.Tensor, 
                enable_mask: bool, mask: torch.Tensor)-> torch.Tensor:
        """
        Input Args:
        q,k,v:  torch tensor (b,h,qs,d)
        A_mask: mask for casual attention (b,h,qs,qs)

        Return Args:
        out: torch tensor (qs,bh,d)
        lse: torch tensor (qs,bh) or None
        """

        if enable_pos:
            k,v = self.apply_pos_emb(k,v,pos_ids,cos,sin)
        k = k.permute(0,2,1) 
        A = q@k #.to(torch.float32)
        if enable_mask:
            A = self.masking(A,A_mask)   
        A_max, _ = A.max(dim=-1, keepdim=True) 
        A  = A - A_max
        e = torch.exp2(A).to(v.dtype)
        se = e.sum(dim=-1, keepdim=True)
        out = ((e / se) @ v).permute(1,0,2) 

        return out 
    
      
                 

  



    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,
                k_gpu_cache:  torch.Tensor, v_gpu_cache:  torch.Tensor, k_cpu_cache: List[torch.Tensor], v_cpu_cache:  List[torch.Tensor],
                cos:  torch.Tensor,sin:  torch.Tensor,pos_ids_gpu:  torch.Tensor, pos_ids_cpu: List[torch.Tensor] ,mask:  torch.Tensor) -> torch.Tensor:
       
        qs =  q.size(-2) 
        
        if  self.lse_cpu.size(1)==qs:
            self.out_cpu = torch.empty(qs,self.batch_size*self.head_num,self.head_dim,dtype=torch.float16).pin_memory().contiguous()
            self.lse_cpu = torch.empty(qs,self.batch_size*self.head_num,dtype=torch.float32).pin_memory().contiguous()

        q = q.view(-1,qs,self.head_dim)
        k = k.view(-1,qs,self.head_dim)
        v = v.view(-1,qs,self.head_dim)
        
        enable_pos  = True if self.apply_pos_emb else False 
        enable_mask = True if mask is None else False 
        
        if k_cpu_cache is not None:
            out = self.hybrid_mha_(q,k,v,k_gpu_cache , v_gpu_cache ,k_cpu_cache , v_cpu_cache ,
                  enable_pos , cos , sin , pos_ids_gpu , pos_ids_cpu ,enable_mask , mask )
        else:
            k_gpu = torch.cat([k_gpu_cache,k],dim=-2)
            v_gpu = torch.cat([v_gpu_cache,v],dim=-2)
            out = self.gpu_mha(q,k_gpu,v_gpu,enable_pos,cos,sin,pos_id,enable_mask,mask)
          
       
        return out.reshape(qs,-1,self.head_num,self.head_dim).permute(1,0,2,3),k,v
    
 
hybrid_attn  = hybrid_attention( 10,model.config)
 
 