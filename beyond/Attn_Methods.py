 
import torch 
import flashinfer 
import torch.nn as nn
from torch.jit import fork, wait
from typing import List,Tuple,Optional
 
 
 
def mha_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
            cos: torch.Tensor, sin:  torch.Tensor, pos_ids: torch.Tensor,
            enable_mask: bool, mask: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:

    k,v = apply_pos_emb(k,v,pos_ids,cos,sin )
    k = k.permute(0,2,1) 
        
    A = q@k 
    if enable_mask:
        A = masking(A,mask)   
    A_max, _ = A.max(dim=-1, keepdim=True) 
    A  = A - A_max
    
    e = torch.exp2(A).to(v.dtype)
    se = e.sum(dim=-1, keepdim=True)
    A = e / se
    out = (A @ v).permute(1,0,2) 
    lse = (torch.log2(se) + A_max).squeeze(-1).permute(1,0).float() 
    
    return out,lse,A

 
def mha_lse_wo_pos( q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                    enable_mask: bool, mask: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    

    k = k.permute(0,2,1) 
    A = q@k  
    if enable_mask:
        A = masking(A,mask)   
    
    A_max, _ = A.max(dim=-1, keepdim=True) 
    A  = A - A_max
    
    e = torch.exp2(A).to(v.dtype)
    se = e.sum(dim=-1, keepdim=True)
    A = e / se
    out = (A @ v).permute(1,0,2) 
    lse = (torch.log2(se) + A_max).squeeze(-1).permute(1,0).float() 
    
    return out,lse,A

 

def masking_(config):
        
    match config.model_type:
 
        case "llama":
            def fn(A,A_mask):
                qs = A.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask  
                return  A

        case "opt":
            def fn(A,A_mask):
                qs = A.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask 
                A = torch.max(A, torch.tensor(-65504.0, device=A.device))
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


def get_pos_emb_cache_(config):
    head_num = config.num_attention_heads
    head_dim = config.hidden_size/head_num 
   
    match  config.model_type:

        case "llama":
          
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            if scaling_type == "linear":
                from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding
                cpu_pos_emb_cache = LlamaLinearScalingRotaryEmbedding(
                    head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=config.rope_theta,
                    device='cpu'
                )
            elif scaling_type == "dynamic":
                from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding
                cpu_pos_emb_cache = LlamaDynamicNTKScalingRotaryEmbedding(
                    head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=config.rope_theta,
                    device='cpu'
                )

          

        case  "gpt-neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding 
            cpu_pos_emb_cache = GPTNeoXRotaryEmbedding(
                int(head_dim * config.rotary_pct),
                max_position_embeddings=config.max_position_embeddings,
                device='cpu')
           
        case "opt":
            cpu_pos_emb_cache = None
            

        case "flexgen-opt":
            cpu_pos_emb_cache = None
 
    return  cpu_pos_emb_cache


def apply_pos_emb_(config):
    
    head_num = config.num_attention_heads
    head_dim = config.hidden_size/head_num 
   
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
            
                
            def fn(k,v,pos_ids,cos ,sin ):
                k = apply_rotary_pos_emb_single(k,  cos, sin, pos_ids)
                k = repeat_kv(k,  num_key_value_groups)
                v = repeat_kv(v,  num_key_value_groups)
                return k,v

        case  "gpt-neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import rotate_half
            rotary_ndims = int(head_dim * config.rotary_pct)

           
            def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
                gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
                gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
                cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
                sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
                x_embed = (x * cos) + (rotate_half(x) * sin)
                return x_embed
            
            def fn(k,v,pos_ids,cos,sin):
                k_rot  = k[..., : rotary_ndims]
                k_pass = k[..., rotary_ndims :]
                k_rot  = apply_rotary_pos_emb_single(k_rot,  cos, sin, pos_ids)
                k = torch.cat((k_rot,k_pass), dim=-1)
                return k,v


        case "opt":
            def fn(k,v,pos_ids,cos,sin):
                return k,v
                 

        case "flexgen-opt":
            def fn(k,v,pos_ids,cos,sin):
                return k,v
            
    return fn 
    
 
    


class AttnMethods(nn.Module):

    def __init__(self,batch_size,config,enable_pos=False,attn_device="cpu"):
        super(AttnMethods,self).__init__()
        
        self.model   =  config.model_type
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size//self.head_num 
        self.batch_size = batch_size
        self.enable_pos = enable_pos
        self.attn_device = attn_device
          

        self.split_dim  =  8192//self.head_dim
      
        self.gpu_mha    = torch.jit.script(self.mha)  
        if self.enable_pos:
            self.hybrid_mha = torch.jit.script(self.hybrid_mha_pos)
        else:
            self.hybrid_mha = torch.jit.script(self.hybrid_mha_wo_pos)  
        

    def merge_state(self,out_gpu: torch.Tensor,lse_gpu: torch.Tensor,
                         out_cpu: torch.Tensor,lse_cpu: torch.Tensor)-> torch.Tensor:
        
        bh = out_gpu.size(1)
        
        # merge appaned states 
        if out_gpu.size(0)!=1:
            out = torch.empty_like(out_gpu)
            out_cpu,lse_cpu = out_cpu.to(out_gpu.device),lse_cpu.to(out_gpu.device)
            for idx in range((bh+self.split_dim-1)//self.split_dim):
                st,ed = idx* self.split_dim,min(bh,idx* self.split_dim+ self.split_dim)
                va,sa = out_gpu[:,st:ed,:].contiguous(),lse_gpu[:,st:ed].contiguous()
                vb,sb = out_cpu[:,st:ed,:].contiguous(),lse_cpu[:,st:ed].contiguous()
                out[:,st:ed,:],_ = flashinfer.merge_state(va,sa,vb,sb)
                                                                        
        # merge decode states    
        else:
           
            out_gpu,lse_gpu = out_gpu.contiguous(),lse_gpu.contiguous()
            out_cpu,lse_cpu = out_cpu.contiguous(),lse_cpu.contiguous()
            for idx in range((bh+self.split_dim-1)//self.split_dim):
                st,ed = idx* self.split_dim,min(bh,idx* self.split_dim+ self.split_dim)
                flashinfer.merge_state_in_place(out_gpu[:,st:ed,:],
                                                lse_gpu[:,st:ed],
                                                out_cpu[:,st:ed,:],
                                                lse_cpu[:,st:ed])
            out = out_gpu
        return out 
 
  

    @staticmethod
    def hybrid_mha_pos( q: torch.Tensor, k: torch.Tensor, v: torch.Tensor ,
                        k_cache_gpu: torch.Tensor, v_cache_gpu: torch.Tensor,
                        k_cache_cpu: List[torch.Tensor], v_cache_cpu:List[torch.Tensor], 
                        cos_gpu: torch.Tensor, sin_gpu: torch.Tensor,pos_ids_gpu: torch.Tensor,
                        cos_cpu: List[torch.Tensor], sin_cpu: List[torch.Tensor], pos_ids_cpu:List[torch.Tensor],
                        enable_mask: bool, mask: torch.Tensor
                        )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,
                                 torch.Tensor,torch.Tensor,List[torch.Tensor]]:
        
        batch_size, head_num, qs, head_dim = q.size()
        q = q.view(-1,qs,head_dim)
     

        q_cpu = q.to('cpu' )


        # launch cpu attn 
        tasks = torch.jit.annotate(List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor]]], [])
        parts = torch.jit.annotate(List[Tuple[int,int]], [])
        
        st,ed = 0,0
        
        for idx in range(len(k_cache_cpu)):
            ed = st+k_cache_cpu[idx].size(0)
            
            tasks.append(fork( mha_lse, q_cpu[st:ed,:,:],k_cache_cpu[idx],v_cache_cpu[idx],
                                        cos_cpu[idx],sin_cpu[idx],pos_ids_cpu[idx],False,mask))
   
            parts.append((st,ed))
            st = ed 
        out_cpu = torch.empty(qs,batch_size*head_num,head_dim,dtype=torch.float16).pin_memory() 
        lse_cpu = torch.empty(qs,batch_size*head_num,dtype=torch.float32).pin_memory() 
        A_cpu = torch.jit.annotate(List[torch.Tensor], [])
        
        # launch gpu attn 
        k_gpu = torch.cat([k_cache_gpu,k.view(-1,qs,head_dim)],dim=-2)
        v_gpu = torch.cat([v_cache_gpu,v.view(-1,qs,head_dim)],dim=-2)
        out_gpu,lse_gpu,A_gpu = mha_lse(q ,k_gpu,v_gpu,enable_mask,mask,cos_gpu,sin_gpu,pos_ids_gpu,enable_mask,mask)
        

       

        # sync cpu attn 
        for task,(st,ed) in zip(tasks,parts):
            out_cpu[:,st:ed,:],lse_cpu[:,st:ed],hold = wait(task)
            A_cpu.append(hold)
                
    
        
        return out_gpu,lse_gpu,A_gpu ,out_cpu,lse_cpu,A_cpu
    

    @staticmethod
    def hybrid_mha_wo_pos(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor ,
                          k_cache_gpu: torch.Tensor, v_cache_gpu: torch.Tensor,
                          k_cache_cpu: List[torch.Tensor], v_cache_cpu:List[torch.Tensor],
                          enable_mask: bool, mask: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,
                                                                        torch.Tensor,torch.Tensor,List[torch.Tensor]]:
            
        batch_size, head_num, qs, head_dim = q.size()
        q = q.view(-1,qs,head_dim)
        # q_cpu = q.to("cpu", non_blocking=True)
        q_cpu = q.to(k_cache_cpu[0].device, non_blocking=True)
         

        # launch cpu attn 
        tasks = torch.jit.annotate(List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], [])
        parts = torch.jit.annotate(List[Tuple[int,int]], [])
        
        st,ed = 0,0
        
        for idx in range(len(k_cache_cpu)):
            ed = st+k_cache_cpu[idx].size(0)
             
            tasks.append(fork( mha_lse_wo_pos, q_cpu[st:ed,:,:],k_cache_cpu[idx],v_cache_cpu[idx],False,mask))
            parts.append((st,ed))
            st = ed 
        out_cpu = torch.empty(qs,batch_size*head_num,head_dim,dtype=torch.float16).pin_memory() 
        lse_cpu = torch.empty(qs,batch_size*head_num,dtype=torch.float32).pin_memory() 
        A_cpu = torch.jit.annotate(List[torch.Tensor], [])
        
        # launch gpu attn 
        k_gpu = torch.cat([k_cache_gpu,k.view(-1,qs,head_dim)],dim=-2)
        v_gpu = torch.cat([v_cache_gpu,v.view(-1,qs,head_dim)],dim=-2)
        out_gpu,lse_gpu,A_gpu = mha_lse_wo_pos(q ,k_gpu,v_gpu,enable_mask,mask)
        
        # sync cpu attn
         
        for task,(st,ed) in zip(tasks,parts):
            out_cpu[:,st:ed,:],lse_cpu[:,st:ed],hold = wait(task)
            A_cpu.append(hold)
        return out_gpu,lse_gpu,A_gpu ,out_cpu,lse_cpu,A_cpu
    

    
    @staticmethod
    def mha(q: torch.Tensor,k: torch.Tensor,v: torch.Tensor, 
             enable_pos:  bool, cos: torch.Tensor, sin:  torch.Tensor, pos_ids:  torch.Tensor, 
             enable_mask: bool, mask: torch.Tensor)-> Tuple[torch.Tensor,torch.Tensor]:
 
        
        if enable_pos:
            k,v = apply_pos_emb(k,v,pos_ids,cos,sin)
        k = k.permute(0,2,1) 
        A = q@k 
        if enable_mask:
            A =  masking(A,mask)   
        A_max, _ = A.max(dim=-1, keepdim=True) 
        A  = A - A_max
        e = torch.exp2(A).to(v.dtype)
        se = e.sum(dim=-1, keepdim=True)
        A  = e / se
        out = (A @ v) 

        return out,A
     

   

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,
                k_cache_gpu:  torch.Tensor, v_cache_gpu:  torch.Tensor, k_cache_cpu: List[torch.Tensor], v_cache_cpu:  List[torch.Tensor],
                mask: Optional[torch.Tensor]=None,
                cos_gpu: Optional[torch.Tensor]=None, sin_gpu: Optional[torch.Tensor]=None,pos_ids_gpu:  Optional[torch.Tensor]=None,
                cos_cpu: Optional[List[torch.Tensor]]=None, sin_cpu: Optional[List[torch.Tensor]]=None, pos_ids_cpu: Optional[List[torch.Tensor]]=None
                ) -> torch.Tensor:
       

        qs =  q.size(-2) 

   
        
        if mask:
            enable_mask = True  
        else:
            enable_mask = False  
            mask = torch.tensor(0,device=q.device)
        
        if k_cache_cpu:
            if self.enable_pos:
                out_gpu,lse_gpu,out_cpu,lse_cpu = self.hybrid_mha(q,k,v,k_cache_gpu , v_cache_gpu ,k_cache_cpu , v_cache_cpu, 
                                                                  cos_gpu, sin_gpu, cos_cpu , sin_cpu , pos_ids_gpu , pos_ids_cpu ,enable_mask,mask )
            else:
                out_gpu,lse_gpu,A_gpu,out_cpu,lse_cpu,A_cpu = self.hybrid_mha(q,k,v,k_cache_gpu , v_cache_gpu ,k_cache_cpu , v_cache_cpu, enable_mask,mask )
                 
            out = self.merge_state(out_gpu,lse_gpu,out_cpu,lse_cpu)
            out = out.reshape(qs,self.batch_size,self.head_num,self.head_dim).permute(1,0,2,3) 
            
        else:
                
            q = q.view(-1,qs,self.head_dim)
            if k_cache_gpu:
                k_gpu = torch.cat([k_cache_gpu,k.view(-1,qs,self.head_dim)],dim=-2)  
                v_gpu = torch.cat([v_cache_gpu,v.view(-1,qs,self.head_dim)],dim=-2)  
            else:
                k_gpu = k.view(-1,qs,self.head_dim)
                v_gpu = v.view(-1,qs,self.head_dim)
            out,A_gpu = self.mha(q,k_gpu,v_gpu,self.enable_pos,cos_gpu, sin_gpu,pos_ids_gpu,enable_mask,mask).reshape(-1,self.head_num,qs,self.head_dim)
            A_cpu = None
       
        return out,A_gpu,A_cpu
 



#####################
#      For Test     # 
#####################
 


def run_test(args):
     
    torch.set_num_threads(args.num_thread) 
    config = AutoConfig.from_pretrained(args.model_name)
     
    print_args_info(args)
    

     
    config = config
    repeat = args.repeat
    batch_size = args.batch_size
    head_num = config.num_attention_heads
    head_dim = config.hidden_size//head_num 
    gpu_cache_size = args.gpu_cache_size
    cpu_cache_size = args.cpu_cache_size
    qs = args.qs
    head_split_num = args.head_split_num
    print(head_num)
    global apply_pos_emb
    global masking   

    id = 0
    apply_pos_emb = apply_pos_emb_(config)
    masking = masking_(config)
    
    hybrid_attn  = AttnMethods( batch_size, config)
    head_num = config.num_attention_heads
    head_dim = config.hidden_size//head_num 
    
    
    q = torch.randn(batch_size,head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    k = torch.randn(batch_size,head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    v = torch.randn(batch_size,head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    k_cache_cpu_   = torch.randn(batch_size*head_num,cpu_cache_size,head_dim,dtype=torch.float16)
    v_cache_cpu_   = torch.randn(batch_size*head_num,cpu_cache_size,head_dim,dtype=torch.float16)

    k_cache_gpu = torch.randn(batch_size*head_num,gpu_cache_size,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    v_cache_gpu = torch.randn(batch_size*head_num,gpu_cache_size,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    
    hdim = head_num//head_split_num
    k_cache_cpu = [k_cache_cpu_[idx*hdim:idx*hdim+hdim,:,:].contiguous() for idx in  range(batch_size*head_split_num)]
    v_cache_cpu = [v_cache_cpu_[idx*hdim:idx*hdim+hdim,:,:].contiguous() for idx in  range(batch_size*head_split_num)]
        
 
     
    
    

    k_cache_cpu_gpu = k_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim)
    v_cache_cpu_gpu = v_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim)

    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q.view(-1,qs,head_dim)
        k_ = k.view(-1,qs,head_dim)
        v_ = v.view(-1,qs,head_dim)
        k_gpu = torch.cat([k_cache_cpu_gpu,k_cache_gpu,k_],dim=-2)
        v_gpu = torch.cat([v_cache_cpu_gpu,v_cache_gpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_gpu,v_gpu,False,None,None,None,False,None)
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3) 
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_gpu
        del v_gpu
    print(f"gpu compute: {np.mean(t[repeat//2:])} ")

    
    
    k_cache_gpu_cpu = k_cache_gpu.cpu()
    v_cache_gpu_cpu = v_cache_gpu.cpu()
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q.view(-1,qs,head_dim).cpu()
        k_ = k.view(-1,qs,head_dim).cpu()
        v_ = v.view(-1,qs,head_dim).cpu()
        k_cpu = torch.cat([k_cache_cpu_.view(-1,cpu_cache_size,head_dim),k_cache_gpu_cpu,k_],dim=-2)
        v_cpu = torch.cat([v_cache_cpu_.view(-1,cpu_cache_size,head_dim),v_cache_gpu_cpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_cpu,v_cpu,False,None,None,None,False,None)
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3)
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_cpu
        del v_cpu
       
        
        
    print(f"cpu compute: {np.mean(t[repeat//2:])} ")


        
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q.view(-1,qs,head_dim)
        k_ = k.view(-1,qs,head_dim)
        v_ = v.view(-1,qs,head_dim)
        k_gpu = torch.cat([k_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim),k_cache_gpu,k_],dim=-2)
        v_gpu = torch.cat([v_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim),v_cache_gpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_gpu,v_gpu,False,None,None,None,False,None)
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3) 
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_gpu
        del v_gpu
    print(f"load gpu compute: {np.mean(t[repeat//2:])} ")
    
     
   
    # k_cache_cpu = [x[:,:-i-1500,:].contiguous() for i,x in enumerate(k_cache_cpu)]
    # v_cache_cpu = [x[:,:-i-1500,:].contiguous() for i,x in enumerate(v_cache_cpu)]
    
    print(len(k_cache_cpu))
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        out_test,_,_  =  hybrid_attn(q,k,v,k_cache_gpu,v_cache_gpu, k_cache_cpu,v_cache_cpu) 
        torch.cuda.synchronize()
        t.append(time.time()-st)  
    print(f"hybrid compute: {np.mean(t[repeat//2:])}")

 

    acc = check_eq(out_test,out_ref)  
    assert  (acc>0.9),  f"accuracy {acc*100:.4}%, merge state fail..."
    print(f"Merge accuracy {acc*100:.4}%")


    
    




if __name__ == "__main__": 
    
    import time 
    import logging 
    import argparse        
    import numpy as np
    from beyond.Utils import *
    from transformers import AutoConfig
    torch.cuda.current_device()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--qs", type=int, default=1)
    parser.add_argument("--head_split_num", type=int, default=4)
    parser.add_argument("--cpu_cache_size", type=int, default=1024)
    parser.add_argument("--gpu_cache_size", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model_name", type=str, default= "facebook/opt-66b")
    parser.add_argument("--num_thread", type=int, default=16)
    
    args = parser.parse_args()

    run_test(args)
     