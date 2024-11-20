import math
import logging
 
import types
import os 
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

 

from beyond.utils import *
from beyond.KVcache_manager import KVCache_manager_
from beyond.attention_methods import (
    mha_lse_methods,
    merge_state_,
)
 
logger = logging.getLogger(__name__)


from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
)
 

__all__ = ["modify_llama_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    
    cos = cos.squeeze(1).squeeze(0)       # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)       # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed
 

def modified_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  
    batch_size, q_len, _ = hidden_states.size()  
      
    q_states = self.q_proj(hidden_states)
    k_states = self.k_proj(hidden_states)
    v_states = self.v_proj(hidden_states)

    # b,h,s,d
    q_states = q_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)/ math.sqrt(self.head_dim)
    k_states = k_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
 
    #####################
    # Get Layer KVCache # 
    #####################
     
     
    
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCache_manager(self.attn_layer_idx) # b,h,s,d

    torch.cuda.synchronize()

    
    kv_seq_len = k_states.size(-2)



    if k_cache_gpu is not None:

        if self.attn_layer_idx==0:
            info = f"GPU cache size: {k_cache_gpu.size(-2)}"
            if k_cache_cpu is not None:  info +=  f" | CPU cache size: {k_cache_cpu.size(-2)}"
            logger.info(info)
            

        kv_seq_len += k_cache_gpu.size(-2) 
        k_cache_gpu = torch.cat([k_cache_gpu, k_states], dim=2)
        v_cache_gpu = torch.cat([v_cache_gpu, v_states], dim=2)
    else:
        k_cache_gpu = k_states
        v_cache_gpu = v_states
     
     
     
    if k_cache_cpu is not None:
        cpu_attn_size = min(k_cache_cpu.size(-2) ,self.KVCache_manager.cpu_attn_sizes[self.attn_layer_idx])
        kv_seq_len += cpu_attn_size
        start_size  = self.KVCache_manager.start_sizes[self.attn_layer_idx]
        
    
    cos_gpu, sin_gpu = self.rotary_emb(v_states, seq_len=kv_seq_len)
    q_position_ids = torch.arange(kv_seq_len-q_len,kv_seq_len,device=q_states.device).unsqueeze(0)
    q_states = apply_rotary_pos_emb_single(q_states, cos_gpu, sin_gpu, q_position_ids)
    
     # Mix CPU&GPU attention
    if k_cache_cpu is not None and cpu_attn_size!=0:
    
       

        attention_mask_q = attention_mask if q_len!=1 else None 
        #####################
        #   CPU Attention   # 
        #####################
        with torch.cuda.stream(self.cpu_stream):
            

            # load appended token stats to CPU
            k_cache_cpu = torch.cat([k_cache_cpu, k_states.to('cpu')], dim=2)
            v_cache_cpu = torch.cat([v_cache_cpu, v_states.to('cpu')], dim=2)
            q_cpu = q_states.detach().to('cpu')
            attention_mask_q_cpu = attention_mask_q.to('cpu') if attention_mask_q is not None else attention_mask_q 
          

            cos_cpu, sin_cpu = self.rotary_emb_cpu(v_cache_cpu, seq_len=kv_seq_len)
            position_ids_cpu = torch.cat([  
                torch.arange(start_size, start_size+cpu_attn_size, device='cpu'),
                torch.arange(kv_seq_len-q_len, kv_seq_len, device='cpu')
                ],dim=0).unsqueeze(0)
             
            k_cache_cpu = apply_rotary_pos_emb_single(k_cache_cpu,  cos_cpu, sin_cpu, position_ids_cpu)
            k_cache_cpu = repeat_kv(k_cache_cpu, self.num_key_value_groups)
            v_cache_cpu = repeat_kv(v_cache_cpu, self.num_key_value_groups)
            v_cpu,s_cpu = self.mha_lse(q_cpu,k_cache_cpu,v_cache_cpu,attention_mask_q_cpu)
        
         
        
        #####################   
        #   GPU Attention   # 
        ##################### 

        position_ids_gpu = torch.cat([
                torch.arange(0,start_size, device=k_cache_gpu.device),
                torch.arange(start_size+cpu_attn_size, kv_seq_len, device=k_cache_gpu.device)
                ],dim=0).unsqueeze(0)
        
      
        k_cache_gpu = apply_rotary_pos_emb_single(k_cache_gpu,  cos_gpu, sin_gpu, position_ids_gpu)
         
        
        k_cache_gpu = repeat_kv(k_cache_gpu, self.num_key_value_groups)
        v_cache_gpu = repeat_kv(v_cache_gpu, self.num_key_value_groups)
        
       
        v_gpu,s_gpu = self.mha_lse(q_states,k_cache_gpu,v_cache_gpu,attention_mask_q)
         
        
        #####################
        #    Merge State    # 
        ##################### 
        
        self.cpu_stream.synchronize()
        attn_output,_ = self.merge_state(v_cpu,s_cpu,v_gpu,s_gpu)
        torch.cuda.synchronize()
       
    # Default Full GPU attention
    else:   
    
        
        
        position_ids_gpu = torch.arange(kv_seq_len, device=k_cache_gpu.device).unsqueeze(0)
       
        k_cache_gpu = apply_rotary_pos_emb_single(k_cache_gpu, cos_gpu, sin_gpu, position_ids_gpu)
        
        
        k_cache_gpu = repeat_kv(k_cache_gpu, self.num_key_value_groups)
        v_cache_gpu = repeat_kv(v_cache_gpu, self.num_key_value_groups)
      
        
        # subtract maximum value to improve numerical stability
        attn_weights = torch.matmul(q_states, k_cache_gpu.transpose(2, 3))  
        max_scores, _ = attn_weights.max(dim=-1, keepdim=True) 
        attn_weights = attn_weights - max_scores

        

        if attention_mask is not None: 
            
            attn_weights[:,:,:,-q_len:] = attn_weights[:,:,:,-q_len:] + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(q_states.dtype)
        
        attn_output = torch.matmul(attn_weights, v_cache_gpu)
        attn_output = attn_output.transpose(1, 2).contiguous()
 
     
        

    kv_cache_2add = (k_states, v_states) # if use_cache else None
    
    self.KVCache_manager.add_kv_cache_by_layer(self.attn_layer_idx, kv_cache_2add)
     
     
     
    attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)
     
    attn_output = self.o_proj(attn_output)
    
    if not output_attentions:
        attn_weights = None


    
     
    return attn_output,attn_weights,past_key_value

  
 
def modify_llama_attention(model,KVCache_manager):
    global layer_idx
    config = model.config 
    layer_idx = config.num_hidden_layers-1
    head_dim = config.hidden_size//config.num_attention_heads
     
    cpu_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    KVCache_manager.copy_stream = copy_stream
    rotary_emb_cpu = LlamaRotaryEmbedding(head_dim,device='cpu')
    merge_state    = merge_state_(config.num_attention_heads)
    mha_lse        = mha_lse_methods('llama')
 
    def replace_layer(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                replace_layer( module,)
 
            if isinstance(module, LlamaAttention):
                global layer_idx
                model._modules[name].attn_layer_idx  = layer_idx
                model._modules[name].rotary_emb_cpu  = rotary_emb_cpu
                model._modules[name].merge_state  = merge_state
                model._modules[name].mha_lse  = mha_lse
                model._modules[name].KVCache_manager = KVCache_manager
                model._modules[name].cpu_stream = cpu_stream
                model._modules[name].forward = types.MethodType( modified_llama_attention_forward, model._modules[name])
                layer_idx -= 1  # layer are reverseved travesed 
    replace_layer(model)
 
 