import math
import logging
 
import types
import os 
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

 

from beyond.utils import *
from beyond.selection_methods import selection_methods_
from beyond.attention_methods import (
    merge_state_,
    mha_lse_methods,
  
)
 
logger = logging.getLogger(__name__)


from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    rotate_half,
    GPTNeoXAttention,
    GPTNeoXRotaryEmbedding,
)
import types

__all__ = ["enable_gpt_neox_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def modified_GPTNeoX_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size] --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)] --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # b h s d 
    q = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    k = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    v = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
 
     



    #####################
    # Get Layer KVCache # 
    #####################
     
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCache_manager(self.attn_layer_idx) # b,h,s,d
    torch.cuda.synchronize()
    

    kv_seq_len = k.size(-2)

    batch_size,  num_heads,q_len,head_dim =  q.size()
    if k_cache_gpu is not None:

        
        if self.attn_layer_idx==35:
            info = f"GPU cache size: {k_cache_gpu.size(-2)}"
            if k_cache_cpu is not None:  info +=  f" | CPU cache size: {k_cache_cpu.size(-2)}"
            logger.info(info)

        kv_seq_len += k_cache_gpu.size(-2) 
        k_cache_gpu = torch.cat([k_cache_gpu, k], dim=2)
        v_cache_gpu = torch.cat([v_cache_gpu, v], dim=2)
    else:
        k_cache_gpu = k
        v_cache_gpu = v
     
     
     
    if k_cache_cpu is not None:
     
        cpu_attn_size = min(k_cache_cpu.size(-2) ,self.KVCache_manager.cpu_attn_sizes[self.attn_layer_idx])
        kv_seq_len += cpu_attn_size
        start_size  = self.KVCache_manager.start_sizes[self.attn_layer_idx]
      
    cos_gpu, sin_gpu = self.rotary_emb(v, seq_len=kv_seq_len)
    q_position_ids = torch.arange(kv_seq_len-q_len,kv_seq_len,device=q.device).unsqueeze(0)
    q_rot  = q[..., : self.rotary_ndims]
    q_pass = q[..., self.rotary_ndims :]

    q_rot = apply_rotary_pos_emb_single(q_rot, cos_gpu, sin_gpu, q_position_ids)
    q = torch.cat((q_rot, q_pass), dim=-1)


  
    if k_cache_cpu is not None:
        q = q * self.norm_factor
        if kv_seq_len > self.bias.shape[-1]:
            self._init_bias(kv_seq_len, device=k.device)
        attention_mask = self.bias[:, :, kv_seq_len - q_len : kv_seq_len, :kv_seq_len]
        attention_mask = attention_mask.squeeze(0)[:,:,-q_len:]
        attention_mask_q = attention_mask if q_len!=1 else None 
        #####################
        #   CPU Attention   # 
        #####################
        with torch.cuda.stream(self.cpu_stream):
            
           
            # load appended token stats to CPU
            k_cache_cpu = torch.cat([k_cache_cpu, k.to('cpu', non_blocking=True)], dim=-2)
            v_cache_cpu = torch.cat([v_cache_cpu, v.to('cpu', non_blocking=True)], dim=-2)
            q_cpu =    q.detach().to('cpu', non_blocking=True)
            attention_mask_q_cpu = attention_mask_q.to('cpu', non_blocking=True) if attention_mask_q is not None else attention_mask_q 
            

            cos_cpu, sin_cpu = self.rotary_emb_cpu(v_cache_cpu, seq_len=kv_seq_len)
            position_ids_cpu = torch.cat([  
                torch.arange(start_size, start_size+cpu_attn_size, device='cpu'),
                torch.arange(kv_seq_len-q_len, kv_seq_len, device='cpu')
                ],dim=0).unsqueeze(0)
            
            k_rot_cpu  = k_cache_cpu[..., : self.rotary_ndims]
            k_pass_cpu = k_cache_cpu[..., self.rotary_ndims :]
             
            k_rot_cpu = apply_rotary_pos_emb_single(k_rot_cpu,  cos_cpu, sin_cpu, position_ids_cpu)
            k_cache_cpu =  torch.cat((k_rot_cpu,k_pass_cpu), dim=-1)
            
             
            v_cpu,s_cpu = self.mha_lse(q_cpu,k_cache_cpu,v_cache_cpu,attention_mask_q_cpu)
        
         
        
        #####################   
        #   GPU Attention   # 
        ##################### 

        position_ids_gpu = torch.cat([
                torch.arange(0,start_size, device=k_cache_gpu.device),
                torch.arange(start_size+cpu_attn_size, kv_seq_len, device=k_cache_gpu.device)
                ],dim=0).unsqueeze(0)
        


        k_rot  = k_cache_gpu[..., : self.rotary_ndims]
        k_pass = k_cache_gpu[..., self.rotary_ndims :]
        k_rot  = apply_rotary_pos_emb_single(k_rot,  cos_gpu, sin_gpu, position_ids_gpu)
        k_cache_gpu = torch.cat((k_rot,k_pass), dim=-1)
         
     
        v_gpu,s_gpu = self.mha_lse(q,k_cache_gpu,v_cache_gpu,attention_mask_q)
         
        
        #####################
        #    Merge State    # 
        ##################### 
        
        self.cpu_stream.synchronize()
        attn_output,_ = self.merge_state(v_cpu,s_cpu,v_gpu,s_gpu)
        attn_output = attn_output.permute(0,2,1,3) # b h s d 

 
    else:
     

         
        k_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        k_rot  = k_cache_gpu[..., : self.rotary_ndims]
        k_pass = k_cache_gpu[..., self.rotary_ndims :]
         
        k_rot = apply_rotary_pos_emb_single(k_rot,  cos_gpu, sin_gpu, k_position_ids)
        
         
        k_cache_gpu = torch.cat((k_rot,k_pass), dim=-1)
      
       
        attn_output, attn_weights = self._attn(q,k_cache_gpu, v_cache_gpu, attention_mask, head_mask)
 
   


    kv_cache_2add = (k, v) if use_cache else None
    self.KVCache_manager.add_kv_cache_by_layer(self.attn_layer_idx, kv_cache_2add)
     
    # Reshape outputs
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    attn_output = self.dense(attn_output)

    outputs = (attn_output, None)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs

 

def modify_GPTNeoX_attention(model,KVCache_manager):
    global layer_idx
    config = model.config 
    layer_idx = config.num_hidden_layers-1
    head_dim = config.hidden_size//config.num_attention_heads
     
    cpu_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    KVCache_manager.copy_stream = copy_stream
    rotary_emb_cpu = GPTNeoXRotaryEmbedding(
                    int(head_dim * config.rotary_pct),
                    max_position_embeddings=config.max_position_embeddings,
                    device='cpu')
    merge_state = merge_state_(config.num_attention_heads)
    mha_lse     = mha_lse_methods('gpt-neox')
    
 
    def replace_layer(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                replace_layer( module,)
 
            if isinstance(module, GPTNeoXAttention):
                global layer_idx
                
                model._modules[name].attn_layer_idx  = layer_idx
                model._modules[name].rotary_emb_cpu  = rotary_emb_cpu
                model._modules[name].merge_state  = merge_state
                model._modules[name].mha_lse  = mha_lse
                model._modules[name].KVCache_manager = KVCache_manager
                model._modules[name].cpu_stream = cpu_stream
                model._modules[name].forward = types.MethodType( modified_GPTNeoX_attention_forward, model._modules[name])
                layer_idx -= 1  # layer are reverseved travesed 
    replace_layer(model)
 