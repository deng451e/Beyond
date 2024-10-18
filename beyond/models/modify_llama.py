import math
from typing import Optional, Tuple

import torch
from torch import nn
 
import torch.utils.checkpoint

import torch.nn.functional as F

from beyond.selection_methods import selection_methods_
from beyond.KVcache_manager import KVCache_manager_
from beyond.attention_methods import (
    mha_logSum,
    merge_state_,
)
from transformer import Llamaconfig 
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["modify_llama_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def modify_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    batch_size, q_len, _ = hidden_states.size()
      
    q_states = self.q_proj(hidden_states)
    k_states = self.k_proj(hidden_states)
    v_states = self.v_proj(hidden_states)


    #####################
    # Get Layer KVCache # 
    #####################

    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCache_manager(self.attn_layer_idx) # b,s,h,d
    self.increase_layer_idx()

    kv_seq_len = k_states.shape[-2]
    if k_cache_gpu is not None:
        kv_seq_len += k_cache_gpu[0].shape[-2] 


    if k_cache_cpu is not None:
        kv_seq_len += k_cache_gpu[0].shape[-2] 



    cos, sin = self.rotary_emb( seq_len=kv_seq_len)

    recent_size = self.KVCache_manager.recent_sizes(self.attn_layer_idx)
    start_size  = self.KVCache_manager.start_sizes(self.attn_layer_idx)
    block_size  = self.KVCache_manager.block_sizes(self.attn_layer_idx)
 
    #####################
    #   CPU Attention   # 
    #####################
    if k_cache_cpu is not None:
        with torch.cuda.stream(self.cpu_stream):
            q_cpu = q_states.clone().detach().to('cpu', non_blocking=True)
            v_cpu,s_cpu = mha_logSum(q_cpu,k_cache_cpu,v_cache_cpu)

    #####################
    #   GPU Attention   # 
    ################ 


    ### Shift Pos: query pos is min(cache_size, idx)
  
    q_states = apply_rotary_pos_emb_single(q_states, cos, sin, position_ids)
   

    if k_cache_gpu is not None:
        # reuse k, v, self_attention
        k_cache_gpu = torch.cat([k_cache_gpu, k_states], dim=2)
        v_cache_gpu = torch.cat([v_cache_gpu, v_states], dim=2)

    kv_cache_2add = (k_states, v_states) if use_cache else None
    self.KVCache_manager.add_kv_cache_by_layer( kv_cache_2add,self.attn_layer_idx)
    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    k_states = apply_rotary_pos_emb_single(k_states, cos, sin, key_position_ids)
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    k_cache_gpu = repeat_kv(k_cache_gpu, self.num_key_value_groups)
    v_cache_gpu = repeat_kv(v_cache_gpu, self.num_key_value_groups)

    v_gpu,s_gpu = mha_logSum(q_states,k_cache_gpu,v_cache_gpu,attention_mask)

    
 
    attn_output,_  = self.merge_state(v_cpu,s_cpu,v_gpu,s_gpu)
    

    return attn_output

 

def increase_layer_idx(self,):
    self.attn_layer_idx  = (self.attn_layer_idx+1)%self.attn_layer_num
 
 
def modify_llama_attention(model,config):
    model.KVCache_manager = KVCache_manager_()
    model.merge_state  = merge_state_(config.num_attention_heads)
    model.attn_layer_idx = 0
    model.cpu_stream  = torch.cuda.Stream()
    model.copy_stream  = torch.cuda.Stream()
    model.attn_layer_num = config.num_hidden_layers 
    model.increase =  types.MethodType(increase,model)
      
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            modify_llama_attention( module,)

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType( modify_llama_attention_forward, model._modules[name])