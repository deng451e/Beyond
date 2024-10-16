import math
from typing import Optional, Tuple

import torch
from torch import nn
 
import torch.utils.checkpoint

import torch.nn.functional as F

from beyond.attention_methods import (
    attention_methods_,
    merge_state_,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["enable_modify_llama_attention"]


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
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = past_key_value

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp

        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )

        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        q_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        q_states = torch.cat(q_states, dim=-1)

        k_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        k_states = torch.cat(k_states, dim=-1)

        v_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        v_states = torch.cat(v_states, dim=-1)

    else:
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

    q_states = q_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k_states   = k_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v_states = v_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = k_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2] 
    cos, sin = self.rotary_emb(v_states, seq_len=kv_seq_len)
    ### Shift Pos: query pos is min(cache_size, idx)
    # q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin, position_ids)
    q_states = apply_rotary_pos_emb_single(q_states, cos, sin, position_ids)
   

    if past_key_value is not None:
        # reuse k, v, self_attention
        k_states = torch.cat([past_key_value[0], k_states], dim=2)
        v_states = torch.cat([past_key_value[1], v_states], dim=2)

    past_key_value = (k_states, v_states) if use_cache else None

    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    k_states = apply_rotary_pos_emb_single(k_states, cos, sin, key_position_ids)
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    k_states = repeat_kv(k_states, self.num_key_value_groups)
    v_states = repeat_kv(v_states, self.num_key_value_groups)

    attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q_states.dtype
    )
    attn_output = torch.matmul(attn_weights, v_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_modify_llama_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_modify_llama_attention( module,)

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType( modify_llama_attention_forward, model._modules[name])