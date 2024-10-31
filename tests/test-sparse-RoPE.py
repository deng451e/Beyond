# Simplified example to apply RoPE to a split sequence

import torch

# Define Rotary Embedding
class LlamaRotaryEmbedding:
    def __init__(self, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
      
    def apply_rotary_pos_emb(self, x, cos, sin):
       
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        
        return x_rot
    def _rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def compute_cos_sin(self, position_ids):
        position_ids = position_ids.squeeze(0)
        # Compute cos and sin embeddings based on global positions
        freqs = torch.einsum('i,j->ij', position_ids, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        
        return cos, sin

# Split sequence for parallelism
sequence_len = 128
num_splits = 4  # Parallel devices
split_size = sequence_len // num_splits

# Assume batch size = 1, num_heads = 8, head_dim = 64
batch_size, num_heads, head_dim = 1, 8, 64

# Global position IDs for the sequence

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
)
from transformers import LlamaConfig  
# Rotary embedding
 
 
config = LlamaConfig()
rotary_emb = LlamaRotaryEmbedding(dim=128)
x= rotary_emb(torch.randn(10,10),20)
print(x[0].shape)