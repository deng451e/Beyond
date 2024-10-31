from transformers.models.llama.modeling_llama import _make_causal_mask
import torch
import time 
# create causal mask
# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len  = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(  attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
        print(combined_attention_mask.shape)
        print(combined_attention_mask)
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        print(combined_attention_mask.shape)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

attention_mask = torch.ones(
                (1, 11), dtype=torch.bool, device='cpu'
            )
x = _prepare_decoder_attention_mask(
            attention_mask, (1, 5), torch.randn(1,5,128), 6
        )
# print(x)
# print(x.shape)
combined_attention_mask = _make_causal_mask(
        (1,5),
        torch.float16,
        device='cuda',
        past_key_values_length=2,
    )
print(combined_attention_mask)
# st = time.time()
# for _ in range(1000):
#     combined_attention_mask = _make_causal_mask(
#         (1,20),
#         torch.float16,
#         device='cuda',
#         past_key_values_length=0,
#     )
# print(combined_attention_mask.shape)
# print(f"time taken:{time.time()-st}")


# st = time.time()
# for _ in range(1000):
#     combined_attention_mask  = combined_attention_mask.to('cpu', non_blocking=True)


# print(f"time taken:{time.time()-st}")
 