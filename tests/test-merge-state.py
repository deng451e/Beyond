import torch
import flashinfer
seq_len = 2048
num_heads = 32
head_dim = 128

device  = "cpu"

cpu_stream =  torch.cuda.Stream()
with torch.cuda.stream(cpu_stream):
     
    v = torch.randn(seq_len, num_heads, head_dim).half().pin_memory()
    s = torch.randn(seq_len, num_heads, dtype=torch.float32).pin_memory()
    v_other = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    s_other = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    v_original = torch.clone(v_other)
    s_original = torch.clone(s_other)
    flashinfer.merge_state_in_place(v, s, v_other, s_other)
    print(f"v tensor device:{v.device}")
    print(f"v_other tensor device:{v_other.device}")
    print(s_original[0],s_other[0])
    print(s_other.dtype)
    if torch.equal(v_original,v_other): print("v_other updated")