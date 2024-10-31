# import torch
# import time 
# # Create multiple CPU tensors
# t1 = torch.randn(1000, 1000)
# t2 = torch.randn(1000, 1000)
# t3 = torch.randn(1000, 1000)

# # Concatenate them into a single tensor
# st = time.time()
# for _ in range(1000):
#     t_combined = torch.cat([t1, t2, t3], dim=0)

#     # Transfer the combined tensor to the GPU
#     t_combined_gpu = t_combined.to('cuda')

#     # Split the tensor back into original shapes
#     t1_gpu, t2_gpu, t3_gpu = torch.split(t_combined_gpu, 1000, dim=0)
# print(f"time taken:{time.time()-st}")
 
# # Create multiple CPU tensors
# t1 = torch.randn(1000, 1000)
# t2 = torch.randn(1000, 1000)
# t3 = torch.randn(1000, 1000)

# # Create a new CUDA stream
# stream = torch.cuda.Stream()

# st = time.time()
# for _ in range(1000):
# # Transfer tensors asynchronously using the stream
#     with torch.cuda.stream(stream):
#         t1_gpu = t1.to('cuda', non_blocking=True)
#         t2_gpu = t2.to('cuda', non_blocking=True)
#         t3_gpu = t3.to('cuda', non_blocking=True)

#     # Wait for all transfers to finish
#     stream.synchronize()
# print(f"time taken:{time.time()-st}")
# # Now you can use t1_gpu, t2_gpu, and t3_gpu on the GPU






# # Create multiple CPU tensors
# t1 = torch.randn(1000, 1000)
# t2 = torch.randn(1000, 1000)
# t3 = torch.randn(1000, 1000)

# # Create a new CUDA stream
# stream = torch.cuda.Stream()

# st = time.time()
# for _ in range(1000):
# # Transfer tensors asynchronously using the stream
#     with torch.cuda.stream(stream):
#         t1_gpu = t1.to('cuda')
#         t2_gpu = t2.to('cuda')
#         t3_gpu = t3.to('cuda')

#     # Wait for all transfers to finish
#     stream.synchronize()
# print(f"time taken:{time.time()-st}")
# # Now you can use t1_gpu, t2_gpu, and t3_gpu on the GPU
 
from transformers.modeling_outputs  import (
    BaseModelOutputWithPast,
)

