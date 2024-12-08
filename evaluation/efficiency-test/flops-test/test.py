import torch
import time

# Create a custom CUDA stream
gpu_stream = torch.cuda.Stream()
cpu_stream = torch.cuda.Stream()
# Dummy CPU task: Simulating a computationally intensive task
 
# Dummy GPU task: Matrix multiplication on the GPU
def gpu_task():
    print("GPU task started")
    start_time = time.time()

    for _ in range(10000):
        a = torch.randn(10000, 10000, device='cuda')
        b = torch.randn(10000, 10000, device='cuda')
        result = torch.mm(a, b)  # Matrix multiplication
    print("GPU task queued")
    print(f"GPU task queued in {time.time() - start_time:.3f} seconds")

# Measure total time
start_time = time.time()

# Start CPU and GPU tasks asynchronously
with torch.cuda.stream(gpu_stream):  #
    gpu_task()  # GPU computation (queued asynchronously)
print("CPU task started")
gpu_stream.synchronize()

print(f"Total execution time: {time.time() - start_time:.3f} seconds")