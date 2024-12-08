import torch
import time

# Create custom CUDA stream
gpu_stream = torch.cuda.Stream()

# Dummy CPU work
def cpu_task():
    print("CPU task started")
    result = sum(i**2 for i in range(10**6))  # Simulate heavy computation
    print("CPU task completed, result:", result)

# GPU work
def gpu_task():
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    with torch.cuda.stream(gpu_stream):
        for _ in range(100):  # Smaller loop for demonstration
            result = torch.mm(a, b)
    print("GPU task queued")

# Measure time
start_time = time.time()

# Start GPU task
gpu_task()

# Start CPU task in parallel
cpu_task()

# Wait for GPU to finish
gpu_stream.synchronize()

print(f"Total time: {time.time() - start_time:.2f} seconds")