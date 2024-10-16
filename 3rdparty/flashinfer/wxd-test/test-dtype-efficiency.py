import torch 
import time 
import argparse
import numpy as np 
from utils import * 
def main(args,log):
   
     
    
    
    for power in range(1,13):
        dim = 2**power
        x = torch.randn(dim,dim,dtype= torch.float16) 
        y = torch.randn(dim,dim,dtype= torch.float16) 
        x = x.to('cuda')
        y = y.to('cuda')
        _ = x@y

    for dtype in ['fp16','fp32']:
        if dtype == 'fp16': type_ = torch.float16
        if dtype == 'fp32': type_ = torch.float32
        for power in range(1,13):
            dim = 2**power  
            cpu_compute_times = []
            gpu_compute_times = []
            transfer_times = []
            x = torch.randn(dim,dim,dtype=type_) 
            y = torch.randn(dim,dim,dtype=type_) 
            for _ in range(args.repeat):
                
                st = time.time()
                cpu_ref = x@y
                cpu_compute_times.append(time.time()-st)
                cpu_ref = cpu_ref.detach()
                #########################
                st = time.time()
                x_ = x.to('cuda')
                y_ = y.to('cuda')
                transfer_times.append(time.time()-st)
                #########################
            
                st = time.time()
                gpu_ref = x_@y_
                gpu_compute_times.append(time.time()-st)
                gpu_ref = gpu_ref.detach().cpu()
             
                assert check_eq(gpu_ref,cpu_ref)>0.9, 'results not equal...'
            log += f"dim:{dim}, dtype:{dtype}, cpu compute time:{np.mean(cpu_compute_times)}, gpu compute time:{np.mean(gpu_compute_times)}, transfer time:{np.mean(transfer_times)}\n"

    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    
    main(args,"")