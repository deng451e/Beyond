import torch 
import time 
import argparse
import numpy as np 
import random
def main(args,log):
   
    
    

     
     
   
    log += "==================================\n"
    
    

    for dtype  in ['fp16','fp32']:
        if dtype == 'fp16': type_ = torch.float16
        if dtype == 'fp32': type_ = torch.float32
        for seq_len in range(1000,10001,1000):
            
            for dim in [64,80,128]:
                cache = torch.randn(seq_len,dim ,dtype=type_)
                indices = list(range(seq_len))
                for ratio in range(1,10):
                    ratio /= 10 
                    selected_indices = torch.tensor(random.sample(indices,int(len(indices)*ratio))) 
                    for device in ['cpu','cuda']:
                    
                        times  = []
                        for _ in range(args.repeat):
                            cache_ = cache.to(device)
                            selected_indices_ = selected_indices.to(device)
                            
                            st = time.time()
                            _ = torch.index_select(cache_, dim=0, index=selected_indices_)  # Using index_select
                            # _ = cache_[selected_indices_]
                            times.append(time.time()-st)
                        log += f"dtype:{dtype}, device:{device}, seq_len:{seq_len}, dim:{dim}, ratio:{ratio}, time:{np.mean(times[-10:])} \n"

    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    
    main(args,"")