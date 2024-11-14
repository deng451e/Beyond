from beyond.attention_methods import *
from beyond.utils import mha_normal
import numpy as np 





#####################
#      For Test     # 
#####################
 
def test(args,log):
    log = add_info(args,log)
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    repeat = args.repeat
    seq_len = args.seq_len
    q_len = args.q_len
    head_dim = hidden_size//num_heads
      
    k_dim   =  2
    k_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    v_cache = torch.randn(batch_size,num_heads,seq_len,head_dim, device='cpu').half()
    q       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    k       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    v       = torch.randn(batch_size,num_heads,q_len,head_dim, device='cuda:0').half() 
    
     
    
    
    out= f"batch_size:{batch_size},seq_len:{seq_len},q_len:{q_len},num_heads:{num_heads},head_dim:{head_dim},"
    #####################
    #   CPU Attention   # 
    #####################
    q_cpu = q.cpu() 
    k_cpu = torch.cat([k_cache,k.cpu()],dim=k_dim)
    v_cpu = torch.cat([v_cache,v.cpu()],dim=k_dim)

    cpu_t = []
    for _ in range(repeat+3):
        torch.cuda.synchronize()
        st = time.time()
        OUTPUT_cpu = mha_normal(q_cpu,k_cpu,v_cpu)
        torch.cuda.synchronize()
        cpu_t.append(time.time()-st)
   
    out+=f"cpu_time:{np.mean(cpu_t[3:])},"
   
    
    #####################
    #   PCIe Transfer   # 
    #####################
    pcie_t = []
    for _ in range(repeat+3):
        torch.cuda.synchronize()
        st = time.time() 
        k_gpu = k_cache.cuda()
        v_gpu = v_cache.cuda()
        torch.cuda.synchronize()
        pcie_t.append(time.time()-st)
    
    out+=f"transfer_time:{np.mean(pcie_t[3:])},"

    k_gpu = torch.cat([k_gpu,k],dim=k_dim )
    v_gpu = torch.cat([v_gpu,v],dim=k_dim )

    #####################
    #   GPU Attention   # 
    #####################
    gpu_t = []
    for _ in range(repeat+3):
        torch.cuda.synchronize()
        st = time.time() 
        OUTPUT_gpu = mha_normal(q,k_gpu ,v_gpu)
        torch.cuda.synchronize()
        gpu_t.append(time.time()-st)

    out+=f"gpu_time: {np.mean(gpu_t[3:])}"
    


    acc = check_eq(OUTPUT_cpu,OUTPUT_gpu.cpu())  
    assert  (acc>0.9),  f"accuracy {acc*100:.4}%,   fail..."
    print(out)


    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=1000 )
    parser.add_argument("--q_len", type=int, default=10)

    # model config 
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
   

    # test config 
    parser.add_argument("--repeat", type=int, default=10)

    args = parser.parse_args()
    
    test(args,"")
     