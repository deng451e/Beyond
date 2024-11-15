import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import re 
import torch 
from beyond.utils import parse_file


 
 
preliminary_test_path =os.getcwd()+"/results"
filename =  preliminary_test_path+'/sparse-attention-test.log'  # Replace with your file path
parsed_data = parse_file(filename)
df = pd.DataFrame(parsed_data)
ratios = [0.1,0.3,0.5,0.7,0.9]
seq_lens = [ 1000,10000]
q_lens = [1,10]

seq_len = 100
q_len = 1
fig, axs = plt.subplots(1, 2, figsize=(10, 3))

for i,q_len in enumerate([1,100]):
    for seq_len in seq_lens:
        
        cpu_ops_ratios = []
        gpu_ops_ratios = []
        for ratio in ratios:
            line = df[(df['top_k'] == int(seq_len*ratio))  & (df['seq_len']==seq_len) & (df['q_len']==q_len) ]
            select_cpu_t = line['cpu select time']
            select_gpu_t = line['gpu select time']
            
            attn_cpu_t = line['cpu attn time']
            attn_gpu_t = line['gpu attn time']

            cpu_ops_ratios.append(select_cpu_t/attn_cpu_t)  
            gpu_ops_ratios.append(select_gpu_t/attn_gpu_t)  
        if i==0:
            axs[i].plot(ratios,cpu_ops_ratios,label=f'cpu ops:{seq_len}',linestyle="dashed")
            axs[i].plot(ratios,gpu_ops_ratios,label=f'gpu ops:{seq_len}',linestyle="solid")
        else:
            axs[i].plot(ratios,cpu_ops_ratios,linestyle="dashed")
            axs[i].plot(ratios,gpu_ops_ratios,linestyle="solid")
        axs[i].set_title(f"Query Size:{q_len}",pad=29,fontsize=14)
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xlabel("Sparse Ratio",fontsize=14)


axs[0].set_ylabel("ops latency ratio",fontsize=14)
fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.55, 0.91),fontsize=12)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig(f'results/{torch.cuda.get_device_name()}-device-ops-ratio.pdf', bbox_inches='tight')

 
#####################################################
preliminary_test_path =os.getcwd()+"/results"
filename =  preliminary_test_path+'/dense-attention-test.log'  # Replace with your file path
parsed_data = parse_file(filename)
df = parsed_data

dims = [2**n for n in range(1,6)]
bar_width = 0.35  # Width of the bars
index = np.arange(len(dims))  # X locations for the bars
q_len = 100
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs = axs.flatten()
for i,batch_size in enumerate([1,10]):
    for j,q_len in enumerate([1,100]):
        hold = df[(df['batch_size']==batch_size) & (df['q_len']==q_len)]
        x_labels = list( df[(df['batch_size']==batch_size) & (df['q_len']==q_len)]["seq_len"])

        cpu_attn_t  = hold['cpu_time']
        gpu_attn_t  = hold['gpu_time']
        pcie_t      = hold['transfer_time']
        
        axs[i*2+j].bar(index, cpu_attn_t, bar_width, label='cpu compute time', color='blue')

        axs[i*2+j].bar(index + bar_width, pcie_t , bar_width, label='pcie transfer time', color='orange')
        axs[i*2+j].bar(index + bar_width, pcie_t + gpu_attn_t, bar_width, bottom=pcie_t     , label='gpu compute time', color='gray')




        # Set title, labels, and legend for the first subplot
        axs[i*2+j].set_title(f'Query Size:{q_len},Batch Size:{batch_size}',fontsize=16)
        axs[i*2+j].set_xlabel('kv cache size',fontsize=16)
        
        axs[i*2+j].set_xticks(index + bar_width / 2)
        axs[i*2+j].set_xticklabels(x_labels)
        axs[i*2+j].set_yscale('log')
         
        

        # Adjust layout to prevent overlap
        plt.tight_layout()

        axs[i*2+j].minorticks_off()
        
        axs[i*2+j].spines[['right', 'top']].set_visible(False)
        

#  
axs[0].set_ylabel('time taken (sec)',fontsize=16)
axs[2].set_ylabel('time taken (sec)',fontsize=16)
axs[0].legend(fontsize=12)
 
 
plt.savefig(f'results/{torch.cuda.get_device_name()}-device-time.pdf', bbox_inches='tight')
 