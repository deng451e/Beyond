import re
import matplotlib.pyplot as plt

def parse_log(lines):
    data_list = []
    for line in lines:
        data_dict = {}
        elements_list = line.strip().split(',')
        for element in elements_list:
            key, value = element.strip().split(':')
            data_dict[key.strip()] = float(value.strip())
            data_list.append(data_dict)
    return data_list

def find(L, keyx, **kwargs):
    for item in L:
        F = True
        if item.get(keyx, None) is None:
            F = False
        if F:
            for key,value in kwargs.items():
                if item.get(key, None) == value:
                    continue
                else:
                    F = False
                    break
        if F:
            # print(item[keyx])
            return item[keyx]
    # print(0)
    return 0

if __name__ == '__main__':
    with open("attention-achieved-flops.log","r") as f:
        lines = f.readlines()
        
    # for batch_size in [1,10]:
    batch_size = 1 
    seq_len = [ 10, 50, 100, 500, 1000, 5000, 10000  ]
    fig, axs = plt.subplots(1,2, figsize=(13, 4), sharex=True ) # , sharey=True
    axs = axs.flatten()
    for i,q_len in enumerate([1,32]):
             
            
            x = list(range(1,len(seq_len)+1))
            parsed_data = parse_log(lines)
            
            y1 = [find(parsed_data, 'cpu attention flops'    , q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len] # cpu attention flops
            y2 = [find(parsed_data, 'gpu attention flops'    , q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len ] # gpu attention flops
            y3 = [find(parsed_data, 'offload attention flops', q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len ] # offload attention flops

            
            if i==0:
                axs[i].plot(x, y1, marker='o', label='CPU Decode')
                axs[i].plot(x, y2, marker='s', label='GPU Decode')
                axs[i].plot(x, y3, marker='^', label='Load & GPU Decode')
            else:
                axs[i].plot(x, y1, marker='o',c='red', label='CPU Append 32')
                axs[i].plot(x, y2, marker='s',c='blue', label='GPU  Append 32')
                axs[i].plot(x, y3, marker='^',c='black', label='Load & GPU  Append 32')

            # axs[i].set_xlabel('KV Cache Length')
            axs[0].set_ylabel('Achieved Attention FLOPs', fontsize=16)

             
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].tick_params(axis='y', which='both', length=0)

            axs[i].set_xticks(x)
            # xlabel = [f"{ss}/{2*ss*4096*2/1024**2.:.4}"for ss in seq_len]
            # print(xlabel)
            axs[i].set_xticklabels(seq_len, rotation=0 , fontsize=12)
            axs[i].set_yticklabels(y2,  fontsize=12)
            axs[i].set_yscale("log")
    

    # Collect legend handles and labels from all subplots
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    print(labels)
    # Reorder the labels and handles
    order = [0,3,1,4,2,5]  # Desired order of the labels (indices in the list)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # plt.tight_layout()
    # axs[0].legend(loc='upper left', ncol=3,frameon=False, bbox_to_anchor=(0, 1.1) )
    fig.legend(handles, labels,frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.51, 1.2) , fontsize=14)

    fig.text(0.55, 0.04, 'KV Cache Length' , ha='center', va='center' , fontsize=16)  # Common y label , fontsize=16
    # fig.text(0.04, 0.5,  'Achieved Attention Flops', ha='center', va='center', rotation='vertical', fontsize=16)  # Common x label
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig(f'gpu-4090-cpu-6430.pdf', bbox_inches='tight')


     