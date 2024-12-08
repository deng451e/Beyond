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
    
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    parsed_data = parse_log(lines)
     
    y1 = [find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 100), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 500), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 1000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 5000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 10000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 50000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 100000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=1, seq_len = 500000)] # cpu attention flops
    y2 = [find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 100), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 500), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 1000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 5000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 10000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 50000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 100000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=1, seq_len = 500000)] # gpu attention flops
    y3 = [find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 100), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 500), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 1000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 5000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 10000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 50000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 100000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=1, seq_len = 500000)] # offload attention flops

    fig, ax = plt.subplots(figsize=(8, 3))
    
    ax.plot(x, y1, marker='o', label='cpu attention flops')
    ax.plot(x, y2, marker='s', label='gpu attention flops')
    ax.plot(x, y3, marker='^', label='offload attention flops')

    ax.set_xlabel('KV Cache Length')
    ax.set_ylabel('Achieved Attention Flops')

    legend = ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(['100', '500', '1000', '5000', '10000','50000','100000','500000'], rotation=0)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig('attention-achieved-flops-batchsize-1.pdf', bbox_inches='tight')
 


   
    y1 = [find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 100), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 500), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 1000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 5000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 10000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 50000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 100000), find(parsed_data, 'cpu attention flops', q_len = 1,batch_size=10, seq_len = 500000)] # cpu attention flops
    y2 = [find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 100), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 500), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 1000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 5000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 10000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 50000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 100000), find(parsed_data, 'gpu attention flops', q_len = 1,batch_size=10, seq_len = 500000)] # gpu attention flops
    y3 = [find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 100), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 500), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 1000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 5000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 10000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 50000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 100000), find(parsed_data, 'offload attention flops', q_len = 1,batch_size=10, seq_len = 500000)] # offload attention flops

    fig, ax = plt.subplots(figsize=(8, 3))
    
    ax.plot(x, y1, marker='o', label='cpu attention flops')
    ax.plot(x, y2, marker='s', label='gpu attention flops')
    ax.plot(x, y3, marker='^', label='offload attention flops')

    ax.set_xlabel('KV Cache Length')
    ax.set_ylabel('Achieved Attention Flops')

    legend = ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(['100', '500', '1000', '5000', '10000','50000','100000','500000'], rotation=0)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig('attention-achieved-flops-batchsize-10.pdf', bbox_inches='tight')
 

    