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
        
    for batch_size in [1,10]:
        for q_len in [1,100]:
            seq_len = [ 10, 50, 100, 500, 1000, 5000, 10000  ]
            
            x = list(range(1,len(seq_len)+1))
            parsed_data = parse_log(lines)
            
            y1 = [find(parsed_data, 'cpu attention flops'    , q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len] # cpu attention flops
            y2 = [find(parsed_data, 'gpu attention flops'    , q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len ] # gpu attention flops
            y3 = [find(parsed_data, 'offload attention flops', q_len = q_len,batch_size=batch_size, seq_len = i) for i in seq_len ] # offload attention flops

            fig, ax = plt.subplots(figsize=(7, 3))
            
            ax.plot(x, y1, marker='o', label='CPU attention')
            ax.plot(x, y2, marker='s', label='GPU attention')
            ax.plot(x, y3, marker='^', label='Load & GPU Attention')

            ax.set_xlabel('KV Cache Length')
            ax.set_ylabel('Achieved Attention Flops')

            legend = ax.legend(loc='upper left', ncol=3,frameon=False, bbox_to_anchor=(0, 1.1) )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', which='both', length=0)

            ax.set_xticks(x)
            ax.set_xticklabels(seq_len, rotation=0)
            ax.set_yscale("log")
            plt.tight_layout()
            plt.savefig(f'attention-achieved-flops-batchsize-{batch_size}-qlen-{q_len}.pdf', bbox_inches='tight')
    

     