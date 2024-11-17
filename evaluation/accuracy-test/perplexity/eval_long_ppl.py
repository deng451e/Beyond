import torch
from tqdm import tqdm
import os



from beyond.utils import *
from beyond.loading import * 
from beyond.models.modify_opt import modify_opt_attention
from beyond.KVcache_manager import KVCache_manager_

 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss 

 
def eval(args):
    device = 'cuda'
    data = load_dataset(args.dataset_name, args.task, split=args.split)
    model, tokenizer = load_model(args.model_name_or_path)

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None


    if "llama" in model.config.model_type:
        from beyond.models.modify_llama import modify_llama_attention as modify_attention
    elif "opt" in model.config.model_type:
        from beyond.models.modify_opt import modify_opt_attention as modify_attention
    elif "gpt_neox" in model.config.model_type:
        from beyond.models.modify_gptNeox import modify_GPTNeoX_attention as modify_attention

    
    else:
        raise ValueError(f"got {model.config.model_type}")
    
    k_seq_dim = v_seq_dim = 2
    KVCache_manager = None

    if args.enable_modify:
            
        config = model.config 
        KVCache_manager = KVCache_manager_(
            start_size=4,
            recent_size=40,
            k_seq_dim=2,
            v_seq_dim=2,
            head_dim=config.hidden_size//config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            gpu_cache_max=2000,
            cpu_attn_size=2000,
            gpu_cache_device="cuda",
        )


        # if args.config_file_path:
        #     print(f"Loading KV manager from {args.config_file_path} ...")
        #     KVCache_manager = set_kv_manager_config(KVCache_manager,args.config_file_path)

        KVCache_manager.print_coverage()

        modify_attention(model,KVCache_manager)
       


    # os.makedirs(args.output_dir, exist_ok=True)
    # f = open(f"{args.output_dir}/log.txt", "w")
    
    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        # print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                 
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            #print(neg_log_likelihood.item() )
            # print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    # 
    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl.item())
    # with open(f"{args.output_dir}/ppl.txt", "w") as f:
    #     f.write(f"{ppl.item()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=50) 
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    # parser.add_argument("--config_file_path", type=str, default="kv_manager_InitConfig/llama-vicuna-13b-v1.3.json")
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--enable_modify", action="store_true")
    parser.add_argument("--model_type", type=str, default="opt")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    
    parser.add_argument("--num_eval_tokens", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_samples",type=int,default=10)
    args = parser.parse_args()
    eval(args)
