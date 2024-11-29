import torch
from tqdm import tqdm
import os
import time 
import numpy as np 
from beyond.utils import *
from beyond.loading import *  
import logging
logger = logging.getLogger(__name__)

 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss 

 
def eval(args):
    device = 'cuda'
    data = load_dataset(args.dataset_name, args.task, split=args.split)
    model, tokenizer = load_model(args.model_name_or_path)

    nlls = []
    latencys = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None

 
     
    k_seq_dim = v_seq_dim = 2
    KVCache_manager = None
    ##############################
    #          beyond            #
    ##############################
    if args.enable_beyond:
                
         
        from beyond.KVcache_manager import KVCache_manager_
        if "llama" in model.config.model_type:
            from beyond.models.modify_llama import modify_llama_attention as modify_attention
        elif "opt" in model.config.model_type:
            from beyond.models.modify_opt import modify_opt_attention as modify_attention
        elif "gpt_neox" in model.config.model_type:
            from beyond.models.modify_gptNeox import modify_GPTNeoX_attention as modify_attention

        config = model.config 
        KVCache_manager = KVCache_manager_(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            head_dim=config.hidden_size//config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            gpu_cache_max=2000000,
            cpu_attn_size=2000000,
            gpu_cache_device="cuda",
        )
        if args.config_file_path:
            print(f"Loading KV manager from {args.config_file_path} ...")
            KVCache_manager = set_kv_manager_config(KVCache_manager,args.config_file_path)
        KVCache_manager.print_coverage()
        modify_attention(model,KVCache_manager)
    ##############################
    #        streamllm           #
    ##############################
    if  args.enable_streamllm:
        from streaming_llm.kv_cache import StartRecentKVCache
        kv_cache = StartRecentKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        if "llama" in model.config.model_type:
            from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

            enable_llama_pos_shift_attention(model)
    
        elif "gpt_neox" in model.config.model_type:
            from streaming_llm.pos_shift.modify_gpt_neox import (
                enable_gpt_neox_pos_shift_attention,
            )

            enable_gpt_neox_pos_shift_attention(model)
       
    

    
    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        
        encodings = tokenizer(text, return_tensors="pt")
  
        seq_len = encodings.input_ids.size(1)
        
        pbar = tqdm(range(0, seq_len - 1))

        st = time.time()
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            
           
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)

                if args.enable_beyond:
                    past_key_values = None 
                elif  args.enable_streamllm:
                    past_key_values = kv_cache(past_key_values)
                else:
                    past_key_values = outputs.past_key_values
                    
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                 
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
             
           
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        latencys.append(time.time()-st)
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
         
  
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"perplexity: {ppl.item():.4}, latency: {np.mean(latencys):.4}")
   
 
if __name__ == "__main__":
    # if os.path.exists("KV_cache_statics.log"): os.remove("KV_cache_statics.log")
    # logging.basicConfig(filename='KV_cache_statics.log', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=500) 
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--config_file_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-13b")
     
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    
    parser.add_argument("--num_eval_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_samples",type=int,default=100)

    parser.add_argument("--enable_beyond",    action="store_true")
    parser.add_argument("--enable_streamllm", action="store_true")
    
    args = parser.parse_args()


    eval(args)
