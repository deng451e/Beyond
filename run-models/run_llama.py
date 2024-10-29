import warnings

warnings.filterwarnings("ignore")
import logging
import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from beyond.utils import *
from beyond.loading import * 
from beyond.models.modify_llama import modify_llama_attention


logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values,KVCache_manager, max_gen_len,enable_modify):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values if not  enable_modify else None 
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
   
        
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if enable_modify:
            past_key_values = None 
             

        else:
            past_key_values = outputs.past_key_values
             
         
             
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
          
            break
     
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def inference(model, tokenizer, prompts, KVCache_manager=None, max_gen_len=1000,enable_modify=False):
    past_key_values = None
  
    for idx, prompt in enumerate(prompts):
        
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        logger.info(f"================================")
        logger.info(f"Index {idx}: {input_ids.size(1)}")
        logger.info(f"================================")
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
     
        
        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values,KVCache_manager, max_gen_len=max_gen_len,enable_modify=enable_modify
        )
     

def main(args):
    
    

    # load dataset 
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    if not os.path.exists(test_filepath):
        download_url("https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",args.data_root,)
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_json(test_filepath)
    cnt = 0
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]
        cnt += 1 
    print(cnt)

    # load model 
    print(f"Loading model from {args.model_name_or_path} ...")
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load_model(model_name_or_path)
    
    # load KV manager 
    KVCache_manager = None

    if args.enable_modify:
         
        config = model.config 
        KVCache_manager = KVCache_manager_(
            start_size=4,
            recent_size=1000,
            k_seq_dim=2,
            v_seq_dim=2,
            head_dim=config.hidden_size//config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            gpu_cache_max=2000,
            cpu_attn_size=2000,
            gpu_cache_device="cuda",
        )

    
        if args.config_file_path:
            print(f"Loading KV manager from {args.config_file_path} ...")
            KVCache_manager = set_kv_manager_config(KVCache_manager,args.config_file_path)
 
        KVCache_manager.print_coverage()

        modify_llama_attention(model,KVCache_manager)
    
    # start inference 
    inference(
        model,
        tokenizer,
        prompts,
        KVCache_manager=KVCache_manager,
        enable_modify=args.enable_modify,
    )


if __name__ == "__main__":
    if os.path.exists("KV_cache_statics.log"): os.remove("KV_cache_statics.log")
    logging.basicConfig(filename='KV_cache_statics.log', level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="kv_manager_InitConfig/llama-vicuna-13b-v1.3.json")
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_modify", action="store_true")
    args = parser.parse_args()
    main(args)