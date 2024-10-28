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
            KVCache_manager.copy_stream.synchronize()

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
  
    for idx, prompt in enumerate(prompts[0:10]):
        
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
    model_name_or_path = args.model_name_or_path

    # load model 
    model, tokenizer = load_model(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    

    # load dataset 
    print(f"Loading data from {test_filepath} ...")
    if not os.path.exists(test_filepath):
        download_url("https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",args.data_root,)
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_json(test_filepath)
    
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]
 

    
  
     # load KV manager 
    config = model.config 
    KVCache_manager = KVCache_manager_(
        start_size=10,
        recent_size=50,
        k_seq_dim=2,
        v_seq_dim=2,
        head_dim=config.hidden_size//config.num_attention_heads,
        num_heads=config.num_attention_heads,
        num_layers=config.num_hidden_layers,
        gpu_cache_max=2000,
        gpu_cache_device="cuda",
    )

 
    if args.config_file_path:
        KVCache_manager = set_kv_manager_config(KVCache_manager,args.config_file_path)
 
   

    if args.enable_modify:
        modify_llama_attention(model,KVCache_manager)
    


    inference(
        model,
        tokenizer,
        prompts,
        KVCache_manager,
        enable_modify=args.enable_modify,
    )


if __name__ == "__main__":
    os.remove("DEBUG.log")
    logging.basicConfig(filename='DEBUG.log', level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_modify", action="store_true")
    args = parser.parse_args()
    main(args)