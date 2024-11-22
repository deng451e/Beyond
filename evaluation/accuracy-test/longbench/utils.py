import os
import argparse
from datasets import load_dataset
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Tuple

from fastchat.model import get_conversation_template
import numpy as np 
import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib
from tqdm import tqdm
from typing import List
from collections import Counter
from rouge import Rouge
import time
 
 
import random


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_eos(text):
        return text.replace("</s>", " ")

    return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    # use this for llama-3
    # return white_space_fix(remove_articles(remove_punc(lower(remove_eos(s)))))



def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')

    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes, length_range='all'):
    if length_range == 'all':
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
    elif length_range == "0-4k":
        scores = {"0-4k": []}
    elif length_range == "4-8k":
        scores = {"4-8k": []}
    elif length_range == "8k+":
        scores = {"8k+": []}
    else:
        raise ValueError("Invalid length range")
    
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    # # save the score list into a csv
    # for i in range(len(scores['0-4k'])):
    #     print(f"{i}: {scores['0-4k'][i]}")

    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)
# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
        stop_token_ids = tokenizer.eos_token_id
    else:
        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    return prompt

def build_stop_token(model_name, tokenizer):
    conv = get_conversation_template(model_name)
    if conv.stop_token_ids is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have eos_token_id")
        else:
            return [tokenizer.eos_token_id]
    else:
        return conv.stop_token_ids  

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


@torch.no_grad()
def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, length_range='all',enable_beyond=False,KVCache_manager=None,kv_cache=None):
    preds = []
    if length_range == 'all':
        pass
    elif length_range == '0-4k':
        data = [d for d in data if d["length"] < 4000]
    elif length_range == '4-8k':
        data = [d for d in data if d["length"] >= 4000 and d["length"] < 8000]
    elif length_range == '8k+':
        data = [d for d in data if d["length"] >= 8000]
    else:
        raise ValueError("Invalid length range")

    stop_token_ids = build_stop_token(model_name, tokenizer)
    cnt =  0
     
    latency = []
    for json_obj in tqdm(data):
        if cnt==3: break 
        cnt += 1
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
      
        if  enable_beyond: 
            KVCache_manager.clear_kv_cache()
            KVCache_manager.modify_recent_size(input.input_ids.shape[-1]-10)
        context_length = input.input_ids.shape[-1]
        st = time.time()
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]] if stop_token_ids is None else stop_token_ids+[tokenizer.encode("\n", add_special_tokens=False)[-1]]
            
            ###################################################
            past_key_values=None
            
            
            outputs = model(
                input_ids=input.input_ids.to(device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values if not  enable_beyond else None 
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = [pred_token_idx.item() ]
             
            
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if enable_beyond:
                    past_key_values = None 
                elif kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
                else:
                    past_key_values = outputs.past_key_values
                    # print(pred_token_idx.shape,past_key_values[0][0].shape,past_key_values[0][0].dtype)
                
                    
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                
                generated_ids.append(pred_token_idx.item() )
                
                if pred_token_idx == stop_token_ids:
                
                    break
    
 
            output  = generated_ids
            ###########################################################
        else:
            try:
                ###################################################
                past_key_values=None
                
               
                outputs = model(
                    input_ids=input.input_ids.to(device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values if not  enable_beyond else None 
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = [pred_token_idx.item() ]
                pos = 0
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    if enable_beyond:
                        past_key_values = None 
                        
                    else:
                        past_key_values = outputs.past_key_values
                     
                    
                        
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    
                    generated_ids.append(pred_token_idx.item() )
                    
                    if pred_token_idx == stop_token_ids:
                    
                        break
                    
                output  = generated_ids
    

                ###########################################################
            except Exception as e:
                print(e)
                # if generate fails, return empty output result
                output = torch.tensor([tokenizer.eos_token_id] * (context_length + 1)).to(device)
        latency.append(time.time()-st)
        pred = tokenizer.decode(output , skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        
    return preds,np.mean(latency)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

