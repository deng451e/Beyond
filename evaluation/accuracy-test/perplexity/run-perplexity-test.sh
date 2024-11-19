# full cache 
echo "==    full cahe   =="
 
seqlen=2048
# python opt.py --model "facebook/opt-13b" \
#   --eval_dataset "wikitext2" \
#   --seq_len ${seqlen} \
#   --eval_samples 0 \
#   --beyond \
#   --print_blk_ppl \
#   --model_name "opt-13b" 
   
python opt.py --model "facebook/opt-13b" \
  --eval_dataset "wikitext2" \
  --seq_len ${seqlen} \
  --eval_samples 0 \
  --print_blk_ppl \
  --model_name "opt-13b" 
   

# cmd="python opt.py --model "facebook/opt-13b" \
#   --eval_dataset "wikitext2" \
#   --seq_len ${seqlen} \
#   --eval_samples 0 \
#   --model_name "opt-13b" \
#   --print_blk_ppl "
  
# outpt=$($cmd 2>&1 | grep -e "Perplexity:" -e"time taken:")
 
# echo $outpt
