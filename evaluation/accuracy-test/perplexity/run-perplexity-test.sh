
rm perplexity-test-results.log 
for method in   "normal"  "beyond" "streamllm" 
do
    for model in   "lmsys/vicuna-7b-v1.5-16k"  "lmsys/vicuna-13b-v1.3"        "facebook/opt-1.3b"  "facebook/opt-6.7b"  "facebook/opt-13b" "hakurei/lotus-12B"
    do
        for recent_size in 100 200 300 400
        do
          cmd="python eval_long_ppl.py --model $model \
            --dataset_name "wikitext" \
            --num_eval_tokens 500 \
            --num_samples 100 \
            --start_size 10\
            --recent_size $recent_size\
            "
          if [ "$method" == "beyond" ];then
            cmd+=" --enable_beyond"   
          fi 

          if [ "$method" == "streamllm" ];then
            cmd+=" --enable_streamllm"   
          fi 
        
          outpt="eval_size: 500, model: $model, method: $method, recent_size: $recent_size, "
          outpt+=$($cmd 2>&1 | grep -e "perplexity:" -e"latency:")
          echo "$outpt" | tee -a perplexity-test-results.log
        done
    done
done 