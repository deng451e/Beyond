
rm perplexity-test-results.log
for method in "streamllm" #    "beyond"   "streamllm"  "normal"
do
    for model in  "facebook/opt-6.7b"  # "lmsys/vicuna-7b-v1.5-16k"      "lmsys/vicuna-13b-v1.3"  "facebook/opt-1.3b" "facebook/opt-2.7b"   
    do
        echo "======================================="
        cmd="python eval_long_ppl.py --model $model \
          --dataset_name "wikitext" \
          --num_eval_tokens 300 \
          --num_samples 100 \
          --start_size 10\
          --recent_size 200\
          "
        if [ "$method" == "beyond" ];then
          cmd+=" --enable_beyond"   
        fi 

        if [ "$method" == "streamllm" ];then
          cmd+=" --enable_streamllm"   
        fi 
        $cmd
        outpt="model: $model, method: $method, "
        # outpt+=$($cmd 2>&1 | grep -e "perplexity:" -e"latency:")
        # echo "$outpt" | tee -a perplexity-test-results.log
    done
done 