rm results -r
rm longbench-test-results.log 
for method in     "beyond"    "normal"  "streamllm" 
do
    for model in   "lmsys/vicuna-7b-v1.5-16k"               "facebook/opt-1.3b"  "facebook/opt-6.7b"  "facebook/opt-13b"     "lmsys/vicuna-13b-v1.3"  
    do
       
        cmd="python longbench_evaluate.py --model $model \
          --start_size 10\
          --recent_size 1800\
          "
        if [ "$method" == "beyond" ];then
          cmd+=" --enable_beyond"   
        fi 

        if [ "$method" == "streamllm" ];then
          cmd+=" --enable_streamllm"   
        fi 
      
        outpt="======================================="$'\n'
        outpt+="model: $model, method: $method"$'\n'
        outpt+=$($cmd 2>&1 | grep -e "score:" -e "latency:" -e "dataset:")
        echo "$outpt" | tee -a longbench-test-results.log
    done
done 