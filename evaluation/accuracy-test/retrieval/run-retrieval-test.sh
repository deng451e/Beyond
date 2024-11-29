# CUDA_VISIBLE_DEVICES=0 python retrieval_evaluate.py --enable_beyond --start_size 50 --recent_size 2000
export CUDA_VISIBLE_DEVICES=0 


rm retrieval-test-results.log 
for method in  "streamllm" # "beyond" #  "normal"  "streamllm" 
do
    for model in   "lmsys/vicuna-7b-v1.5-16k"   #    "lmsys/vicuna-13b-v1.3"     "facebook/opt-6.7b"     "facebook/opt-1.3b" "facebook/opt-2.7b"   
    do
        echo "======================================="
        cmd="python  retrieval_evaluate.py --model $model \
          --start_size 50\
          --recent_size 3000\
          "
        if [ "$method" == "beyond" ];then
          cmd+=" --enable_beyond"   
        fi 

        if [ "$method" == "streamllm" ];then
          cmd+=" --enable_streamllm"   
        fi 
        $cmd
        # outpt="model: $model, method: $method, "
        # outpt+=$($cmd 2>&1 | grep -e "perplexity:" -e"latency:")
        # echo "$outpt" | tee -a retrieval-test-results.log
    done
done 