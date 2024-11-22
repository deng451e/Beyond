CUDA_VISIBLE_DEVICES=0 python retrieval_evaluate.py --enable_beyond --start_size 50 --recent_size 2000



# rm retrieval-test-results.log 
# for method in   "normal"  "beyond" "streamllm" 
# do
#     for model in   "lmsys/vicuna-7b-v1.5-16k"  "lmsys/vicuna-13b-v1.3"   #     "facebook/opt-6.7b"     "facebook/opt-1.3b" "facebook/opt-2.7b"   
#     do
#         echo "======================================="
#         cmd="python eval_long_ppl.py --model $model \
#           --dataset_name "wikitext" \
#           --num_eval_tokens 300 \
#           --num_samples 100 \
#           --start_size 10\
#           --recent_size 200\
#           "
#         if [ "$method" == "beyond" ];then
#           cmd+=" --enable_beyond"   
#         fi 

#         if [ "$method" == "streamllm" ];then
#           cmd+=" --enable_streamllm"   
#         fi 
       
#         outpt="model: $model, method: $method, "
#         outpt+=$($cmd 2>&1 | grep -e "perplexity:" -e"latency:")
#         echo "$outpt" | tee -a retrieval-test-results.log
#     done
# done 