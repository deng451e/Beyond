home_path="$PWD/../../.."
flexgen_path="$home_path/3rdparty/InfiniGen/speedup/flexgen"

export CUDA_LAUNCH_BLOCKING=0 # Debug use 
export CUDA_VISIBLE_DEVICES=0 
rm opt-1.3b-results.log
model="huggingface/opt-1.3b"
for prompt_len in  1024 #  500 600
    do 
    for gen_len in  512
    do  
        for bsz in 2 # 5 10   1 2 5
        do
            cmd_="--model $model --percent 100 0 0 100 100 0  --gpu-batch-size $bsz \
            --num-gpu-batches 1 --prompt-len $prompt_len --gen-len $gen_len --warmup-input-path $flexgen_path/pg19_firstbook.txt \
            --test-input-path  $flexgen_path/pg19_firstbook.txt"
 
            
            # beyond
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $home_path/beyond/modified_flexGen/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $home_path/beyond/modified_flexGen/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false --start_size 1 --recent_size 100"
            outpt="==================================================================================="$'\n'
            outpt+="method: beyond, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            # python -m flexgen.flex_opt $cmd 
            SECONDS=0 
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            outpt+="  Execution time: $SECONDS seconds"
            echo "$outpt"  | tee -a opt-1.3b-results.log
            

            # infinigen
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/infinigen/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/infinigen/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_"  --alpha 4 --partial-weight-ratio 0.2 --max-num-kv `expr \( $prompt_len + 128 \) / 5` --overlap false"
            outpt="==================================================================================="$'\n'
            outpt+="method: infinigen, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            SECONDS=0 
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            outpt+="  Execution time: $SECONDS seconds"
            echo "$outpt"  | tee -a opt-1.3b-results.log
            


            # flexgen 
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false"
            outpt="==================================================================================="$'\n'
            outpt+="method: flexgen, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n' 
            SECONDS=0 
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            outpt+="  Execution time: $SECONDS seconds"
            echo "$outpt"  | tee -a opt-1.3b-results.log
            

            # flexgen-overlap
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap true"
            outpt="==================================================================================="$'\n'
            outpt+="method: flexgen-overlap, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            SECONDS=0 
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            outpt+="  Execution time: $SECONDS seconds"
            echo "$outpt"  | tee -a opt-1.3b-results.log


            

            # h2o 
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/h2o/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/h2o/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false  --max-num-kv `expr \( $prompt_len + 128 \) / 5` --hh-ratio 0.1 --hh-all"
            outpt="==================================================================================="$'\n'
            outpt+="method: h2o, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            SECONDS=0 
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            outpt+="  Execution time: $SECONDS seconds"
            echo "$outpt"  | tee -a opt-1.3b-results.log
        
        done 
        
    done 
done 