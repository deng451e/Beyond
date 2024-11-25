home_path="$PWD/../../.."
flexgen_path="$home_path/3rdparty/InfiniGen/speedup/flexgen"

gen_len=600
prompt_len=600
 

rm flexgen-opt-results.log
 
for gen_len in 500 600 700
    do 
    for model in  "huggingface/opt-6.7b"   "huggingface/opt-1.3b" #    "huggingface/opt-1.3b"
    do  
        for bsz in 1 2 5
        do
            cmd_="--model $model --percent 100 0 0 100 100 0  --gpu-batch-size $bsz \
            --num-gpu-batches 1 --prompt-len $prompt_len --gen-len $gen_len --warmup-input-path $flexgen_path/pg19_firstbook.txt \
            --test-input-path  $flexgen_path/pg19_firstbook.txt"

        


            # beyond
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $home_path/beyond/modified_flexGen/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $home_path/beyond/modified_flexGen/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false --start_size 10 --recent_size 1000"
            outpt="==================================================================================="$'\n'
            outpt+="method: beyond, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            echo "$outpt"  | tee -a flexgen-opt-results.log
            

            #infinigen
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/infinigen/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/infinigen/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_"  --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 102 --overlap false"
            outpt="==================================================================================="$'\n'
            outpt+="method: infinigen, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            echo "$outpt"  | tee -a flexgen-opt-results.log
            



            # flexgen 
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false"
            outpt="==================================================================================="$'\n'
            outpt+="method: flexgen, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            echo "$outpt"  | tee -a flexgen-opt-results.log
            

            # flexgen-overlap
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap true"
            outpt="==================================================================================="$'\n'
            outpt+="method: flexgen-overlap, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            echo "$outpt"  | tee -a flexgen-opt-results.log


            

            # h2o 
            rm $flexgen_path/flexgen/flex_opt.py
            rm $flexgen_path/flexgen/pytorch_backend.py
            ln -s  $flexgen_path/h2o/flex_opt.py $flexgen_path/flexgen/flex_opt.py
            ln -s  $flexgen_path/h2o/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
            cmd=$cmd_" --overlap false  --max-num-kv 102 --hh-ratio 0.1 --hh-all"
            outpt="==================================================================================="$'\n'
            outpt+="method: h2o, model: $model, intput:$prompt_len, output:$gen_len , bzs: $bsz"$'\n'
            outpt+=$( python -m flexgen.flex_opt $cmd 2>&1 | grep   -e"Total:" -e"Prefill:" -e"Decode:")
            echo "$outpt"  | tee -a flexgen-opt-results.log
        done 
        
    done 
done 