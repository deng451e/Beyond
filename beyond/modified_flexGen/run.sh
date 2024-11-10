
flexgen_path="$PWD/../../3rdparty/InfiniGen/speedup/flexgen"
cmd_="--model huggingface/opt-6.7b --percent 100 0 0 100 100 0  --gpu-batch-size 5 \
--num-gpu-batches 1 --prompt-len 384 --gen-len 128 --warmup-input-path $flexgen_path/pg19_firstbook.txt \
--test-input-path  $flexgen_path/pg19_firstbook.txt"



# # beyond
# rm $flexgen_path/flexgen/flex_opt.py
# rm $flexgen_path/flexgen/pytorch_backend.py
# ln -s  $PWD/flex_opt.py $flexgen_path/flexgen/flex_opt.py
# ln -s  $PWD/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
# cmd=$cmd_" --overlap false --start_size 10 --recent_size 30"
# python -m flexgen.flex_opt $cmd


 
# #infinigen
# rm $flexgen_path/flexgen/flex_opt.py
# rm $flexgen_path/flexgen/pytorch_backend.py
# ln -s  $flexgen_path/infinigen/flex_opt.py $flexgen_path/flexgen/flex_opt.py
# ln -s  $flexgen_path/infinigen/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
# cmd=$cmd_"  --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 102 --overlap false"
# python -m flexgen.flex_opt $cmd
 

# # original-overlap
# rm $flexgen_path/flexgen/flex_opt.py
# rm $flexgen_path/flexgen/pytorch_backend.py
# ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
# ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
# cmd=$cmd_" --overlap true"
# python -m flexgen.flex_opt $cmd


# # original 
# rm $flexgen_path/flexgen/flex_opt.py
# rm $flexgen_path/flexgen/pytorch_backend.py
# ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
# ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
# cmd=$cmd_" --overlap false"
# python -m flexgen.flex_opt $cmd




# # original 
# rm $flexgen_path/flexgen/flex_opt.py
# rm $flexgen_path/flexgen/pytorch_backend.py
# ln -s  $flexgen_path/original/flex_opt.py $flexgen_path/flexgen/flex_opt.py
# ln -s  $flexgen_path/original/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
# cmd=$cmd_" --overlap false"
# python -m flexgen.flex_opt $cmd
   



# h2o 
rm $flexgen_path/flexgen/flex_opt.py
rm $flexgen_path/flexgen/pytorch_backend.py
ln -s  $flexgen_path/h2o/flex_opt.py $flexgen_path/flexgen/flex_opt.py
ln -s  $flexgen_path/h2o/pytorch_backend.py $flexgen_path/flexgen/pytorch_backend.py
cmd=$cmd_" --overlap false  --max-num-kv 102 --hh-ratio 0.1 --hh-all"
python -m flexgen.flex_opt $cmd
   

 