
flexgen_path="{$PWD}/../../3rdparty/InfiniGen/speedup/flexgen"
python flex_opt.py --model huggingface/opt-13b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 20 /
--num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt /
--test-input-path pg19_firstbook.txt