rm ${PWD}/bench_mergeState_zeroCopy_rtx4090.log
rm ${PWD}/bench_mergeState_rtx4090.log
export CUDA_VISIBLE_DEVICES=0
${PWD}/../build/bench_all_mergeState_ZeroCopy_methods > ${PWD}/bench_mergeState_zeroCopy_rtx4090.log
${PWD}/../build/bench_all_mergeState_methods > ${PWD}/bench_mergeState_rtx4090.log