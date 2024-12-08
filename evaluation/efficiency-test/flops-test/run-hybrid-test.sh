rm  hybrid-attention-flops.log

for seq_len in   1000 5000   10000 50000 100000 500000 1000000 # 5000000 10000000
do 

       

        

        for  ratio in 0.001  0.05 0.1  0.15 0.2 # 0.25 0.3  # 0.35 0.4 0.45 0.5
        do 
            CMD_="--q_len 1    --batch_size 1 --seq_len $seq_len --ratio $ratio "
                
            CMD=$CMD_" --test_cpu  --test_hybrid "
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a hybrid-attention-flops.log
                  
            CMD=$CMD_" --test_gpu  --test_hybrid "
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a hybrid-attention-flops.log

            CMD=$CMD_" --test_hybrid  "
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a hybrid-attention-flops.log
           


            echo "============================================================" | tee -a hybrid-attention-flops.log

        done
        
done 

 