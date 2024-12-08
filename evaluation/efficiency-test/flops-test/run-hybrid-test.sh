rm  hybrid-attention-flops.log

for seq_len in 100 500 1000 5000 10000 100000
do 

       

        

        for  ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  
        do 
            CMD_="--q_len 1    --batch_size 1 --seq_len $seq_len --ratio $ratio "
                
            
                  
            CMD=$CMD_" --test_gpu  --test_hybrid "
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a hybrid-attention-flops.log
            
            CMD=$CMD_" --test_hybrid  "
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a hybrid-attention-flops.log
           


            echo "============================================================" | tee -a hybrid-attention-flops.log

        done
        
done 

 