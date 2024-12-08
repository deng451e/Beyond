rm  attention-achieved-flops.log

for batch_size in 1 10
do 
  for  q_len in 1 100
  do
      for   seq_len in   10 50 100 500 1000 5000 10000  
        do 
          
              
          
 
 
            CMD_="--q_len $q_len --seq_len $seq_len   --batch_size $batch_size "
          

            CMD=$CMD_" --test_cpu"
            outpt=$(python  measure_operation_intensity.py  $CMD   2>&1  )  
            echo "$outpt" | tee -a attention-achieved-flops.log
 
            CMD=$CMD_" --test_gpu"
            outpt=$(python  measure_operation_intensity.py  $CMD    2>&1  )  
            echo "$outpt" | tee -a attention-achieved-flops.log

            CMD=$CMD_" --test_offload"
            outpt=$(python  measure_operation_intensity.py  $CMD   2>&1  )  
            echo "$outpt" | tee -a attention-achieved-flops.log




      done
  done
done 

 