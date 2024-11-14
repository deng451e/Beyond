rm results/sparse-attention-test.log

for ratio in 0.1  0.3 0.5 0.7 0.9
do 
  for seq_len in 10 100 1000 10000
  do
      for q_len in 1 10 100
        do
          
              
          
            
            topk=$(echo "$ratio * $seq_len" | bc)
 
            CMD="--top_k $topk --seq_len $seq_len   --q_len $q_len"
          
            
            outpt=$(python   sparse-attention-test.py $CMD   2>&1  )  
            echo "$outpt" | tee -a results/sparse-attention-test.log
      done
  done
done


rm results/dense-attention-test.log

for batch_sizse in 1 10 
do 
  for seq_len in 10 100 1000 10000
  do
      for q_len in 1 10 100
        do
           
            
            CMD="--batch_size $batch_sizse --seq_len $seq_len   --q_len $q_len"
          
            
            outpt=$(python   dense-attention-test.py $CMD   2>&1  )  
            echo "$outpt" | tee -a results/dense-attention-test.log
      done
  done
done

 