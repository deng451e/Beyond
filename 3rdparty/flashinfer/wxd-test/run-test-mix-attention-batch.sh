rm mix-attention-batch.log
for arch_name in "opt-1.3b"  "opt-6.7b" "opt-13b"
do 
  for batch_size in 1 2 5 
  do
    for seq_len in 10000 20000 50000
    do
        
          for q_len in 1 20 50   
          do  
              for ratio in   0.01 0.1 0.2 0.5  
              do 
              CMD="test-mix-attention-batch.py --arch_name $arch_name --batch_size $batch_size --seq_len $seq_len --q_len $q_len --ratio $ratio"
              
              output=$(python   $CMD   2>&1 )  
              echo "$output" | tee -a mix-attention-batch.log
              done
          done
    done
  done
done


 

