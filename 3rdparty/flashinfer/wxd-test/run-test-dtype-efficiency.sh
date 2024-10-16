rm dtype-efficienct.log
       
CMD="test-dtype-efficiency.py"
output=$(python   $CMD   2>&1 )  
echo "$output" | tee -a dtype-efficienct.log
