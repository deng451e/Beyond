rm sparse-fetch-efficienct.log
 

CMD="test-sparse-fetch.py"
output=$(python   $CMD   2>&1 )  
echo "$output" | tee -a  sparse-fetch-efficienct.log





