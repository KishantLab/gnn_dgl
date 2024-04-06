#!/bin/bash

# Run the command and save the output to a variable
output=$(python3 node_classification.py --dataset=ogbn-arxiv --batch_size=1024)
#python3 node_classification.py --dataset=ogbn-products --batch_size=1024
# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
sampling_time=0.0
training_time=0.0
# last_epoch_line=false

# Loop through the output lines
while read -r line; do
  # Check if the line contains the string "cuda,sapmling"
  if [[ $line == cuda* ]]; then
    # Extract the time value and add it to the sampling time
    #echo $line
    time_value=$(echo $line | awk '{print $4}')
    #echo $time_value
    sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
    #echo $sampling_time
    #sampling_time=$(($sampling_time + $time_value))
    #result=$(echo "$sampling_time + $time_value" | bc -l)
  # Check if the line contains the string "Epoch"
  #elif [[ $line == Epoch* ]]; then
    # If this is the first "Epoch" line, print it as-is
     #if ! $last_epoch_line; then
     #echo $line | awk '{print $12}'
     #t_time=$(echo $line | awk '{print $12}')
     #training_time=$(echo "$training_time + $t_time" | bc -l)
     #echo $line
     #echo "Epoch ${line#*|} | Sampling Time: $sampling_time"
       #last_epoch_line=true
    # If this is a subsequent "Epoch" line, print it with the total sampling time
     #else
       #echo "Epoch ${line#*|} | Sampling Time: $sampling_time"
       #last_epoch_line=false
    #fi
  # If the line is not "Epoch" or "cuda,sapmling", print it as-is
  #else
    #echo $line 
  fi
done <<< "$output"

# Loop through the output lines
while read -r line; do
  # Check if the line contains the string "cuda,sapmling"
  if [[ $line == Epoch* ]]; then
     t_time=$(echo $line | awk '{print $12}')
     training_time=$(echo "$training_time + $t_time" | bc -l)
     # echo $line
  fi
done <<< "$output"

while read -r line; do
  # Check if the line contains the string "cuda,sapmling"
  if [[ $line == Total* ]]; then
     # t_time=$(echo $line | awk '{print $12}')
     # training_time=$(echo "$training_time + $t_time" | bc -l)
     # echo $line
     tt_time=$line
  fi
done <<< "$output"

while read -r line; do
  # Check if the line contains the string "cuda,sapmling"
  if [[ $line == Test* ]]; then
     # t_time=$(echo $line | awk '{print $12}')
     # training_time=$(echo "$training_time + $t_time" | bc -l)
     test=$line
  fi
done <<< "$output"

echo "Total sampling time : " $sampling_time ", Total training time :" $training_time
echo "Total sampling time : " $sampling_time/10 ", Total training time :" $training_time/10
echo $test
echo $tt_time

# Print the total training time
# echo "Total Training time: $(echo $output | grep -oP 'Total Training time \K[\d.]+')"
