#!/bin/bash

# Run the command and save the output to a variable
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
batch_sizes=(2000 4000 8000)
fanouts=(20 30)

# output=$(python3 node_classification.py --dataset=$1 --batch_size=1024)
#python3 node_classification.py --dataset=ogbn-products --batch_size=1024
# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
for fanout in "${fanouts[@]}"; do
  # Loop through each batch size
  for batch_size in "${batch_sizes[@]}"; do
    sampling_time=0.0
    training_time=0.0
    spmm_time=0.0
    last_spmm_time=""
    last_cuda_sampling_time=""
    add_spmm_time=true

    output=$(python3 node_classification.py --dataset=$1 --batch_size=$batch_size --fan_out=$fanout,$fanout,$fanout --epoch=$2 --spmm=respmm)
    filename="training_time/metis/$1/$1_F${fanout}_B${batch_size}_${epoch}_Sampling_respmm.txt"
    echo "Dataset = $1, batch_size = $batch_size" > $filename
    #python3 node_classification.py --dataset=ogbn-products --batch_size=1024
    #Loop through the output lines
    while read -r line; do
      if [[ $line == Testing...* ]]; then
        add_spmm_time=false
      fi
      # Check if the line contains the string "cuda,sapmling"
      if [[ $line == cusparse\ spmm\ time* ]] && $add_spmm_time; then
        # Extract the time value and add it to the sampling time
        #echo $line
        # spmm_time_value=$(echo $line | awk '{print $3}')
        last_spmm_time=$(echo $line | awk '{print $4}')
        #echo $time_value
        # spmm_time=$(echo "$spmm_time + $spmm_time_value" | bc -l)
        # fi
      elif [[ $line == cuda\ sapmling\ time* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        # time_value=$(echo $line | awk '{print $4}')
        last_cuda_sampling_time=$(echo $line | awk '{print $4}')
        #echo $time_value
        # sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
      fi
      echo $last_spmm_time
      echo $last_cuda_sampling_time

    done <<< "$output"

        # Check if epoch_data.txt exists
        if [ -f "epoch_data.txt" ]; then
          cat "epoch_data.txt" >> $filename
          echo "Data copied successfully!"
        else
          echo "Error: epoch_data.txt does not exist."
        fi
        # echo "Total sampling time :" $sampling_time ", Total training time :" $training_time >> $filename
        # echo "Total spmm time , Total sampling time" >> $filename
        # echo $spmm_time"," $sampling_time >> $filename
        echo "last_spmm_time, last_cuda_sampling_time" >> $filename
        echo "$last_spmm_time, $last_cuda_sampling_time" >> $filename
        # echo "Total sampling time : " $sampling_time/$epoch ", Total training time :" $training_time/$epoch >> $filename
        # echo $test >> $filename
        # echo $tt_time >> $filename
      done
    done


