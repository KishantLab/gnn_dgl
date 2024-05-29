#!/bin/bash

# Run the command and save the output to a variable
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
batch_sizes=(1024 2048 4096)
fanouts=(10 15)

# output=$(python3 node_classification.py --dataset=$1 --batch_size=1024)
#python3 node_classification.py --dataset=ogbn-products --batch_size=1024
# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
for fanout in "${fanouts[@]}"; do
  # Loop through each batch size
  for batch_size in "${batch_sizes[@]}"; do
    sampling_time=0.0
    training_time=0.0

    output=$(python3 node_classification.py --dataset=$1 --batch_size=$batch_size --fan_out=$fanout,$fanout,$fanout --epoch=$2)
    filename="training_time/$1/$1_F${fanout}_B${batch_size}_${epoch}_SMEM.txt"
    echo "Dataset = $1, batch_size = $batch_size" > $filename
    #python3 node_classification.py --dataset=ogbn-products --batch_size=1024
    #Loop through the output lines
    while read -r line; do
      # Check if the line contains the string "cuda,sapmling"
      if [[ $line == cuda* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $4}')
        #echo $time_value
        sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
      fi
    done <<< "$output"

        # Check if epoch_data.txt exists
    if [ -f "epoch_data.txt" ]; then
            cat "epoch_data.txt" >> $filename
            echo "Data copied successfully!"
    else
            echo "Error: epoch_data.txt does not exist."
    fi

    # # Loop through the output lines
    # while read -r line; do
    # # Check if the line contains the string "cuda,sapmling"
    #   if [[ $line == Epoch* ]]; then
    #     t_time=$(echo $line | awk '{print $12}')
    #     training_time=$(echo "$training_time + $t_time" | bc -l)
    #     epoch_line=$line
    #     echo $epoch_line >> $filename
    #   fi
    # done <<< "$output"
    #
    # while read -r line; do
    # # Check if the line contains the string "cuda,sapmling"
    #   if [[ $line == Total* ]]; then
    #   # t_time=$(echo $line | awk '{print $12}')
    #   # training_time=$(echo "$training_time + $t_time" | bc -l)
    #   # echo $line
    #   tt_time=$line
    # fi
    # done <<< "$output"
    #
    # while read -r line; do
    # # Check if the line contains the string "cuda,sapmling"
    #   if [[ $line == Test* ]]; then
    #   # t_time=$(echo $line | awk '{print $12}')
    #   # training_time=$(echo "$training_time + $t_time" | bc -l)
    #   test=$line
    # fi
    # done <<< "$output"
    # Create filename
    # Print output to file
    #echo "$output" > "$filename"
    # Print total sampling time
    #print_total_sampling_time "$output" >> "$filename"
    echo "Total sampling time : " $sampling_time ", Total training time :" $training_time >> $filename
    # echo "Total sampling time : " $sampling_time/$epoch ", Total training time :" $training_time/$epoch >> $filename
    # echo $test >> $filename
    # echo $tt_time >> $filename
  done
done


