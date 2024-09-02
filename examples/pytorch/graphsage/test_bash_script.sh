#!/bin/bash

# Filename of the terminal output
filename="output.txt"

# Initialize variables to store the times
last_spmm_time=""
last_cuda_sampling_time=""

# Read the file line by line
while IFS= read -r line; do
    if [[ $line == spmm\ time* ]]; then
        # Extract the time value for spmm time
        last_spmm_time=$(echo $line | awk '{print $3}')
    elif [[ $line == cuda\ sapmling\ time* ]]; then
        # Extract the time value for cuda sampling time
        last_cuda_sampling_time=$(echo $line | awk '{print $4}')
    fi
done < "$filename"

# Print the last times (or you can use them as needed in your script)
echo "Last spmm time: $last_spmm_time"
echo "Last cuda sapmling time: $last_cuda_sampling_time"

