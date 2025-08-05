#!/bin/bash
show_progress() {
  local start_time=$(date +%s)
  local delay=1
    # Define multiple emoji lists
  # emoji_sets=(
  #   "⏰ ⏱️  ⏳ 🕐 🕜 🕑 🕝"
  #   "🔄 🔁 🔃 🔂 🔃 🔄"
  #   "🧪 🔬 ⚗️  🧫 🧬"
  #   "🛠️  ⚙️  🔧 🔩 🧰"
  #   "🚀 🛸 ✈️  🚁 🚂"
  #   "📡 💾 🖥️  💻 ⌨️"
  # ) 

  # # Pick one emoji set randomly
  # selected_index=$((RANDOM % ${#emoji_sets[@]}))
  # IFS=' ' read -r -a emoji_list <<< "${emoji_sets[$selected_index]}"
  local emoji_list=("🕐" "🕑" "🕒" "🕓" "🕔" "🕕" "🕖" "🕗" "🕘" "🕙" "🕚" "🕛")
  local i=0 

  while true; do
    local now=$(date +%s)
    local elapsed=$((now - start_time))
        
    # Format into HH:MM:SS
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
        
    printf "\r⏳ Elapsed: %02d:%02d:%02d %s Running..." $hours $minutes $seconds "${emoji_list[$i]}"
    ((i=(i+1)%${#emoji_list[@]}))
    sleep 1
    # sleep $delay
  done
}

# Run the command and save the output to a variable
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
# batch_sizes=(1024 2048 4096 8192 16384 32768 65536 131072)
batch_sizes=(128 256 512 1024 2048 4096)
fanouts=(-1)

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

    echo -e "\n▶️ Running config: Dataset=$dataset | Fanout=$fanout | Batch=$batch_size | Epoch=$epoch"
    # Start progress animation in background
    show_progress &
    progress_pid=$!

    output=$(python3 node_classification.py --dataset=$1 --batch_size=$batch_size --fan_out=$fanout,$fanout,$fanout --epoch=$2 --spmm=gespmm --sampling default --sampler full)
    
# Kill the progress spinner
    kill $progress_pid >/dev/null 2>&1
    wait $progress_pid 2>/dev/null

    filename="training_time/$1/$1_F${fanout}_B${batch_size}_${epoch}_Sampling_gespmm_full_sampler.txt"
    echo "Dataset = $1, batch_size = $batch_size" > $filename
    #python3 node_classification.py --dataset=ogbn-products --batch_size=1024
    #Loop through the output lines
    while read -r line; do
      if [[ $line == Testing...\* ]]; then
        add_spmm_time=false
      fi
      # Check if the line contains the string "cuda,sapmling"
      # if [[ $line == cusparse\ spmm\ time* ]] && $add_spmm_time; then
      if [[ $line == ge-spmm\ time* ]] && $add_spmm_time; then
        # Extract the time value and add it to the sampling time
        #echo $line
        # spmm_time_value=$(echo $line | awk '{print $3}')
        last_spmm_time=$(echo $line | awk '{print $3}')
        #echo $time_value
        # spmm_time=$(echo "$spmm_time + $spmm_time_value" | bc -l)
        # fi
      elif [[ $line == default\ cuda\ sapmling\ time* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        # time_value=$(echo $line | awk '{print $4}')
        last_cuda_sampling_time=$(echo $line | awk '{print $5}')
        #echo $time_value
        # sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
      fi

    done <<< "$output"

      echo "last_spmm_time: $last_spmm_time , last_cuda_sampling_time: $last_cuda_sampling_time"
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


