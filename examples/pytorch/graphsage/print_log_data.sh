#!/bin/bash

base_dir="./training_time"
datasets=("reddit" "ogbn-products" "ogbn-arxiv")
Bs=(1024 2048)
F=10
spmm_types=("default" "gespmm" "respmm")
headers=("Default, cusparse" "Default, GE-SpMM" "Metis, ReSpMM")

# Print header row 1
echo -e "Dataset\tB = 1024, F = 10, ${headers[0]}\t\t\t\t\tB = 1024, F = 10, ${headers[1]}\t\t\t\t\tB = 1024, F = 10, ${headers[2]}"
echo -e "\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time"

# Function to extract values from a log file
extract_values() {
    local file=$1
    if [[ -f "$file" ]]; then
        spmm=$(grep -m1 "SPMM" "$file" | awk '{print $NF}')
        sampling=$(grep -m1 "sampling" "$file" | awk '{print $(NF-1)}')
        loop=$(grep -m1 "for loop" "$file" | awk '{print $(NF-1)}')
        model=$(grep -m1 "model" "$file" | awk '{print $(NF-1)}')
        loss=$(grep -m1 "loss" "$file" | awk '{print $(NF-1)}')
        total=$(grep -m1 "total" "$file" | awk '{print $(NF-1)}')
        echo -e "$spmm\t$sampling\t$loop\t$model\t$loss\t$total"
    else
        echo -e "N/A\tN/A\tN/A\tN/A\tN/A\tN/A"
    fi
}

# Print each dataset row for B=1024
for dataset in "${datasets[@]}"; do
    row="$dataset"
    for spmm in "${spmm_types[@]}"; do
        log_file="$base_dir/$dataset/${dataset}_F${F}_B1024_100_Sampling_${spmm}.txt"
        row+="\t$(extract_values "$log_file")"
    done
    echo -e "$row"
done

# Print header row 2
echo -e "\tB = 2048, F = 10, ${headers[0]}\t\t\t\t\tB = 2048, F = 10, ${headers[1]}\t\t\t\t\tB = 2048, F = 10, ${headers[2]}"
echo -e "\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time"

# Print each dataset row for B=2048
for dataset in "${datasets[@]}"; do
    row="$dataset"
    for spmm in "${spmm_types[@]}"; do
        log_file="$base_dir/$dataset/${dataset}_F${F}_B2048_100_Sampling_${spmm}.txt"
        row+="\t$(extract_values "$log_file")"
    done
    echo -e "$row"
done

