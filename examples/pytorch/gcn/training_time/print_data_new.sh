#!/bin/bash

echo -e "\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time"

# Define datasets and batch sizes
datasets=("reddit" "ogbn-products" "ogbn-arxiv")
batch_sizes=("1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072")

for B in "${batch_sizes[@]}"; do
    echo -e "\n\t\t\t\t\tB = $B, F = $2, Sampling, $1"
    echo ""
    for dataset in "${datasets[@]}"; do
        dataset_upper=$(echo "$dataset" | sed 's/ogbn-/OGB-/;s/.*/\u&/') # Capitalize + fix ogbn
        #echo -e "$dataset_upper"

        file="$dataset/${dataset}_F$2_B${B}_100_Sampling_$1.txt"
        if [ -f "$file" ]; then
            spmm=$(tail -n 1 "$file" | cut -d',' -f1)
            sampling=$(tail -n 1 "$file" | cut -d',' -f2)
            for_loop=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f1)
            model=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f2)
            loss=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f3)
            total=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f4)
            echo -e "$spmm\t$sampling\t$for_loop\t$model\t$loss\t$total"
        else
            echo -e "N/A\tN/A\tN/A\tN/A\tN/A\tN/A"
        fi
    done
done

