#!/bin/bash

# echo -e "\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time"

# Define datasets and batch sizes
datasets=("reddit" "ogbn-products" "ogbn-arxiv" "amazon-products")
# datasets=("reddit" "ogbn-products" "ogbn-arxiv" "igb-small" "amazon-products" "wiki5M")
batch_sizes=("2048" "4096")
# batch_sizes=("1024" "2048" "4096" "8192" "16384" "32768")

for B in "${batch_sizes[@]}"; do
    echo -e "\n\t\t\tB = $B, F = $2, Sampling, $1, puregpu"
    # echo -e "Dataset              SPMM    Sampling  For_loop_time  Model_time  Loss Time  Total_time"
    echo -e "Dataset\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss Time\tTotal_time\tAccuracy"
    echo ""
    for dataset in "${datasets[@]}"; do
        dataset_upper=$(echo "$dataset" | sed 's/ogbn-/OGB-/;s/.*/\u&/') # Capitalize + fix ogbn
        #echo -e "$dataset_upper"

        file="$dataset/${dataset}_F$2_B${B}_100_Sampling_$1_puregpu.txt"
        if [ -f "$file" ]; then
            spmm=$(tail -n 1 "$file" | cut -d',' -f1)
            sampling=$(tail -n 1 "$file" | cut -d',' -f2)
            for_loop=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f1)
            model=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f2)
            loss=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f3)
            total=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f4)
	    accuracy=$(tail -n 3 "$file" | cut -d ' ' -f3)
            # echo -e "$dataset             $spmm  $sampling  $for_loop      $model    $loss     $total"
            echo -e "$dataset\t$spmm\t$sampling\t$for_loop\t$model\t$loss\t$total\t$accuracy"
        else
            echo -e "$dataset\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A"
        fi
    done
done

