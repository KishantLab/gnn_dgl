echo -e "File\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss_time\tTotal_time"
for file in ogbn-arxiv/*_F10_B*_100_Sampling_gespmm.txt; do
    spmm=$(tail -n 1 "$file" | cut -d',' -f1)
    sampling=$(tail -n 1 "$file" | cut -d',' -f2)
    for_loop=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f1)
    model=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f2)
    loss=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f3)
    total=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f4)
    echo -e "$file\t$spmm\t$sampling\t$for_loop\t$model\t$loss\t$total"
done
echo -e "File\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss_time\tTotal_time"
for file in ogbn-products/*_F10_B*_100_Sampling_gespmm.txt; do
    spmm=$(tail -n 1 "$file" | cut -d',' -f1)
    sampling=$(tail -n 1 "$file" | cut -d',' -f2)
    for_loop=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f1)
    model=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f2)
    loss=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f3)
    total=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f4)
    echo -e "$file\t$spmm\t$sampling\t$for_loop\t$model\t$loss\t$total"
done
echo -e "File\tSPMM\tSampling\tFor_loop_time\tModel_time\tLoss_time\tTotal_time"
for file in reddit/*_F10_B*_100_Sampling_gespmm.txt; do
    spmm=$(tail -n 1 "$file" | cut -d',' -f1)
    sampling=$(tail -n 1 "$file" | cut -d',' -f2)
    for_loop=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f1)
    model=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f2)
    loss=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f3)
    total=$(tail -n 4 "$file" | head -n 1 | cut -d',' -f4)
    echo -e "$file\t$spmm\t$sampling\t$for_loop\t$model\t$loss\t$total"
done

