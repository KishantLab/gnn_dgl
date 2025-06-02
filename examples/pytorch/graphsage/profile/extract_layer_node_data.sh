#!/bin/bash

# Input log file
INPUT_LOG=$1
# Output CSV file
OUTPUT_CSV=$2

# Write CSV header
echo "num_rows,num_x_lenght,nnz,num_cols" > "$OUTPUT_CSV"

# Extract and format the data using awk
awk '/csr num_rows/ {
    for(i=1; i<=NF; i++) {
        if ($i ~ /num_rows/) nr=$(i+1)
        if ($i ~ /num_x_lenght/) nx=$(i+1)
        if ($i ~ /nnz/) nnz=$(i+1)
        if ($i ~ /num_cols/) nc=$(i+1)
    }
    print nr "," nx "," nnz "," nc
}' "$INPUT_LOG" >> "$OUTPUT_CSV"

