#!/bin/bash

input_file=$1
output_file=$2

# Clear the output file
> "$output_file"

# Use awk to parse and extract the required lines
awk '
BEGIN { sum=""; mem="" }
/sm__sass_sectors_mem_global\.sum/ {
    gsub(",", "", $NF)     # Remove commas from the number
    sum=$NF
}
/^ *Memory \[%\]/ {
    mem=$NF
}
sum != "" && mem != "" {
    print sum, mem >> "'"$output_file"'"
    sum=""; mem=""
}
' "$input_file"


