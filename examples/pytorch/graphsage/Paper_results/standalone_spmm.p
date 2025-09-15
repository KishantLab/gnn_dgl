set terminal postscript eps enhanced color solid font 'Arial,45' size 12.0,6.0
set output 'standalone_spmm.eps'
set xlabel 'Dataset'
set ylabel 'Time (second)'
set key top right
set ytics 1500.0

set xtics rotate by -20
set style data histograms
set style fill solid 1.00 border lt -1
set boxwidth 0.9 absolute
plot \
    'standalone_spmm.dat' using 2:xtic(1) title 'Cusparse'  , \
    'standalone_spmm.dat' using 3:xtic(1) title 'GE-SPMM'  
