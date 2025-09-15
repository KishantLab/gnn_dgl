
set terminal pdfcairo size 1200,800 enhanced font 'Arial,12'
set output 'standalon_spmm.pdf'

set title "SPMM Benchmark"
set xlabel "Datasets"
set ylabel "Time (s)"
set grid

set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.9
set xtics rotate by -45

plot 'standalone_spmm.dat' using 2:xtic(1) title 'Cusparse', \
     'standalone_spmm.dat' using 3:xtic(1) title 'GE-SPMM', \
     'standalone_spmm.dat' using 4:xtic(1) title 'SPMM', \
     'standalone_spmm.dat' using 5:xtic(1) title 'Reordered'
