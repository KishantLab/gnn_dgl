def generate_gnuplot_script(data_file, output_file, plot_type, terminal, title, xlabel, ylabel, use_header):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    if use_header:
        header = lines[0].strip().split()
        lines = lines[1:]
    else:
        header = [f"Col{i}" for i in range(1, len(lines[0].split())+1)]

    num_cols = len(header)

    style_block = ""
    plot_cmds = []

    if plot_type == 'histogram':
        style_block += """
set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.9
set xtics rotate by -45
"""
        for i in range(2, num_cols+1):
            plot_cmds.append(f"'{data_file}' using {i}:xtic(1) title '{header[i-1]}'")
    elif plot_type == 'boxes':
        style_block += """
set style fill solid border -1
set boxwidth 0.9
"""
        for i in range(2, num_cols+1):
            plot_cmds.append(f"'{data_file}' using {i}:xtic(1) with boxes title '{header[i-1]}'")
    else:
        for i in range(2, num_cols+1):
            style = 'lines' if plot_type == 'line' else 'points'
            plot_cmds.append(f"'{data_file}' using 0:{i} with {style} title '{header[i-1]}'")

    # Build the script in parts to avoid f-string issues
    plot_block = "plot \\\n     " + ", \\\n     ".join(plot_cmds)

    script = f"""
set terminal {terminal}cairo

