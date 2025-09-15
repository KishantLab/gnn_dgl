import argparse
import os
import subprocess

TEMPLATE = """
set terminal {terminal_type} size 1200,800 enhanced font 'Arial,12'
set output '{output_file}'

set title "{title}"
set xlabel "{xlabel}"
set ylabel "{ylabel}"
set grid

{style_block}

plot {plot_commands}
"""

def generate_plot(data_file, output_file, plot_type, terminal, title, xlabel, ylabel, use_header):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    # Auto-detect columns
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
    else:  # line or scatter
        for i in range(2, num_cols+1):
            style = 'lines' if plot_type == 'line' else 'points'
            plot_cmds.append(f"'{data_file}' using 0:{i} with {style} title '{header[i-1]}'")

    script = TEMPLATE.format(
        terminal_type=f"{terminal}cairo",
        output_file=output_file,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        style_block=style_block.strip(),
        plot_commands=", \\\n     ".join(plot_cmds)
    )

    script_file = output_file.rsplit('.', 1)[0] + '.gp'
    with open(script_file, 'w') as f:
        f.write(script)

    print(f"[+] Gnuplot script written to: {script_file}")
    return script_file


def run_gnuplot(script_file):
    try:
        subprocess.run(['gnuplot', script_file], check=True)
        print("[✓] Plot generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[X] Gnuplot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNUPlot Script Generator CLI")
    parser.add_argument("data", help="Path to input data file (.dat or .csv)")
    parser.add_argument("-o", "--output", default="plot.png", help="Output plot file (e.g., plot.png or plot.pdf)")
    parser.add_argument("-t", "--type", choices=["line", "scatter", "histogram", "boxes"], default="line", help="Type of plot")
    parser.add_argument("--title", default="Plot Title", help="Title of the plot")
    parser.add_argument("--xlabel", default="X-axis", help="Label for X-axis")
    parser.add_argument("--ylabel", default="Y-axis", help="Label for Y-axis")
    parser.add_argument("--terminal", choices=["png", "pdf"], default="png", help="Output format (png/pdf)")
    parser.add_argument("--run", action="store_true", help="Run GNUPlot to generate the plot immediately")
    parser.add_argument("--header", action="store_true", help="Indicates if the first row is a header")

    args = parser.parse_args()

    script_path = generate_plot(
        data_file=args.data,
        output_file=args.output,
        plot_type=args.type,
        terminal=args.terminal,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        use_header=args.header
    )

    if args.run:
        run_gnuplot(script_path)

