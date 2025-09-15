import streamlit as st
import tempfile
import subprocess
import os
from gnuplot_template import generate_gnuplot_script

st.title("🎨 GNUPlot Script & Plot Generator")

uploaded_file = st.file_uploader("Upload your data file (.dat, .csv)", type=["dat", "csv", "txt"])
plot_type = st.selectbox("Select Plot Type", ["line", "scatter", "histogram", "boxes"])
terminal = st.selectbox("Output Format", ["png", "pdf"])
title = st.text_input("Plot Title", "SPMM Benchmark")
xlabel = st.text_input("X-axis Label", "Dataset")
ylabel = st.text_input("Y-axis Label", "Time (ms)")
use_header = st.checkbox("File has header row", value=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as temp_data:
        temp_data.write(uploaded_file.read())
        data_path = temp_data.name

    output_ext = "png" if terminal == "png" else "pdf"
    output_plot = os.path.join(tempfile.gettempdir(), f"plot_output.{output_ext}")
    gp_script = generate_gnuplot_script(
        data_file=data_path,
        output_file=output_plot,
        plot_type=plot_type,
        terminal=terminal,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        use_header=use_header
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gp", mode='w') as script_file:
        script_file.write(gp_script)
        gp_script_path = script_file.name

    st.code(gp_script, language='gnuplot')

    if st.button("Generate Plot"):
        try:
            subprocess.run(["gnuplot", gp_script_path], check=True)
            st.success("Plot generated!")

            with open(output_plot, "rb") as f:
                st.download_button("Download Plot", f, file_name=os.path.basename(output_plot), mime="image/png" if terminal == "png" else "application/pdf")

            st.image(output_plot) if terminal == "png" else st.info("PDF generated. Download above.")
        except Exception as e:
            st.error(f"Error generating plot: {e}")

