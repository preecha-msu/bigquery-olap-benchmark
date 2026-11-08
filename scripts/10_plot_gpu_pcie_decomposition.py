# ไฟล์: scripts/10_plot_gpu_pcie_decomposition.py
# สร้างกราฟ 300 DPI (2-Panel) เปรียบเทียบ DuckDB Native CPU vs. CUDA GPU (Stacked PCIe H2D + Warm Kernel) และ Speedup Crossover
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XTICK_LABELS = [
    "L1\nScan/Filter\n(CSCI=21.0)",
    "L2\nMulti-Join\n(CSCI=28.8)",
    "L3\nNested CTE\n(CSCI=33.5)",
    "L4\nWindow\n(CSCI=34.2)",
    "L5\nUNNEST\n(CSCI=36.4)"
]

def generate_gpu_decomposition_figure(root: Path, sf: int = 1):
    sum_csv = root / "results" / "summary_tables" / f"gpu_vs_cpu_vs_bq_summary_sf{sf}.csv"
    if not sum_csv.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {sum_csv} กรุณารันสคริปต์ 09 ก่อน")

    df = pd.read_csv(sum_csv)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.8,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 9.0
    })

    fig_dir = root / "results" / "figures_300dpi"
    ms_fig_dir = root / "manuscript" / "figures"
    for d in [fig_dir, ms_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=300)
    x = np.arange(len(df))
    w = 0.34

    cpu_ms = df["DuckDB_Native_CPU_ms"].values
    h2d_ms = df["GPU_PCIe_H2D_ms"].values
    warm_ms = df["GPU_Warm_Kernel_ms"].values
    cold_ms = df["GPU_Cold_Total_ms"].values

    # -------------------------------------------------------------------------
    # Panel (a): DuckDB Vectorized CPU vs. Stacked GPU Latency (PCIe H2D + Warm Kernel)
    # -------------------------------------------------------------------------
    bars_cpu = ax1.bar(
        x - w/2, cpu_ms, w,
        label="DuckDB Native Columnar (Vectorized CPU)",
        color="#4C72B0", edgecolor="black", linewidth=0.7
    )
    bars_warm = ax1.bar(
        x + w/2, warm_ms, w,
        label="GPU Warm Kernel Compute (In-VRAM)",
        color="#55A868", edgecolor="black", linewidth=0.7
    )
    bars_h2d = ax1.bar(
        x + w/2, h2d_ms, w, bottom=warm_ms,
        label="PCIe Host-to-Device (H2D) Transfer Overhead",
        color="#DD8452", edgecolor="black", linewidth=0.7, hatch="//"
    )

    for i in range(len(x)):
        ax1.text(x[i] - w/2, cpu_ms[i] + 7, f"{cpu_ms[i]:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#2B4C7E")
        ax1.text(x[i] + w/2, cold_ms[i] + 7, f"{cold_ms[i]:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#8C4A27")

    # ชี้จุด PCIe Crossover ที่ L1
    ax1.annotate(
        "L1 PCIe Crossover:\nCold GPU (55.0 ms) >\nCPU (50.8 ms)",
        xy=(0 + w/2, cold_ms[0]),
        xytext=(0.55, 235),
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2),
        fontsize=8.2, fontweight="bold", color="#C44E52",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec="#C44E52", lw=0.8)
    )

    ax1.set_ylabel("Execution Latency (ms, Linear Scale)")
    ax1.set_title("(a) Vectorized CPU vs. CUDA GPU Latency Decomposition (PCIe H2D + Kernel)", pad=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(XTICK_LABELS)
    ax1.set_ylim(0, max(cpu_ms) * 1.16)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper left")

    # -------------------------------------------------------------------------
    # Panel (b): Architectural Speedup Trajectory & Break-Even Threshold (1.0x)
    # -------------------------------------------------------------------------
    sp_warm_cpu = df["Speedup_WarmGPU_vs_CPU"].values
    sp_cold_cpu = df["Speedup_ColdGPU_vs_CPU"].values
    sp_warm_bq = df["Speedup_WarmGPU_vs_BQ"].values

    ax2.plot(x, sp_warm_bq, marker="^", markersize=7, linewidth=2.0, linestyle="-.", color="#8172B2", label="Warm GPU vs. BigQuery Clustered (Cloud)")
    ax2.plot(x, sp_warm_cpu, marker="s", markersize=7, linewidth=2.2, linestyle="-", color="#55A868", label="Warm GPU (In-VRAM) vs. DuckDB Native CPU")
    ax2.plot(x, sp_cold_cpu, marker="o", markersize=7, linewidth=2.2, linestyle="-", color="#C44E52", label="Cold GPU (Incl. PCIe H2D) vs. DuckDB Native CPU")

    # เส้นประ Break-Even (1.0x)
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="CPU-GPU Break-Even Threshold (1.0x)")

    for i in range(len(x)):
        ax2.text(x[i], sp_cold_cpu[i] * 0.72 if i == 0 else sp_cold_cpu[i] * 1.15, f"{sp_cold_cpu[i]:.2f}x",
                 ha="center", va="center", fontsize=8.0, fontweight="bold", color="#C44E52")
        ax2.text(x[i], sp_warm_cpu[i] * 1.18, f"{sp_warm_cpu[i]:.2f}x",
                 ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#2E7D32")
        ax2.text(x[i], sp_warm_bq[i] * 1.15, f"{sp_warm_bq[i]:.1f}x",
                 ha="center", va="bottom", fontsize=7.8, color="#5E4FA2")

    ax2.set_yscale("log")
    ax2.set_ylabel("Speedup Ratio (Log Scale, Higher is Faster)")
    ax2.set_title("(b) GPU Acceleration Trajectory Across AST Complexity Tiers (L1-L5)", pad=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(XTICK_LABELS)
    ax2.set_ylim(0.5, 260)
    ax2.grid(True, which="both", linestyle="--", alpha=0.45)
    ax2.legend(frameon=True, loc="upper left")

    plt.tight_layout(w_pad=2.4)

    pdf_name = f"fig_gpu_pcie_decomposition_sf{sf}_300dpi.pdf"
    png_name = f"fig_gpu_pcie_decomposition_sf{sf}_300dpi.png"

    fig.savefig(fig_dir / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / png_name, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(ms_fig_dir / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 100)
    print(f" [OK] Generated 300 DPI GPU Decomposition PDF : {ms_fig_dir / pdf_name}")
    print(f" [OK] Generated 300 DPI GPU Decomposition PNG : {fig_dir / png_name}")
    print("=" * 100)

def inject_figure3_into_main_tex(root: Path):
    tex_path = root / "manuscript" / "main.tex"
    if not tex_path.exists():
        return
    tex = tex_path.read_text(encoding="utf-8")
    if "fig:gpu_pcie_decomposition" in tex:
        print(" [OK] Figure 3 (fig:gpu_pcie_decomposition) is already included in manuscript/main.tex")
        return

    fig3_block = r"""
\begin{figure*}[t!]
\centering
\IfFileExists{figures/fig_gpu_pcie_decomposition_sf1_300dpi.pdf}{%
    \includegraphics[width=0.96\textwidth]{figures/fig_gpu_pcie_decomposition_sf1_300dpi.pdf}%
}{%
    \fbox{\parbox{0.92\textwidth}{\centering\vspace{1.2em}\textbf{Figure 3 Placeholder:} Upload \texttt{figures/fig\_gpu\_pcie\_decomposition\_sf1\_300dpi.pdf} to Overleaf.\vspace{1.2em}}}%
}
\caption{Hardware-timed CUDA GPU latency decomposition on NVIDIA GeForce RTX 2050 ($4.0\text{ GB}$ VRAM, $N=375$ trials at $\text{SF}=1\text{ GB}$): (a) Direct comparison between DuckDB Native Columnar CPU latency and stacked GPU latency (in-VRAM Warm Kernel execution plus PCIe Host-to-Device transfer overhead), highlighting the L1 PCIe bottleneck crossover ($55.02\text{ ms}$ Cold GPU vs. $50.84\text{ ms}$ CPU); and (b) Logarithmic speedup trajectory across AST complexity tiers L1--L5 relative to the $1.0\times$ CPU--GPU break-even threshold.}
\label{fig:gpu_pcie_decomposition}
\end{figure*}
"""
    target_anchor = r"\section{EMPIRICAL RESULTS AND ANALYSIS}"
    if target_anchor in tex:
        tex = tex.replace(target_anchor, fig3_block + "\n" + target_anchor)
        tex = tex.replace(
            "Table~\\ref{tab:gpu_three_tier} presents",
            "Table~\\ref{tab:gpu_three_tier} and Fig.~\\ref{fig:gpu_pcie_decomposition} present"
        )
        tex_path.write_text(tex, encoding="utf-8")
        print(" [OK] Injected Figure 3 (fig:gpu_pcie_decomposition) into manuscript/main.tex")

if __name__ == "__main__":
    generate_gpu_decomposition_figure(PROJECT_ROOT, sf=1)
    inject_figure3_into_main_tex(PROJECT_ROOT)
