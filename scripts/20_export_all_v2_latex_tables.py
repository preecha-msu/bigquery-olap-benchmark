#!/usr/bin/env python3
r"""
scripts/20_export_all_v2_latex_tables.py
แปลงไฟล์ CSV ข้อมูลจริงทั้งหมดใน results/ ให้เป็นไฟล์ตาราง LaTeX (.tex) ใน manuscript/
ครบทุกประเด็นของ Reviewer (3.1 - 3.8 และ 4.1 - 4.2) แบบ Zero-Discrepancy
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def export_all_tables() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ตาราง Official TPC-H Q1-Q22 ข้าม Engine (15 คอลัมน์ล่าสุด)
    df_tpch = pd.read_csv(RESULTS_DIR / "official_tpch_q1_q22_external_validation.csv")
    lines_tpch = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{External benchmark validation on the official TPC-H query suite (Q1--Q22): static AST structural metrics, revised Composite Structural Complexity Index ($\text{CSCI}_{v2}$), cross-engine latency at $SF=1$ GB (DuckDB Native, DuckDB Parquet, and Apache DataFusion), and scaling at $SF=10$ GB.}",
        r"\label{tab:official_tpch_validation}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Query} & \textbf{$D$} & \textbf{$N$} & \textbf{$J$} & \textbf{$S$} & \textbf{$\text{CSCI}_{v2}$} & \textbf{Duck Nat (ms)} & \textbf{Duck PQ (ms)} & \textbf{DataFusion (ms)} & \textbf{SF10 Nat (ms)} & \textbf{Scale Ratio} \\",
        r"\midrule",
    ]
    for _, r in df_tpch.iterrows():
        q_lbl = str(r["Query_ID"]).replace("_", r"\_")
        lines_tpch.append(
            f"{q_lbl} & {int(r['AST_Depth'])} & {int(r['AST_Nodes'])} & {int(r['Joins'])} & "
            f"{int(r['Subqueries_CTEs'])} & {r['CSCI_v2']:.2f} & "
            f"{r['SF1_DuckDB_Native_ms']:.2f} & {r['SF1_DuckDB_Parquet_ms']:.2f} & "
            f"{r['SF1_DataFusion_Rust_ms']:.2f} & {r['SF10_DuckDB_Native_ms']:.2f} & "
            f"{r['Scaling_Ratio_SF10_SF1']:.2f}$\\times$ \\\\"
        )
    lines_tpch.extend([
        r"\midrule",
        f"\\textbf{{Mean (Q1--Q22)}} & {df_tpch['AST_Depth'].mean():.1f} & {df_tpch['AST_Nodes'].mean():.1f} & "
        f"{df_tpch['Joins'].mean():.1f} & {df_tpch['Subqueries_CTEs'].mean():.1f} & {df_tpch['CSCI_v2'].mean():.2f} & "
        f"{df_tpch['SF1_DuckDB_Native_ms'].mean():.2f} & {df_tpch['SF1_DuckDB_Parquet_ms'].mean():.2f} & "
        f"{df_tpch['SF1_DataFusion_Rust_ms'].mean():.2f} & {df_tpch['SF10_DuckDB_Native_ms'].mean():.2f} & "
        f"{df_tpch['Scaling_Ratio_SF10_SF1'].mean():.2f}$\\times$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (MANUSCRIPT_DIR / "table_official_tpch_validation.tex").write_text("\n".join(lines_tpch), encoding="utf-8")

    # 2. ตาราง 3-Mode GPU vs CPU Fairness Comparison (Reviewer 3.2)
    df_fair = pd.read_csv(RESULTS_DIR / "gpu_cpu_3mode_fairness_comparison_sf1.csv")
    lines_fair = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Four-stage latency decomposition and three-mode fairness evaluation between vectorized CPU engines and SIMT GPU execution across AST complexity tiers ($SF=1$ GB).}",
        r"\label{tab:gpu_cpu_fairness}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Tier} & \textbf{Plan} & \textbf{I/O} & \textbf{PCIe H2D} & \textbf{M1: Duck} & \textbf{M1: GPU} & \textbf{M1 Speedup} & \textbf{M2: GPU Cold} & \textbf{M2 Speedup} & \textbf{M3: Duck PQ} & \textbf{M3: GPU Disk} & \textbf{M3 Speedup} \\",
        r"\midrule",
    ]
    for _, r in df_fair.iterrows():
        t_lbl = str(r["Complexity_Tier"]).replace("_", r"\_")
        lines_fair.append(
            f"{t_lbl} & {r['AST_Plan_ms']:.2f} & {r['Storage_IO_ms']:.2f} & {r['PCIe_H2D_ms']:.2f} & "
            f"{r['M1_Kernel_DuckDB_ms']:.2f} & {r['M1_Kernel_GPU_Warm_ms']:.2f} & {r['M1_Speedup_GPU_vs_Duck']:.2f}$\\times$ & "
            f"{r['M2_Transfer_GPU_Cold_ms']:.2f} & {r['M2_Speedup_GPU_vs_Duck']:.2f}$\\times$ & "
            f"{r['M3_EndToEnd_Duck_PQ_ms']:.2f} & {r['M3_EndToEnd_GPU_Disk_ms']:.2f} & {r['M3_Speedup_GPU_vs_Duck']:.2f}$\\times$ \\\\"
        )
    lines_fair.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_gpu_cpu_fairness.tex").write_text("\n".join(lines_fair), encoding="utf-8")

    # 3. ตาราง Multi-Scale GPU VRAM & Rust Arrow Streaming (Reviewer 3.3)
    df_gpu_ms = pd.read_csv(RESULTS_DIR / "multiscale_gpu_vram_and_arrow_streaming.csv")
    lines_gms = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Multi-scale execution mode transition, host RAM footprint, GPU VRAM consumption, and naive out-of-memory (OOM) verification for Rust Arrow streaming engines and NVIDIA RTX 2050 GPU (4.0 GB VRAM) across $SF=1$, $10$, and $100$ GB.}",
        r"\label{tab:multiscale_gpu_arrow}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"\textbf{Scale Factor} & \textbf{Engine} & \textbf{Execution Mode} & \textbf{Latency (ms)} & \textbf{Peak RAM (GB)} & \textbf{Peak VRAM (GB)} & \textbf{Naive OOM} & \textbf{$\Phi_{PCIe}$ (\%)} \\",
        r"\midrule",
    ]
    for _, r in df_gpu_ms.iterrows():
        lines_gms.append(
            f"{r['Scale_Factor']} & {str(r['Engine']).replace('_', r'\_')} & {str(r['Execution_Mode']).replace('_', r'\_')} & "
            f"{r['Canonical_Scan_Agg_ms']:.2f} & {r['Peak_Host_RAM_GB']:.3f} & {r['Peak_GPU_VRAM_GB']:.3f} & "
            f"{int(r['Naive_FullLoad_OOM'])} & {r['PCIe_Transfer_Share_Pct']:.1f}\\% \\\\"
        )
    lines_gms.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_multiscale_gpu_arrow.tex").write_text("\n".join(lines_gms), encoding="utf-8")

    # 4. ตาราง Two-Way Statistical Interaction (Reviewer 3.4)
    df_inter = pd.read_csv(RESULTS_DIR / "two_way_interaction_complexity_selectivity_sf1.csv")
    lines_int = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Two-way non-parametric factorial analysis (Scheirer--Ray--Hare $H$ and Aligned Rank Transform $F_{\text{ART}}$ interaction tests) evaluating main effects and interaction between structural complexity (L1--L5) and selectivity (V1--V3).}",
        r"\label{tab:two_way_interaction}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Engine} & \textbf{$H_{\text{Complexity}}$} & \textbf{$p_{\text{Complexity}}$} & \textbf{$H_{\text{Selectivity}}$} & \textbf{$p_{\text{Selectivity}}$} & \textbf{$F_{\text{ART}} (L \times V)$} & \textbf{$p_{\text{Interaction}}$} & \textbf{V3/V1 Ratio} \\",
        r"\midrule",
    ]
    for _, r in df_inter.iterrows():
        lines_int.append(
            f"{str(r['Engine']).replace('_', r'\_')} & {r['H_Complexity(L1-L5)']:.2f} & {r['p_Complexity']:.2e} & "
            f"{r['H_Selectivity(V1-V3)']:.2f} & {r['p_Selectivity']:.2e} & "
            f"{r['F_ART_Interaction(LxV)']:.2f} & {r['p_ART_Interaction']:.4f} & {r['V3_vs_V1_Ratio']:.2f}$\\times$ \\\\"
        )
    lines_int.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_two_way_interaction.tex").write_text("\n".join(lines_int), encoding="utf-8")

    print("[OK] สร้างตาราง LaTeX เพิ่มเติมทั้ง 4 ตารางในโฟลเดอร์ manuscript/ สำเร็จเรียบร้อยแล้ว!")


if __name__ == "__main__":
    export_all_tables()