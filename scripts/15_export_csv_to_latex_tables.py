#!/usr/bin/env python3
r"""
scripts/15_export_csv_to_latex_tables.py
แปลงไฟล์ CSV ผลลัพธ์การทดลองและการวิเคราะห์ทางสถิติให้เป็นโค้ดตาราง LaTeX (.tex)
แบบ Zero-Discrepancy (ไม่มี SyntaxWarning และคอมไพล์ผ่าน 100%)
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def export_reconciled_master_table_to_latex() -> None:
    """สร้างตาราง LaTeX หลักที่แก้ไขตัวเลขข้ามตาราง (Table 1-6) ให้ตรงกับไฟล์ CSV จริง 100%"""
    csv_path = RESULTS_DIR / "reconciled_sf1_master_table.csv"
    if not csv_path.exists():
        print(f"[WARN] ไม่พบไฟล์ {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)

    latex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Reconciled six-engine cross-paradigm execution latency (ms), GPU PCIe overhead fraction ($\Phi_{PCIe}$), and architectural speedup ratios across five AST complexity tiers ($N=75$ queries, TPC-H $SF=1$ GB).}",
        r"\label{tab:reconciled_master_sf1}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Complexity Tier} & \textbf{SQLite3} & \textbf{BQ Unopt} & \textbf{BQ Clust} & \textbf{Duck PQ} & \textbf{Duck Nat} & \textbf{Polars} & \textbf{DataFusion} & \textbf{GPU H2D} & \textbf{GPU Warm} & \textbf{GPU Cold} & \textbf{$\Phi_{PCIe}$} \\",
        r"\midrule",
    ]

    for tier_name, row in df.iterrows():
        label = str(tier_name).replace("_", r"\_")
        if tier_name == "Overall_Mean":
            latex_lines.append(r"\midrule")
            label = r"\textbf{Overall Mean ($N=75$)}"

        line = (
            f"{label} & "
            f"{row['SQLite3_ms']:.2f} & "
            f"{row['BQ_Unopt_ms']:.2f} & "
            f"{row['BQ_Clust_ms']:.2f} & "
            f"{row['Duck_Parquet_ms']:.2f} & "
            f"{row['Duck_Native_ms']:.2f} & "
            f"{row['Polars_ms']:.2f} & "
            f"{row['DataFusion_ms']:.2f} & "
            f"{row['GPU_PCIe_H2D_ms']:.2f} & "
            f"{row['GPU_Warm_Kernel_ms']:.2f} & "
            f"{row['GPU_Cold_Total_ms']:.2f} & "
            f"{row['Phi_PCIe_pct']:.1f}\\% \\\\"
        )
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = MANUSCRIPT_DIR / "table_reconciled_master_sf1.tex"
    out_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"[OK] สร้างตาราง LaTeX สำเร็จ: {out_tex}")


def export_hierarchical_stats_to_latex() -> None:
    csv_path = RESULTS_DIR / "hierarchical_seed_cluster_stats_sf1.csv"
    if not csv_path.exists():
        print(f"[WARN] ไม่พบไฟล์ {csv_path}")
        return

    df = pd.read_csv(csv_path)

    latex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Hierarchical seed-cluster statistical evaluation and non-parametric comparison across OLAP engines ($N=75$ queries, $N_{seeds}=25$ independent seed clusters, TPC-H $SF=1$ GB).}",
        r"\label{tab:hierarchical_stats}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{OLAP Engine} & \textbf{Mean $\pm$ SD (ms)} & \textbf{95\% Seed-CI} & \textbf{Median [IQR] (ms)} & \textbf{Wilcoxon $p$ (Seeds)} & \textbf{Cliff's Delta ($\delta$)} & \textbf{Holm-Adj $p$} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        eng = str(row["Engine"]).replace("_", r"\_")
        mean_sd = str(row["Mean±SD (ms)"]).replace("±", r"$\pm$")
        ci = str(row["95% Seed-Bootstrap CI"])
        med_iqr = str(row["Median [IQR] (ms)"])
        p_val = f"{row['Wilcoxon p (N=25 Seeds)']:.2e}"
        delta = str(row["Cliff's Delta (vs DuckDB)"])
        holm_p = f"{row['Holm-Adj p (N=25 Seeds)']:.2e}"
        latex_lines.append(f"{eng} & {mean_sd} & {ci} & {med_iqr} & {p_val} & {delta} & {holm_p} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = MANUSCRIPT_DIR / "table_hierarchical_stats.tex"
    out_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"[OK] สร้างตาราง LaTeX สำเร็จ: {out_tex}")


def export_router_summary_to_latex() -> None:
    csv_path = RESULTS_DIR / "ast_query_router_evaluation_sf1.csv"
    if not csv_path.exists():
        print(f"[WARN] ไม่พบไฟล์ {csv_path}")
        return

    df = pd.read_csv(csv_path)

    latex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Performance evaluation of the proposed AST-aware query routing framework against single-engine baselines and Oracle optimal bounds (5-fold out-of-seed group cross-validation, $N=75$ queries).}",
        r"\label{tab:router_evaluation}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Routing Strategy} & \textbf{Mean (ms)} & \textbf{Median (ms)} & \textbf{P95 (ms)} & \textbf{Regret (ms)} & \textbf{Exact (\%)} & \textbf{Within-5ms (\%)} & \textbf{Speedup} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        strat = str(row["Routing Strategy"]).replace("&", r"\&").replace("_", r"\_")
        mean_v = f"{row['Mean Latency (ms)']:.2f}"
        med_v = f"{row['Median Latency (ms)']:.2f}"
        p95_v = f"{row['P95 Tail Latency (ms)']:.2f}"
        regret = f"{row['Mean Regret vs Oracle (ms)']:.2f}"
        exact = f"{row['Exact Match (%)']:.1f}\\%"
        within5 = f"{row['Within-5ms Optimal (%)']:.1f}\\%"
        speedup = str(row["Speedup vs Always-DuckDB"])
        latex_lines.append(f"{strat} & {mean_v} & {med_v} & {p95_v} & {regret} & {exact} & {within5} & {speedup} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    out_tex = MANUSCRIPT_DIR / "table_router_evaluation.tex"
    out_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"[OK] สร้างตาราง LaTeX สำเร็จ: {out_tex}")


if __name__ == "__main__":
    print("=" * 70)
    print("กำลังแปลงไฟล์ CSV ผลลัพธ์เป็นตาราง LaTeX (Zero-Warning & Zero-Discrepancy)...")
    print("=" * 70)
    export_reconciled_master_table_to_latex()
    export_hierarchical_stats_to_latex()
    export_router_summary_to_latex()
    print("[DONE] แปลงตาราง LaTeX ทั้ง 3 ตารางเสร็จสิ้นสมบูรณ์")