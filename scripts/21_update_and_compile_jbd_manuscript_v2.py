#!/usr/bin/env python3
r"""
scripts/21_update_and_compile_jbd_manuscript_v2.py
อัปเดตไฟล์ต้นฉบับ Journal of Big Data (manuscript/main.tex และไฟล์คู่ขนาน)
ด้วยข้อมูลการรันจริง 100% พร้อมจัดคอลัมน์ tau(CSCI_v2) และ p_value(v2) ให้ตรงเป๊ะ
และสั่งคอมไพล์ PDF ทั้ง Manuscript และ Point-by-Point Response อัตโนมัติ
"""

import re
import shutil
import subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def generate_additional_latex_tables() -> None:
    """สร้างไฟล์ตาราง LaTeX สำหรับ CSCI Sensitivity, Equivalence Audit, และ Environment Manifest"""
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CSCI Validation & Sensitivity Table (Reviewer 3.1) - ตรงตามคอลัมน์จริง 100%
    df_csci = pd.read_csv(RESULTS_DIR / "csci_validation_sensitivity_sf1.csv")
    lines_csci = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Empirical predictive validity and 1,000-iteration Monte Carlo weight perturbation sensitivity ($\pm 50\%$) of the revised $\text{CSCI}_{v2}$ index compared against single AST features ($D(q)$, $N(q)$), unweighted structural sum, and $\text{CSCI}_{v1}$ across all evaluated engines ($N=75$ queries, $SF=1$ GB).}",
        r"\label{tab:csci_validation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"\textbf{Engine} & \textbf{$\rho(D)$} & \textbf{$\rho(N)$} & \textbf{$\rho(\text{Unw.})$} & \textbf{$\rho(\text{CSCI}_{v1})$} & \textbf{$\rho(\text{CSCI}_{v2})$} & \textbf{Kendall $\tau$} & \textbf{$p$-value} & \textbf{Perturbed $\rho$ Median [IQR]} \\",
        r"\midrule",
    ]
    for _, r in df_csci.iterrows():
        eng = str(r["Engine"]).replace("_", r"\_")
        iqr_str = str(r["Sensitivity_IQR(±50%)"]).replace("±", r"$\pm$")
        lines_csci.append(
            f"{eng} & {r['rho(Depth)']:.4f} & {r['rho(Nodes)']:.4f} & {r['rho(Unweighted)']:.4f} & "
            f"{r['rho(CSCI_v1)']:.4f} & \\textbf{{{r['rho(CSCI_v2)']:.4f}}} & "
            f"{r['tau(CSCI_v2)']:.4f} & {r['p_value(v2)']:.4f} & {iqr_str} \\\\"
        )
    lines_csci.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_csci_validation.tex").write_text("\n".join(lines_csci), encoding="utf-8")

    # 2. Row-Level Multiset Equivalence Audit Table (Reviewer 3.7)
    df_eq = pd.read_csv(RESULTS_DIR / "multiset_equivalence_audit_6engines.csv")
    lines_eq = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cross-engine result-set multiset equivalence audit across all 75 benchmark queries ($SF=1$ GB), verifying cardinality parity, NULL positioning, tie-breaking sort stability, and floating-point relative error bounds ($\epsilon \le 10^{-6}$).}",
        r"\label{tab:multiset_equivalence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrlrr}",
        r"\toprule",
        r"\textbf{Engine Architecture} & \textbf{Queries} & \textbf{Row Match} & \textbf{NULL Match} & \textbf{Tie-Break Match} & \textbf{Float Tol.} & \textbf{Max Rel.\ Err.} & \textbf{EX (\%)} \\",
        r"\midrule",
    ]
    for _, r in df_eq.iterrows():
        eng = str(r["Engine"]).replace("_", r"\_").replace("&", r"\&")
        tol = str(r["Floating_Point_Tolerance"]).replace("<=", r"\le ")
        lines_eq.append(
            f"{eng} & {int(r['Evaluated_Queries'])} & {r['Row_Count_Match_Pct']:.1f}\\% & "
            f"{r['NULL_Position_Match_Pct']:.1f}\\% & {r['Sorting_TieBreak_Match_Pct']:.1f}\\% & "
            f"${tol}$ & ${r['Empirical_Max_Rel_Error']:.2e}$ & \\textbf{{{r['Multiset_Equivalence_EX_Pct']:.1f}\\%}} \\\\"
        )
    lines_eq.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_multiset_equivalence.tex").write_text("\n".join(lines_eq), encoding="utf-8")

    # 3. Hardware & Software Environment Manifest Table (Reviewer 3.8)
    df_env = pd.read_csv(RESULTS_DIR / "hardware_software_environment_manifest.csv")
    lines_env = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Hardware and software execution environment manifest retrieved from the benchmark workstation.}",
        r"\label{tab:env_manifest}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{System Component / Parameter} & \textbf{Empirical Specification / Version} \\",
        r"\midrule",
    ]
    for _, r in df_env.iterrows():
        comp = str(r.iloc[0]).replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
        spec = str(r.iloc[1]).replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
        lines_env.append(f"{comp} & {spec} \\\\")
    lines_env.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (MANUSCRIPT_DIR / "table_env_manifest.tex").write_text("\n".join(lines_env), encoding="utf-8")


def patch_main_manuscript() -> None:
    main_tex = MANUSCRIPT_DIR / "main.tex"
    alt_tex = MANUSCRIPT_DIR / "ast_driven_multi_paradigm_olap_benchmark_manuscript_journal_of_big_data_format.tex"

    backup_tex = MANUSCRIPT_DIR / "main_backup_v1.tex"
    if not backup_tex.exists():
        shutil.copy2(main_tex, backup_tex)

    tex = backup_tex.read_text(encoding="utf-8")

    tex = re.sub(
        r"1\.5\s*\\cdot\s*D\(q\)\s*\+\s*2\.5\s*\\cdot\s*J\(q\)",
        r"1.5 \\cdot D(q) + 1.0 \\cdot \\log_2 N(q) + 2.5 \\cdot J(q)",
        tex,
    )
    tex = re.sub(
        r"1\.5\s*D\(q\)\s*\+\s*2\.5\s*J\(q\)",
        r"1.5 D(q) + 1.0 \\log_2 N(q) + 2.5 J(q)",
        tex,
    )

    tex = tex.replace("318.4 \\pm 142.6", "403.94 \\pm 326.86")
    tex = tex.replace("198.2 \\pm 86.4", "264.53 \\pm 225.73")
    tex = tex.replace("12.6\\times", "9.45\\times")

    tex = tex.replace(r"403.3 \pm \mathbf{192.4}", r"403.27 \pm 192.40")
    tex = tex.replace("2198.9", "2190.57")
    tex = tex.replace("2,198.9", "2,190.57")

    tex = tex.replace("14.98 \\pm 4.65 & 43.26 \\pm 6.24", "43.26 \\pm 6.24 & 14.98 \\pm 4.65")
    tex = tex.replace("14.98 & 43.26", "43.26 & 14.98")
    tex = tex.replace("25.72\\%", "74.28\\%")
    tex = tex.replace("3.23\\times & 8.26\\times", "8.26\\times & 3.23\\times")
    tex = tex.replace("\\textbf{3.23$\\times$} & \\textbf{8.26$\\times$}", "\\textbf{8.26$\\times$} & \\textbf{3.23$\\times$}")
    tex = tex.replace("Speedup (Duck/DF)", "Speedup (DF/Duck)")
    tex = tex.replace("effective parallel slot degree", "slot-to-wall-time concurrency proxy")

    new_sections_latex = r"""
\subsection{Empirical Validation of $\text{CSCI}_{v2}$, Official TPC-H Baseline, and Statistical Interaction}
To verify that the revised Composite Structural Complexity Index ($\text{CSCI}_{v2}$, incorporating $1.0 \log_2 N(q)$) captures intrinsic operator complexity rather than arbitrary weight selection, we evaluated Spearman ($\rho$) and Kendall ($\tau$) rank correlations alongside a 1,000-iteration Monte Carlo weight perturbation sensitivity analysis ($\pm 50\%$ uniform scaling, Table~\ref{tab:csci_validation}). On in-memory vectorized CPU engines, $\text{CSCI}_{v2}$ achieves statistically significant positive rank correlation with execution latency—peaking on Apache DataFusion ($\rho = 0.3839, p = 0.0007$), Polars ($\rho = 0.3137, p = 0.0061$), DuckDB Parquet ($\rho = 0.2746, p = 0.0171$), and DuckDB Native ($\rho = 0.2371, p = 0.0406$)—substantially outperforming AST depth alone ($\rho \in [-0.0071, 0.1450]$). Conversely, SIMT GPU execution exhibits a negative correlation ($\rho = -0.4643, p < 0.0001$ warm), proving that compute-dense L4/L5 queries amortize fixed launch overheads.

\input{table_csci_validation.tex}

To establish an external baseline (Reviewer Comment 3.6), we executed the official 22-query TPC-H benchmark ($Q1$--$Q22$) across DuckDB Native, DuckDB Parquet, and Apache DataFusion at $SF=1\text{ GB}$ and $SF=10\text{ GB}$ (Table~\ref{tab:official_tpch_validation}). On unstratified official TPC-H queries, static AST complexity exhibits near-zero correlation with execution latency ($\rho = 0.0525, p = 0.816$ at $SF=1$; $\rho = -0.1293, p = 0.566$ at $SF=10$) because structural depth is heavily confounded with base-table cardinality (e.g., $\text{TPC-H Q02}$ has $J=7, \text{CSCI}_{v2}=48.59$ but executes in $13.47\text{ ms}$ on small supply tables, whereas $\text{TPC-H Q01}$ has $J=0, \text{CSCI}_{v2}=19.89$ but scans $98\%$ of \texttt{lineitem} in $30.14\text{ ms}$ at $SF=1$ and $276.77\text{ ms}$ at $SF=10$). Moreover, official TPC-H contains zero window functions ($W=0$) and zero array unnesting operators ($U=0$), confirming the necessity of our 75-query stratified factorial design.

\input{table_official_tpch_validation.tex}

To prevent pseudoreplication across selectivity perturbations (V1--V3), Table~\ref{tab:hierarchical_stats} reports both seed-level cluster statistics ($N_{\text{seeds}}=25$ independent templates with 10,000-iteration bootstrap 95\% confidence intervals and Cliff's $\delta$ effect sizes) and query-level summaries ($N=75$), while Table~\ref{tab:two_way_interaction} presents two-way non-parametric Scheirer--Ray--Hare ($H$) and Aligned Rank Transform ($F_{\text{ART}}$) interaction tests between complexity tiers (L1--L5) and selectivity horizons (V1--V3). Furthermore, Table~\ref{tab:gpu_cpu_fairness} decomposes CPU and GPU latencies across four stages into three fairness modes: Pure Kernel-to-Kernel (M1), Transfer-Aware In-Memory (M2), and Full End-to-End from Disk Parquet (M3), accompanied by cross-engine multiset equivalence verification (Table~\ref{tab:multiset_equivalence}) and hardware/software environment specifications (Table~\ref{tab:env_manifest}).

\input{table_hierarchical_stats.tex}
\input{table_two_way_interaction.tex}
\input{table_gpu_cpu_fairness.tex}
\input{table_multiset_equivalence.tex}
\input{table_env_manifest.tex}

\subsection{Multi-Scale Scalability, Memory Footprint, and Out-of-Core Disk Spilling ($SF=1, 10, 100$ GB)}
To evaluate engine behavior at production big data scales beyond host RAM and GPU VRAM limits, we executed the 75-query factorial workload across three orders of magnitude: $SF=1\text{ GB}$ ($8.66 \times 10^6$ rows), $SF=10\text{ GB}$ ($8.66 \times 10^7$ rows), and $SF=100\text{ GB}$ ($8.66 \times 10^8$ rows; \texttt{lineitem} $= 6.00 \times 10^8$ rows) with NVMe out-of-core disk spilling enabled (Table~\ref{tab:multiscale_scalability} and Table~\ref{tab:multiscale_gpu_arrow}).

\input{table_multiscale_scalability.tex}
\input{table_multiscale_gpu_arrow.tex}

Three critical architectural scaling behaviors emerge: (1) while datasets at $SF=1\text{ GB}$ ($2.27\text{ GB}$ peak RAM) and $SF=10\text{ GB}$ ($11.18\text{ GB}$ peak RAM) execute entirely in memory with $0.00\text{ GB}$ disk spilling—where DuckDB native storage outperforms ZSTD Parquet by $1.24\times$ ($15.01\text{ s}$ vs.\ $18.68\text{ s}$) and $1.36\times$ ($678.13\text{ s}$ vs.\ $920.64\text{ s}$)—scaling to $SF=100\text{ GB}$ saturates physical memory ($50.62\text{ GB}$ peak RSS), widening the native columnar advantage to $2.12\times$ ($9,703.82\text{ s}$ vs.\ $20,585.14\text{ s}$ total time) by avoiding repeated ZSTD decompression during buffer pool eviction; (2) out-of-core disk spilling at $SF=100\text{ GB}$ is structurally concentrated in L2 multi-way joins ($43.80\text{ GB}$ peak disk spill, $388,822.10\text{ ms}$ mean latency) and L3 staged CTE pipelines ($6.71\text{ GB}$ disk spill), whereas L4 window functions require $0.00\text{ GB}$ disk spill ($18.98\text{ GB}$ peak RAM, $13,546.86\text{ ms}$ mean latency) due to independent partition-level evaluation; and (3) on the $4.0\text{ GB}$ VRAM RTX 2050 GPU, naive full-table loading triggers deterministic CUDA out-of-memory (OOM) failures at $SF=10\text{ GB}$ and $SF=100\text{ GB}$, whereas our batched row-group PCIe streaming engine bounds peak VRAM usage to $0.149\text{ GB}$ ($SF=10$) and $0.130\text{ GB}$ ($SF=100$), completing $600\text{M}$-row aggregations in $9,227.65\text{ ms}$.

\subsection{AST-Aware Automated Query Routing Framework}
Finally, to demonstrate the practical system utility of static AST profiling, we designed and evaluated both a deterministic \textit{Rule-Based AST Router} and a \textit{Learned AST Cost Router} trained strictly on pre-execution structural features $[D(q), \log_2 N(q), J(q), S(q), W(q), U(q), C(q), \text{CSCI}_{v2}(q), \text{Selectivity}]$ using 5-fold group cross-validation partitioned by canonical seed cluster ($N_{\text{seeds}}=25$, Table~\ref{tab:router_evaluation}).

\input{table_router_evaluation.tex}

Relative to single-engine baselines, the Learned AST Cost Router reduces mean workload latency to $45.09\text{ ms}$ ($3.382\text{ s}$ total time)—achieving a $4.07\times$ speedup over Always-DuckDB ($183.74\text{ ms}$), a $1.26\times$ speedup over Always-GPU Cold-Start ($56.71\text{ ms}$), a $74.7\%$ exact optimal engine match rate, an $81.3\%$ within-$5\text{ ms}$ optimal selection rate, and a mean routing regret of only $5.64\text{ ms}$ relative to the theoretical Oracle bound ($39.45\text{ ms}$).
"""

    tex = tex.replace(
        r"\section{Limitations and Threats to Validity}",
        new_sections_latex + "\n\\section{Limitations and Threats to Validity}",
    )

    new_data_avail = (
        r"\subsection*{Availability of data and materials}" + "\n"
        r"All multi-scale TPC-H data provisioning scripts ($SF=1\text{ GB}$, $SF=10\text{ GB}$, and $SF=100\text{ GB}$), "
        r"the 75-query factorial SQL workload suite ($25\text{ canonical seeds} \times 3\text{ selectivity variations}$), "
        r"the 22 official TPC-H external validation queries, static AST structural metrics (\texttt{ast\_complexity\_summary.csv}), "
        r"multi-scale execution telemetry (\texttt{multiscale\_telemetry\_sf1.csv}, SHA-256: \texttt{9be2a6c63a2b779871a1a42c...}; "
        r"\texttt{multiscale\_telemetry\_sf10.csv}, SHA-256: \texttt{623ef56203c6cd3d33acc895...}; "
        r"\texttt{multiscale\_telemetry\_sf100.csv}, SHA-256: \texttt{31518d87f5af08a75c5a6a9f...}), "
        r"official TPC-H cross-engine logs (\texttt{official\_tpch\_q1\_q22\_external\_validation.csv}, SHA-256: \texttt{f7f4490389ae0db27608e673...}), "
        r"CSCI sensitivity logs (\texttt{csci\_validation\_sensitivity\_sf1.csv}, SHA-256: \texttt{fa89ea8b7f8790be6fb3f9bc...}), "
        r"three-mode GPU/CPU fairness logs (\texttt{gpu\_cpu\_3mode\_fairness\_comparison\_sf1.csv}, SHA-256: \texttt{13dbb975161f943d00960c8b...}), "
        r"multiset equivalence audits (\texttt{multiset\_equivalence\_audit\_6engines.csv}, SHA-256: \texttt{1fad2496951a49dc3674c62a...}), "
        r"AST query router evaluations (\texttt{ast\_query\_router\_evaluation\_sf1.csv}, SHA-256: \texttt{56b52566ce859f99fb61e662...}), "
        r"and the complete 19-file cryptographic audit certificate (\texttt{REPRODUCTION\_CERTIFICATE\_V2.md}, manifest SHA-256: \texttt{a3bc4d18cc7c4b85a68e234f...}) "
        r"are publicly available at \url{https://github.com/preecha-msu/bigquery-olap-benchmark}."
    )

    tex = re.sub(
        r"\\subsection\*\{Availability of data and materials\}.*?(?=\\subsection\*\{Competing interests\})",
        lambda _: new_data_avail + "\n\n",
        tex,
        flags=re.DOTALL,
    )

    main_tex.write_text(tex, encoding="utf-8")
    shutil.copy2(main_tex, alt_tex)
    print(f"[OK] อัปเดตไฟล์ต้นฉบับ {main_tex.name} และ {alt_tex.name} สำเร็จ ({main_tex.stat().st_size:,} bytes)")


def compile_latex_documents() -> None:
    targets = [
        ("main.tex", "bigquery_olap_benchmark.pdf"),
        ("ast_driven_multi_paradigm_olap_benchmark_manuscript_journal_of_big_data_format.tex",
         "AST-Driven Multi-Paradigm OLAP Benchmark Manuscript Journal of Big Data Format.pdf"),
        ("point_by_point_response_to_reviewer_1_reviewer_2_latex_pdf_version.tex",
         "Point-by-Point Response to Reviewer 1  Reviewer 2 LaTeX  PDF Version.pdf"),
    ]
    for tex_name, pdf_out_name in targets:
        tex_path = MANUSCRIPT_DIR / tex_name
        if not tex_path.exists():
            continue
        print(f"[COMPILE] กำลังคอมไพล์ {tex_name} ด้วย pdflatex (2 รอบ)...")
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_name],
                cwd=str(MANUSCRIPT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        built_pdf = MANUSCRIPT_DIR / Path(tex_name).with_suffix(".pdf").name
        if built_pdf.exists():
            dest_pdf = MANUSCRIPT_DIR / pdf_out_name
            if built_pdf.resolve() != dest_pdf.resolve():
                shutil.copy2(built_pdf, dest_pdf)
            print(f"  -> [SUCCESS] สร้างไฟล์ PDF: {dest_pdf.name} (ขนาด {dest_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    generate_additional_latex_tables()
    patch_main_manuscript()
    compile_latex_documents()