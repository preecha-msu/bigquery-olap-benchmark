#!/usr/bin/env python3
r"""
scripts/22_build_overleaf_single_tex_and_zip.py
แก้ปัญหาการนำโค้ดไปรันบน Overleaf แล้วเกิด Error:
1. สร้างตารางใหม่ทั้งหมดด้วย LaTeX Syntax มาตรฐานบริสุทธิ์ (ไม่มีอักขระ UTF-8 แปลกปลอม ไม่พึ่งพา \input{})
2. ฝัง (Inline) ทุกตารางเข้าไปในไฟล์เดียว: manuscript/overleaf_ready_single_file.tex (และ main.tex)
3. ทดสอบคอมไพล์ด้วย pdflatex -halt-on-error เพื่อตรวจจับและยืนยันว่าไม่มี LaTeX Error แม้แต่บรรทัดเดียว
4. สร้างไฟล์ manuscript/overleaf_project_bundle.zip (รวม main.tex, references.bib, และโฟลเดอร์ figures/)
"""

import shutil
import subprocess
import zipfile
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def tex_escape(val) -> str:
    r"""แปลงอักขระพิเศษให้ปลอดภัยต่อ LaTeX / Overleaf 100%"""
    s = str(val).strip()
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    s = s.replace("±", r"$\pm$")
    s = s.replace("<=", r"$\le$")
    s = s.replace(">=", r"$\ge$")
    return s


def fmt_num(val, decimals: int = 2, math_mode: bool = True) -> str:
    r"""จัดรูปแบบตัวเลขหรือสตริงอย่างปลอดภัยโดยไม่เกิด ValueError"""
    try:
        f = float(val)
        res = f"{f:.{decimals}f}"
        return f"${res}$" if math_mode else res
    except (ValueError, TypeError):
        return tex_escape(val)


def fmt_sci(val) -> str:
    r"""จัดรูปแบบตัวเลขวิทยาศาสตร์ (เช่น p-value)"""
    try:
        f = float(val)
        return f"${f:.2e}$"
    except (ValueError, TypeError):
        return tex_escape(val)


def build_clean_inline_tables() -> dict:
    r"""สร้างโค้ด LaTeX ของทุกตารางใหม่แบบ Self-Contained (ไม่ต้องใช้ \input)"""
    tables = {}

    # 1. Table: CSCI Validation & Sensitivity
    df_csci = pd.read_csv(RESULTS_DIR / "csci_validation_sensitivity_sf1.csv")
    rows = []
    for _, r in df_csci.iterrows():
        eng = tex_escape(r.iloc[0])
        iqr_s = tex_escape(r.iloc[-1])
        rows.append(
            f"{eng} & {fmt_num(r.iloc[1], 4)} & {fmt_num(r.iloc[2], 4)} & {fmt_num(r.iloc[3], 4)} & "
            f"{fmt_num(r.iloc[4], 4)} & $\\mathbf{{{fmt_num(r.iloc[5], 4, False)}}}$ & "
            f"{fmt_num(r.iloc[6], 4)} & {fmt_num(r.iloc[7], 4)} & {iqr_s} \\\\"
        )
    tables["table_csci_validation.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\caption{Empirical predictive validity and 1,000-iteration Monte Carlo weight perturbation sensitivity ($\pm 50\%$) of the revised $\text{CSCI}_{v2}$ index compared against single AST features ($D(q)$, $N(q)$), unweighted structural sum, and $\text{CSCI}_{v1}$ across all evaluated engines ($N=75$ queries, $\text{SF}=1$ GB).}",
        r"\label{tab:csci_validation}",
        r"\begin{tabular}{@{}lrrrrrrrl@{}}",
        r"\toprule",
        r"\textbf{Engine} & \textbf{$\rho(D)$} & \textbf{$\rho(N)$} & \textbf{$\rho(\text{Unw.})$} & \textbf{$\rho(\text{CSCI}_{v1})$} & \textbf{$\rho(\text{CSCI}_{v2})$} & \textbf{Kendall $\tau$} & \textbf{$p$-value} & \textbf{Perturbed $\rho$ Median [IQR]} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 2. Table: Official TPC-H Q1-Q22 Validation
    df_tpch = pd.read_csv(RESULTS_DIR / "official_tpch_q1_q22_external_validation.csv")
    rows = []
    for _, r in df_tpch.iterrows():
        q_lbl = tex_escape(r["Query_ID"])
        rows.append(
            f"{q_lbl} & ${int(r['AST_Depth'])}$ & ${int(r['AST_Nodes'])}$ & ${int(r['Joins'])}$ & "
            f"${int(r['Subqueries_CTEs'])}$ & ${r['CSCI_v2']:.2f}$ & "
            f"${r['SF1_DuckDB_Native_ms']:.2f}$ & ${r['SF1_DuckDB_Parquet_ms']:.2f}$ & "
            f"${r['SF1_DataFusion_Rust_ms']:.2f}$ & ${r['SF10_DuckDB_Native_ms']:.2f}$ & "
            f"${r['Scaling_Ratio_SF10_SF1']:.2f}\\times$ \\\\"
        )
    rows.append(r"\midrule")
    rows.append(
        f"\\textbf{{Mean (Q1--Q22)}} & ${df_tpch['AST_Depth'].mean():.1f}$ & ${df_tpch['AST_Nodes'].mean():.1f}$ & "
        f"${df_tpch['Joins'].mean():.1f}$ & ${df_tpch['Subqueries_CTEs'].mean():.1f}$ & ${df_tpch['CSCI_v2'].mean():.2f}$ & "
        f"${df_tpch['SF1_DuckDB_Native_ms'].mean():.2f}$ & ${df_tpch['SF1_DuckDB_Parquet_ms'].mean():.2f}$ & "
        f"${df_tpch['SF1_DataFusion_Rust_ms'].mean():.2f}$ & ${df_tpch['SF10_DuckDB_Native_ms'].mean():.2f}$ & "
        f"${df_tpch['Scaling_Ratio_SF10_SF1'].mean():.2f}\\times$ \\\\"
    )
    tables["table_official_tpch_validation.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\caption{External benchmark validation on the official TPC-H query suite (Q1--Q22): static AST structural metrics, revised Composite Structural Complexity Index ($\text{CSCI}_{v2}$), cross-engine latency at $\text{SF}=1$ GB (DuckDB Native, DuckDB Parquet, and Apache DataFusion), and scaling at $\text{SF}=10$ GB.}",
        r"\label{tab:official_tpch_validation}",
        r"\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Query} & \textbf{$D$} & \textbf{$N$} & \textbf{$J$} & \textbf{$S$} & \textbf{$\text{CSCI}_{v2}$} & \textbf{Duck Nat (ms)} & \textbf{Duck PQ (ms)} & \textbf{DataFusion (ms)} & \textbf{SF10 Nat (ms)} & \textbf{Scale Ratio} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 3. Table: Hierarchical Seed-Cluster vs Query-Level Statistics (สร้างจากคอลัมน์จริงใน CSV)
    df_hier = pd.read_csv(RESULTS_DIR / "hierarchical_seed_cluster_stats_sf1.csv")
    n_cols_h = len(df_hier.columns)
    col_headers_h = " & ".join([f"\\textbf{{{tex_escape(c)}}}" for c in df_hier.columns])
    col_align_h = "@{}l" + "l" * (n_cols_h - 1) + "@{}"
    rows = []
    for _, r in df_hier.iterrows():
        cells = []
        for idx, val in enumerate(r):
            if idx == 0:
                cells.append(tex_escape(val))
            elif isinstance(val, (int, float)) and not pd.isna(val):
                if abs(val) < 0.01 and val != 0:
                    cells.append(f"${val:.2e}$")
                else:
                    cells.append(f"${val:.3f}$")
            else:
                cells.append(tex_escape(val))
        rows.append(" & ".join(cells) + r" \\")
    tables["table_hierarchical_stats.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Hierarchical statistical inference comparing independent seed-cluster analysis ($N_{\text{seeds}}=25$, 10,000-iteration bootstrap 95\% CI) against query-level analysis ($N=75$) with Cliff's $\delta$ non-parametric effect sizes ($\text{SF}=1$ GB).}",
        r"\label{tab:hierarchical_stats}",
        f"\\begin{{tabular}}{{{col_align_h}}}",
        r"\toprule",
        col_headers_h + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 4. Table: Two-Way Non-Parametric Interaction
    df_inter = pd.read_csv(RESULTS_DIR / "two_way_interaction_complexity_selectivity_sf1.csv")
    rows = []
    for _, r in df_inter.iterrows():
        eng = tex_escape(r["Engine"])
        rows.append(
            f"{eng} & ${r['H_Complexity(L1-L5)']:.2f}$ & ${r['p_Complexity']:.2e}$ & "
            f"${r['H_Selectivity(V1-V3)']:.2f}$ & ${r['p_Selectivity']:.2e}$ & "
            f"${r['F_ART_Interaction(LxV)']:.2f}$ & ${r['p_ART_Interaction']:.4f}$ & ${r['V3_vs_V1_Ratio']:.2f}\\times$ \\\\"
        )
    tables["table_two_way_interaction.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Two-way non-parametric factorial analysis (Scheirer--Ray--Hare $H$ and Aligned Rank Transform $F_{\text{ART}}$ interaction tests) evaluating main effects and interaction between structural complexity (L1--L5) and selectivity (V1--V3).}",
        r"\label{tab:two_way_interaction}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Engine} & \textbf{$H_{\text{Complexity}}$} & \textbf{$p_{\text{Complexity}}$} & \textbf{$H_{\text{Selectivity}}$} & \textbf{$p_{\text{Selectivity}}$} & \textbf{$F_{\text{ART}} (L \times V)$} & \textbf{$p_{\text{Interaction}}$} & \textbf{V3/V1 Ratio} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 5. Table: 3-Mode GPU vs CPU Fairness
    df_fair = pd.read_csv(RESULTS_DIR / "gpu_cpu_3mode_fairness_comparison_sf1.csv")
    rows = []
    for _, r in df_fair.iterrows():
        t_lbl = tex_escape(r["Complexity_Tier"])
        rows.append(
            f"{t_lbl} & ${r['AST_Plan_ms']:.2f}$ & ${r['Storage_IO_ms']:.2f}$ & ${r['PCIe_H2D_ms']:.2f}$ & "
            f"${r['M1_Kernel_DuckDB_ms']:.2f}$ & ${r['M1_Kernel_GPU_Warm_ms']:.2f}$ & ${r['M1_Speedup_GPU_vs_Duck']:.2f}\\times$ & "
            f"${r['M2_Transfer_GPU_Cold_ms']:.2f}$ & ${r['M2_Speedup_GPU_vs_Duck']:.2f}\\times$ & "
            f"${r['M3_EndToEnd_Duck_PQ_ms']:.2f}$ & ${r['M3_EndToEnd_GPU_Disk_ms']:.2f}$ & ${r['M3_Speedup_GPU_vs_Duck']:.2f}\\times$ \\\\"
        )
    tables["table_gpu_cpu_fairness.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Four-stage latency decomposition and three-mode fairness evaluation between vectorized CPU engines and SIMT GPU execution across AST complexity tiers ($\text{SF}=1$ GB).}",
        r"\label{tab:gpu_cpu_fairness}",
        r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Tier} & \textbf{Plan} & \textbf{I/O} & \textbf{PCIe} & \textbf{M1:Duck} & \textbf{M1:GPU} & \textbf{M1 Spd.} & \textbf{M2:GPU} & \textbf{M2 Spd.} & \textbf{M3:Duck} & \textbf{M3:GPU} & \textbf{M3 Spd.} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 6. Table: Row-Level Multiset Equivalence Audit
    df_eq = pd.read_csv(RESULTS_DIR / "multiset_equivalence_audit_6engines.csv")
    rows = []
    for _, r in df_eq.iterrows():
        eng = tex_escape(r["Engine"])
        rows.append(
            f"{eng} & ${int(r['Evaluated_Queries'])}$ & ${r['Row_Count_Match_Pct']:.1f}\\%$ & "
            f"${r['NULL_Position_Match_Pct']:.1f}\\%$ & ${r['Sorting_TieBreak_Match_Pct']:.1f}\\%$ & "
            f"$\\le 10^{{-6}}$ & ${r['Empirical_Max_Rel_Error']:.2e}$ & $\\mathbf{{{r['Multiset_Equivalence_EX_Pct']:.1f}\\%}}$ \\\\"
        )
    tables["table_multiset_equivalence.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Cross-engine result-set multiset equivalence audit across all 75 benchmark queries ($\text{SF}=1$ GB), verifying cardinality parity, NULL positioning, tie-breaking sort stability, and floating-point relative error bounds ($\epsilon \le 10^{-6}$).}",
        r"\label{tab:multiset_equivalence}",
        r"\begin{tabular}{@{}lrrrrlrr@{}}",
        r"\toprule",
        r"\textbf{Engine Architecture} & \textbf{Queries} & \textbf{Row Match} & \textbf{NULL Match} & \textbf{Tie-Break Match} & \textbf{Float Tol.} & \textbf{Max Rel.\ Err.} & \textbf{EX (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 7. Table: Environment Manifest
    df_env = pd.read_csv(RESULTS_DIR / "hardware_software_environment_manifest.csv")
    rows = []
    for _, r in df_env.iterrows():
        rows.append(f"{tex_escape(r.iloc[0])} & {tex_escape(r.iloc[1])} \\\\")
    tables["table_env_manifest.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{6.0pt}",
        r"\caption{Hardware and software execution environment manifest retrieved from the benchmark workstation.}",
        r"\label{tab:env_manifest}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"\textbf{System Component / Parameter} & \textbf{Empirical Specification / Version} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 8. Table: Multi-Scale Scalability (SF1, SF10, SF100)
    df_ms = pd.read_csv(RESULTS_DIR / "multiscale_scalability_summary_table.csv")
    rows = []
    for _, r in df_ms.iterrows():
        rows.append(
            f"{tex_escape(r['Scale_Factor'])} & {tex_escape(r['Engine_Mode'])} & "
            f"${r['Total_75Q_Time_s']:.2f}$ & ${r['Mean_Latency_ms']:.2f}$ & "
            f"${r['Median_Latency_ms']:.2f}$ & ${r['P95_Latency_ms']:.2f}$ & "
            f"${r['Peak_RAM_GB']:.2f}$ & ${r['Peak_Disk_Spill_GB']:.2f}$ & "
            f"${int(r['Spilled_Queries'])}$ & ${int(r['OOM_Limit_Events'])}$ \\\\"
        )
    tables["table_multiscale_scalability.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\caption{Multi-scale empirical scalability, memory footprint, and out-of-core disk spilling evaluation across TPC-H $\text{SF}=1$ GB, $\text{SF}=10$ GB, and $\text{SF}=100$ GB ($N=75$ queries per scale factor).}",
        r"\label{tab:multiscale_scalability}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Scale Factor} & \textbf{Storage Mode} & \textbf{Total (s)} & \textbf{Mean (ms)} & \textbf{Median (ms)} & \textbf{P95 (ms)} & \textbf{Peak RAM (GB)} & \textbf{Peak Spill (GB)} & \textbf{Spill Qs} & \textbf{OOM/Limit} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 9. Table: Multi-Scale GPU VRAM & Rust Arrow Streaming
    df_gms = pd.read_csv(RESULTS_DIR / "multiscale_gpu_vram_and_arrow_streaming.csv")
    rows = []
    for _, r in df_gms.iterrows():
        rows.append(
            f"{tex_escape(r['Scale_Factor'])} & {tex_escape(r['Engine'])} & {tex_escape(r['Execution_Mode'])} & "
            f"${r['Canonical_Scan_Agg_ms']:.2f}$ & ${r['Peak_Host_RAM_GB']:.3f}$ & ${r['Peak_GPU_VRAM_GB']:.3f}$ & "
            f"${int(r['Naive_FullLoad_OOM'])}$ & ${r['PCIe_Transfer_Share_Pct']:.1f}\\%$ \\\\"
        )
    tables["table_multiscale_gpu_arrow.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\caption{Multi-scale execution mode transition, host RAM footprint, GPU VRAM consumption, and naive out-of-memory (OOM) verification for Rust Arrow streaming engines and NVIDIA RTX 2050 GPU (4.0 GB VRAM) across $\text{SF}=1$, $10$, and $100$ GB.}",
        r"\label{tab:multiscale_gpu_arrow}",
        r"\begin{tabular}{@{}lllrrrrr@{}}",
        r"\toprule",
        r"\textbf{Scale Factor} & \textbf{Engine} & \textbf{Execution Mode} & \textbf{Latency (ms)} & \textbf{Peak RAM (GB)} & \textbf{Peak VRAM (GB)} & \textbf{Naive OOM} & \textbf{$\Phi_{\text{PCIe}}$ (\%)} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    # 10. Table: AST-Aware Query Router Evaluation
    df_rt = pd.read_csv(RESULTS_DIR / "ast_query_router_evaluation_sf1.csv")
    n_cols_rt = len(df_rt.columns)
    col_headers_rt = " & ".join([f"\\textbf{{{tex_escape(c)}}}" for c in df_rt.columns])
    col_align_rt = "@{}l" + "r" * (n_cols_rt - 1) + "@{}"
    rows = []
    for _, r in df_rt.iterrows():
        cells = []
        for idx, val in enumerate(r):
            if idx == 0:
                cells.append(tex_escape(val))
            elif isinstance(val, (int, float)) and not pd.isna(val):
                cells.append(f"${val:.2f}$")
            else:
                cells.append(tex_escape(val))
        rows.append(" & ".join(cells) + r" \\")
    tables["table_router_evaluation.tex"] = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Out-of-sample 5-fold group cross-validation evaluation of the proposed AST-Aware Query Routing Framework ($N=75$ queries partitioned across $N_{\text{seeds}}=25$ independent seed templates, $\text{SF}=1$ GB).}",
        r"\label{tab:router_evaluation}",
        f"\\begin{{tabular}}{{{col_align_rt}}}",
        r"\toprule",
        col_headers_rt + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    for fname, content in tables.items():
        (MANUSCRIPT_DIR / fname).write_text(content, encoding="utf-8")

    return tables


def build_self_contained_overleaf_tex(inline_tables: dict) -> Path:
    r"""ฝังโค้ดตารางทั้งหมดลงในไฟล์ .tex หลักโดยตรง เพื่อไม่ให้ Overleaf ฟ้องหาไฟล์ \input{}"""
    # รัน patch_main_manuscript จากสคริปต์ 21 ก่อนเพื่อให้ได้โครงสร้างที่มี \input{} ครบ
    import importlib.util
    s21_path = PROJECT_ROOT / "scripts" / "21_update_and_compile_jbd_manuscript_v2.py"
    if s21_path.exists():
        spec = importlib.util.spec_from_file_location("s21", str(s21_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.patch_main_manuscript()

    main_tex = MANUSCRIPT_DIR / "main.tex"
    alt_tex = MANUSCRIPT_DIR / "ast_driven_multi_paradigm_olap_benchmark_manuscript_journal_of_big_data_format.tex"
    overleaf_tex = MANUSCRIPT_DIR / "overleaf_ready_single_file.tex"

    tex = main_tex.read_text(encoding="utf-8")

    for tbl_name, tbl_code in inline_tables.items():
        tex = tex.replace(f"\\input{{{tbl_name}}}", tbl_code)

    tex = tex.replace("±", r"\pm ")

    overleaf_tex.write_text(tex, encoding="utf-8")
    main_tex.write_text(tex, encoding="utf-8")
    shutil.copy2(main_tex, alt_tex)

    print(f"[OK] สร้างไฟล์เดี่ยว Self-Contained สำหรับ Overleaf สำเร็จ: {overleaf_tex.name} ({overleaf_tex.stat().st_size:,} bytes)")
    return overleaf_tex


def verify_zero_latex_errors() -> None:
    r"""รัน pdflatex -halt-on-error เพื่อตรวจว่ามี Error บรรทัดไหนหรือไม่"""
    print("[VERIFY] กำลังทดสอบคอมไพล์ด้วยโหมดเข้มงวด (pdflatex -halt-on-error)...")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "overleaf_ready_single_file.tex"],
        cwd=str(MANUSCRIPT_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "overleaf_ready_single_file.tex"],
            cwd=str(MANUSCRIPT_DIR),
            capture_output=True,
            text=True,
        )
        pdf_built = MANUSCRIPT_DIR / "overleaf_ready_single_file.pdf"
        shutil.copy2(pdf_built, MANUSCRIPT_DIR / "bigquery_olap_benchmark.pdf")
        shutil.copy2(
            pdf_built,
            MANUSCRIPT_DIR / "AST-Driven Multi-Paradigm OLAP Benchmark Manuscript Journal of Big Data Format.pdf",
        )
        print("  -> [ZERO ERRORS VERIFIED] คอมไพล์ผ่าน 100% โดยไม่มี LaTeX Error แม้แต่จุดเดียว!")
    else:
        print("  -> [FOUND LATEX ERROR] รายละเอียดจาก Log:")
        lines = proc.stdout.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("!"):
                print("\n".join(lines[max(0, i - 3):min(len(lines), i + 12)]))
                break


def create_overleaf_zip_bundle() -> None:
    r"""สร้างไฟล์ ZIP ที่รวม main.tex (แบบรวมตารางแล้ว), references.bib และโฟลเดอร์ figures/ สำหรับอัปโหลดขึ้น Overleaf"""
    zip_path = MANUSCRIPT_DIR / "overleaf_project_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MANUSCRIPT_DIR / "overleaf_ready_single_file.tex", arcname="main.tex")
        bib_path = MANUSCRIPT_DIR / "references.bib"
        if bib_path.exists():
            zf.write(bib_path, arcname="references.bib")
        fig_dir = MANUSCRIPT_DIR / "figures"
        if fig_dir.exists():
            for f in fig_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")
    print(f"[OK] สร้างไฟล์ ZIP พร้อมอัปโหลดขึ้น Overleaf: {zip_path.name} ({zip_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    inline_tbls = build_clean_inline_tables()
    build_self_contained_overleaf_tex(inline_tbls)
    verify_zero_latex_errors()
    create_overleaf_zip_bundle()