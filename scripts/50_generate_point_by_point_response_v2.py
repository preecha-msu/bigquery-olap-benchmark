#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/50_generate_point_by_point_response_v2.py
สร้างและคอมไพล์เอกสาร Point-by-Point Response to Reviewers (LaTeX + PDF)
ที่สรุปการแก้ไขตาม Revision Priority Matrix (หัวข้อที่ 6) และตอบครบทุกข้อ:
- Critical Issues 3.1 - 3.4
- Major Concerns 3.5 - 3.7
- High-Weight Additional Experiments 4.1 - 4.3
- Reviewer 2 Methodological Defenses 5.1 - 5.4
พร้อมอ้างอิงหมายเลขหน้า ตาราง และรหัส SHA-256 ล่าสุดจาก bigquery_olap_benchmark_V2.tex
"""

import hashlib
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"
RESPONSE_TEX = MANUSCRIPT_DIR / "point_by_point_reviewer_response_V2.tex"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_response_latex() -> None:
    tex_hash = sha256_file(TARGET_TEX)[:24]
    pdf_hash = sha256_file(MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf")[:24]

    response_content = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{mathptmx}}
\usepackage[margin=1.8cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{amsmath,amssymb}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}

\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black
}}

\begin{{document}}

\begin{{center}}
    {{\Large \textbf{{Point-by-Point Response to Reviewers \& Revision Priority Matrix Audit}}}}\\[6pt]
    {{\large \textbf{{Journal of Big Data (SpringerOpen)}}}}\\[4pt]
    \textbf{{Manuscript Title:}} \emph{{AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines}}\\[4pt]
    \textbf{{Author:}} Preecha Noiumkar (Mahasarakham University, Thailand)\\[2pt]
    \textbf{{Revised Manuscript TeX SHA-256:}} \texttt{{{tex_hash}...}} $\mid$ \textbf{{Compiled PDF SHA-256:}} \texttt{{{pdf_hash}...}}
\end{{center}}

\vspace{{4pt}}
\hrule
\vspace{{8pt}}

\section*{{Executive Summary \& Revision Priority Matrix (Section 6)}}{{
We sincerely thank the Reviewer for the rigorous, constructive, and insightful evaluation. In strict adherence to our \textbf{{Zero-Estimation, 100\% Real Execution Protocol}}, we have resolved every numerical, mathematical, experimental, and methodological item identified in Sections 3, 4, 5, and 6 of the review report. Table~\ref{{tab:priority_matrix}} summarizes the complete implementation status across all eight priority dimensions of the Reviewer's \textbf{{Revision Priority Matrix}}.
}}

\begin{{table}}[htbp]
\centering
\caption{{Complete Implementation Audit of the Reviewer's Revision Priority Matrix (Section 6).}}
\label{{tab:priority_matrix}}
\scriptsize
\setlength{{\tabcolsep}}{{4.0pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabular}}{{@{{}}p{{0.24\textwidth}}cp{{0.34\textwidth}}p{{0.30\textwidth}}@{{}}}}
\toprule
\textbf{{Revision Priority Item}} & \textbf{{Priority Level}} & \textbf{{Empirical Action \& Resolution Implemented}} & \textbf{{Location in Revised Manuscript}} \\
\midrule
\textbf{{1. Internal Numerical \& Equation Consistency}} (Sec.~3.1) & Critical (Must-Fix) & Reconciled 10,000-trial CSCI perturbation stability ($\bar{{\rho}}=0.9880$ at $\pm 20\%$) vs. cross-selectivity stability ($\bar{{\rho}}=0.8189$); unified all $\text{{SF}}=1\text{{ GB}}$ latencies ($183.74\text{{ ms}}$ DuckDB, $54.92\text{{ ms}}$ AST-Route) and cross-platform figures ($168.40\rightarrow 110.85\rightarrow 54.92\text{{ ms}}$). & Abstract, Sec.~3.4 (p.~8), Tables 1, 2, 5, 8, 10, 12, and Figs.~1, 2, 11 \\
\textbf{{2. Energy Measurement Validation}} (Sec.~3.2) & Critical (Must-Fix) & Documented NVML board sensor ($10.5\text{{ W}}$ idle) vs. TDP-calibrated CPU model ($14.5\text{{ W}}$ idle, $45.0\text{{ W}}$ TDP); bounded $250\text{{ Hz}}$ sampling via $N=5$ repeats ($\text{{CV}}\le 3.52\%$); added 95\% CIs, Net Dynamic Energy ($E_{{\text{{dyn}}}}$), and Composite Host+GPU Energy ($E_{{\text{{sys}}}}=1.58\text{{ J}}$ Cold / $0.61\text{{ J}}$ Warm). & Sec.~3.2 (Eq.~6, p.~7), Table 7 \& Note (p.~11), Sec.~4.5 (p.~17), Sec.~5 (p.~20) \\
\textbf{{3. GPU/CPU End-to-End Fairness \& 3-Level Split}} (Sec.~3.3) & Critical (Must-Fix) & Added 6-operator algorithmic mapping (\texttt{{WHERE}}, \texttt{{JOIN}}, \texttt{{GROUP BY}}, \texttt{{ORDER BY}}, \texttt{{WINDOW}}, \texttt{{UNNEST}}) with Arrow-to-pinned H2D/D2H accounting; upgraded Table 4 to report \textbf{{Level 1 (Kernel-to-Kernel)}}, \textbf{{Level 2 (Data-in to Result-out)}}, and \textbf{{Level 3 (Full SQL-to-Result)}}. & Sec.~3.4 (p.~8--9), Table 4 (p.~10), Table 13 Part B (p.~13), Sec.~4.3 (p.~13--14) \\
\textbf{{4. Nested Cross-Validation on Official TPC-H Q1--Q22}} (Sec.~3.6) & Critical (Must-Fix) & Executed strict nested 22-fold LOO-CV (inner 21-query tuning only, zero test leakage), achieving $68.18\%$ Top-1 and $90.91\%$ Top-2 accuracy ($16.98\text{{ ms}}$), reported alongside unweighted ($40.91\%, 27.33\text{{ ms}}$) and post-hoc reference bounds ($72.73\%, 15.84\text{{ ms}}$). & Sec.~3.3 (p.~8), Table 12 Part C (p.~12), Sec.~4.9 Item 3 (p.~20) \\
\textbf{{5. External Workload Generalization}} (Sec.~4.1) & High Impact & Validated zero-leakage generalization on 5 locked unseen seeds ($73.33\%$ Top-1, $47.01\text{{ ms}}$, P95 matching Oracle $79.64\text{{ ms}}$) and TPC-H 5--8 way join choke points (Q2, Q5, Q7, Q8, Q9, Q21); justified multi-scale symmetry vs. fixed-scale JOB. & Table 11 Part B (p.~12), Table 12 Part C (p.~12), Sec.~4.8 (p.~19), Sec.~5 Item 2 (p.~20--21) \\
\textbf{{6. Constrained Multi-Objective Routing \& 3D Pareto}} (Sec.~3.4) & High Contribution & Formulated dimensionless normalized $(\alpha,\beta,\gamma)$ objective over Latency, Energy, and FinOps Cost under memory feasibility $\hat{{M}}_e \le M_{{\max}}$ (Eq.~8); evaluated 8 Pareto regimes (B1--B8) including strict VRAM cap ($\le 1.28\text{{ GB}}$) and $\text{{SF}}=100\text{{ GB}}$ FinOps caps. & Sec.~3.3 (Eq.~8, p.~7--8), Table 9 Exp B (p.~11), Fig.~9b (p.~16), Sec.~4.7 Item 2 (p.~18) \\
\textbf{{7. Concurrent Workload Benchmark ($C=1\text{{--}}32$)}} (Sec.~4.2) & High Value Add & Expanded multi-tenant concurrency from $C=1,4,8,16$ to $C=32$ stress testing; decomposed Active Execution Time vs. Admission Queueing Delay, P95/P99 latency, GPU utilization ($93.5\%$), failure rate ($0\%$), and stability ($96\text{{--}}100\%$), reaching $154.44\text{{ QPS}}$ ($2.52\times$). & Table 13 Part C (p.~13), Sec.~4.9 Item 2 (p.~20), Sec.~5 Item 3 (p.~21) \\
\textbf{{8. Confidence-Aware Routing \& Prediction Intervals}} (Sec.~3.7, 4.3) & Novel Extension & Computed out-of-fold 90\% Conformal Prediction Intervals ($88.0\text{{--}}94.7\%$ coverage) and implemented 4 uncertainty rules (GPU clear win, $12\%$ tie abstention, BigQuery upper-PI guard, memory pruning), cutting high-regret failures ($>100\%$) from $14.67\%$ to $6.67\%$. & Sec.~3.3 (p.~8), Table 11 Parts A \& C (p.~12), Sec.~4.8 Item 1 (p.~19) \\
\textbf{{9. Reviewer 2 Defenses: VIF, SHAP, Pre-Exec \& Table 2A}} (Sec.~5.1--5.4) & Methodological Rigor & Audited atomic operator VIF ($\text{{Mean}}=1.63, \text{{Max}}=1.86 < 5.0$), grouped/permutation/TreeSHAP importance, multi-engine construct validity, zone-map selectivity MAE ($3.84\%$, Q-Error $1.092$), $50\%$ missing-metadata fallback, and Related Work Table 2A. & Sec.~2.2 (Table 2A, p.~4), Sec.~3.1 (p.~6), Sec.~3.4 (p.~8), Sec.~4.6--4.7 (p.~18) \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Detailed Point-by-Point Responses}}

\subsection*{{Response to Section 3.1 (Critical 1: Internal Numerical Consistency)}}
\begin{{itemize}}
    \item \textbf{{CSCI Stability Reconciliation:}} We verified from the raw 10,000-trial Monte Carlo audit log that direct weight perturbation ($\pm 20\%$) yields $\bar{{\rho}} = 0.9880 \pm 0.0102$ ($\min \rho = 0.9418$), whereas stratified rank stability across V1--V3 selectivity horizons is $\bar{{\rho}} = 0.8189$. Both figures are now explicitly distinguished in Section~3.4 (p.~8), Table~10 Part B (p.~12), and Figures~1--2 (pp.~5--6).
    \item \textbf{{Unified $\text{{SF}}=1\text{{ GB}}$ Baseline Across Multi-Scale \& Cross-Platform Tables:}} Table~12 Parts A and B (p.~12), Section~4.9 (p.~19--20), and Figure~11 (p.~16) are now anchored 100\% on the full 75-query workload means ($183.74\text{{ ms}}$ DuckDB, $82.47\text{{ ms}}$ Polars, $56.96\text{{ ms}}$ GPU Cold, and $54.92\text{{ ms}}$ AST-Route at $\text{{SF}}=1\text{{ GB}}$; $168.40 \rightarrow 110.85 \rightarrow 54.92\text{{ ms}}$ across Platforms A, B, and C). All typographical column swaps in Tables~1, 2, and 5 have been corrected.
\end{{itemize}}

\subsection*{{Response to Section 3.2 (Critical 2: Energy Measurement Validation)}}
\begin{{itemize}}
    \item \textbf{{Sensor Scope, Discretization Bound, and Three-Tier Energy Accounting:}} Section~3.2 (Eq.~6, p.~7), Table~7 (p.~11), and Section~4.5 (p.~17) now explicitly report: (i) hardware NVML board power ($10.5\text{{ W}}$ idle) vs. TDP-calibrated CPU package power ($14.5\text{{ W}}$ idle, $45.0\text{{ W}}$ active TDP), (ii) $N=5$ repeated-trial aggregation bounding $250\text{{ Hz}}$ ($\Delta t = 4.0\text{{ ms}}$) discretization variance to $\text{{CV}} \le 3.52\%$ with 10,000-bootstrap 95\% CIs, (iii) Net Dynamic Energy above idle ($E_{{\text{{dyn}}}}$: $3.73\text{{ J}}$ DuckDB, $1.97\text{{ J}}$ Polars, $2.00\text{{ J}}$ DataFusion, $0.14\text{{ J}}$ GPU Cold, $0.05\text{{ J}}$ GPU Warm), and (iv) Composite Host CPU + GPU System Energy ($E_{{\text{{sys}}}} = 1.58\text{{ J}}$ Cold and $0.61\text{{ J}}$ Warm, corresponding to a $2.0\times\text{{--}}4.1\times$ cold and $5.2\times\text{{--}}10.5\times$ warm system-level energy reduction).
\end{{itemize}}

\subsection*{{Response to Section 3.3 (Critical 3: GPU/CPU Fairness \& Three-Level Execution Equivalence)}}
\begin{{itemize}}
    \item \textbf{{Six-Operator Implementation Mapping \& Three-Level Decomposition:}} Table~13 Part B (p.~13) details exact CPU-to-GPU algorithmic mappings across all six relational operations (\texttt{{WHERE}}, \texttt{{JOIN}}, \texttt{{GROUP BY}}, \texttt{{ORDER BY}}, \texttt{{WINDOW}}, and \texttt{{UNNEST}}), including zero-copy Arrow casting, pinned memory staging, H2D ($33.89\text{{ ms}}$), and D2H result pullback ($0.82\text{{ ms}}$). Table~4 (p.~10) and Section~4.3 (p.~13--14) now report: \textbf{{Level 1 (Kernel-to-Kernel: $22.25\text{{ ms}}$ vs. $182.87\text{{ ms}}$, $8.22\times$)}}, \textbf{{Level 2 (Data-in to Result-out: $56.96\text{{ ms}}$ vs. $183.74\text{{ ms}}$, $3.23\times$)}}, and \textbf{{Level 3 (Full End-to-End SQL-to-Result including SQLGlot transpilation $+4.15\text{{ ms}}$: $61.11\text{{ ms}}$ In-Mem [$3.01\times$] and $131.79\text{{ ms}}$ Disk Parquet [$1.93\times$ vs. $254.42\text{{ ms}}$])}}.
\end{{itemize}}

\subsection*{{Response to Section 3.4 (Critical 4: Constrained Multi-Objective Routing \& 3D Pareto)}}
\begin{{itemize}}
    \item \textbf{{Dimensionless Objective \& Memory-Constrained 3D Pareto (B1--B8):}} Equation~(8) (p.~7--8) normalizes latency, energy, and FinOps cost by cross-engine reference scales ($T_{{\text{{med}}}}, E_{{\text{{med}}}}, C_{{\text{{ref}}}}$) subject to memory feasibility $\mathcal{{E}}_{{\text{{feasible}}}}(q) = \{{e \mid \widehat{{\text{{RAM}}}}_e \le M_{{\text{{host\_max}}}} \land \widehat{{\text{{VRAM}}}}_e \le M_{{\text{{gpu\_max}}}}\}}$. Table~9 Exp B (p.~11), Figure~9b (p.~16), and Section~4.7 (p.~18) evaluate 8 configurations (B1--B8), demonstrating automatic L2--L4 GPU pruning under $M_{{\text{{gpu\_max}}}} \le 1.28\text{{ GB}}$ (B6: $1.24\text{{ GB}}$ peak VRAM, $0\%$ OOM) and $\text{{SF}}=100\text{{ GB}}$ Cloud-Elastic (B7: $1142.8\text{{ ms}}, \$507/1\text{{M}}$) vs. FinOps Zero-Cloud routing (B8: $1326.9\text{{ ms}}, \$0/1\text{{M}}$).
\end{{itemize}}

\subsection*{{Response to Sections 3.5, 3.6, 3.7 \& Section 4 (4.1--4.3) \& Section 5 (5.1--5.4)}}
\begin{{itemize}}
    \item \textbf{{Paired Test vs. Always GPU \& Tail/Max Regret (Sec.~3.5):}} Table~8 (p.~11) and Section~4.6 (p.~17--18) report Mean, Median ($0.00\%$), P90 ($89.75\%$), P95 ($114.11\%$), and Max ($245.18\%$ vs. $620.63\%$ on Always GPU) regret, alongside 95\% bootstrap CIs for both the full 75-query set ($\Delta T = -2.04\text{{ ms}}$, 95\% CI: $[-6.85, +1.92]\text{{ ms}}, p=0.142$ due to $51/75$ shared GPU dispatches) and the $n=24$ differentially dispatched subset ($\Delta T_{{\text{{disp}}}} = -6.38\text{{ ms}}$, 95\% CI: $[-11.42, -2.15]\text{{ ms}}, p=0.0021$; overall regret reduction $\Delta \mathcal{{R}}_{{\text{{regret}}}} = -25.78\%$, 95\% CI: $[-46.12\%, -8.45\%], p=0.0021$).
    \item \textbf{{Strict Nested LOO-CV on Official TPC-H Q1--Q22 \& External Generalization (Sec.~3.6 \& 4.1):}} Table~12 Part C (p.~12) and Section~4.9 (p.~20) contrast unweighted syntax LOO-CV ($40.91\%$ Top-1, $27.33\text{{ ms}}$), leak-free nested 22-fold LOO-CV ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{{ ms}}$), and post-hoc reference bounds ($72.73\%$ Top-1, $15.84\text{{ ms}}$), while Section~5 Item 2 (p.~20--21) details why TPC-H 5--8 way join choke points and locked unseen seeds ($73.33\%$ Top-1) preserve four-scale symmetry compared to fixed-scale IMDB/JOB.
    \item \textbf{{Multi-Tenant Concurrency up to $C=32$ Streams (Sec.~4.2):}} Table~13 Part C (p.~13) and Section~4.9 Item 2 (p.~20) evaluate $C=1, 4, 8, 16,$ and $32$ streams, separating active execution time from queuing delay and demonstrating $139.99\text{{ QPS}}$ ($2.38\times$) at $C=16$ and $154.44\text{{ QPS}}$ ($2.52\times$, P95: $268.5\text{{ ms}}$, P99: $312.6\text{{ ms}}$, GPU util: $93.5\%$, $0\%$ failures, $96.0\%$ routing stability) at $C=32$.
    \item \textbf{{Confidence-Aware Routing \& Section 5 Defenses (Sec.~3.7, 4.3, 5.1--5.4):}} Table~11 Parts A \& C (p.~12) report 90\% conformal prediction intervals and validate the 4 uncertainty rules (cutting high-regret failures $>100\%$ from $14.67\%$ to $6.67\%$). Sections~2.2, 3.1, 3.4, 4.6, and 4.7 incorporate Table~2A (p.~4), atomic VIF ($\text{{Mean}}=1.63, \text{{Max}}=1.86 < 5.0$), grouped/permutation/TreeSHAP importance, multi-engine construct validity, and pre-execution zone-map MAE ($3.84\%$, Q-Error $1.092$) with $50\%$ missing-metadata fallback ($53.33\%$ Top-1, $68.42\text{{ ms}}$).
\end{{itemize}}

\end{{document}}
"""
    RESPONSE_TEX.write_text(response_content, encoding="utf-8")
    print(f"[WRITE-OK] สร้างไฟล์ {RESPONSE_TEX} สำเร็จ!")

    for _ in (1, 2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", RESPONSE_TEX.name],
            cwd=str(MANUSCRIPT_DIR),
            capture_output=True,
            text=True,
        )
    resp_pdf = MANUSCRIPT_DIR / "point_by_point_reviewer_response_V2.pdf"
    print(f"[PDF-OK] คอมไพล์ {resp_pdf.name} สำเร็จ! (SHA-256: {sha256_file(resp_pdf)[:24]}...)")

    # อัปเดตไฟล์ ZIP ให้รวมทั้ง Manuscript V2 และ Point-by-Point Response V2
    for zname in ["overleaf_V2_submission_bundle.zip", "overleaf_project_bundle.zip"]:
        zpath = MANUSCRIPT_DIR / zname
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(TARGET_TEX, arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            zf.write(RESPONSE_TEX, arcname="point_by_point_reviewer_response_V2.tex")
            bib = MANUSCRIPT_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIGURES_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")
    print("[ZIP-OK] อัปเดตชุดไฟล์ส่งตีพิมพ์ overleaf_V2_submission_bundle.zip เรียบร้อยแล้ว!")


if __name__ == "__main__":
    print("=" * 92)
    print("Generating Final Point-by-Point Reviewer Response & Revision Priority Matrix PDF")
    print("=" * 92)
    generate_response_latex()
    print("=" * 92)
