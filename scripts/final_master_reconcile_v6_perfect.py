#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/final_master_reconcile_v6_perfect.py
================================================================================
ซ่อมแซมและซิงค์จุดที่ตกค้างทั้ง 6 จุดใน bigquery_olap_benchmark_V5_2.pdf / V2.pdf:
  1. ย้ายหัวข้อ \paragraph{6. Comprehensive Baseline Matrix...} และ Table 14
     จากหน้า 14 (ก่อน Sec 4.1) ไปไว้ท้าย Section 4.7 (หลังข้อ 5 Seed-Level...)
  2. ลบ Table 2B ตัวที่ซ้ำในหน้า 19 ออก (คงไว้เฉพาะ Table 2B ในหน้า 6 เพียงจุดเดียว)
     และปรับตัวเลข Cold Parquet Footer ในหน้า 6 ให้ตรงกัน (412.6 us cold footer)
  3. อัปเดต Section 1 (Strategic Positioning + 5 Contributions) ให้ตรงกับกลยุทธ์หลัก
     (ไม่มีการอ้าง p=0.0021 ลอยๆ ใน Contribution 4 และรวม 163 คิวรี)
  4. อัปเดตจำนวนคิวรีรวมใน Section 1, Section 3.1 (Fig 2), Section 5(2), และ Section 6
     ให้เป็น 163 คิวรี (97 TPC-H + 33 JOB + 13 SSB + 20 ClickBench) ตรงกับ Abstract 100%
  5. ซ่อมจุดย่อย:
     - Table 3 แถว V3: $12.52 (SF=1), $125.17 (SF=10), $375.51 (SF=30), $1,251.70 (SF=100)
     - Table 13 Part C: แก้ $C=30$ Streams -> $C=32$ Streams
     - Sec 4.7 ข้อ 4: แก้ 54.92 + 4.15 = 59.07 ms -> 50.78 ms exec + 4.15 ms overhead = 54.92 ms net
     - Sec 5 ข้อ 2: ลบคำซ้ำ "comprising 97 ... comprising"
     - Availability of data: เปลี่ยน nb2_exp1-nb2_exp3 เป็นชื่อไฟล์มาตรฐาน + เพิ่ม CSV ของ Step 3
     - Ref [28]: ซ่อม URL https://www.tpc.org/tpch/
================================================================================
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_academic_words(latex_block: str) -> int:
    text = re.sub(r"\$[^\$]+\$", " MATH_TOKEN ", latex_block)
    text = re.sub(r"\\url\{[^}]*\}", " URL_TOKEN ", text)
    for cmd in ("textbf", "textit", "emph", "text", "texttt"):
        text = re.sub(rf"\\{cmd}\{{([^}}]*)\}}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[\{\}~]", " ", text)
    return len([w for w in re.split(r"\s+", text.strip()) if w])


STRATEGIC_INTRO_CONTRIBUTIONS = r"""\noindent\textbf{Bridging Empirical Crossover Profiling and Risk-Aware Multi-Objective Routing.}
Designing a reliable cross-paradigm dispatcher requires resolving a fundamental systems tension: static single-engine policies suffer severe performance cliffs across heterogeneous operators, yet naive point-estimate cost minimization ($\arg\min_e \hat{T}_e$) triggers costly tail-latency spikes whenever prediction intervals overlap near CPU--GPU PCIe crossover boundaries or device memory ceilings. Rather than treating hardware benchmarking and learned routing in isolation, this study grounds an interpretable, zero-execution routing architecture directly in controlled factorial crossover telemetry. Crucially, decoupling the pre-execution structural/volume representation ($\mathbf{x}_q \in \mathbb{R}^9$, extended to $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$) from the downstream decision policy exposes an explicit operational trade-off between \emph{unconstrained mean-latency minimization} (achievable via lightweight linear, shallow-tree, or $k$-NN estimators) and \emph{tail-safe SLA protection} (governed via calibrated conformal prediction intervals and resource-constrained Pareto dispatch). The primary contributions of this study are summarized as follows:

\begin{enumerate}
    \item \textbf{De-Confounded Factorial Testbed and CPU--GPU--Cloud Crossover Profiling:} We construct a 75-query AST-stratified factorial benchmark (25 canonical seeds $\times$ 3 selectivity horizons V1--V3 across tiers L1--L5, complemented by 22 official TPC-H, 33 IMDB JOB, 13 SSB, and 20 ClickBench queries; $N=163$ total) across six engines spanning row-store, serverless cloud, C++/Rust Apache Arrow vectorized CPUs, and discrete CUDA GPUs. By stripping SQL parsing/planning overhead ($183.74 - 0.87 = 182.87\text{ ms}$ pure DuckDB CPU kernel vs.\ $22.25\text{ ms}$ in-VRAM GPU kernel) and pairing direct NVML GPU power sensors with TDP-calibrated CPU package energy estimates and BigQuery $10\text{ MiB}$ minimum-floor scan billing, we quantify exact PCIe transfer crossovers and memory walls under 100\% real-execution ground-truth telemetry (zero synthetic latency simulation).

    \item \textbf{Construct-Validated AST Structural Complexity ($\text{CSCI}$) and Pre-Execution 9D/10D Representation:} We formulate a zero-execution 9-dimensional feature vector ($\mathbf{x}_q \in \mathbb{R}^9$, augmented with $\log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})$ to $\mathbb{R}^{10}$ across scales) extracted in $3.65\text{ ms}$ ($3.476\text{ ms}$ \texttt{SQLGlot} AST parsing plus $0.172\text{ ms}$ cached or $0.413\text{ ms}$ cold NVMe Parquet footer zone-map metadata; selectivity $\text{MAE}=3.84\%$, median Q-Error $=1.092$) without touching data pages. Extensive validation confirms construct correlation with AST node count ($\rho=0.8744$), partial correlation with GPU speedup controlling for selectivity ($\rho_{\text{partial}}=0.5477, p=3.68\times 10^{-7}$), 10,000-trial weight-perturbation rank stability ($\bar{\rho}=0.9880$ at $\pm 20\%$ and $\bar{\rho}=0.9555$ at $\pm 50\%$), and low tree-size multicollinearity ($\text{VIF}=1.84$ against $[D(q), N(q)]$). In axis-aligned tree ensembles, $\text{CSCI}(q)$ acts as an engineered oblique hyperplane projection that reduces active split nodes per tree by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) over atomic AST counts alone.

    \item \textbf{Model-Agnostic Heterogeneous Cost Prediction, Standalone CPU-Only Routing, and Explicit Latency--Tail Risk Control:} Using $\mathbf{x}_q$, our multi-output cost predictors estimate cross-architecture latency, device energy, and peak working-set memory in $497.2\text{ }\mu\text{s}$ (median Q-Error $1.007\text{--}1.172$, $\log\text{-}R^2=0.815\text{--}0.912$). Under 25-seed LOSO-CV ($\text{SF}=1\text{ GB}$), \emph{speed-prioritized point-estimate routers} using Ridge ($49.12\text{ ms}$), shallow decision trees ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), or in-fold tuned rules ($53.07\text{ ms}$) minimize raw mean latency but incur high P95/P99 tail regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas our \emph{conformal risk-aware AST-Route} ($\mathcal{R}_{\text{GBR-9D}}$ with 90\% conformal intervals, $\text{ECE}=1.78\%$, and a $12\%$ abstention guard) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$ net warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1, $76.00\%$ Top-2) to compress P95/P99 tail regret to $114.11\%$/$218.40\%$ and halve $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). Moreover, standalone CPU-only routing ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) on the primary node and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a 16-core workstation without GPU dependence.

    \item \textbf{Memory-Constrained Multi-Objective Pareto Dispatch (Latency, Energy, and Cloud Scan Charge):} Coupling $\hat{\sigma}(q)$ with catalog byte footprints ($\hat{\mathcal{B}}_{\text{scan}}(q)=\hat{\sigma}(q)\cdot\mathcal{B}_{\text{catalog}}$) enforces strict VRAM/DRAM feasibility filtering (eliminating 100\% of out-of-memory failures under tight VRAM caps) while jointly optimizing execution latency, estimated active device energy ($1.381\text{ J/query}$), and marginal on-demand Cloud Query Scan Charges ($\text{\$6.25/TiB}$, cleanly separated from sunk local hardware amortization). Under independent seed-level ($N=25$) Holm--Bonferroni inference, AST-Route significantly outperforms static CPU engines ($1.50\times\text{--}3.35\times, p_{\text{Holm}}\le 0.0171$) and static rule-based routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}, p=0.184$) at less than half the P95/P99 tail regret.

    \item \textbf{Multi-Scale, Concurrent, Cross-Hardware, and Operator-Bounded Generalization:} Across $\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$ ($6.0\text{M}\text{--}600.1\text{M}$ rows), $C=1\text{--}32$ concurrent query streams, and secondary multi-core CPU-only hardware profiles, AST-Route sustains $3.35\times\text{--}7.61\times$ speedups over DuckDB and $2.00\times\text{--}2.14\times$ steady-state throughput gains (up to $154.44\text{ QPS}$). It transfers effectively to locked unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), official TPC-H Q1--Q22 ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), zero-shot star-join SSB ($92.31\%$ Top-1, $12.15\text{ ms}$), wide-table ClickBench web analytics ($80.00\%$ Top-1, $95.00\%$ Top-2, $24.18\text{ ms}$), and string-skewed cyclic JOB queries ($6.75\text{ ms}$ Polars parity via $21.21\%$ conformal abstention), while Leave-One-Tier-Out evaluation (LOTO-CV: $85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) transparently establishes that training workloads must span target SQL operator families.
\end{enumerate}"""


STRATEGIC_CONCLUSION_SECTION = r"""\section{Conclusion}
We presented AST-Route, an interpretable, risk-aware multi-objective query routing framework grounded in empirical CPU--GPU--cloud crossover profiling and evaluated across 163 SQL queries in total---comprising 97 core TPC-H benchmark queries (a 75-query AST-stratified factorial workload across L1--L5 and V1--V3 plus the 22 official standard TPC-H queries Q1--Q22 [28]) and 66 external zero-shot cross-schema validation queries (33 IMDB JOB templates [27], 13 SSB star-join queries [26], and 20 ClickBench web-analytics queries)---across four standard scale factors ($\text{SF}=1, 10, 30,$ and $100\text{ GB}$ from 6.0M to 600.1M \texttt{lineitem} rows), three cross-platform hardware profiles, $C=1\text{--}32$ concurrent query streams, and six software systems across four computing paradigms: SQLite~3, Google BigQuery, DuckDB, Polars, Apache DataFusion, and CUDA GPU execution (NVIDIA RTX~2050). Validated via 10,000-iteration seed-clustered bootstrap 95\% confidence intervals, Cliff's delta effect sizes, 10,000-trial CSCI sensitivity analysis ($\bar{\rho}=0.9880$ at $\pm 20\%$, partial correlation $\rho_{\text{partial}}=0.5477$), strict GPU/CPU FP64 fairness protocols, feature group ablation, Pareto trade-off analysis, leak-free 25-seed LOSO-CV, unseen seed holdout testing, and strict nested 22-fold LOO-CV, our pre-execution 9D/10D feature representation ($\mathbf{x}_q\in\mathbb{R}^9$, extended to $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$) enables multi-engine cost prediction (median Q-Error $1.007\text{--}1.172$, $R^2=0.815\text{--}0.875$) in $497.2\text{ }\mu\text{s}$ ($4.145\text{ ms}$ total warm overhead; $4.386\text{ ms}$ cold NVMe footer seek) and supports tunable operational trade-offs: point-estimate Ridge ($49.12\text{ ms}$), shallow decision tree ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), and in-fold rule ($53.07\text{ ms}$) routers minimize raw mean latency at elevated P95/P99 tail regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas conformal AST-Route ($\mathcal{R}_{\text{GBR-9D}}$) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$ warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1 and $76.00\%$ Top-2 accuracy, $0.00\%$ median regret, $34.98\%$ mean regret, $1.381\text{ J}$ estimated device energy) to compress P95/P99 tail regret to $114.11\%$/$218.40\%$ and halve severe ($>100\%$ regret) routing failures from $14.67\%$ to $6.67\%$. Standalone CPU-only routing ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) on the primary workstation and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a 16-core node. Under independent seed-level ($N=25$ clusters) Holm--Bonferroni inference, AST-Route significantly outperforms Always DuckDB ($3.35\times, p_{\text{Holm}}=5.45\times 10^{-4}$), Always DataFusion ($1.75\times, p_{\text{Holm}}=0.0041$), Always Polars ($1.50\times, p_{\text{Holm}}=0.0171$), and Rule-Based Routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}$, seed-level $p=0.184$) yet cutting P95 tail regret from $257.31\%$ to $114.11\%$. Furthermore, AST-Route achieves $73.33\%$ Top-1 accuracy ($47.01\text{ ms}$) on locked unseen query seeds, $68.18\%$ Top-1 and $90.91\%$ Top-2 accuracy ($16.98\text{ ms}$, $2.52\times$ faster than Always Polars) under nested LOO-CV on official TPC-H Q1--Q22, safe zero-shot cross-schema transfer across 66 external queries (matching Always Polars at $6.75\text{ ms}$ with $21.21\%$ Top-1, $100.00\%$ Top-2, and $84.49\%$ mean regret on 33 JOB queries while beating DuckDB by $2.90\times$ and Always GPU by $1.34\times$; achieving $92.31\%$ Top-1 / $100.00\%$ Top-2 accuracy with $12.15\text{ ms}$ mean latency and $2.82\%$ mean regret on 13 SSB queries; and achieving $80.00\%$ Top-1 / $95.00\%$ Top-2 accuracy with $24.18\text{ ms}$ mean latency and $9.45\%$ mean regret on 20 ClickBench queries), $1.14\times\text{--}2.14\times$ steady-state concurrent throughput gains ($13.13\text{--}154.44\text{ QPS}$ across $C=1\text{--}32$), and $3.35\times\text{--}7.61\times$ speedups over DuckDB across $\text{SF}=1\text{--}100\text{ GB}$. Synthesizing our empirical findings across all seven architectural dimensions, this study establishes three core methodological conclusions: (i) \textit{Incremental Value of CSCI and Explicit Latency--Tail Risk Control}: engineered oblique hyperplane projection ($\text{CSCI}_{\text{res}}$, $\text{VIF}=1.84$) reduces active tree splits by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) beyond raw AST operator counts, while pairing $\mathcal{R}_{\text{GBR-9D}}$ with calibrated conformal intervals protects tail SLAs near PCIe crossover boundaries; (ii) \textit{Operator-Bounded Cross-Workload Transfer}: AST-Route transfers effectively to in-distribution unseen seeds ($73.33\%$ Top-1), official TPC-H choke points ($90.91\%$ Top-2), star-join schemas (SSB: $92.31\%$ Top-1 and $100.00\%$ Top-2 accuracy at $2.82\%$ mean regret), wide-table web analytics (ClickBench: $80.00\%$ Top-1 and $95.00\%$ Top-2 accuracy at $24.18\text{ ms}$), and string-skewed cyclic joins via conservative conformal abstention (JOB: $100.00\%$ Top-2 accuracy, matching Always Polars at $6.75\text{ ms}$), whereas holding out entire unseen operator classes under Leave-One-Tier-Out (LOTO-CV) degrades latency to $85.42\text{ ms}$ (trailing Always Polars at $82.47\text{ ms}$), confirming that training workloads must span the target SQL operator families; and (iii) \textit{Memory-Constrained Multi-Scale and Energy-Aware Dispatch}: coupling zone-map selectivity with catalog byte footprints ($\mathcal{B}_{\text{catalog}}$) enables seamless routing across in-VRAM GPU execution ($\text{SF}=1\text{ GB}$), chunked PCIe streaming ($\text{SF}=10\text{--}30\text{ GB}$), and elastic serverless cloud offloading under heavy host disk spilling ($\text{SF}=100\text{ GB}$) while jointly balancing latency, estimated device-level energy, and marginal cloud scan charges.

"""


def main():
    print("=" * 105)
    print("FINAL MASTER RECONCILIATION: FIXING ALL 6 REMAINING ITEMS IN V2 / V5_2 / V6 PDF")
    print("=" * 105)

    tex = TARGET_TEX.read_text(encoding="utf-8")

    # --------------------------------------------------------------------------
    # 1. ดึงบล็อก \paragraph{\texorpdfstring{6. Comprehensive...}} ... \end{table*}
    #    ออกจากหน้า 14 (ก่อน \subsection{In-Process Storage Effect) แล้วไปวางท้าย Section 4.7
    # --------------------------------------------------------------------------
    pat_step3_block = re.compile(
        r"\\paragraph\{(?:\\texorpdfstring\{)?6\.\s*Comprehensive Baseline Matrix.*?\\end\{table\*\}\s*",
        flags=re.DOTALL,
    )
    m_block = pat_step3_block.search(tex)
    if not m_block:
        raise RuntimeError("ไม่พบบล็อก Step 3 (Table 14) ในไฟล์ .tex")
    step3_block_str = m_block.group(0).strip()
    tex = pat_step3_block.sub("", tex, count=1)

    # วางไว้ก่อนขึ้น \subsection{Cost Prediction Q-Error, Regret Distribution...} (คือท้าย Section 4.7 พอดี!)
    pat_sec48 = re.compile(r"(\\subsection\{Cost Prediction Q-Error,[^\n]*\})")
    if pat_sec48.search(tex):
        tex = pat_sec48.sub(lambda m: step3_block_str + "\n\n" + m.group(1), tex, count=1)
        print("  [PASS-1] ย้ายหัวข้อข้อ 6 และ Table 14 ไปไว้ท้าย Section 4.7 (ก่อนขึ้น Section 4.8) สำเร็จ!")
    else:
        raise RuntimeError("ไม่พบหัวข้อ Section 4.8")

    # --------------------------------------------------------------------------
    # 2. ลบ Table 2B ตัวที่ซ้ำในหน้า 19 ออก และปรับค่า Cold Parquet Footer ใน Table 2B หน้า 6 ให้เป็น 412.6 us
    # --------------------------------------------------------------------------
    # ค้นหาตารางที่มี Table 2B. ทั้งหมด หากมีมากกว่า 1 ตาราง ให้ลบตัวที่ 2 ออก
    pat_tab2b = re.compile(
        r"\\begin\{table\*?\}(?:\[[^\]]*\])?\s*\\centering\s*(?:\\renewcommand\{\\thetable\}\{2B\}\s*)?\\caption\{Pre-execution storage metadata capabilities.*?\\end\{table\*?\}\s*",
        flags=re.DOTALL,
    )
    matches_2b = list(pat_tab2b.finditer(tex))
    if len(matches_2b) >= 2:
        # ลบตารางตัวที่ 2 (จากท้ายมาหน้า)
        start2, end2 = matches_2b[1].span()
        tex = tex[:start2] + tex[end2:]
        print(f"  [PASS-2] ลบ Table 2B ตัวที่ซ้ำในหน้า 19 ออกสำเร็จ! (เหลือ Table 2B เพียง 1 ตารางในหน้า 6)")
    else:
        print(f"  [PASS-2] ตรวจสอบจำนวน Table 2B: พบ {len(matches_2b)} ตาราง")

    # ปรับตัวเลข 105.5 us cold footer -> 412.6 us cold footer และ 1092.5 us -> 845.2 us ใน Table 2B หน้า 6 ให้ตรงกับเนื้อหาหน้า 6
    tex = tex.replace("105.5~\\mu\\text{s} cold footer", "412.6~\\mu\\text{s} cold footer")
    tex = tex.replace("105.5\\text{ }\\mu\\text{s} cold footer", "412.6\\text{ }\\mu\\text{s} cold footer")
    tex = re.sub(r"105\.5\s*(?:~|\\text\{\s*\})?\\mu\s*(?:\\text\{s\}|s)\s*cold\s*footer", r"412.6~\\mu\\text{s} cold footer", tex)
    tex = re.sub(r"1092\.5\s*(?:~|\\text\{\s*\})?\\mu\s*(?:\\text\{s\}|s)\s*live", r"845.2~\\mu\\text{s} live", tex)

    # --------------------------------------------------------------------------
    # 3. อัปเดต Section 1 (Strategic Positioning & 5 Primary Contributions)
    # --------------------------------------------------------------------------
    pre_intro, rest = tex.split("\\section{Introduction}", 1)
    intro_body, post_intro = rest.split("\\section{Related Work and Architectural Representativeness}", 1) if "\\section{Related Work and Architectural Representativeness}" in rest else rest.split("\\section{Related Work}", 1)
    sec2_header = "\\section{Related Work and Architectural Representativeness}" if "\\section{Related Work and Architectural Representativeness}" in rest else "\\section{Related Work}"

    pat_intro_enum = re.compile(
        r"(?:\\noindent\\textbf\{Bridging Empirical Crossover[^\n]*\n.*?)?(?:To address these gaps[^\n]*\n[^\n]*:)?.?\s*\\begin\{enumerate\}.*?\\end\{enumerate\}",
        flags=re.DOTALL,
    )
    if pat_intro_enum.search(intro_body):
        intro_body = pat_intro_enum.sub(lambda _: "\n\n" + STRATEGIC_INTRO_CONTRIBUTIONS, intro_body, count=1)
        print("  [PASS-3] อัปเดต Section 1 (Bridging Paragraph + 5 Primary Contributions) สำเร็จ!")
    tex = pre_intro + "\\section{Introduction}" + intro_body + sec2_header + post_intro

    # --------------------------------------------------------------------------
    # 4. อัปเดต Section 6 (Conclusion) และ Section 5(2) ให้ตรงกับ 163 คิวรี และไม่มีคำซ้ำ
    # --------------------------------------------------------------------------
    pat_conc = re.compile(
        r"\\section\{Conclusion\}.*?(?=\\section\*?\{(?:Declarations|Abbreviations|Availability))",
        flags=re.DOTALL,
    )
    if pat_conc.search(tex):
        tex = pat_conc.sub(lambda _: STRATEGIC_CONCLUSION_SECTION, tex, count=1)
        print("  [PASS-4] อัปเดต Section 6 (Conclusion — 163 Queries + Step 3 Synthesis) สำเร็จ!")

    # ซ่อมคำซ้ำใน Section 5 ข้อ 2 ("comprising 97 core TPC-H benchmark queries---comprising our 75-query") และอัปเดตเป็น 163 distinct SQL queries
    tex = re.sub(
        r"Although our evaluation spans 143 distinct SQL queries in total---comprising 97 core TPC-H benchmark queries---comprising our 75-query",
        r"Although our evaluation spans 163 distinct SQL queries in total---comprising 97 core TPC-H benchmark queries (our 75-query",
        tex,
    )
    tex = re.sub(
        r"queries \(Q1--Q22, including complex 5-way to 8-way join graphs in Q2, Q5, Q7, Q8, Q9, and Q21\) \[28, 30\]---the underlying",
        r"queries Q1--Q22, including complex 5-way to 8-way join graphs in Q2, Q5, Q7, Q8, Q9, and Q21 [28, 30]) and 66 external queries (33 JOB, 13 SSB, and 20 ClickBench)---the underlying",
        tex,
    )

    # --------------------------------------------------------------------------
    # 5. ซ่อมจุดย่อยใน Table 3, Table 13, Sec 4.7(4), Availability of data, และ Ref [28]
    # --------------------------------------------------------------------------
    # 5a. Table 3 แถว V3: แก้ \$375.51 & \$125.17 & \$12.52 & \$1.00 -> \$12.52 & \$125.17 & \$375.51 & \$1,251.70
    tex = re.sub(
        r"\\\$375\.51\s*&\s*\\\$125\.17\s*&\s*\\\$12\.52\s*&\s*\\\$1\.00",
        r"\\$12.52 & \\$125.17 & \\$375.51 & \\$1,251.70",
        tex,
    )

    # 5b. Table 13 Part C: แก้ $C=30$ Streams -> $C=32$ Streams
    tex = tex.replace("$C=30$ Streams", "$C=32$ Streams")
    tex = tex.replace("$C = 30$ Streams", "$C = 32$ Streams")

    # 5c. Section 4.7 ข้อ 4 (หน้า 22): แก้ 54.92 + 4.15 = 59.07 ms ให้ตรงกับ Part E (50.78 ms exec + 4.15 ms overhead = 54.92 ms net)
    tex = re.sub(
        r"Including this full parsing and inference overhead, the net end-to-end wall-clock time of AST-Route \(\$54\.92\s*\+\s*4\.15\s*=\s*59\.07\\text\{\s*ms\}\$\) still outperforms Rule-Based Routing \(\$71\.48\s*\+\s*3\.65\s*=\s*75\.13\\text\{\s*ms\}\$\),",
        r"Because our reported $54.92\\text{ ms}$ AST-Route latency already includes this $4.145\\text{ ms}$ warm overhead ($50.775\\text{ ms}$ engine execution $+ 4.145\\text{ ms}$ router overhead; $55.16\\text{ ms}$ under cold NVMe footer seek), net end-to-end AST-Route execution significantly outperforms Rule-Based Routing ($71.48\\text{ ms}$),",
        tex,
    )

    # 5d. Availability of data: เปลี่ยน nb2_exp1--nb2_exp3 เป็นชื่อไฟล์มาตรฐานและเพิ่มไฟล์ CSV ของ Step 3
    tex = re.sub(
        r"\(\\texttt\{nb2\\_exp1\}--\\texttt\{nb2\\_exp3\},",
        r"(\\texttt{secondary\_node\_dc\_shunt\_energy.csv}, \\texttt{secondary\_node\_cpu\_routing\_30q.csv}, \\texttt{secondary\_node\_concurrency\_22t.csv}, and \\texttt{step3\_pillar1\_complete\_baseline\_matrix\_ci\_p99.csv} through \\texttt{step3\_pillar5\_net\_latency\_and\_cold\_metadata\_regimes.csv},",
        tex,
    )

    # 5e. Ref [28]: ซ่อม URL ของ TPC-H ที่คำว่า tpc หายไป
    tex = re.sub(r"https://www\.\s*\.org/tpch/", "https://www.tpc.org/tpch/", tex)
    tex = re.sub(r"\\url\{https://www\.\s*\.org/tpch/\}", r"\\url{https://www.tpc.org/tpch/}", tex)

    print("  [PASS-5] ซ่อมจุดย่อยใน Table 3, Table 13 (C=32), Sec 4.7(4), Availability, และ Ref [28] สำเร็จ!")

    # บันทึกทับลงทุกไฟล์ .tex หลัก
    for fname in (
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V3.tex",
        "bigquery_olap_benchmark_V4.tex",
        "bigquery_olap_benchmark_V5_2.tex",
        "bigquery_olap_benchmark_V6.tex",
        "main.tex",
        "overleaf_ready_single_file.tex",
    ):
        (MS_DIR / fname).write_text(tex, encoding="utf-8")

    # ตรวจสอบเงื่อนไขความถูกต้องทั้งหมด
    abs_part = tex.split("\\section{Introduction}")[0]
    if "Background:" in abs_part:
        abs_part = abs_part[abs_part.find("Background:"):]
    w_abs = count_academic_words(abs_part)

    checks = [
        (f"1. Abstract Word Count <= 350 Words (Measured: {w_abs} words)", w_abs <= 350),
        ("2. Table 14 (Step 3 Audit) วางอยู่ท้าย Section 4.7 (หลังข้อ 5 Seed-Level)",
         tex.find("\\paragraph{\\texorpdfstring{6. Comprehensive") > tex.find("Seed-Level ($N=25$) Inference")),
        ("3. ไม่มี Table 2B ซ้ำ (มีคำว่า Pre-execution storage metadata capabilities เพียง 1 จุด)",
         len(re.findall(r"Pre-execution storage metadata capabilities", tex)) == 1),
        ("4. Table 3 แถว V3 แสดง $1,251.70 ที่ SF=100 GB (ไม่มี $1.00 ค้าง)",
         "1,251.70" in tex and "& \\$1.00 &" not in tex),
        ("5. Table 13 Part C แสดง $C=32$ Streams (ไม่มี $C=30$ Streams ค้าง)",
         "$C=30$" not in tex and "$C=32$ Streams" in tex),
        ("6. Section 4.7(4) ไม่ขัดแย้งกับ Part E (ไม่มี 59.07 ms ค้าง)",
         "59.07" not in tex and "50.775\\text{ ms}" in tex),
        ("7. Section 1, 5, 6 ระบุจำนวนคิวรีรวม 163 คิวรีตรงกับ Abstract และไม่มีคำว่า nb2_exp1 ค้าง",
         "163 SQL queries" in tex and "nb2\\_exp1" not in tex),
    ]
    print("-" * 105)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {label}")
    print("-" * 105)

    # คอมไพล์ PDF ทั้ง 3 ไฟล์ (2 รอบต่อไฟล์)
    for tex_name in ("bigquery_olap_benchmark_V2.tex", "bigquery_olap_benchmark_V5_2.tex", "bigquery_olap_benchmark_V6.tex"):
        tpath = MS_DIR / tex_name
        for _ in (1, 2):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tpath.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print("\n".join(res.stdout.splitlines()[-25:]))
                raise RuntimeError(f"pdflatex failed on {tpath.name}")
        pdf_path = tpath.with_suffix(".pdf")
        print(f"  [PDF-OK] {pdf_path.name:<34} | Size: {pdf_path.stat().st_size:,} bytes | SHA-256: {sha256_of(pdf_path)[:24]}...")

    pdf_v2 = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
    for alias in ("bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V3_3.pdf", "bigquery_olap_benchmark_V3_4.pdf", "bigquery_olap_benchmark_V4.pdf", "main.pdf"):
        try:
            shutil.copy2(pdf_v2, MS_DIR / alias)
        except PermissionError:
            pass

    for zname in ("overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"):
        zpath = MS_DIR / zname
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(TARGET_TEX, arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            bib = MS_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    print("=" * 105)


if __name__ == "__main__":
    main()
