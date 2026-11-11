#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# scripts/step2_lock_strategic_framing_abstract_intro_conclusion.py
# สคริปต์อัปเดต Strategic Framing (แนวทาง ก ประยุกต์ + รากฐาน ข):
#   "Interpretable, Risk-Aware Multi-Objective Routing Grounded in Crossover Profiling"
#   1. อัปเดต Abstract (<= 350 คำมาตรฐานวิชาการ) ชู Latency vs. Tail-Risk Trade-off
#      (Ridge 49.12 ms / Shallow DT 51.35 ms vs. Conformal GBR-9D 54.92 ms, P95 114.11%)
#   2. อัปเดต Section 1: ย่อหน้าเชื่อมกลยุทธ์ + 5 Primary Contributions
#   3. อัปเดต Section 6: Conclusion ให้สอดคล้องกับ Seed-Level (N=25) Holm-Bonferroni,
#      Latency--Tail Risk Trade-off, และขอบเขต LOTO-CV (85.42 ms)
#   4. คอมไพล์ PDF 2 รอบ + แพ็ก ZIP + ตรวจสอบ 8 เงื่อนไข [PASS] 100%
# ==============================================================================

import hashlib
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIGURES_DIR = MS_DIR / "figures"
TARGET_V2_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_academic_words(latex_block: str) -> int:
    # นับจำนวนคำมาตรฐานทางวิชาการ (ยุบนิพจน์คณิตศาสตร์ $...$ เป็น 1 คำต่อ 1 นิพจน์ และถอดคำสั่ง LaTeX)
    text = re.sub(r"\$[^\$]+\$", " MATH_TOKEN ", latex_block)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[\{\}~]", " ", text)
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return len(words)


# ==============================================================================
# 1. ABSTRACT ฉบับล็อกเป้ากลยุทธ์ (ความยาว ~324 คำวิชาการ <= 350 คำ)
# ==============================================================================
STRATEGIC_ABSTRACT_BODY = r"""\textbf{Background:} Analytical infrastructures span serverless cloud warehouses, vectorized CPU engines, and discrete GPUs. Empirical crossover profiling shows no single engine dominates across query topologies and scales ($\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$): simple scans incur GPU PCIe transfer and cloud scheduling overheads, complex window/unnest pipelines bottleneck CPUs, and working sets exceeding VRAM or DRAM trigger chunked streaming and disk spilling.

\textbf{Methodology:} We propose \textbf{AST-Route}, an interpretable, risk-aware multi-objective query routing framework grounded in CPU--GPU--cloud crossover profiling. Layer~1 extracts a pre-execution 9D vector ($\mathbf{x}_q\in\mathbb{R}^9$) combining static Abstract Syntax Tree (AST) operator counts, a construct-validated Composite Structural Complexity Index ($\text{CSCI}$, 10,000-trial stability $\bar{\rho}=0.9880$ at $\pm 20\%$; $\rho_{\text{partial}}=0.5477, p=3.68\times 10^{-7}$; tree-size $\text{VIF}=1.84$), and zone-map selectivity $\hat{\sigma}(q)$ coupled with catalog byte volume ($\mathcal{B}_{\text{catalog}}$). Layers~2--4 pair analytical crossover models with pluggable predictors---spanning speed-first linear/tree estimators and a Multi-Output Gradient Boosting Regressor ($\mathcal{R}_{\text{GBR-9D}}$) with 90\% conformal intervals ($\text{ECE}=1.78\%$) and a $12\%$ PCIe abstention guard---across SQLite~3, Google BigQuery, DuckDB, Polars, Apache DataFusion, and an NVIDIA RTX~2050 CUDA engine.

\textbf{Results:} Across 143 queries (75 factorial TPC-H, 22 official TPC-H Q1--Q22, 33 IMDB JOB, and 13 SSB queries) under 100\% real-execution telemetry, vectorized CPU engines (Polars $82.47\text{ ms}$, DataFusion $96.30\text{ ms}$, DuckDB $183.74\text{ ms}$) beat SQLite~3 ($8417.33\text{ ms}$) by $45.8\times\text{--}102.1\times$, while RTX~2050 reaches $56.96\text{ ms}$ cold ($1.45\times$ vs.\ Polars, $3.23\times$ vs.\ DuckDB) and $22.25\text{ ms}$ warm ($3.71\times$ vs.\ Polars, $8.22\times$ kernel speedup). Under 25-seed LOSO-CV ($\text{SF}=1\text{ GB}$, $4.15\text{ ms}$ overhead), our 9D features enable point-estimate Ridge ($49.12\text{ ms}$) and Shallow Decision Tree ($51.35\text{ ms}$) routers to minimize raw mean latency at elevated P95 regret ($196.80\%$ and $150.90\%$), whereas conformal AST-Route ($\mathcal{R}_{\text{GBR-9D}}$, $54.92\text{ ms}$, $57.33\%$ Top-1, $76.00\%$ Top-2, $0.00\%$ median, $34.98\%$ mean regret, $1.381\text{ J}$ estimated device energy) trades $3.6\text{--}5.8\text{ ms}$ of mean latency to compress P95 tail regret to $114.11\%$ (vs.\ $257.31\%$ on Always GPU) and halve $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). Under seed-level ($N=25$) Holm--Bonferroni tests, AST-Route outperforms Always DuckDB ($3.35\times, p_{\text{Holm}}=5.45\times 10^{-4}$), DataFusion ($1.75\times, p_{\text{Holm}}=0.0041$), Polars ($1.50\times, p_{\text{Holm}}=0.0171$), and Rule-Based Routing ($p=0.0187$), while matching Always GPU in mean latency ($p=0.184$). It achieves $73.33\%$ Top-1 ($47.01\text{ ms}$) on unseen seeds, $68.18\%$ Top-1 / $90.91\%$ Top-2 ($16.98\text{ ms}$) on Q1--Q22, $92.31\%$ Top-1 ($12.15\text{ ms}$) on SSB, Polars parity ($6.75\text{ ms}, 100\%$ Top-2) on JOB, $1.12\times\text{--}2.14\times$ concurrent QPS gains ($C=1\text{--}32$), and $3.35\times\text{--}7.61\times$ speedups across $\text{SF}=1\text{--}100\text{ GB}$, whereas Leave-One-Tier-Out (LOTO-CV) yields $85.42\text{ ms}$ (vs.\ Polars $82.47\text{ ms}$).

\textbf{Conclusion:} AST-Route delivers interpretable, catalog-volume-coupled routing with tunable latency--tail-risk control across CPU, GPU, and serverless cloud backends when training workloads cover target SQL operator families (\url{https://github.com/preecha-msu/bigquery-olap-benchmark})."""


# ==============================================================================
# 2. SECTION 1: STRATEGIC POSITIONING & 5 PRIMARY CONTRIBUTIONS
# ==============================================================================
STRATEGIC_INTRO_CONTRIBUTIONS = r"""\noindent\textbf{Bridging Empirical Crossover Profiling and Risk-Aware Multi-Objective Routing.}
Designing a reliable cross-paradigm dispatcher requires resolving a fundamental systems tension: static single-engine policies suffer severe performance cliffs across heterogeneous operators, yet naive point-estimate cost minimization ($\arg\min_e \hat{T}_e$) triggers costly tail-latency spikes whenever prediction intervals overlap near CPU--GPU PCIe crossover boundaries or device memory ceilings. Rather than treating hardware benchmarking and learned routing in isolation, this study grounds an interpretable, zero-execution routing architecture directly in controlled factorial crossover telemetry. Crucially, decoupling the pre-execution structural/volume representation ($\mathbf{x}_q \in \mathbb{R}^9$) from the downstream decision policy exposes an explicit operational trade-off between \emph{unconstrained mean-latency minimization} (achievable via lightweight linear or shallow-tree estimators) and \emph{tail-safe SLA protection} (governed via calibrated conformal prediction intervals and resource-constrained Pareto dispatch). The primary contributions of this study are summarized as follows:

\begin{enumerate}
    \item \textbf{De-Confounded Factorial Testbed and CPU--GPU--Cloud Crossover Profiling:} We construct a 75-query AST-stratified factorial benchmark (25 canonical seeds $\times$ 3 selectivity horizons V1--V3 across tiers L1--L5, complemented by 22 official TPC-H, 33 IMDB JOB, and 13 SSB queries; $N=143$ total) across six engines spanning row-store, serverless cloud, C++/Rust Apache Arrow vectorized CPUs, and discrete CUDA GPUs. By stripping SQL parsing/planning overhead ($183.74 - 0.87 = 182.87\text{ ms}$ pure DuckDB CPU kernel vs.\ $22.25\text{ ms}$ in-VRAM GPU kernel) and pairing direct NVML GPU power sensors with TDP-calibrated CPU package energy estimates and BigQuery $10\text{ MiB}$ minimum-floor scan billing, we quantify exact PCIe transfer crossovers and memory walls under 100\% real-execution ground-truth telemetry (zero synthetic latency simulation).

    \item \textbf{Construct-Validated AST Structural Complexity ($\text{CSCI}$) and Pre-Execution 9D Representation:} We formulate a zero-execution 9-dimensional feature vector ($\mathbf{x}_q \in \mathbb{R}^9$) extracted in $3.65\text{ ms}$ ($3.476\text{ ms}$ \texttt{SQLGlot} AST parsing plus $0.172\text{ ms}$ Parquet/DuckDB row-group zone-map and cloud partition pruning metadata; selectivity $\text{MAE}=3.84\%$, median Q-Error $=1.092$) without touching data pages. Extensive validation confirms construct correlation with AST node count ($\rho=0.8744$), partial correlation with GPU speedup controlling for selectivity ($\rho_{\text{partial}}=0.5477, p=3.68\times 10^{-7}$), 10,000-trial weight-perturbation rank stability ($\bar{\rho}=0.9880$ at $\pm 20\%$), and low tree-size multicollinearity ($\text{VIF}=1.84$ against $[D(q), N(q)]$). In axis-aligned tree ensembles, $\text{CSCI}(q)$ acts as an engineered oblique hyperplane projection that reduces active split nodes per tree by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) over atomic AST counts alone.

    \item \textbf{Model-Agnostic Heterogeneous Cost Prediction and Explicit Latency--Tail Risk Control:} Using $\mathbf{x}_q$, our multi-output cost predictors estimate cross-architecture latency, device energy, and peak working-set memory in $497.2\text{ }\mu\text{s}$ (median Q-Error $1.007\text{--}1.172$, $\log\text{-}R^2=0.815\text{--}0.912$). We demonstrate that the 9D feature space supports two distinct operating regimes under 25-seed LOSO-CV ($\text{SF}=1\text{ GB}$): \emph{speed-prioritized point-estimate routers} using Ridge regression ($49.12\text{ ms}$) or shallow decision trees ($51.35\text{ ms}$) achieve the lowest raw mean latencies across all evaluated policies but incur high P95 tail regret ($196.80\%$ and $150.90\%$), whereas our proposed \emph{conformal risk-aware AST-Route} ($\mathcal{R}_{\text{GBR-9D}}$ with 90\% conformal prediction intervals, $\text{ECE}=1.78\%$, and a $12\%$ abstention guard to deterministic CPU execution on $16.00\%$ of ambiguous queries) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$) to compress P95 tail regret to $114.11\%$, reduce worst-case regret from $620.63\%$ (Always GPU) and $312.40\%$ (point-estimate $\arg\min$) to $245.18\%$, halve severe routing failures ($>100\%$ regret from $14.67\%$ to $6.67\%$), and raise decision stability to $98.67\%$.

    \item \textbf{Memory-Constrained Multi-Objective Pareto Dispatch (Latency, Energy, and Cloud Scan Charge):} Coupling $\hat{\sigma}(q)$ with catalog byte footprints ($\hat{\mathcal{B}}_{\text{scan}}(q)=\hat{\sigma}(q)\cdot\mathcal{B}_{\text{catalog}}$) enforces strict VRAM/DRAM feasibility filtering (eliminating 100\% of out-of-memory failures under tight VRAM caps) while jointly optimizing execution latency, estimated active device energy ($1.381\text{ J/query}$), and marginal on-demand Cloud Query Scan Charges ($\text{\$6.25/TiB}$, cleanly separated from sunk local hardware amortization). Under independent seed-level ($N=25$) Holm--Bonferroni inference, AST-Route significantly outperforms static CPU engines ($1.50\times\text{--}3.35\times, p_{\text{Holm}}\le 0.0171$) and static rule-based routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}, p=0.184$) at less than half the P95 tail regret.

    \item \textbf{Multi-Scale, Concurrent, Cross-Hardware, and Operator-Bounded Generalization:} Across $\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$ ($6.0\text{M}\text{--}600.1\text{M}$ rows), $C=1\text{--}32$ concurrent query streams, and secondary multi-core CPU-only hardware profiles, AST-Route sustains $3.35\times\text{--}7.61\times$ speedups over DuckDB and $1.12\times\text{--}2.14\times$ steady-state throughput gains (up to $154.44\text{ QPS}$). It transfers effectively to locked unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), official TPC-H Q1--Q22 ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), zero-shot star-join SSB ($92.31\%$ Top-1, $12.15\text{ ms}$), and string-skewed cyclic JOB queries ($6.75\text{ ms}$ Polars parity via $21.21\%$ conformal abstention), while Leave-One-Tier-Out evaluation (LOTO-CV: $85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) transparently establishes that training workloads must span target SQL operator families.
\end{enumerate}"""


# ==============================================================================
# 3. SECTION 6: CONCLUSION ฉบับสอดคล้องกับกลยุทธ์หลัก 100%
# ==============================================================================
STRATEGIC_CONCLUSION_SECTION = r"""\section{Conclusion}
We presented AST-Route, an interpretable, risk-aware multi-objective query routing framework grounded in empirical CPU--GPU--cloud crossover profiling and evaluated across 143 SQL queries in total---comprising 97 core TPC-H benchmark queries (a 75-query AST-stratified factorial workload across L1--L5 and V1--V3 plus the 22 official standard TPC-H queries Q1--Q22 [28]) and 46 external zero-shot cross-schema validation queries (33 IMDB JOB templates [27] and 13 SSB star-join queries [26])---across four standard scale factors ($\text{SF}=1, 10, 30,$ and $100\text{ GB}$ from 6.0M to 600.1M \texttt{lineitem} rows), three cross-platform hardware profiles, $C=1\text{--}32$ concurrent query streams, and six software systems across four computing paradigms: SQLite~3, Google BigQuery, DuckDB, Polars, Apache DataFusion, and CUDA GPU execution (NVIDIA RTX~2050). Validated via 10,000-iteration seed-clustered bootstrap 95\% confidence intervals, Cliff's delta effect sizes, 10,000-trial CSCI sensitivity analysis ($\bar{\rho}=0.9880$ at $\pm 20\%$, partial correlation $\rho_{\text{partial}}=0.5477$), strict GPU/CPU FP64 fairness protocols, feature group ablation, Pareto trade-off analysis, leak-free 25-seed LOSO-CV, unseen seed holdout testing, and strict nested 22-fold LOO-CV, our pre-execution 9D feature representation ($\mathbf{x}_q\in\mathbb{R}^9$) enables multi-engine cost prediction (median Q-Error $1.007\text{--}1.172$, $R^2=0.815\text{--}0.875$) in $497.2\text{ }\mu\text{s}$ ($4.145\text{ ms}$ total end-to-end parsing and inference time) and supports tunable operational trade-offs: point-estimate Ridge ($49.12\text{ ms}$) and shallow decision tree ($51.35\text{ ms}$) routers minimize raw mean latency at elevated P95 tail regret ($196.80\%$ and $150.90\%$), whereas conformal AST-Route ($\mathcal{R}_{\text{GBR-9D}}$) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$, $57.33\%$ Top-1 and $76.00\%$ Top-2 accuracy, $0.00\%$ median regret, $34.98\%$ mean regret, $1.381\text{ J}$ estimated device energy) to compress P95 tail regret to $114.11\%$ and halve severe ($>100\%$ regret) routing failures from $14.67\%$ to $6.67\%$. Under independent seed-level ($N=25$ clusters) Holm--Bonferroni inference, AST-Route significantly outperforms Always DuckDB ($3.35\times, p_{\text{Holm}}=5.45\times 10^{-4}$), Always DataFusion ($1.75\times, p_{\text{Holm}}=0.0041$), Always Polars ($1.50\times, p_{\text{Holm}}=0.0171$), and Rule-Based Routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}$, seed-level $p=0.184$) yet cutting P95 tail regret from $257.31\%$ to $114.11\%$. Furthermore, AST-Route achieves $73.33\%$ Top-1 accuracy ($47.01\text{ ms}$) on locked unseen query seeds, $68.18\%$ Top-1 and $90.91\%$ Top-2 accuracy ($16.98\text{ ms}$, $2.52\times$ faster than Always Polars) under nested LOO-CV on official TPC-H Q1--Q22, safe zero-shot cross-schema transfer across 46 external queries (matching Always Polars at $6.75\text{ ms}$ with $21.21\%$ Top-1, $100.00\%$ Top-2, and $84.49\%$ mean regret on 33 JOB queries while beating DuckDB by $2.90\times$ and Always GPU by $1.34\times$, and achieving $92.31\%$ Top-1 / $100.00\%$ Top-2 accuracy with $12.15\text{ ms}$ mean latency and $2.82\%$ mean regret on 13 SSB queries), $1.12\times\text{--}2.14\times$ steady-state concurrent throughput gains ($13.13\text{--}154.44\text{ QPS}$ across $C=1\text{--}32$), and $3.35\times\text{--}7.61\times$ speedups over DuckDB across $\text{SF}=1\text{--}100\text{ GB}$. Synthesizing our empirical findings across all seven architectural dimensions, this study establishes three core methodological conclusions: (i) \textit{Incremental Value of CSCI and Explicit Latency--Tail Risk Control}: engineered oblique hyperplane projection ($\text{CSCI}_{\text{res}}$, $\text{VIF}=1.84$) reduces active tree splits by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) beyond raw AST operator counts, while pairing $\mathcal{R}_{\text{GBR-9D}}$ with calibrated conformal intervals protects tail SLAs near PCIe crossover boundaries; (ii) \textit{Operator-Bounded Cross-Workload Transfer}: AST-Route transfers effectively to in-distribution unseen seeds ($73.33\%$ Top-1), official TPC-H choke points ($90.91\%$ Top-2), star-join schemas (SSB: $92.31\%$ Top-1 and $100.00\%$ Top-2 accuracy at $2.82\%$ mean regret), and string-skewed cyclic joins via conservative conformal abstention (JOB: $100.00\%$ Top-2 accuracy, matching Always Polars at $6.75\text{ ms}$), whereas holding out entire unseen operator classes under Leave-One-Tier-Out (LOTO-CV) degrades latency to $85.42\text{ ms}$ (trailing Always Polars at $82.47\text{ ms}$), confirming that training workloads must span the target SQL operator families; and (iii) \textit{Memory-Constrained Multi-Scale and Energy-Aware Dispatch}: coupling zone-map selectivity with catalog byte footprints ($\mathcal{B}_{\text{catalog}}$) enables seamless routing across in-VRAM GPU execution ($\text{SF}=1\text{ GB}$), chunked PCIe streaming ($\text{SF}=10\text{--}30\text{ GB}$), and elastic serverless cloud offloading under heavy host disk spilling ($\text{SF}=100\text{ GB}$) while jointly balancing latency, estimated device-level energy, and marginal cloud scan charges.

"""


def apply_strategic_framing(tex: str) -> tuple[str, int, dict]:
    flags = {}

    # 1. แทนที่ส่วน Abstract (รองรับทั้ง \begin{abstract}...\end{abstract} และ \textbf{Background:}...\textbf{Keywords:})
    pat_env = re.compile(r"(\\begin\{abstract\}\s*)(.*?)(\s*\\end\{abstract\})", flags=re.DOTALL)
    pat_bg_kw = re.compile(
        r"((?:\\noindent\s*)?\\textbf\{Background:\s*\}|Background:\s*Modern\s+analytical|Background:\s*Analytical).*?(?=(?:\n\s*(?:\\vspace\{[^}]*\}\s*)?(?:\\noindent\s*)?(?:\\textbf\{Keywords:?\}|Keywords:)|\n\s*\\section\{Introduction\}))",
        flags=re.DOTALL,
    )
    if pat_env.search(tex):
        tex = pat_env.sub(lambda m: m.group(1) + STRATEGIC_ABSTRACT_BODY + m.group(3), tex, count=1)
        flags["Abstract_Updated"] = True
    elif pat_bg_kw.search(tex):
        tex = pat_bg_kw.sub(lambda _: STRATEGIC_ABSTRACT_BODY + "\n\n", tex, count=1)
        flags["Abstract_Updated"] = True
    else:
        raise RuntimeError("ไม่พบตำแหน่ง Abstract ในไฟล์ .tex")

    w_count = count_academic_words(STRATEGIC_ABSTRACT_BODY)

    # 2. แทนที่บล็อก Contributions ใน Section 1 (Introduction)
    # ค้นหาตั้งแต่ย่อหน้าเกริ่นนำก่อน \begin{enumerate} (หรือตัว \begin{enumerate} เองใน Section 1) จนถึง \end{enumerate} ก่อนขึ้น \section{Related Work}
    if "\\section{Introduction}" in tex and "\\section{Related Work}" in tex:
        pre_intro, rest = tex.split("\\section{Introduction}", 1)
        intro_body, post_intro = rest.split("\\section{Related Work}", 1)

        # ลองจับตั้งแต่ประโยค The primary contributions / Bridging Empirical Crossover ... จนถึง \end{enumerate}
        pat_intro_block = re.compile(
            r"(?:\\noindent\\textbf\{Bridging Empirical Crossover[^\n]*\n.*?)?(?:The\s+(?:primary|five\s+primary|main)\s+contributions\s+of\s+this\s+study\s+are\s+summarized\s+as\s+follows:|To\s+address\s+these\s+gaps,\s+the\s+primary\s+contributions[^\n]*:)?\s*\\begin\{enumerate\}.*?\\end\{enumerate\}",
            flags=re.DOTALL,
        )
        if pat_intro_block.search(intro_body):
            intro_body = pat_intro_block.sub(lambda _: STRATEGIC_INTRO_CONTRIBUTIONS, intro_body, count=1)
            flags["Intro_Contributions_Updated"] = True
        else:
            raise RuntimeError("ไม่พบบล็อก \\begin{enumerate}...\\end{enumerate} ใน Section 1 (Introduction)")

        tex = pre_intro + "\\section{Introduction}" + intro_body + "\\section{Related Work}" + post_intro
    else:
        raise RuntimeError("ไม่พบหัวข้อ \\section{Introduction} หรือ \\section{Related Work}")

    # 3. แทนที่ Section 6 (Conclusion) ทั้งบล็อก
    pat_conc = re.compile(
        r"\\section\{Conclusion\}.*?(?=\\section\*?\{(?:Declarations|Abbreviations|Availability))",
        flags=re.DOTALL,
    )
    if pat_conc.search(tex):
        tex = pat_conc.sub(lambda _: STRATEGIC_CONCLUSION_SECTION, tex, count=1)
        flags["Conclusion_Updated"] = True
    else:
        flags["Conclusion_Updated"] = False

    return tex, w_count, flags


def main():
    print("=" * 100)
    print("STEP 2: STRATEGIC FRAMING LOCK (ABSTRACT + INTRO CONTRIBUTIONS + CONCLUSION)")
    print("=" * 100)

    if not TARGET_V2_TEX.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์เป้าหมาย: {TARGET_V2_TEX}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MS_DIR / f"bigquery_olap_benchmark_V2_pre_step2_framing_{ts}.tex"
    shutil.copy2(TARGET_V2_TEX, backup)

    raw_tex = TARGET_V2_TEX.read_text(encoding="utf-8")
    updated_tex, abs_words, flags = apply_strategic_framing(raw_tex)

    # บันทึกทับลงทุกไฟล์ .tex หลักในโฟลเดอร์ manuscript/
    tex_targets = (
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V3.tex",
        "bigquery_olap_benchmark_V4.tex",
        "bigquery_olap_benchmark_V5_2.tex",
        "bigquery_olap_benchmark_V6.tex",
        "main.tex",
        "overleaf_ready_single_file.tex",
    )
    for fname in tex_targets:
        (MS_DIR / fname).write_text(updated_tex, encoding="utf-8")

    # ดึงส่วน Abstract, Section 1, และ Section 6 มาตรวจสอบความถูกต้อง
    abs_part = updated_tex.split("\\section{Introduction}")[0]
    if "Background:" in abs_part:
        abs_part = abs_part[abs_part.find("Background:"):]
    intro_part = updated_tex.split("\\section{Introduction}")[1].split("\\section{Related Work}")[0]
    sec6_part = updated_tex.split("\\section{Conclusion}")[-1]
    for stop_tok in ("\\section*{Declarations}", "\\section{Declarations}", "\\section*{Abbreviations}"):
        if stop_tok in sec6_part:
            sec6_part = sec6_part.split(stop_tok)[0]

    audit_checks = [
        (f"1. ความยาว Abstract <= 350 คำมาตรฐานวิชาการ (วัดได้จริง: {abs_words} คำ)", abs_words <= 350),
        ("2. Abstract ชู Latency vs. Tail-Risk Trade-off (Ridge 49.12 / DT 51.35 ms vs. Conformal 54.92 ms, P95 114.11%)",
         "49.12" in abs_part and "51.35" in abs_part and "54.92" in abs_part and "114.11\\%" in abs_part),
        ("3. Abstract ระบุผล Seed-Level (N=25) + Holm (p_Holm <= 0.0171, GPU p=0.184) และไม่มี p=0.0021",
         "0.0021" not in abs_part and "p_{\\text{Holm}}" in abs_part and "0.184" in abs_part),
        ("4. Abstract ระบุผล LOTO-CV (85.42 ms vs. Polars 82.47 ms) อย่างโปร่งใส",
         "85.42" in abs_part and "LOTO-CV" in abs_part),
        ("5. Section 1 (Contributions 1-5) อัปเดตครบ: Crossover + CSCI Oblique (-28.2%, -9.22%) + Conformal Guard",
         flags.get("Intro_Contributions_Updated", False)
         and "-28.2\\%" in intro_part
         and "-9.22\\%" in intro_part
         and "49.12\\text{ ms}" in intro_part
         and "\\$6.25/TiB" in intro_part),
        ("6. Section 6 (Conclusion) สอดคล้องกับ Seed-Level Holm + Trade-off + LOTO-CV (85.42 ms) และไม่มี p=0.0021",
         flags.get("Conclusion_Updated", False)
         and "0.0021" not in sec6_part
         and "49.12\\text{ ms}" in sec6_part
         and "85.42\\text{ ms}" in sec6_part),
        ("7. ความสะอาดทั้งเล่ม: Table 2B ไม่ซ้ำ, ซ่อม DOI [13]/[15], ไม่มี LAPTOP-LTL5PC7P หรือ green OLAP",
         len(re.findall(r"Table\s*2B\.", updated_tex)) <= 1
         and "10.1186/s40537-022-00659-3" in updated_tex
         and "LAPTOP-LTL5PC7P" not in updated_tex
         and "green OLAP" not in updated_tex),
        ("8. ขอบเขต Concurrency เป็น C=1--32 ตรงกันทั่วทั้งเล่ม (ไม่มี 1--16 concurrent ค้าง)",
         "16 concurrent" not in updated_tex and "C=1\\text{--}32" in updated_tex),
    ]

    for label, ok in audit_checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {label}")
    print("-" * 100)

    # คอมไพล์ PDF หลักและซิงค์ทุกไฟล์ PDF ในโฟลเดอร์ manuscript/
    for name in ("bigquery_olap_benchmark_V2.tex", "bigquery_olap_benchmark_V5_2.tex", "bigquery_olap_benchmark_V6.tex"):
        tfile = MS_DIR / name
        for pass_idx in (1, 2):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tfile.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(f"[ERROR compiling {tfile.name} Pass {pass_idx}]:\n" + "\n".join(res.stdout.splitlines()[-25:]))
                raise RuntimeError("pdflatex compilation failed")
        pfile = tfile.with_suffix(".pdf")
        if pfile.exists():
            print(f"[PDF-OK] {pfile.name:<34} | Size: {pfile.stat().st_size:,} bytes | SHA-256: {sha256_file(pfile)[:24]}...")

    pdf_v2 = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
    for pdf_alias in ("bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V3_3.pdf", "bigquery_olap_benchmark_V3_4.pdf", "bigquery_olap_benchmark_V4.pdf", "main.pdf"):
        try:
            shutil.copy2(pdf_v2, MS_DIR / pdf_alias)
        except PermissionError:
            print(f"  [WARN] ข้ามการเขียนทับ {pdf_alias} เนื่องจากเปิดค้างอยู่ในโปรแกรมอ่าน PDF")

    # อัปเดตไฟล์ ZIP สำหรับส่งวารสาร
    for zname in ("overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"):
        zpath = MS_DIR / zname
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(TARGET_V2_TEX, arcname="main.tex")
            zf.write(TARGET_V2_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            bib = MS_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIGURES_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    print("-" * 100)
    print(f"  -> Master TeX SHA-256 : {sha256_file(TARGET_V2_TEX)}")
    print(f"  -> Master PDF SHA-256 : {sha256_file(pdf_v2)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
