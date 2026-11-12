#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/91_reframe_3_pillars_jbd_positioning_to_V2_tex.py
================================================================================
Reframes the manuscript's narrative positioning specifically for Journal of Big Data
around the 3 Core Systems & Orchestration Pillars—eliminating any misperception of
a simple engine benchmark while preserving 100% of real-execution numbers & SHA-256 logs:

  Pillar 1: Big Data Systems Innovation (Query management across heterogeneous
            serverless cloud, C++ vectorized CPU, Rust Arrow, and SIMT GPU engines)
  Pillar 2: Intelligent Resource Orchestration (Pre-execution AST structural
            profiling, CSCI, and multi-output cost/energy/memory prediction)
  Pillar 3: Scalable and Risk-Aware Execution (Conformal tail-risk control, hard
            VRAM/DRAM memory caps, C=1-32 & bursty SLA compliance, SF=1-100 GB)

Updates 5 high-visibility zones in bigquery_olap_benchmark_V2.tex:
  (A) Abstract (leads Results with AST-Route intelligent orchestration + keeps all metrics)
  (B) SpringerOpen JBD Keywords block (explicitly centered on Intelligent Query Routing)
  (C) Section 1 Contributions (restructured explicitly into Pillars 1, 2, and 3)
  (D) Section 4 Opening Systems Roadmap (frames Sec 4.1-4.5 as physical crossover
      laws parameterizing AST-Route in Sec 4.6-4.9, NOT a standalone benchmark)
  (E) Section 6 Conclusion Takeaways (aligned 1-to-1 with the 3 Core Pillars)

Overwrites & compiles (2 passes):
  E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark\manuscript\bigquery_olap_benchmark_V2.tex
================================================================================
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_plain_words(latex_snippet: str) -> int:
    clean = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", latex_snippet)
    clean = re.sub(r"[{}$%&_~^\\]", " ", clean)
    tokens = [w for w in clean.split() if any(ch.isalnum() for ch in w)]
    return len(tokens)


def reframe_manuscript() -> dict:
    if not TARGET_TEX.exists():
        raise FileNotFoundError(f"Target manuscript missing: {TARGET_TEX}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = MANUSCRIPT_DIR / f"bigquery_olap_benchmark_V2_pre_3pillars_{ts}.tex"
    shutil.copy2(TARGET_TEX, backup_path)

    tex = TARGET_TEX.read_text(encoding="utf-8")

    # Safety cleanup if repeated "Est. Est." header exists in Table 8
    tex = re.sub(r"(?:Est\.\\?\s+){2,}", lambda _: r"Est.\ ", tex)

    # ==========================================================================
    # ZONE A & B: REFRAME ABSTRACT (ORCHESTRATION-FIRST) + ADD JBD KEYWORDS
    # ==========================================================================
    new_abstract_and_keywords = (
        r"\begin{abstract}" "\n"
        r"\textbf{Background:} Modern big data analytics span heterogeneous computing infrastructures---elastic serverless cloud warehouses, "
        r"in-process C++/Rust vectorized CPUs, and discrete SIMT GPUs---yet static single-engine execution suffers severe performance, "
        r"memory, and FinOps cliffs across scale factors ($\text{SF}\in\{1,10,30,100\}\text{ GB}$): simple scans incur GPU PCIe and cloud "
        r"scheduling overheads, window/unnest pipelines bottleneck CPUs, and large working sets trigger streaming or out-of-core disk spilling." "\n\n"
        r"\textbf{Methodology:} Rather than benchmarking engines in isolation, we present \textbf{AST-Route}, an intelligent, pre-execution "
        r"multi-objective query routing and resource orchestration framework. Layer~1 extracts a primary $9\text{D}$ vector ($\mathbf{x}_q \in \mathbb{R}^{9}$ "
        r"for fixed-scale $\text{SF}=1\text{ GB}$ routing, augmented with $\log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})$ into $\mathbf{x}_q^{\text{scale}} \in \mathbb{R}^{10}$ "
        r"exclusively for multi-scale $\text{SF}=1\text{--}100\text{ GB}$ routing) combining static AST operator counts, a Composite Structural Complexity Index "
        r"($\text{CSCI}$, $10{,}000$-trial $\bar{\rho}=0.9880$; $\rho_{\text{partial}}=0.5477$, $p=3.68\times 10^{-7}$; $\text{VIF}=1.84$), and catalog zone-map "
        r"selectivity $\hat{\sigma}(q)$ ($0.172\text{ ms}$ cached, $0.413\text{ ms}$ cold footer) without query compilation. Layers~2--4 couple physical PCIe/memory "
        r"crossover models with pluggable cost predictors (Ridge, shallow trees, $k$-NN, in-fold rules, and multi-output $\mathcal{R}_{\text{GBR-9D}}$) governed "
        r"by $90\%$ conformal intervals ($\text{ECE}=1.78\%$), a $12\%$ PCIe guard, and memory-constrained Pareto dispatch across SQLite~3, BigQuery, "
        r"DuckDB, Polars, DataFusion, and an RTX~2050 GPU." "\n\n"
        r"\textbf{Results:} Evaluated across $163$ queries ($75$ factorial TPC-H, $22$ Q1--Q22, $33$ JOB, $13$ SSB, $20$ ClickBench) under $25$-seed "
        r"LOSO-CV ($\text{SF}=1\text{ GB}$), unconstrained point-estimate routers (Ridge $49.12\text{ ms}$, Shallow DT $51.35\text{ ms}$, $k$-NN $52.84\text{ ms}$, "
        r"In-Fold Rules $53.07\text{ ms}$) minimize mean latency at high P95/P99 regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas "
        r"conformal AST-Route ($54.92\text{ ms}$ warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1, $76.00\%$ Top-2, $34.98\%$ mean regret, "
        r"$1.381\text{ J}$ estimated energy) compresses P95/P99 regret to $114.11\%/218.40\%$ (vs.\ $257.31\%$ Always GPU) and halves $>100\%$ regret "
        r"failures ($14.67\%\rightarrow 6.67\%$) across heterogeneous backends (Polars $82.47\text{ ms}$, DataFusion $96.30\text{ ms}$, DuckDB $183.74\text{ ms}$ "
        r"with active energy $6.40\text{ J}$ DuckDB, $3.17\text{ J}$ Polars vs.\ SQLite~3 $8417.33\text{ ms}$; RTX~2050 $56.96\text{ ms}$ cold [$1.45\times$ vs.\ Polars, "
        r"$3.23\times$ vs.\ DuckDB] and $22.25\text{ ms}$ warm [$3.71\times$ vs.\ Polars, $8.22\times$ kernel]). Under seed-cluster ($K=25$, $m=5$ Holm-adjusted) "
        r"Wilcoxon tests, AST-Route beats DuckDB ($3.35\times$, $p_{\text{Holm,seed}}=5.45\times 10^{-4}$), DataFusion ($1.75\times$, $p_{\text{Holm,seed}}=0.0041$), "
        r"and Polars ($1.50\times$, $p_{\text{Holm,seed}}=0.0171$), beats Rule-Based on paired queries ($N=75$, unadjusted $p_{\text{raw,query}}=0.0187$), and "
        r"matches Always GPU in mean latency ($p_{\text{raw,seed}}=0.184$ across $K=25$ seeds; $p_{\text{raw,query}}=0.142$ across $N=75$ queries). Standalone "
        r"CPU-only orchestration ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a $16$-core node, "
        r"while scaling across unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), Q1--Q22 ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), SSB ($92.31\%$ Top-1, "
        r"$12.15\text{ ms}$), ClickBench ($80.00\%$ Top-1, $24.18\text{ ms}$), JOB ($6.75\text{ ms}$), $C=1\text{--}32$ concurrency ($154.44\text{ QPS}$), and "
        r"$\text{SF}=1\text{--}100\text{ GB}$ ($3.35\times\text{--}7.61\times$), whereas LOTO-CV ($85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) bounds unseen-operator transfer." "\n\n"
        r"\textbf{Conclusion:} AST-Route delivers lightweight, risk-aware multi-engine resource orchestration across local CPU/GPU and serverless cloud "
        r"infrastructures when training workloads span target SQL operator families (\url{https://github.com/preecha-msu/bigquery-olap-benchmark})." "\n\n"
        r"\noindent\textbf{Keywords:} Intelligent query routing; Heterogeneous big data systems; AST-aware cost prediction; Resource orchestration; "
        r"Conformal risk control; Multi-objective OLAP optimization" "\n"
        r"\end{abstract}"
    )

    abs_replaced = False
    if "\\begin{abstract}" in tex and "\\end{abstract}" in tex:
        tex = re.sub(
            r"\\begin\{abstract\}.*?\\end\{abstract\}(?:\s*\\noindent\\textbf\{Keywords:\}[^\n]*)?",
            lambda _: new_abstract_and_keywords,
            tex,
            count=1,
            flags=re.DOTALL,
        )
        abs_replaced = True

    abs_words = count_plain_words(new_abstract_and_keywords)

    # ==========================================================================
    # ZONE C: SECTION 1 CONTRIBUTIONS -> EXPLICIT 3 CORE JBD INNOVATION PILLARS
    # ==========================================================================
    sec1_pos = tex.find("\\section{Introduction}")
    sec2_pos = tex.find("\\section{Related Work")
    contrib_replaced = False

    if sec1_pos != -1 and sec2_pos != -1:
        intro_block = tex[sec1_pos:sec2_pos]
        enum_start = intro_block.find("\\begin{enumerate}")
        enum_end = intro_block.rfind("\\end{enumerate}")
        if enum_start != -1 and enum_end != -1:
            three_pillars_intro_lead = (
                r"To position our work clearly within big data systems architecture rather than standalone engine benchmarking, "
                r"the primary contributions of this study are organized around three core pillars:" "\n"
            )
            three_pillars_enum = (
                r"\begin{enumerate}" "\n"
                r"\item \textbf{Big Data Systems Innovation --- Cross-Paradigm Query Management Across Heterogeneous Infrastructure:} "
                r"We design a unified, zero-execution query routing plane across four distinct computing paradigms---row-store baselines, "
                r"serverless cloud MPP warehouses (Google BigQuery with $10\text{ MiB}$ minimum-floor scan billing), in-process C++ vectorized "
                r"DBMSs (DuckDB), Rust Apache Arrow streaming engines (Polars, DataFusion), and discrete CUDA GPUs---grounded in a de-confounded "
                r"$163$-query multi-suite testbed ($75$ AST-stratified factorial queries across $25$ canonical seeds $\times$ $3$ selectivity horizons "
                r"V1--V3 in tiers L1--L5, plus $22$ official TPC-H, $33$ IMDB JOB, $13$ SSB, and $20$ ClickBench queries). Rather than treating "
                r"engine profiling as an end in itself, stripping SQL planning overhead ($183.74 - 0.87 = 182.87\text{ ms}$ pure DuckDB CPU kernel "
                r"vs.\ $22.25\text{ ms}$ in-VRAM GPU kernel) and pairing physical NVML GPU board sensors with TDP-calibrated CPU package energy "
                r"estimates establish the exact PCIe transfer crossover laws and memory walls needed to govern automated cross-engine dispatch "
                r"under $100\%$ real-execution telemetry." "\n\n"
                r"\item \textbf{Intelligent Resource Orchestration --- Pre-Execution AST Structural Profiling ($\text{CSCI}$) and Multi-Output Cost Prediction:} "
                r"We formulate a lightweight pre-execution $9\text{D}$ feature representation ($\mathbf{x}_q \in \mathbb{R}^{9}$ for fixed-scale routing, "
                r"extended with $\log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})$ to $\mathbf{x}_q^{\text{scale}} \in \mathbb{R}^{10}$ across scales and $\mathbb{R}^{12}$ "
                r"with catalog string/skew sketches) extracted in $3.65\text{ ms}$ ($3.476\text{ ms}$ SQLGlot AST parsing plus $0.172\text{ ms}$ cached "
                r"or $0.413\text{ ms}$ cold NVMe Parquet footer lookup; selectivity $\text{MAE}=3.84\%$, median $\text{Q-Error}=1.092$) without compiling "
                r"physical operator plans or touching data pages. Construct validation confirms correlation with AST node count ($\rho=0.8744$), "
                r"partial correlation with GPU speedup controlling for selectivity ($\rho_{\text{partial}}=0.5477$, $p=3.68\times 10^{-7}$), $10{,}000$-trial "
                r"perturbation stability ($\bar{\rho}=0.9880$ at $\pm 20\%$ and $\bar{\rho}=0.9555$ at $\pm 50\%$), and low tree-size multicollinearity "
                r"($\text{VIF}=1.84$ against $[D(q), N(q)]$), where $\text{CSCI}(q)$ acts as an oblique hyperplane projection reducing GBDT split nodes "
                r"by $-28.2\%$ and routing regret by $-9.22\%$ ($p=0.0038$). In Layer~3, multi-output regressors predict heterogeneous latency, "
                r"estimated device energy, and working-set memory in $497.2\,\mu\text{s}$ (median Q-Error $1.007\text{--}1.172$, $\log\text{-}R^{2}=0.815\text{--}0.912$), "
                r"enabling both heterogeneous CPU+GPU+Cloud dispatch and standalone CPU-only orchestration ($M_{\text{gpu\_max}}=0$: $74.67\%$ Top-1, "
                r"$69.12\text{ ms}$ on the primary node; $86.67\%$ Top-1, $62.05\text{ ms}$ on a $16$-core workstation)." "\n\n"
                r"\item \textbf{Scalable and Risk-Aware Execution --- Conformal Tail-SLA Control, Memory/FinOps Pareto, Concurrency, and Multi-Scale Resilience:} "
                r"Under $25$-seed LOSO-CV ($\text{SF}=1\text{ GB}$), AST-Route exposes a tunable operational trade-off: whereas unconstrained point-estimate "
                r"routers using Ridge ($49.12\text{ ms}$), shallow trees ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), or in-fold rules ($53.07\text{ ms}$) "
                r"incur severe P95/P99 tail regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), our conformal risk-aware $\mathcal{R}_{\text{GBR-9D}}$ "
                r"($90\%$ conformal intervals, $\text{ECE}=1.78\%$, $12\%$ PCIe guard) trades $3.6\text{--}5.8\text{ ms}$ ($54.92\text{ ms}$ warm, $55.16\text{ ms}$ "
                r"cold footer, $57.33\%$ Top-1, $76.00\%$ Top-2) to compress P95/P99 tail regret to $114.11\%/218.40\%$ and halve $>100\%$ regret failures "
                r"($14.67\%\rightarrow 6.67\%$). Coupling $\hat{\sigma}(q)\cdot\mathcal{B}_{\text{catalog}}$ enforces strict VRAM/DRAM feasibility filtering "
                r"($0\%$ OOM violations) across latency, estimated device energy ($1.381\text{ J/query}$), and marginal BigQuery scan charges ($\$6.25/\text{TiB}$), "
                r"significantly beating static CPU engines under $K=25$ seed-cluster Holm inference ($1.50\times\text{--}3.35\times$, $p_{\text{Holm,seed}}\le 0.0171$) "
                r"and static rule-based routing ($p_{\text{raw,query}}=0.0187$ across $N=75$ queries) while matching Always GPU in mean latency ($54.92$ vs.\ "
                r"$56.96\text{ ms}$, $p_{\text{raw,seed}}=0.184$ across $K=25$ seeds; $p_{\text{raw,query}}=0.142$ across $N=75$ queries) at less than half the tail regret. "
                r"Finally, AST-Route sustains $3.35\times\text{--}7.61\times$ speedups across $\text{SF}\in\{1,10,30,100\}\text{ GB}$ ($6.0\text{M--}600.1\text{M}$ rows), "
                r"$2.00\times\text{--}2.14\times$ concurrent throughput gains across $C=1\text{--}32$ streams (up to $154.44\text{ QPS}$) with bursty contention SLA "
                r"protection and fault-tolerant fallback, and robust transfer to locked unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), official TPC-H Q1--Q22 "
                r"($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), SSB ($92.31\%$ Top-1, $12.15\text{ ms}$), ClickBench ($80.00\%$ Top-1, $24.18\text{ ms}$), "
                r"and string-skewed JOB ($6.75\text{ ms}$ Polars parity via $21.21\%$ abstention, elevated to $69.70\%\text{--}96.97\%$ Top-1 via $12\text{D}$ catalog sketches), "
                r"while LOTO-CV ($85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) transparently bounds unseen-operator extrapolation." "\n"
                r"\end{enumerate}"
            )

            # Replace lead sentence ("The primary contributions of this study are summarized as follows:") + enumerate block
            lead_anchor = "The primary contributions of this study are summarized as follows:"
            if lead_anchor in intro_block[:enum_start]:
                rel_lead = intro_block.rfind(lead_anchor, 0, enum_start)
                new_intro_block = intro_block[:rel_lead] + three_pillars_intro_lead + three_pillars_enum + intro_block[enum_end + len("\\end{enumerate}"):]
            else:
                new_intro_block = intro_block[:enum_start] + three_pillars_intro_lead + three_pillars_enum + intro_block[enum_end + len("\\end{enumerate}"):]

            tex = tex[:sec1_pos] + new_intro_block + tex[sec2_pos:]
            contrib_replaced = True

    # ==========================================================================
    # ZONE D: SECTION 4 OPENING ROADMAP (EXPLICITLY PREVENTS "JUST A BENCHMARK" VIEW)
    # ==========================================================================
    sec4_roadmap = (
        r"\noindent\textbf{Evaluation Organization (From Physical Crossover Laws to Intelligent Query Orchestration).} "
        r"Rather than presenting an isolated engine bake-off, our experimental evaluation is structured in two linked stages "
        r"reflecting AST-Route's architecture: Sections~4.1--4.5 first establish the physical storage, serverless slot-pruning, "
        r"PCIe transfer ($\Phi_{\text{PCIe}}$), SIMD vectorization, and device-energy crossover laws that explain why static "
        r"single-engine assignment fails and directly parameterize Layer~2's analytical guards; Sections~4.6--4.9 then evaluate "
        r"AST-Route's intelligent cost-prediction accuracy, conformal tail-risk control, memory/FinOps Pareto orchestration, "
        r"cross-schema generalization, concurrent/bursty SLA resilience, and $\text{SF}=1\text{--}100\text{ GB}$ scaling." "\n\n"
    )

    roadmap_marker = "Evaluation Organization (From Physical Crossover Laws to Intelligent Query Orchestration)."
    if roadmap_marker not in tex:
        tex = re.sub(
            r"(\\section\{Experimental Validation and Empirical Analysis\}\s*)",
            lambda m: m.group(1) + "\n" + sec4_roadmap,
            tex,
            count=1,
        )

    # ==========================================================================
    # ZONE E: SECTION 6 CONCLUSION -> ALIGN 3 SYNTHESIS PILLARS (i), (ii), (iii)
    # ==========================================================================
    tex = tex.replace(
        "(i) Incremental Value of CSCI and Explicit Latency-Tail Risk Control:",
        r"(i) \emph{Intelligent AST-Aware Resource Orchestration and Tail-Risk Control}:",
    )
    tex = tex.replace(
        r"(i) \textbf{Incremental Value of CSCI and Explicit Latency-Tail Risk Control}:",
        r"(i) \textbf{Intelligent AST-Aware Resource Orchestration and Tail-Risk Control}:",
    )
    tex = tex.replace(
        "(ii) Operator-Bounded Cross-Workload Transfer:",
        r"(ii) \emph{Big Data Systems Generalization Across Heterogeneous Schemas and Infrastructures}:",
    )
    tex = tex.replace(
        r"(ii) \textbf{Operator-Bounded Cross-Workload Transfer}:",
        r"(ii) \textbf{Big Data Systems Generalization Across Heterogeneous Schemas and Infrastructures}:",
    )
    tex = tex.replace(
        "(iii) Memory-Constrained Multi-Scale and Energy-Aware Dispatch:",
        r"(iii) \emph{Scalable, Memory-Safe, and Multi-Objective Execution}:",
    )
    tex = tex.replace(
        r"(iii) \textbf{Memory-Constrained Multi-Scale and Energy-Aware Dispatch}:",
        r"(iii) \textbf{Scalable, Memory-Safe, and Multi-Objective Execution}:",
    )

    # ==========================================================================
    # HARD GUARD: VERIFY ALL 18 CANONICAL METRICS REMAIN INTACT BEFORE SAVING
    # ==========================================================================
    required_tokens = [
        "57.33\\%", "76.00\\%", "34.98\\%", "114.11\\%", "257.31\\%", "1.381",
        "183.74", "82.47", "96.30", "56.96", "54.92",
        "6.40", "3.17", "3.73", "1.97", "1.58", "0.61",
        "73.33\\%", "47.01", "68.18\\%", "90.91\\%", "16.98", "40.91\\%", "27.33",
    ]
    missing_tokens = [tk for tk in required_tokens if tk not in tex]
    if missing_tokens:
        raise AssertionError(f"[SAFETY ABORT] Reframing accidentally omitted canonical tokens: {missing_tokens}")

    # Write directly to target manuscript & mirrors
    TARGET_TEX.write_text(tex, encoding="utf-8")
    for mirror in (
        MANUSCRIPT_DIR / "main.tex",
        MANUSCRIPT_DIR / "overleaf_ready_single_file.tex",
        MANUSCRIPT_DIR / "bigquery_olap_benchmark_V6.tex",
        MANUSCRIPT_DIR / "bigquery_olap_benchmark_V6_2.tex",
        MANUSCRIPT_DIR / "bigquery_olap_benchmark_V6_3.tex",
        PROJECT_ROOT / "main.tex",
    ):
        mirror.write_text(tex, encoding="utf-8")

    print("[COMPILE] Running pdflatex -interaction=nonstopmode -halt-on-error (2 passes) on bigquery_olap_benchmark_V2.tex ...")
    for pass_i in (1, 2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MANUSCRIPT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"[LATEX ERROR Pass {pass_i}]:")
            print("\n".join(proc.stdout.splitlines()[-30:]))
            raise RuntimeError("pdflatex compilation failed")

    pdf_v2 = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf"
    for alias in ("bigquery_olap_benchmark_V6_3.pdf", "bigquery_olap_benchmark_V6_4.pdf", "main.pdf"):
        try:
            shutil.copy2(pdf_v2, MANUSCRIPT_DIR / alias)
        except PermissionError:
            print(f"[NOTE] {alias} is currently open in a PDF viewer; close viewer to overwrite {alias} (V2.pdf is updated!).")

    zip_out = MANUSCRIPT_DIR / "overleaf_V2_submission_bundle.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
        zf.write(TARGET_TEX, arcname="main.tex")
        bib_p = MANUSCRIPT_DIR / "references.bib"
        if bib_p.exists():
            zf.write(bib_p, arcname="references.bib")
        if FIGURES_DIR.exists():
            for fg in sorted(FIGURES_DIR.glob("*")):
                if fg.is_file():
                    zf.write(fg, arcname=f"figures/{fg.name}")

    log_txt = (MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.log").read_text(encoding="utf-8", errors="ignore")
    undef_refs = len(re.findall(r"LaTeX Warning: Reference .* undefined", log_txt))

    # Save SHA-256 framing audit log
    audit_df = pd.DataFrame([
        {"Zone": "Abstract_Orchestration_Lead", "Applied": abs_replaced, "Abstract_Plain_Words": abs_words},
        {"Zone": "JBD_Keywords_Added", "Applied": "Intelligent query routing; Heterogeneous big data systems" in tex, "Abstract_Plain_Words": abs_words},
        {"Zone": "Sec1_Three_Core_Pillars", "Applied": contrib_replaced and ("Big Data Systems Innovation" in tex) and ("Intelligent Resource Orchestration" in tex) and ("Scalable and Risk-Aware Execution" in tex), "Abstract_Plain_Words": abs_words},
        {"Zone": "Sec4_Systems_Evaluation_Roadmap", "Applied": roadmap_marker in tex, "Abstract_Plain_Words": abs_words},
        {"Zone": "Sec6_Conclusion_Three_Pillars", "Applied": "Scalable, Memory-Safe, and Multi-Objective Execution" in tex, "Abstract_Plain_Words": abs_words},
    ])
    p_log = RESULTS_DIR / "jbd_3pillars_positioning_audit_log.csv"
    audit_df.to_csv(p_log, index=False, encoding="utf-8")

    return {
        "audit_df": audit_df,
        "p_log": p_log,
        "sha_log": sha256_file(p_log),
        "sha_tex": sha256_file(TARGET_TEX),
        "sha_pdf": sha256_file(pdf_v2),
        "pdf_size": pdf_v2.stat().st_size,
        "undef_refs": undef_refs,
    }


def main() -> None:
    print("=" * 108)
    print("JOURNAL OF BIG DATA POSITIONING UPGRADE: REFRAMING CONTRIBUTIONS AROUND THE 3 CORE ORCHESTRATION PILLARS")
    print("=" * 108)

    res = reframe_manuscript()
    print("\n[3-PILLAR POSITIONING VERIFICATION TABLE]")
    print(res["audit_df"].to_string(index=False))
    print("-" * 108)
    print(f"  * Positioning Log CSV : {res['p_log'].name} | SHA-256: {res['sha_log']}")
    print(f"  * Updated V2.tex      : {TARGET_TEX.name} | SHA-256: {res['sha_tex']}")
    print(f"  * Compiled V2.pdf     : bigquery_olap_benchmark_V2.pdf ({res['pdf_size']:,} B | UndefRefs={res['undef_refs']}) | SHA-256: {res['sha_pdf']}")
    print("=" * 108)


if __name__ == "__main__":
    main()
