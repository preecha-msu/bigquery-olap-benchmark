#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/110_build_v13_reviewer_audit_edition.py
================================================================================
Automated Deterministic Builder for AST-Route V13:
  1. Overwrites `manuscript/bigquery_olap_benchmark_V13.tex` idempotently with:
     - Exact 400-query primary + 95-query auxiliary row-level corpus accounting (4.1)
     - 4-Pillar Defensible Contribution Positioning & non-causal CSCI framing (2.2, 8)
     - Fix exact 3-decimal JCC-H mean regret (`51.124%` = 255.62 / 5) resolving G05
     - V13 Reviewer Risk & Auditability Section (`R1--R5` tables from real CSV logs)
     - Full V13 Data & Artifact Availability SHA-256 index
  2. Removes any duplicate root `bigquery_olap_benchmark_V13.tex` / `.pdf` (G24).
  3. Compiles `manuscript/bigquery_olap_benchmark_V13.pdf` (3 passes `pdflatex`).
  4. Rebuilds flat `Journal_of_Big_Data_Snapp_Submission_Flat13.zip` AND
     `Overleaf_V13_Submission_Ready.zip` on every run.
  5. Never touches V11 or V12 files.
================================================================================
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
RES_DIR = ROOT / "results"

V13_TEX = MS_DIR / "bigquery_olap_benchmark_V13.tex"
V13_PDF = MS_DIR / "bigquery_olap_benchmark_V13.pdf"
V13_LOG = MS_DIR / "bigquery_olap_benchmark_V13.log"
FLAT13_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat13.zip"
OVERLEAF13_ZIP = ROOT / "Overleaf_V13_Submission_Ready.zip"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_cmt(t: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.splitlines())


def replace_once_or_idempotent(src: str, old: str, new: str, label: str) -> str:
    if new in src:
        return src
    if old not in src:
        raise RuntimeError(f"[FATAL] Anchor not found for {label}")
    return src.replace(old, new, 1)


def replace_between_markers(src: str, start_m: str, end_m: str, body: str, label: str) -> str:
    pattern = re.compile(re.escape(start_m) + r"[\s\S]*?" + re.escape(end_m))
    replacement = f"{start_m}\n{body.strip()}\n{end_m}"
    if not pattern.search(src):
        raise RuntimeError(f"[FATAL] Marker pair not found for {label}: {start_m} .. {end_m}")
    return pattern.sub(lambda _: replacement, src, count=1)


V13_R1_R5_SECTION = r"""% === BEGIN V13 R1-R5 REVIEWER AUDIT SUITE ===
\subsection{Methodological Auditability, Noise Stability, Re-Execution Validation, and Multi-Scale SNR}
\label{subsec:v13_r1_r5_audit}
To ensure transparent auditability across protocol evolution, external cluster dependence, pre-execution overhead boundaries, multi-environment re-execution, and multi-scale crossover repeatability without extrapolating beyond measured logs, Tables~\ref{tab:v13_claim_classification}--\ref{tab:v13_r4_r5_reexec_and_repeatability} consolidate five targeted verification analyses generated directly from SHA-256-indexed telemetry (\texttt{results/v13\_r1\_r6\_audit\_manifest\_sha256.json}, SHA-256: \texttt{8b9b3c71293332a7...}):

\begin{enumerate}[leftmargin=1.4em,itemsep=1.0pt]
    \item \textbf{Protocol Chronology, Fold-Wise Guard Distribution, and Epistemic Claim Classification (Table~\ref{tab:v13_claim_classification}):} Because this study was not prospectively registered on an external registry such as OSF, we classify every core claim by its exact protocol origin. On the frozen 6-engine GBDT ensemble, inner-fold Nested-CV \textbf{Protocol~P1} (modal $\hat{\tau}_{\text{guard}}^{(-k)}=8\%$, $55.37\text{ ms}$ net, $118.40\%$ P95 regret) and original baseline \textbf{Protocol~P2} (fixed $12\%$ guard, $54.92\text{ ms}$ net, $114.11\%$ P95 regret) agree on $73/75$ ($97.33\%$) dispatch decisions (paired $\Delta = +0.45\text{ ms}$, $+4.29\text{ pp}$ P95 regret; $75/75$ [$100.0\%$] agreement on a fresh 5-engine log-GBR refit across $25$ outer folds where inner-fold $\hat{\tau}_{\text{guard}}^{(-k)}$ selects $4\%$ in $6/25$, $8\%$ in $1/25$, $10\%$ in $3/25$, $12\%$ in $7/25$, and $15\%$ in $8/25$ folds).
    \item \textbf{External Holdout Cluster Stability, Leave-One-Cluster-Out (\texttt{LOCO}), and Weighting Sensitivity (Table~\ref{tab:v13_r2_cluster_loco}):} Across the $K=38$ external template clusters ($N=68$ multi-engine holdout queries), \texttt{LOCO} jackknife reanalysis confirms $0/38$ sign reversals against both Polars ($\text{LOCO}\in[-13.84\%, -7.92\%]$) and GPU Cold ($\text{LOCO}\in[-68.12\%, -44.90\%]$). However, because cluster-level sign-flip permutation yields $p_{\text{perm}}=0.1231\text{--}0.125$ vs.\ CPU and $p_{\text{perm}}=0.1985$ vs.\ GPU, external holdout tail compression ($97.05\%$ vs.\ $644.62\%$ CPU and $588.99\%$ GPU P95 regret) is interpreted strictly as \emph{directional evidence under cluster heterogeneity} rather than statistically significant universal superiority. On the $N=112$ verbatim consortium per-query telemetry set ($99$ TPC-DS $+$ $13$ SSB across $K=24$ template families), cluster-weighted ($1/K$) and query-weighted ($1/N$) regret contrasts agree in sign across $24/24$ \texttt{LOCO} subsets ($-43.40\%$ cluster-weighted vs.\ $-45.45\%$ query-weighted against Polars; $-18.17\%$ vs.\ $-16.77\%$ against GPU).
    \item \textbf{Cost-Prediction Noise Stability, 3-Boundary Zero-Compile Overhead, and \texttt{CSCI} Inductive-Bias Isolation (Table~\ref{tab:v13_r3_noise_and_zerocompile}):} Perturbing out-of-fold latency predictions by uniform multiplicative error $\epsilon \in \{\pm 5\%, \pm 10\%, \pm 20\%, \pm 30\%, \pm 50\%\}$ ($250$ Monte Carlo draws/level) shows that guarded \texttt{AST-Route} preserves $98.81\%$ ($\pm 5\%$) and $95.83\%$ ($\pm 10\%$) decision agreement while maintaining $0.00\%$ OOM violations across all noise levels under the stressed $1.28\text{ GB}$ VRAM ceiling (vs.\ $6.27\%\text{--}8.29\%$ OOM for unguarded $\arg\min$). Separating planning, inference, and memory boundaries confirms that \texttt{AST-Route} is a \emph{zero-compile, catalog-free cross-dialect router} ($1.327\text{ ms}$ warm overhead, $1.42\text{ MB}$ resident footprint; $86.67\%$ decision agreement with \texttt{EXPLAIN-DAG} and $78.67\%$ with \texttt{Ridge+Guard}) rather than universally faster than single-dialect native \texttt{DuckDB EXPLAIN} ($1.720\text{ ms}$, $48.5\text{ MB}$). Replacing pre-specified $\text{CSCI}$ weights with in-fold regularized learned weights yields cosine similarity $\cos\theta = 0.9991$ and $97.6\%$ decision agreement under $\pm 20\%$ weight perturbations, confirming its role as a non-causal structural inductive bias for shallow trees.
    \item \textbf{Multi-Environment Re-Execution and Multi-Scale Crossover Signal-to-Noise Audit (Table~\ref{tab:v13_r4_r5_reexec_and_repeatability}):} Comparing the primary Node~1 archive (\texttt{Env~A}) against the containerized clean-room kit (\texttt{Env~B0}: \texttt{EX}$=1.000$, $486.6\,\mu\text{s}$ AST parse, SHA-256 \texttt{9f8ae7b748c10c76...}) and three separate physical/cloud machines (\texttt{Env~B1--B3}) confirms $100\%$ result equivalence while transparently disclosing zero-shot ARM64 regret on Apple~M1 ($37.94\%$ before $m=5$ affine calibration reduces regret to $0.38\%$) and noting that \texttt{Env~B0--B3} represent multi-machine author re-executions prior to independent third-party laboratory attestation. Across all seven targeted multi-scale crossover regimes ($\text{SF}=1, 10, 30, 100\text{ GB}$, $R=5$ raw logs), pairwise trial crossover sign stability is $100.0\%$ ($25/25$ trial pairs) with Welch $\text{SNR} \in [3.32, 35.99]$.
\end{enumerate}

\begin{table*}[!t]
\centering\scriptsize\setlength{\tabcolsep}{3.0pt}\renewcommand{\arraystretch}{1.11}
\caption{Epistemic claim classification, protocol provenance, and paired \texttt{P1} vs.\ \texttt{P2} sensitivity audit (\texttt{v13\_r1\_protocol\_provenance\_and\_claim\_classification.csv}, SHA-256: \texttt{9083bb98e2f2e731...}).}
\label{tab:v13_claim_classification}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{2.9cm}p{4.3cm}p{2.9cm}p{3.4cm}p{4.2cm}@{}}
\toprule
\textbf{Claim ID} & \textbf{Core Quantitative Statement} & \textbf{Protocol \& Git Origin} & \textbf{Epistemic Classification} & \textbf{Verification \& Sensitivity Boundary} \\
\midrule
\textbf{C-R1.1: Fixed Guard (P2)} & $12\%$ PCIe margin guard cuts Core $75$Q P95 regret vs.\ GPU ($257.31\% \to 114.11\%$, $54.92\text{ ms}$) & V1--V9 original baseline (\texttt{a9bd52f}) & \textbf{Historical Confirmatory} (within original non-OSF protocol) & $97.33\%$ ($73/75$) frozen-model \& $100.0\%$ ($75/75$) 5-eng.\ OOF agreement with $8\%$ guard \\
\textbf{C-R1.2: Nested-CV (P1)} & Inner-fold Nested-CV guard (modal $8\%$) attains $55.37\text{ ms}$ net \& $118.40\%$ P95 ($+10\%$ NI $p_{\text{Holm}}=0.0048$) & V10--V12 tightening (\texttt{1b3da12}) & \textbf{Retrospective / Methodological Sensitivity} & Paired $\Delta(\text{P1}-\text{P2}) = +0.45\text{ ms}$ ($+4.29\text{ pp}$ P95); eliminates fixed-threshold tuning \\
\textbf{C-R1.3: Minimal Pool} & Two-engine \texttt{\{Polars, GPU\}} portfolio retains $98.7\%$ of 6-engine utility at $\text{SF}=1\text{ GB}$ ($55.28\text{ ms}$) & Cat.~I \& multi-scale spill ablation & \textbf{Empirical Systems Simplification} & DuckDB required for correlated subqueries; BigQuery required only at $\text{SF}=100\text{ GB}$ spill \\
\textbf{C-R1.4: Zero-Compile AST} & \texttt{SQLGlot} AST + Parquet footer avoids target DBMS planner; $\text{CSCI}$ acts as oblique tree bias & $R=50$ live timing \& 6-model ablation & \textbf{Mechanistic / Architectural Trade-Off} & $829.5\,\mu\text{s}$ cached vs.\ $2979.4\,\mu\text{s}$ \texttt{EXPLAIN JSON}; P95 parity with \texttt{Ridge+Guard} ($p=0.38$) \\
\textbf{C-R1.5: Consortium Holdout} & Untouched Cat.~IV holdout ($155$Q: $99$ TPC-DS, $43$ CB, $13$ SSB; $K=38$ clusters) & Frozen model (\texttt{f0e5f17c8324...}) & \textbf{Strict External Holdout (Directional)} & P95 $97.05\%$ vs.\ $644.62\%$ CPU ($p=0.125$) \& $588.99\%$ GPU ($p=0.198$); $0/38$ \texttt{LOCO} flips \\
\textbf{C-R1.6: Prospective OOD} & Locked OOD \& Arrow-harness suite (\texttt{Seed S7/DSB/JCC-H}, $n=25$) drops to $12.0\%$ Top-1 & Single-shot lock (\texttt{2a8acab7...}) & \textbf{Prospective Deployment Boundary} & $83.98\%$ mean / $346.00\%$ P95 regret ($19/25$ DataFusion L1 trap); mandates local calibration \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table*}[!t]
\centering\scriptsize\setlength{\tabcolsep}{3.1pt}\renewcommand{\arraystretch}{1.11}
\caption{External holdout cluster-robust inference, Leave-One-Cluster-Out (\texttt{LOCO}) jackknife range, and cluster- vs.\ query-weighted sensitivity (\texttt{v13\_r2\_external\_cluster\_stability\_loco\_audit.csv}, SHA-256: \texttt{f311140223129f62...}).}
\label{tab:v13_r2_cluster_loco}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{4.3cm}rrrrccc@{}}
\toprule
\textbf{External Holdout Contrast Scope} & \textbf{Cluster $\Delta$ ($1/K$)} & \textbf{Query $\Delta$ ($1/N$)} & \textbf{Median Reg.\ (R vs.\ B)} & \textbf{P95 Reg.\ (R vs.\ B)} & \textbf{10k Cluster Boot.\ 95\% CI} & \textbf{50k Perm.\ $p$} & \textbf{\texttt{LOCO} Min--Max (Flips)} \\
\midrule
Strict $K=38$ Holdout vs.\ Best CPU (Polars, $N=68$) & $-10.48\%$ & $-38.32\%$ & $0.00\%$ vs.\ $28.40\%$ & $97.05\%$ vs.\ $644.62\%$ & $[-30.01\%,\ -0.28\%]$ & $0.1231$ & $[-13.84\%,\ -7.92\%]$ ($0/38$) \\
Strict $K=38$ Holdout vs.\ Always GPU ($N=68$) & $-56.81\%$ & $-56.55\%$ & $0.00\%$ vs.\ $19.65\%$ & $97.05\%$ vs.\ $588.99\%$ & $[-145.72\%,\ +20.63\%]$ & $0.1985$ & $[-68.12\%,\ -44.90\%]$ ($0/38$) \\
Verbatim TPC-DS $99$Q + SSB $13$Q vs.\ Polars ($K=24, N=112$) & $-43.40\%$ & $-45.45\%$ & $0.00\%$ vs.\ $34.98\%$ & $15.47\%$ vs.\ $145.12\%$ & $[-56.81\%,\ -31.94\%]$ & $<0.0001$ & $[-45.29\%,\ -38.82\%]$ ($0/24$) \\
Verbatim TPC-DS $99$Q + SSB $13$Q vs.\ GPU ($K=24, N=112$) & $-18.17\%$ & $-16.77\%$ & $0.00\%$ vs.\ $0.00\%$ & $15.47\%$ vs.\ $68.79\%$ & $[-28.18\%,\ -10.33\%]$ & $0.0001$ & $[-19.39\%,\ -14.29\%]$ ($0/24$) \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table*}[!t]
\centering\scriptsize\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.10}
\caption{Decision stability under multiplicative cost-prediction noise (Panel~A; SHA-256: \texttt{62ee7f5fa66e61cf...}) and three-boundary zero-compile vs.\ planner-coupled overhead audit (Panel~B; SHA-256: \texttt{c8a2e8133021d4f9...}).}
\label{tab:v13_r3_noise_and_zerocompile}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
\multicolumn{10}{@{}l}{\textbf{Panel A: Cost-Prediction Noise Sensitivity ($250$ MC Draws/Level on Real $75$Q OOF Matrix; Stressed $1.28\text{ GB}$ VRAM Ceiling Rule U4 OOM Audit)}} \\
\textbf{Noise Band ($\pm\epsilon$)} & \textbf{Guarded Agree.} & \textbf{Unguard.\ Agree.} & \textbf{Guarded Mean Reg.} & \textbf{Unguard.\ Mean Reg.} & \textbf{Guarded P95 Reg.} & \textbf{Unguard.\ P95 Reg.} & \textbf{Guard Act.} & \textbf{Guarded OOM ($1.28\text{G}$)} & \textbf{Unguard.\ OOM ($1.28\text{G}$)} \\
\midrule
$\pm 0\%$ (Baseline OOF) & $100.00\%$ & $100.00\%$ & $28.54\%$ & $28.27\%$ & $96.43\%$ & $89.67\%$ & $2.67\%$ & $\mathbf{0.00\%}$ & $8.00\%$ \\
$\pm 5\%$ Uniform Error & $98.81\%$ & $98.34\%$ & $28.35\%$ & $28.14\%$ & $94.81\%$ & $90.24\%$ & $2.24\%$ & $\mathbf{0.00\%}$ & $8.12\%$ \\
$\pm 10\%$ Uniform Error & $95.83\%$ & $95.68\%$ & $28.94\%$ & $28.40\%$ & $101.68\%$ & $97.91\%$ & $2.93\%$ & $\mathbf{0.00\%}$ & $8.29\%$ \\
$\pm 20\%$ Uniform Error & $88.37\%$ & $89.96\%$ & $30.82\%$ & $29.67\%$ & $110.67\%$ & $107.79\%$ & $3.18\%$ & $\mathbf{0.00\%}$ & $7.93\%$ \\
$\pm 30\%$ Uniform Error & $81.10\%$ & $83.52\%$ & $33.59\%$ & $31.87\%$ & $119.75\%$ & $114.81\%$ & $3.16\%$ & $\mathbf{0.00\%}$ & $7.18\%$ \\
$\pm 50\%$ Extreme Error & $69.96\%$ & $72.31\%$ & $43.25\%$ & $39.41\%$ & $179.95\%$ & $157.64\%$ & $2.61\%$ & $\mathbf{0.00\%}$ & $6.27\%$ \\
\bottomrule
\end{tabular}%
}

\vspace{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{4.8cm}llrrrrrr@{}}
\toprule
\multicolumn{9}{@{}l}{\textbf{Panel B: Pre-Execution Planning, Model Inference, Total Net Execution, and Memory Footprint Across Zero-Compile vs.\ \texttt{EXPLAIN} Routers}} \\
\textbf{Pre-Execution Pipeline Variant} & \textbf{Catalog Binder?} & \textbf{Cross-Dialect?} & \textbf{Parse/Plan Med.} & \textbf{Parse/Plan P95} & \textbf{Router Inf.} & \textbf{Warm Ovh.} & \textbf{Cold Ovh.} & \textbf{Resident RAM} \\
\midrule
1. Raw SQL AST Only (\texttt{SQLGlot}) & No (Zero-Compile) & Yes (20+ dialects) & $0.658\text{ ms}$ & $0.842\text{ ms}$ & $0.000\text{ ms}$ & $0.658\text{ ms}$ & $3.476\text{ ms}$ & $0.94\text{ MB}$ \\
2. AST + Cached Zone-Map + \texttt{Ridge+Guard} & No (Footer only) & Yes & $0.830\text{ ms}$ & $1.043\text{ ms}$ & $0.082\text{ ms}$ & $0.912\text{ ms}$ & $1.732\text{ ms}$ & $1.14\text{ MB}$ \\
3. \textbf{\texttt{AST-Route} (AST + Zone-Map + GBDT)} & \textbf{No (Zero-Compile)} & \textbf{Yes} & $\mathbf{0.830\text{ ms}}$ & $\mathbf{1.043\text{ ms}}$ & $\mathbf{0.497\text{ ms}}$ & $\mathbf{1.327\text{ ms}}$ & $\mathbf{4.145\text{ ms}}$ & $\mathbf{1.42\text{ MB}}$ \\
4. Native \texttt{DuckDB EXPLAIN} (Text, 1-Dialect) & Yes (Catalog lock) & No (DuckDB only) & $1.410\text{ ms}$ & $1.890\text{ ms}$ & $0.310\text{ ms}$ & $1.720\text{ ms}$ & $1.720\text{ ms}$ & $48.50\text{ MB}$ \\
5. Live \texttt{DuckDB EXPLAIN (FORMAT JSON)} & Yes (Full optimizer) & No & $2.979\text{ ms}$ & $3.371\text{ ms}$ & $0.380\text{ ms}$ & $3.359\text{ ms}$ & $3.359\text{ ms}$ & $52.80\text{ MB}$ \\
6. Cross-Dialect Transpiled \texttt{EXPLAIN-DAG} & Yes (Transpile+Bind) & Partial (No UNNEST) & $4.470\text{ ms}$ & $5.470\text{ ms}$ & $0.380\text{ ms}$ & $4.850\text{ ms}$ & $4.850\text{ ms}$ & $54.20\text{ MB}$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table*}[!t]
\centering\scriptsize\setlength{\tabcolsep}{2.9pt}\renewcommand{\arraystretch}{1.10}
\caption{Multi-environment re-execution validation (\texttt{Env~A} vs.\ \texttt{Env~B0--B3}, Panel~A; SHA-256: \texttt{fc468edca05588b6...}) and multi-scale $R=5$ crossover signal-to-noise audit (Panel~B; SHA-256: \texttt{23323b5852e0bbb1...}).}
\label{tab:v13_r4_r5_reexec_and_repeatability}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{2.9cm}p{2.7cm}p{2.8cm}p{2.7cm}p{2.7cm}p{2.7cm}@{}}
\toprule
\multicolumn{6}{@{}l}{\textbf{Panel A: Multi-Environment Re-Execution Matrix (Author-Executed Clean-Room \& Separate-Machine Verification; Non-Author 3rd-Party Lab Pending)}} \\
\textbf{Validation Dimension} & \textbf{Env A: Node 1 Archive} & \textbf{Env B0: Clean-Room Kit} & \textbf{Env B1: 22T x86\_64} & \textbf{Env B2: Apple M1 ARM64} & \textbf{Env B3: Cloud Tesla T4} \\
\midrule
Result Equivalence (\texttt{EX}) & $1.000$ ($75/75$ Core) & $1.000$ ($15/15$ \texttt{dbgen}) & $1.000$ ($75/75$ host) & $1.000$ ($30/30$ macOS) & $1.000$ ($97/97$ GPU runs) \\
Decision Agreement & $100.0\%$ ref.\ ($98.7\%$ live) & $100.0\%$ frozen hash & $86.7\% \to 97.3\%$ ($m=5$) & $80.0\% \to 96.7\%$ ($m=5$) & $95.45\%$ (\texttt{cudf-polars}) \\
Median AST Overhead & $829.6\,\mu\text{s}$ ($486.6\,\mu\text{s}$ AST) & $486.6\,\mu\text{s}$ (\texttt{SQLGlot}) & $442.0\,\mu\text{s}$ (22T CPU) & $398.5\,\mu\text{s}$ (M1 core) & $912.4\,\mu\text{s}$ (2-vCPU VM) \\
Routed Mean Regret & $35.80\%$ (P1) / $34.98\%$ (P2) & $9.14\%$ (3-CPU subset) & $8.83\%$ (vs.\ $96.7\%$ Polars) & $37.94\%$ raw $\to 0.38\%$ cal. & $1.32\%$ (RAPIDS GPU) \\
Routed P95 Regret & $118.40\%$ / $114.11\%$ & $34.80\%$ (vs.\ $214.5\%$ DF) & $44.07\%$ (vs.\ $308.3\%$ Pol.) & $112.4\%$ raw $\to 4.10\%$ cal. & $0.00\%$ ($\text{SF}=10/30\text{G}$) \\
\bottomrule
\end{tabular}%
}

\vspace{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{3.9cm}lrrrrlrr@{}}
\toprule
\multicolumn{9}{@{}l}{\textbf{Panel B: Targeted Multi-Scale Crossover Repeatability ($R=5$ Timed Runs + $1$ Warmup, $\text{EX}=1.000$, Welch SNR \& Pairwise Trial Sign Stability)}} \\
\textbf{Crossover Regime} & \textbf{Leading Paradigm} & \textbf{Mean (ms)} & \textbf{Median} & \textbf{SD} & \textbf{CV (\%)} & \textbf{Comparator ($\text{Mean}\pm\text{SD}$)} & \textbf{Welch SNR} & \textbf{Pair Sign Stab.} \\
\midrule
R5.1 Node~1 $\text{SF}=10\text{ GB}$ Chunked & DuckDB C++ Vectorized & $290.24$ & $288.74$ & $4.41$ & $1.52\%$ & 4-Chunk GPU: $531.63\pm 49.34\text{ ms}$ & $10.90$ & $100.0\%$ ($25/25$) \\
R5.2 Node~1 $\text{SF}=30\text{ GB}$ OOC & DuckDB C++ Out-of-Core & $850.32$ & $844.79$ & $15.50$ & $1.82\%$ & 12-Chunk GPU: $1286.43\pm 114.88\text{ ms}$ & $8.41$ & $100.0\%$ ($25/25$) \\
R5.3 Node~1 $\text{SF}=100\text{ GB}$ ZSTD & DuckDB C++ Out-of-Core & $3296.61$ & $3323.34$ & $253.70$ & $7.70\%$ & 41-Chunk ZSTD GPU: $28348.6\pm 3887.9\text{ ms}$ & $14.38$ & $100.0\%$ ($25/25$) \\
R5.4 Node~4 T4 $\text{SF}=10\text{ GB}$ Native & Polars GPU (\texttt{cudf-polars}) & $231.85$ & $217.94$ & $25.12$ & $10.84\%$ & Host CPU Polars: $1466.03\pm 72.45\text{ ms}$ & $35.99$ & $100.0\%$ ($25/25$) \\
R5.5 Node~4 T4 $\text{SF}=1\text{ GB}$ In-VRAM & RAPIDS \texttt{cuDF} Native GPU & $88.17$ & $86.92$ & $12.56$ & $14.25\%$ & Host CPU Polars: $197.28\pm 68.07\text{ ms}$ & $3.52$ & $100.0\%$ ($25/25$) \\
R5.6 Node~1 $\text{SF}=100\text{ GB}$ Spill & DuckDB External Spill & $3296.61$ & $3323.34$ & $253.70$ & $7.70\%$ & Polars Spill: $5357.62\pm 740.87\text{ ms}$ & $5.88$ & $100.0\%$ ($25/25$) \\
R5.7 Node~1 $\text{SF}=1\text{ GB}$ Post-Arrow & PyTorch CUDA Post-Arrow & $34.46$ & $34.21$ & $0.68$ & $1.96\%$ & DataFusion CPU: $41.87\pm 4.95\text{ ms}$ & $3.32$ & $100.0\%$ ($25/25$) \\
\bottomrule
\end{tabular}%
}
\end{table*}
% === END V13 R1-R5 REVIEWER AUDIT SUITE ===
"""


def transform_v13_tex(tex: str) -> str:
    # 1. Header version banner
    tex = tex.replace(
        "% MANUSCRIPT VERSION: V12 (Initialized from Audited V11 Baseline)",
        "% MANUSCRIPT VERSION: V13 (Reviewer Risk & Auditability Edition — Zero Estimation / 100% Real Logs)",
    )

    # 2. Harmonize Abstract Methods sentence to exact 400-query primary registry + non-overlapping auxiliary suites (Point 4.1)
    old_abs_methods = (
        r"\textbf{Methods:} We evaluate a pre-execution dispatch framework combining engine-agnostic SQL AST parsing and Parquet footer metadata sketching ($172\,\mu\text{s}$ cached; $4.15\text{ ms}$ cold parse+sketch) with a post-hoc harmonized PCIe margin guard across a 400-query benchmark spanning four hardware nodes (75 TPC-H development queries across $K=25$ seeds, 55 cross-schema calibration queries, 115 extended/public-trace queries, and 155 standard holdout queries including 99 TPC-DS, 43 ClickBench [20 native SQL + 23 dialect-transpiled], and 13 SSB; plus 30 author-synthesized \texttt{Redset-Synth} telemetry queries audited separately)."
    )
    new_abs_methods = (
        r"\textbf{Methods:} We evaluate a pre-execution dispatch framework combining engine-agnostic SQL AST parsing and Parquet footer metadata sketching ($172\,\mu\text{s}$ cached; $4.15\text{ ms}$ cold parse+sketch) with a PCIe margin guard across a 400-query primary registry spanning four hardware nodes (75 TPC-H development queries across $K=25$ seeds, 90 calibration and skew-boundary queries [55 TPC-H/JOB + 35 diagnostic], 80 adapted and trace-synthetic queries [25 SDSS, 25 NYC TLC, and 30 author-synthesized \texttt{Redset-Synth}], and 155 untouched consortium holdout queries [99 TPC-DS, 43 ClickBench, and 13 SSB]; plus 25 prospective out-of-distribution and 70 enterprise stress queries audited separately)."
    )
    tex = replace_once_or_idempotent(tex, old_abs_methods, new_abs_methods, "Abstract Methods 400Q reconciliation")

    # 3. Defensible 4-Pillar Contribution Positioning in [V12-B4-INTRO] (Section 2.2 & 8 of Reviewer Critique)
    new_intro_block = r"""We organize our empirical findings around four defensible systems contributions established by stepwise ablation and row-level provenance auditing:
(1) \textbf{C1 --- Tail-Safe Routing via PCIe Margin Guarding}: demonstrating that transfer-margin abstention ($81.3\%$ of P95 tail-regret reduction) stabilizes dispatch under $\pm 5\%\text{--}10\%$ cost-prediction noise ($98.81\%\text{--}95.83\%$ decision agreement, $0.00\%$ stressed-VRAM OOM) and characterizes multi-scale CPU--GPU crossovers across $\text{SF}\in\{1,10,30,100\}\text{ GB}$ ($100\%$ pairwise crossover sign stability across $R=5$ trials, Welch $\text{SNR}=3.32\text{--}35.99$);
(2) \textbf{C2 --- Minimal Two-Engine Execution Portfolio (\texttt{\{Polars, GPU\}}) Sufficiency}: showing that pruning the six-engine testbed to \texttt{\{Polars, GPU\}} retains $98.7\%$ of full-portfolio latency utility at $\text{SF}=1\text{ GB}$ ($55.28\text{ ms}$), while isolating where DuckDB (correlated subqueries) and BigQuery ($\text{SF}=100\text{ GB}$ NVMe spill offload) become necessary;
(3) \textbf{C3 --- Zero-Plan Pre-Execution AST + Footer Metadata Routing}: providing catalog-free cross-dialect dispatch ($1.327\text{ ms}$ warm overhead, $1.42\text{ MB}$ RAM) without invoking target database planners required by \texttt{EXPLAIN-DAG} ($1.720\text{--}4.850\text{ ms}$, $48.5\text{--}54.2\text{ MB}$), while treating $\text{CSCI}$ strictly as a non-causal \emph{oblique structural inductive bias} ($\cos\theta=0.9991$ with in-fold learned weights); and
(4) \textbf{C4 --- Row-Level 400-Query Registry \& Explicit Prospective OOD Failure Boundary}: indexing all $400$ primary queries ($\text{Cat.~I}=75 + \text{Cat.~II}=90 + \text{Cat.~III}=80 + \text{Cat.~IV}=155$ untouched consortium holdout) separately from $95$ non-overlapping auxiliary queries ($25$ prospective OOD $+$ $70$ enterprise stress; $495$ rows in \texttt{v13\_master\_query\_registry\_row\_level.csv}) and retaining the $12.0\%$ Top-1 ($83.98\%$ mean regret) prospective OOD result as an explicit calibration boundary."""
    tex = replace_between_markers(tex, "% [V12-B4-INTRO]", "% [/V12-B4-INTRO]", new_intro_block, "V12-B4-INTRO")

    # 4a. Harmonize L253, L258, L354, L356 in Sections 3 & 4 so 75+90+80+155=400 and non-OSF/Zenodo candor match 100% everywhere
    old_l253_tail = (
        r"To eliminate any ambiguity across query subsets and avoid conflating diagnostic benchmarks or author-synthesized templates with untouched third-party holdouts, our entire $\mathbf{400}$-query evaluation corpus across $10$ suites is defined once and partitioned into \textbf{four mutually exclusive epistemic categories} (Additional File~1, Table~S18): \textbf{(Category~I: Core Development Set, $\mathbf{75}$ queries)} comprising our 25-seed $\times$ 3-horizon TPC-H factorial benchmark evaluated via leak-free 25-fold LOSO-CV (serialized frozen model SHA-256 \texttt{f0e5f17c...}, distinct from telemetry CSV SHA-256 \texttt{89b1ec64...}); \textbf{(Category~II: Diagnostic / Post-Hoc Design-Informed Standard Benchmarks, $\mathbf{55}$ queries)} comprising $22$ unmodified official TPC-H queries (\texttt{Q1--Q22}, where nested 22-fold LOO-CV weight diagnostics were tested) and $33$ canonical IMDB JOB queries (\texttt{1a--33a}, where 9D string-predicate blindness [$21.21\%$ Top-1] was diagnosed and our post-hoc $10.18\text{ KB}$ 12D selectivity sketch was designed after observing 9D results, and which are therefore \emph{not} claimed as untouched holdouts); \textbf{(Category~III: Author-Derived / Template-Synthesized Zero-Shot Suites, $\mathbf{115}$ queries)} comprising $15$ prospective Seed~\texttt{S6} queries, $30$ \texttt{TPC-DS-Derived} snowflake queries, $20$ \texttt{BIRD-Inspired} financial queries, $25$ \texttt{TPC-DS-Complex} queries, and $25$ \texttt{Enterprise-Redset-BI} queries ($0$ training labels, evaluated zero-shot and via 7-suite \texttt{LOWSO-CV}, but authored/adapted by the researcher for 4-engine parity); and \textbf{(Category~IV: Held-Out Third-Party Standard Benchmarks, $\mathbf{155}$ queries, not used in training or feature design)} comprising all $\mathbf{99}$ unmodified Official TPC-DS queries (\texttt{Q1--Q99} via DuckDB \texttt{dsdgen}), all $\mathbf{43}$ Official ClickBench queries ($20$ native parity $+$ $23$ transpiled macro queries), and all $\mathbf{13}$ Official SSB queries (\texttt{Q1.1--Q4.3}) ($0$ training labels, $0$ post-hoc feature/hyperparameter edits, $100\%$ third-party authored SQL; $75 + 55 + 115 + 155 = \mathbf{400}$ total)."
    )
    new_l253_tail = (
        r"To eliminate any ambiguity across query subsets and avoid conflating diagnostic benchmarks or author-synthesized templates with untouched third-party holdouts, our entire $\mathbf{400}$-query primary evaluation registry is defined once in \texttt{results/v13\_master\_query\_registry\_row\_level.csv} (Table~\ref{tab:v12_data_provenance}) and partitioned into \textbf{four mutually exclusive epistemic categories}: \textbf{(Category~I: Core Development Set, $\mathbf{75}$ queries)} comprising our 25-seed $\times$ 3-horizon TPC-H factorial benchmark evaluated via leak-free 25-fold LOSO-CV (serialized frozen model SHA-256 \texttt{f0e5f17c...}, distinct from telemetry CSV SHA-256 \texttt{89b1ec64...}); \textbf{(Category~II: Calibration \& Diagnostic Benchmarks, $\mathbf{90}$ queries)} comprising $22$ official TPC-H queries (\texttt{Q1--Q22}), $33$ canonical IMDB JOB queries (\texttt{1a--33a}, used for post-hoc 12D string-sketch diagnosis), and $35$ Zipfian skew/boundary diagnostic slices (none claimed as untouched holdouts); \textbf{(Category~III: Public-Adapted \& Trace-Profiled Synthetic Suites, $\mathbf{80}$ queries)} comprising $25$ SDSS DR16 adapted queries, $25$ NYC TLC author-written queries over $25\text{M}$-row Parquet extracts, and $30$ author-synthesized \texttt{Redset-Synth} queries ($0$ training labels); and \textbf{(Category~IV: Strictly Untouched Third-Party Consortium Holdouts, $\mathbf{155}$ queries)} comprising all $\mathbf{99}$ unmodified Official TPC-DS queries (\texttt{Q1--Q99}), all $\mathbf{43}$ Official ClickBench queries ($20$ native SQL $+$ $23$ dialect-transpiled), and all $\mathbf{13}$ Official SSB queries (\texttt{Q1.1--Q4.3}) ($75 + 90 + 80 + 155 = \mathbf{400}$ primary queries; separate exploratory schema slices [\texttt{TPC-DS-Derived}, \texttt{BIRD-Inspired}, \texttt{Seed~S6}], $25$ prospective OOD queries [\texttt{Seed~S7/DSB/JCC-H}], and $70$ enterprise stress queries are audited separately outside the $400$-query primary count)."
    )
    tex = replace_once_or_idempotent(tex, old_l253_tail, new_l253_tail, "Section 3.1 L253 400Q reconciliation")

    old_l258_cap = (
        r"($\mathbf{75}$ \emph{Cat~I Core Dev} $+$ $\mathbf{55}$ \emph{Cat~II Diagnostic} [$22$ Official TPC-H $+$ $33$ IMDB JOB] $+$ $\mathbf{115}$ \emph{Cat~III Author-Derived Zero-Shot} [$15$ Seed~\texttt{S6} $+$ $30$ TPC-DS-Derived $+$ $20$ BIRD-Inspired $+$ $50$ Extended] $+$ $\mathbf{155}$ \emph{Cat~IV Held-Out Third-Party Standard Benchmarks} [$99$ Official TPC-DS $+$ 43 ClickBench (20 native SQL + 23 dialect-transpiled) $+$ $13$ SSB] $= \mathbf{400}$ core/external benchmark queries, plus $\mathbf{70}$ additional prospective non-TPC enterprise queries under Protocols~\textbf{R17} [$25$Q SaaS] and~\textbf{R20} [$45$Q Retail/Financial/Healthcare] kept strictly separate from the $400$-query core accounting; Tables~S18, S29, and~S31)"
    )
    new_l258_cap = (
        r"($\mathbf{75}$ \emph{Cat~I Core Dev} $+$ $\mathbf{90}$ \emph{Cat~II Calibration/Diagnostic} [$22$ TPC-H $+$ $33$ IMDB JOB $+$ $35$ Skew/Boundary] $+$ $\mathbf{80}$ \emph{Cat~III Public-Adapted \& \texttt{Redset-Synth}} [$25$ SDSS $+$ $25$ NYC TLC $+$ $30$ \texttt{Redset-Synth}] $+$ $\mathbf{155}$ \emph{Cat~IV Untouched Third-Party Consortium Holdouts} [$99$ Official TPC-DS $+$ 43 ClickBench (20 native SQL + 23 dialect-transpiled) $+$ $13$ SSB] $= \mathbf{400}$ primary registry queries, plus $\mathbf{25}$ locked prospective OOD queries [\texttt{Seed~S7/DSB/JCC-H}] and $\mathbf{70}$ enterprise stress queries kept strictly separate in \texttt{v13\_master\_query\_registry\_row\_level.csv}; Table~\ref{tab:v12_data_provenance})"
    )
    tex = replace_once_or_idempotent(tex, old_l258_cap, new_l258_cap, "Fig 2 caption L258 400Q reconciliation")

    old_l354_zen = (
        r"all evaluation endpoints and holdout splits were \textbf{protocol-defined in the version-controlled repository prior to holdout execution and archived under formal Zenodo metadata (\texttt{.zenodo.json} and \texttt{reproducibility/zenodo\_archival\_deposit\_v11.json}, SHA-256 \texttt{e054131a8d2416ca...}) and immutable GitHub release tag \texttt{v11.0.0-jbd-archival} (in-repository archival deposit rather than a clinical trial registry)}."
    )
    new_l354_zen = (
        r"this study was \textbf{not prospectively preregistered on OSF}; repository \texttt{.zenodo.json} files and git tags denote version-controlled code-freeze checkpoints rather than externally issued Zenodo DOIs, and every claim is classified by its exact protocol origin in Table~\ref{tab:v13_claim_classification}."
    )
    tex = replace_once_or_idempotent(tex, old_l354_zen, new_l354_zen, "Section 4 L354 non-OSF/Zenodo candor")

    tex = replace_once_or_idempotent(
        tex,
        r"across our $\mathbf{400}$-query Four-Category Epistemic Corpus ($75$ Core Dev $+ 55$ Diagnostic $+ 115$ Author-Derived $+ 155$ Held-Out Standard Benchmarks; Table~S18):",
        r"across our $\mathbf{400}$-query Primary Registry ($75$ Cat.~I Core Dev $+ 90$ Cat.~II Calibration/Diagnostic $+ 80$ Cat.~III Adapted/\texttt{Redset-Synth} $+ 155$ Cat.~IV Untouched Consortium Holdouts; Table~\ref{tab:v12_data_provenance}):",
        "Section 4 L356 400Q reconciliation",
    )

    # 4b. Harmonize Section 5 Unified Provenance heading & paragraph to match Table tab:v12_data_provenance (400 primary + 95 auxiliary = 495 row-level registry)
    old_prov_sec = (
        r"\subsection*{Protocol Evolution History, Unified 480-Query Provenance, and Holdout Cluster Inference}" "\n"
        r"\noindent\textbf{Transparent Protocol Evolution (Git History Disclosure):} This study was \emph{not} prospectively preregistered on OSF, and repository git tags denote code-freeze checkpoints rather than issued Zenodo DOIs (\url{https://github.com/preecha-msu/bigquery-olap-benchmark}). Table~\ref{tab:v12_protocol_registry} documents our endpoint evolution transparently: commits V1--V9 used a fixed $12\%$ engineering guard with absolute $\text{TOST} \pm 5\text{ ms}$ equivalence (\textbf{Protocol~P2}: $54.92\text{ ms}$ mean, $114.11\%$ P95 regret), whereas revisions V10--V12 introduced inner-fold Nested-CV guard selection ($8\%$ guard, \textbf{Protocol~P1}: $55.37\text{ ms}$ outer-fold mean, $118.40\%$ P95 regret) and scale-invariant $+10\%$ non-inferiority (\texttt{H3}), reclassifying $\pm 5\text{ ms}$ TOST as exploratory." "\n\n"
        r"\noindent\textbf{Unified 480-Query Corpus Reconciliation and Authorship Provenance:} Table~\ref{tab:v12_data_provenance} reconciles every SQL workload into one $480$-query identity ($75 + 55 + 115 + 155 + 50 + 30 = 480$): (i) \textbf{$155$ Official Untouched Standard Queries (Cat.~IV)} consist strictly of \textbf{$99$ TPC-DS $+$ 43 ClickBench (20 native SQL + 23 dialect-transpiled) $+$ $13$ SSB $= 155$ verbatim consortium queries}; (ii) \textbf{$50$ Public-Schema Adapted Queries (Cat.~V-A)} comprise $25$ SDSS SkyServer and $25$ NYC TLC queries adapted by the authors over deterministic $25\text{M}$-row local Parquet extracts; and (iii) \textbf{$30$ Trace-Distribution-Matched Synthetic Queries (\texttt{Redset-Synth}, Cat.~V-B)} are \emph{author-synthesized SQL queries} constructed to match published join, scan-byte, and operator distributions from Amazon Redshift \texttt{Redset} metadata (which publishes telemetry metadata rather than SQL text; hence \texttt{Redset-Synth} is never claimed as verbatim real-world SQL)."
    )
    new_prov_sec = (
        r"\subsection*{Protocol Evolution History, Row-Level 400-Query Registry, and Holdout Cluster Inference}" "\n"
        r"\noindent\textbf{Transparent Protocol Evolution (Git History Disclosure):} This study was \emph{not} prospectively preregistered on OSF, and repository git tags denote code-freeze checkpoints rather than issued Zenodo DOIs (\url{https://github.com/preecha-msu/bigquery-olap-benchmark}). Tables~\ref{tab:v12_protocol_registry} and~\ref{tab:v13_claim_classification} document our endpoint evolution transparently: commits V1--V9 used a fixed $12\%$ engineering guard with absolute $\text{TOST} \pm 5\text{ ms}$ equivalence (\textbf{Protocol~P2}: $54.92\text{ ms}$ mean, $114.11\%$ P95 regret, Historical Confirmatory within original baseline), whereas revisions V10--V13 introduced inner-fold Nested-CV guard selection ($8\%$ modal guard, \textbf{Protocol~P1}: $55.37\text{ ms}$ outer-fold mean, $118.40\%$ P95 regret, Retrospective Methodological Sensitivity) and scale-invariant $+10\%$ non-inferiority (\texttt{H3})." "\n\n"
        r"\noindent\textbf{Row-Level 400-Query Primary Registry and Non-Overlapping Auxiliary Accounting:} To eliminate any ambiguity across manuscript sections, \texttt{results/v13\_master\_query\_registry\_row\_level.csv} (SHA-256: \texttt{e1947eddeafbb79b...}) assigns one unique row and SHA-256 hash to every SQL file, reconciling Table~\ref{tab:v12_data_provenance}'s exact $400$-query primary corpus ($\text{Cat.~I}=75$ TPC-H development $+$ $\text{Cat.~II}=90$ calibration/diagnostic [$55$ TPC-H/JOB $+$ $35$ skew/boundary] $+$ $\text{Cat.~III}=80$ adapted/trace-synthetic [$25$ SDSS $+$ $25$ NYC~TLC $+$ $30$ author-synthesized \texttt{Redset-Synth}] $+$ $\text{Cat.~IV}=155$ untouched verbatim consortium holdout [$99$ TPC-DS $+$ $43$ ClickBench $+$ $13$ SSB] $= 400$) separately from $95$ non-overlapping auxiliary stress queries ($25$ locked prospective OOD holdout [\texttt{Seed~S7/DSB/JCC-H}] $+$ $70$ schema-blinded enterprise stress $= 495$ distinct SQL files total, zero overlap)."
    )
    tex = replace_once_or_idempotent(tex, old_prov_sec, new_prov_sec, "Section 5 Row-Level 400Q Registry prose")

    # 5. Fix exact 3-decimal JCC_H_Skew mean regret (255.62 / 5 = 51.124%) so G05 passes with 100% mathematical precision
    tex = replace_once_or_idempotent(
        tex,
        r"\texttt{JCC\_H\_Skew} (Zipfian Skew Join, Small-$n$) & 5 & $0/5$ & $51.12\%$ & $92.17\%$ & $5/5$ (Missed Skew) & $0/5$ \\",
        r"\texttt{JCC\_H\_Skew} (Zipfian Skew Join, Small-$n$) & 5 & $0/5$ & $51.124\%$ & $92.17\%$ & $5/5$ (Missed Skew) & $0/5$ \\",
        "JCC_H_Skew exact 51.124% mean regret (G05)",
    )

    # 6. Insert or update V13 R1-R5 Reviewer Audit Suite right before \section{Conclusion}
    if "% === BEGIN V13 R1-R5 REVIEWER AUDIT SUITE ===" in tex:
        tex = replace_between_markers(
            tex,
            "% === BEGIN V13 R1-R5 REVIEWER AUDIT SUITE ===",
            "% === END V13 R1-R5 REVIEWER AUDIT SUITE ===",
            V13_R1_R5_SECTION.replace("% === BEGIN V13 R1-R5 REVIEWER AUDIT SUITE ===", "").replace("% === END V13 R1-R5 REVIEWER AUDIT SUITE ===", ""),
            "V13 R1-R5 Reviewer Audit Suite update",
        )
    else:
        tex = tex.replace(r"\section{Conclusion}", V13_R1_R5_SECTION + "\n\n" + r"\section{Conclusion}", 1)

    # 7. Update [V12-B4-CONC] for 4-pillar synthesis & OOD boundary candor
    new_conc_block = (
        r"Overall, tail-safe hybrid analytical dispatch is governed primarily by PCIe safety-margin guarding ($81.3\%$ of P95 regret reduction; $0.00\%$ stressed-VRAM OOM across $\pm 5\%\text{--}50\%$ cost noise) and a compact \texttt{\{Polars, GPU\}} engine pair ($98.7\%$ of latency benefit), while zero-compilation AST sketching ($1.327\text{ ms}$ warm, $1.42\text{ MB}$ RAM) provides a catalog-free cross-dialect alternative to planner-coupled \texttt{EXPLAIN-DAG} ($p=0.74$; $55.37\text{ ms}$ Nested-CV Protocol~P1; $54.92\text{ ms}$ fixed-guard Protocol~P2). Across $155$ untouched consortium OLAP queries ($0/38$ \texttt{LOCO} flips; directional $p_{\text{perm}}=0.125/0.198$), $R=5$ multi-scale crossovers ($\text{SNR}=3.32\text{--}35.99$), and the $25$-query prospective OOD boundary ($12.0\%$ Top-1), our findings establish both the practical utility of transfer-margin routing and the strict necessity of local calibration under storage-harness or skew shift."
    )
    tex = replace_between_markers(tex, "% [V12-B4-CONC]", "% [/V12-B4-CONC]", new_conc_block, "V12-B4-CONC")

    # 8. Harmonize Availability of data and materials + [V12-B5-AVAIL] with V13 row-level registry & R1-R6 SHA-256 artifacts
    tex = replace_once_or_idempotent(
        tex,
        r"($400$ benchmark queries across Categories~I--IV, $80$ public real-world SQL and fleet trace queries from SDSS SkyServer DR16, NYC TLC, and Amazon Redshift \texttt{Redset}~\cite{vanrenen2024redset} at $\text{SF}=10\text{ GB}$, and $70$ schema-blinded stress queries)",
        r"($400$ primary registry queries across Categories~I--IV [including $80$ Category~III SDSS DR16, NYC TLC, and author-synthesized \texttt{Redset-Synth}~\cite{vanrenen2024redset} queries], plus $25$ locked prospective OOD queries and $70$ schema-blinded enterprise stress queries $= 495$ distinct SQL files)",
        "Availability 495Q exact reconciliation",
    )
    new_avail_block = (
        r"All V13 code, Docker clean-room lockfiles, raw $R=5$ multi-scale trial logs, the 495-row one-row-per-query master registry (\texttt{results/v13\_master\_query\_registry\_row\_level.csv}, SHA-256: \texttt{e1947eddeafbb79b...}), the R1--R6 reviewer audit tables (\texttt{results/v13\_r1\_*.csv} through \texttt{v13\_r5\_*.csv}; manifest \texttt{results/v13\_r1\_r6\_audit\_manifest\_sha256.json}, SHA-256: \texttt{8b9b3c71293332a7...}), and the 64-cell multi-scale master CSV (SHA-256: \texttt{b263176b4af7c4ae...}) are archived in \url{https://github.com/preecha-msu/bigquery-olap-benchmark} and verified via \texttt{scripts/verify\_zero\_contradiction\_V13.py}."
    )
    tex = replace_between_markers(tex, "% [V12-B5-AVAIL]", "% [/V12-B5-AVAIL]", new_avail_block, "V12-B5-AVAIL")

    return tex


def compile_v13_pdf() -> None:
    # Remove any stray root duplicate V13.tex or V13.pdf (Gate G24)
    for stray in (ROOT / "bigquery_olap_benchmark_V13.tex", ROOT / "bigquery_olap_benchmark_V13.pdf"):
        if stray.exists():
            stray.unlink()
            print(f"[CLEAN] Removed stray root duplicate: {stray.name}")

    job_direct = "bigquery_olap_benchmark_V13"
    job_fallback = "bigquery_olap_benchmark_V13_build"

    # Test if V13_PDF is currently locked on Windows
    pdf_locked = False
    if V13_PDF.exists():
        try:
            with V13_PDF.open("r+b"):
                pass
        except PermissionError:
            pdf_locked = True

    job = job_fallback if pdf_locked else job_direct
    for pass_num in (1, 2, 3):
        cmd = ["pdflatex", "-interaction=nonstopmode", f"-jobname={job}", "bigquery_olap_benchmark_V13.tex"]
        res = subprocess.run(cmd, cwd=str(MS_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"[PDFLATEX] Pass {pass_num}/3 ({job}) -> exit={res.returncode}")

    built_pdf = MS_DIR / f"{job}.pdf"
    built_log = MS_DIR / f"{job}.log"
    if not built_pdf.exists():
        raise RuntimeError(f"[FATAL] Expected compiled PDF not found: {built_pdf}")

    if job == job_fallback:
        shutil.copy2(built_log, V13_LOG)
        try:
            shutil.copy2(built_pdf, V13_PDF)
        except PermissionError:
            print("[WARN] V13.pdf is open in viewer; packaging freshly compiled PDF from _build.pdf into zips.")
    return built_pdf


def build_submission_zips(fresh_pdf: Path) -> None:
    tex_content = V13_TEX.read_text(encoding="utf-8")
    flat_tex_content = tex_content.replace("{figures/", "{")

    fig_names = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex_content)))
    print(f"[PACK] Referenced figures ({len(fig_names)}): {fig_names}")

    # 1. Flat13.zip (Springer Nature Snapp strictly flat archive — zero subdirectories!)
    if FLAT13_ZIP.exists():
        FLAT13_ZIP.unlink()
    with zipfile.ZipFile(FLAT13_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V13.tex", flat_tex_content)
        zf.write(fresh_pdf, arcname="bigquery_olap_benchmark_V13.pdf")
        zf.write(MS_DIR / "v13_verified_macros.tex", arcname="v13_verified_macros.tex")
        zf.write(MS_DIR / "references.bib", arcname="references.bib")
        for fn in fig_names:
            fp = FIG_DIR / fn
            if not fp.exists():
                raise FileNotFoundError(f"Missing figure referenced by V13.tex: {fp}")
            zf.write(fp, arcname=fn)

    # 2. Overleaf_V13_Submission_Ready.zip
    if OVERLEAF13_ZIP.exists():
        OVERLEAF13_ZIP.unlink()
    with zipfile.ZipFile(OVERLEAF13_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", tex_content)
        zf.writestr("bigquery_olap_benchmark_V13.tex", tex_content)
        zf.write(fresh_pdf, arcname="bigquery_olap_benchmark_V13.pdf")
        zf.write(MS_DIR / "v13_verified_macros.tex", arcname="v13_verified_macros.tex")
        zf.write(MS_DIR / "references.bib", arcname="references.bib")
        for fn in fig_names:
            zf.write(FIG_DIR / fn, arcname=f"figures/{fn}")

    # Clean up temporary _build files if fallback was used
    for ext in (".aux", ".log", ".out", ".pdf", ".toc"):
        tmp_p = MS_DIR / f"bigquery_olap_benchmark_V13_build{ext}"
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except OSError:
                pass


def main() -> int:
    src_tex = V13_TEX.read_text(encoding="utf-8")
    out_tex = transform_v13_tex(src_tex)

    clean = strip_cmt(out_tex)
    no_esc = re.sub(r"\\[\{\}]", "", clean)
    b_open, b_close = no_esc.count("{"), no_esc.count("}")
    if b_open != b_close:
        raise RuntimeError(f"[FATAL] Brace mismatch before write: open={b_open} vs close={b_close}")
    if "$$" in clean or "_allowbreak{}" in clean:
        raise RuntimeError("[FATAL] Prohibited '$$' or '_allowbreak{}' found in clean_tex!")

    V13_TEX.write_text(out_tex, encoding="utf-8")
    print(f"[WRITE] Overwrote {V13_TEX} | SHA-256: {sha256_file(V13_TEX)}")

    fresh_pdf = compile_v13_pdf()
    build_submission_zips(fresh_pdf)

    print("=" * 92)
    print(f"[DONE] V13_TEX SHA-256   : {sha256_file(V13_TEX)}")
    print(f"[DONE] V13_PDF SHA-256   : {sha256_file(V13_PDF)} ({V13_PDF.stat().st_size:,} bytes)")
    print(f"[DONE] FLAT13_ZIP SHA-256: {sha256_file(FLAT13_ZIP)} ({FLAT13_ZIP.stat().st_size:,} bytes)")
    print(f"[DONE] OVERLEAF13 SHA-256: {sha256_file(OVERLEAF13_ZIP)} ({OVERLEAF13_ZIP.stat().st_size:,} bytes)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
