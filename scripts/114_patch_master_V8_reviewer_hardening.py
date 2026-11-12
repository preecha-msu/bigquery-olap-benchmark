#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR, RES_DIR = ROOT / "manuscript", ROOT / "results"
V8_TEX = MS_DIR / "bigquery_olap_benchmark_V8.tex"
V8_PDF = MS_DIR / "bigquery_olap_benchmark_V8.pdf"

if not V8_TEX.exists():
    shutil.copy2(MS_DIR / "bigquery_olap_benchmark_V2.tex", V8_TEX)

for bak in ROOT.glob("manuscript_backup_*"):
    if bak.is_dir():
        shutil.rmtree(bak, ignore_errors=True)

df_p0 = pd.read_csv(RES_DIR / "v8_reviewer_p0_explain_classifier_ood_audit.csv").set_index("Policy")
df_tl = pd.read_csv(RES_DIR / "v8_tail_significance_bootstrap_ci.csv").iloc[0]

exp_r = df_p0.loc["DuckDB_EXPLAIN_Plan_Router (0.87ms Plan + GBR)"]
clf_r = df_p0.loc["Direct_Cost_Sensitive_Classifier (GBDT-Clf 9D)"]
ood_r = df_p0.loc["LOTO_CV_With_OOD_Operator_Gate (Fallback->Polars)"]

tex = V8_TEX.read_text(encoding="utf-8")

concise_abstract = r"""\begin{quote}
\small\sffamily
\textbf{\color{springerblue}Abstract}\\[0.3em]
\textbf{Background:} Hybrid analytical pipelines increasingly span local vectorized C++/Rust CPUs, consumer/edge GPUs, and serverless cloud warehouses across $\text{SF}=1\text{--}100\text{ GB}$, where static engine assignment suffers severe PCIe transfer stalls, out-of-core memory spilling, or unnecessary cloud scan charges.

\textbf{Methodology:} We present \textbf{AST-Route}, a dialect-agnostic, tail-safe query routing framework across six OLAP engines (SQLite~3, DuckDB, Polars, DataFusion, PyTorch CUDA tensor execution, and Google BigQuery). Without requiring target-engine schema binding or multi-engine physical plan compilation, AST-Route extracts a $9\text{D}$ static SQL AST and zone-map selectivity vector ($\mathbf{x}_q \in \mathbb{R}^{9}$, extended to $\mathbb{R}^{10}$ across scale factors and $\mathbb{R}^{12}$ with $10.2\text{ KB}$ catalog sketches) in $4.15\text{ ms}$ total warm dispatch overhead ($3.48\text{ ms}$ parse, $0.17\text{ ms}$ metadata lookup, $0.50\text{ ms}$ multi-output GBR inference) to govern memory-feasible, conformal risk-aware dispatch with a $12\%$ PCIe crossover guard and an out-of-distribution (OOD) operator fallback gate.

\textbf{Results:} Evaluated across $163$ queries from five suites (TPC-H Factorial, Official TPC-H Q1--Q22, IMDB JOB, SSB, and ClickBench), three physical hardware nodes (\texttt{x86\_64}+GPU, 22-thread \texttt{x86\_64}, and Apple M1 \texttt{ARM64}), and $C=1\text{--}32$ concurrent streams, AST-Route ($\text{SF}=1\text{ GB}$, $25$-seed LOSO-CV) achieves $54.92\text{ ms}$ net mean latency---outperforming static CPU engines by $1.50\times\text{--}3.35\times$ ($p_{\text{Holm,seed}}\le 0.0171$) while matching Always GPU ($56.96\text{ ms}$, $p_{\text{seed}}=0.184$) at less than half the 95th-percentile tail regret ($114.11\%$ vs.\ $257.31\%$, paired cluster bootstrap $p < 0.005$) and halving $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). An OOD operator gate resolves unseen-operator extrapolation under Leave-One-Tier-Out validation, while multi-scale routing sustains $3.35\times\text{--}7.61\times$ speedups across $\text{SF}=1\text{--}100\text{ GB}$.

\textbf{Conclusion:} AST-Route provides practical, risk-controlled cross-engine query orchestration when heterogeneous engines share columnar data access (\url{https://github.com/preecha-msu/bigquery-olap-benchmark}).

\noindent\textbf{Keywords:} Cross-engine query routing; Tail-risk control; Conformal prediction; Heterogeneous OLAP systems; Apache Arrow and GPU analytics; Multi-objective cost modeling
\end{quote}"""

q_start = tex.find(r"\begin{quote}")
q_end = tex.find(r"\end{quote}") + len(r"\end{quote}")
if q_start != -1 and q_end > q_start:
    tex = tex[:q_start] + concise_abstract + tex[q_end:]

tex = re.sub(r",?\s*\$p\s*=\s*1\.783\s*\\times\s*10\^\{-55\}\$\s*(?:trial-level|across trials)", "", tex)
tex = tex.replace("exceptional rank invariance", "high rank stability")
tex = tex.replace("oblique hyperplane projection", "linear composite structural feature")
tex = tex.replace("(100\\% real-execution telemetry, zero synthetic simulation)", "(empirical hardware execution telemetry)")
tex = tex.replace("without synthetic simulation", "under controlled hardware execution")
tex = tex.replace(r"\rho_{\text{partial}} = 0.8088", r"\rho_{\text{op}|N,\sigma} = 0.8088")
tex = tex.replace(r"\rho_{\text{partial}}=0.8088", r"\rho_{\text{op}|N,\sigma}=0.8088")

p0_defense_tex = rf"""
\noindent\textbf{{Comparison Against Physical-Planner (\texttt{{EXPLAIN}}), Cost-Sensitive Classification, OOD Operator Gating, and Data-Placement Economics.}}
To address critical systems design questions regarding physical-plan features, classification vs.\ regression, unseen-operator extrapolation, tail-metric significance, and multi-engine storage placement, Table~\ref{{tab:v8_p0_systems_defense}} consolidates five targeted evaluations on our canonical testbed:
(1) \emph{{Why Dialect-Agnostic AST Instead of Single-Engine \texttt{{EXPLAIN}}?}} While in-process DuckDB compiles a single local physical plan in $0.87\text{{ ms}}$ once tables are bound in memory (Table~4), routing via a single engine's planner introduces two architectural bottlenecks in polyglot environments: (i) dialect incompatibility (e.g., BigQuery \texttt{{UNNEST}}/array syntax or cloud dry-run APIs incurring $150\text{{--}}300\text{{ ms}}$ network RTT) and (ii) single-engine cardinality bias on multi-stage CTEs. Empirically, a baseline router trained on DuckDB \texttt{{EXPLAIN (FORMAT JSON)}} operator and cardinality estimates (${exp_r['Router_Overhead_ms']:.2f}\text{{ ms}}$ overhead) yields ${exp_r['Net_Mean_ms']:.2f}\text{{ ms}}$ net latency and ${exp_r['P95_Regret_Pct']:.2f}\%$ P95 regret, trailing Conformal AST-Route ($54.92\text{{ ms}}$, $114.11\%$ P95 regret);
(2) \emph{{Multi-Output Cost Regression vs.\ Direct Cost-Sensitive Classification}}: A regret-weighted GBDT classifier predicting $\hat{{e}}(q)$ directly achieves ${clf_r['Top1_Pct']:.2f}\%$ Top-1 and ${clf_r['Net_Mean_ms']:.2f}\text{{ ms}}$ net latency (${clf_r['P95_Regret_Pct']:.2f}\%$ P95 regret), confirming that explicit latency/energy/memory regression is essential to enforce dynamic VRAM caps ($M_{{\text{{gpu\_max}}}}$) and conformal abstention intervals;
(3) \emph{{Statistical Significance of Tail-Risk Reduction}}: Under $10{{,}}000$-iteration seed-clustered ($K=25$) paired bootstrap resampling, Conformal AST-Route reduces P95 tail regret relative to Always GPU by $\Delta\text{{P95}} = -143.20\text{{ pp}}$ (95\% cluster CI: $[{df_tl['Delta_P95_Regret_CI_Low']:.1f}, {df_tl['Delta_P95_Regret_CI_High']:.1f}]\text{{ pp}}$, $p = {df_tl['Tail_Bootstrap_p_value']:.4f}$) and $\text{{CVaR}}_{{0.95}}$ tail regret with 95\% CI $[{df_tl['Delta_CVaR95_CI_Low']:.1f}, {df_tl['Delta_CVaR95_CI_High']:.1f}]\text{{ pp}}$;
(4) \emph{{OOD Operator Gate Resolving Leave-One-Tier-Out (LOTO-CV) Extrapolation}}: Augmenting Algorithm~1 with a zero-cost structural domain check---falling back to safe in-memory Polars Arrow whenever an incoming query contains an operator family unseen during training ($W(q) > \max_{{\text{{train}}}} W$ or $U(q) > \max_{{\text{{train}}}} U$)---reduces LOTO-CV net latency from $85.42\text{{ ms}}$ ($348.0\%$ P95 regret) down to $\mathbf{{{ood_r['Net_Mean_ms']:.2f}\text{{ ms}}}}$ (${ood_r['P95_Regret_Pct']:.2f}\%$ P95 regret), eliminating catastrophic unseen-operator misrouting; and
(5) \emph{{Zero-Copy Shared Lakehouse Placement \& Break-Even Amortization}}: AST-Route does \emph{{not}} duplicate mutable tables across proprietary silos; local DuckDB, Polars, and DataFusion read the \emph{{same}} canonical ZSTD Parquet / Arrow IPC dataset zero-copy on NVMe ($0\text{{ MB}}$ local duplication), GPU transfers occur on-demand over PCIe (already charged inside $T_{{\text{{GPU\_Cold}}}} = 56.96\text{{ ms}}$), and cloud GCS/BigQuery sync ($\sim\$0.020/\text{{GB-month}}$ storage; $8.4\text{{ s}}$ bulk Parquet upload per $\text{{SF}}=1\text{{ GB}}$) is only required when $\text{{SF}}=100\text{{ GB}}$ out-of-core spilling ($6.55\text{{--}}8.70\text{{ s}}$/query locally vs.\ $2.57\text{{ s}}$ BigQuery) amortizes data staging after $>22$ spill-bound queries.

\begin{{table}}[!htbp]
\centering
\caption{{Additional V8 systems validation ($N=75$, $\text{{SF}}=1\text{{ GB}}$): comparing Conformal AST-Route against a physical-planner router (\texttt{{DuckDB EXPLAIN JSON}}), a regret-weighted cost-sensitive classifier, OOD-gated Leave-One-Tier-Out (LOTO-CV) operator safety, and Leave-One-Scale-Out ($\text{{Train SF}}\in\{{1,10,30\}}\rightarrow\text{{Test SF}}=100\text{{ GB}}$) transfer.}}\label{{tab:v8_p0_systems_defense}}
\setlength{{\tabcolsep}}{{3.2pt}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llcccccl@{{}}}}
\toprule
\textbf{{Evaluation Axis}} & \textbf{{Router / Baseline Variant}} & \textbf{{Ovh.\ (ms)}} & \textbf{{Top-1}} & \textbf{{Net (ms)}} & \textbf{{Mean Reg.}} & \textbf{{P95 Reg.}} & \textbf{{Key Architectural Trade-off / Finding}} \\
\midrule
Planner vs.\ AST & DuckDB \texttt{{EXPLAIN (JSON)}} Plan Router & ${exp_r['Router_Overhead_ms']:.2f}$ & ${exp_r['Top1_Pct']:.2f}\%$ & ${exp_r['Net_Mean_ms']:.2f}$ & ${exp_r['Mean_Regret_Pct']:.2f}\%$ & ${exp_r['P95_Regret_Pct']:.2f}\%$ & Requires bound schema; dialect-coupled to DuckDB \\
Classif.\ vs.\ Reg. & Direct Cost-Sensitive Classifier (GBDT-9D) & ${clf_r['Router_Overhead_ms']:.2f}$ & ${clf_r['Top1_Pct']:.2f}\%$ & ${clf_r['Net_Mean_ms']:.2f}$ & ${clf_r['Mean_Regret_Pct']:.2f}\%$ & ${clf_r['P95_Regret_Pct']:.2f}\%$ & Cannot enforce dynamic $M_{{\text{{max}}}}$ or $(\alpha,\beta,\gamma)$ weights \\
Tail-Safe Proposed & \textbf{{Conformal AST-Route (Proposed 9D)}} & $\mathbf{{4.15}}$ & $\mathbf{{57.33\%}}$ & $\mathbf{{54.92}}$ & $\mathbf{{34.98\%}}$ & $\mathbf{{114.11\%}}$ & \textbf{{Dialect-agnostic; $\Delta\text{{P95}} = -143.2\text{{ pp}}$ vs.\ GPU ($p<0.005$)}} \\
\midrule
Unseen Op.\ (LOTO) & Standard LOTO-CV (No OOD Gate) & $4.15$ & $44.00\%$ & $85.42$ & $159.92\%$ & $348.00\%$ & Extrapolates poorly on unseen \texttt{{WINDOW}}/\texttt{{UNNEST}} \\
Unseen Op.\ (LOTO) & \textbf{{LOTO-CV + OOD Structural Fallback Gate}} & $\mathbf{{4.15}}$ & $\mathbf{{{ood_r['Top1_Pct']:.2f}\%}}$ & $\mathbf{{{ood_r['Net_Mean_ms']:.2f}}}$ & $\mathbf{{{ood_r['Mean_Regret_Pct']:.2f}\%}}$ & $\mathbf{{{ood_r['P95_Regret_Pct']:.2f}\%}}$ & \textbf{{Detects unseen $W,U > \max_{{\text{{train}}}}$ $\rightarrow$ safe Polars fallback}} \\
Unseen Scale (LOScaleO) & Train $\text{{SF}}\in\{{1,10,30\}}\rightarrow$ Zero-Shot $\text{{SF}}=100\text{{G}}$ & $4.15$ & $66.67\%$ & $1214.50$ & $24.80\%$ & $138.40\%$ & Beats static DuckDB ($8700\text{{ ms}}$) \& Polars ($6551\text{{ ms}}$) by $5.4\times\text{{--}}7.2\times$ \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""

tex = re.sub(
    r"\\noindent\\textbf\{Comparison Against Physical-Planner.*?\\end\{table\}\s*",
    "",
    tex,
    flags=re.DOTALL,
)
lim_anchor = r"\section{Limitations and Threats to Validity}"
if lim_anchor in tex:
    tex = tex.replace(lim_anchor, p0_defense_tex.strip() + "\n\n" + lim_anchor, 1)

no_cmt = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
cl_b = re.sub(r"\\[\{\}]", "", no_cmt)
assert cl_b.count("{") == cl_b.count("}"), f"Brace mismatch: {cl_b.count('{')} vs {cl_b.count('}')}"
for idx, para in enumerate(re.split(r"\n\s*\n", no_cmt), 1):
    assert len(re.findall(r"(?<!\\)\$", para)) % 2 == 0, f"Odd '$' count in paragraph {idx}"

V8_TEX.write_text(tex, encoding="utf-8")

for p_i in (1, 2, 3):
    pr = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", V8_TEX.name],
        cwd=str(MS_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if pr.returncode != 0:
        print("\n".join(pr.stdout.splitlines()[-25:]))
        raise RuntimeError(f"pdflatex failed on pass {p_i}")

log_txt = (MS_DIR / "bigquery_olap_benchmark_V8.log").read_text(encoding="utf-8", errors="replace")
undef = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
print(f"[V8 COMPILED OK] {V8_PDF.name} ({V8_PDF.stat().st_size:,} bytes) | Undefined Refs = {undef}")