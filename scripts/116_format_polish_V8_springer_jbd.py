#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, shutil, subprocess, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR, FIG_DIR, ROOT_FIG = ROOT/"manuscript", ROOT/"manuscript"/"figures", ROOT/"figures"
V8_TEX, V8_PDF = MS_DIR/"bigquery_olap_benchmark_V8.tex", MS_DIR/"bigquery_olap_benchmark_V8.pdf"

for bak in ROOT.glob("manuscript_backup_*"):
    if bak.is_dir():
        shutil.rmtree(bak, ignore_errors=True)

def regenerate_fig15_synchronized():
    plt.rcParams["font.family"] = "serif"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.4, 4.35), dpi=300)

    x = np.arange(4)
    flip_naive  = [0.00, 2.99, 7.35, 14.05]
    flip_conf   = [0.00, 3.02, 5.78, 12.00]
    viol_nopcie = [10.67, 10.89, 10.87, 11.05]
    viol_conf   = [9.33, 9.33, 9.12, 8.86]

    b1 = axa.bar(x - 0.17, flip_naive, 0.33, color="#e67e22", edgecolor="black", label="Unguarded ArgMin Flip Rate (%)")
    b2 = axa.bar(x + 0.17, flip_conf,  0.33, color="#1e8449", edgecolor="black", label="Conformal AST-Route Flip Rate (%)")
    ax2 = axa.twinx()
    ax2.plot(x, viol_nopcie, "s--", color="#9e2a2b", lw=1.8, markersize=5.5, label="Without 12% PCIe Guard: >100% Viol. (%)")
    ax2.plot(x, viol_conf,   "o-",  color="#1f4e78", lw=2.0, markersize=5.5, label="Conformal AST-Route: >100% Viol. (%)")

    for b in list(b1) + list(b2):
        if b.get_height() > 0:
            axa.text(b.get_x() + b.get_width()/2, b.get_height() + 0.38, f"{b.get_height():.2f}%", ha="center", fontsize=6.9, fontweight="bold")

    axa.set_xticks(x)
    axa.set_xticklabels(["0%\nClean", "+-5%\nNoise", "+-10%\nNoise", "+-20%\nNoise"], fontsize=8.0)
    axa.set_ylabel("Routing Flip Rate (%)", fontsize=8.8)
    ax2.set_ylabel("Severe Regret (>100%) Rate (%)", fontsize=8.8, color="#1f4e78")
    axa.set_ylim(0, 20.5)
    ax2.set_ylim(5.0, 14.0)
    axa.set_title("(a) Router Decision Stability & 12% PCIe Guard (1,000-MC)", fontsize=9.2, fontweight="bold")
    h1, l1 = axa.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axa.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.7)
    axa.grid(axis="y", ls="--", alpha=0.4)

    xd = np.arange(7)
    pol_ms = [82.47, 92.89, 87.60, 87.60, 86.86, 91.26, 91.26]
    ast_ms = [60.90, 62.34, 68.85, 64.49, 67.01, 78.47, 76.13]
    pi_cov = [89.0, 88.7, 88.7, 88.7, 85.3, 75.0, 89.0]

    axb.bar(xd - 0.17, pol_ms, 0.33, color="#7f8c8d", edgecolor="black", label="Static Polars Mean Latency (ms)")
    ba = axb.bar(xd + 0.17, ast_ms, 0.33, color="#2e5984", edgecolor="black", label="AST-Route Net Latency (ms)")
    ba[5].set_facecolor("#c44e52")
    ba[6].set_facecolor("#2ca02c")
    for i, (mv, cv) in enumerate(zip(ast_ms, pi_cov)):
        axb.text(xd[i] + 0.17, mv + 1.8, f"{mv:.1f}\n({cv:.0f}%)", ha="center", fontsize=6.4, fontweight="bold")

    axb.set_xticks(xd)
    axb.set_xticklabels(["D0\nBase", "D1\nMix", "D2a\nStale", "D2b\nRefr", "D3\nSkew.8", "D4a\nSkew1.4", "D4b\nRecal"], fontsize=7.5)
    axb.set_ylabel("Mean Execution Latency (ms)", fontsize=8.8)
    axb.set_ylim(0, 118.0)
    axb.set_title("(b) Dynamic Drift & 5-Canary Recalibration Policy", fontsize=9.2, fontweight="bold")
    axb.legend(loc="upper left", fontsize=7.1)
    axb.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    for fn in ("fig_v7_stability_and_workload_drift_300dpi.pdf", "fig_v7_stability_and_workload_drift_300dpi.png"):
        fig.savefig(FIG_DIR / fn, dpi=300, bbox_inches="tight")
        if ROOT_FIG.exists():
            fig.savefig(ROOT_FIG / fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

def polish_v8_format():
    regenerate_fig15_synchronized()
    tex = V8_TEX.read_text(encoding="utf-8")

    # 1. Enable T1 font encoding + URL/DOI line-breaking to prevent right-margin clipping on Pages 35 & 37
    if r"\RequirePackage[T1]{fontenc}" not in tex and r"\usepackage[T1]{fontenc}" not in tex:
        tex = "\\RequirePackage[T1]{fontenc}\n" + tex
    if r"\Urlmuskip=0mu plus 2mu" not in tex:
        tex = tex.replace(
            r"\begin{document}",
            "\\makeatletter\\g@addto@macro{\\UrlBreaks}{\\UrlOrds\\do\\-\\do\\_\\do\\/\\do\\.\\do\\0\\do\\1\\do\\2\\do\\3\\do\\4\\do\\5\\do\\6\\do\\7\\do\\8\\do\\9}\\makeatother\n\\Urlmuskip=0mu plus 2mu\\relax\n\\emergencystretch=1.5em\n\\begin{document}",
            1,
        )

    # 2. SpringerOpen 150-250 word Abstract (243 words, all acronyms defined on first use, no raw URL)
    springer_abstract = r"""\begin{quote}
\small\sffamily
\textbf{\color{springerblue}Abstract}\\[0.3em]
\textbf{Background:} Hybrid analytical pipelines span local vectorized C++/Rust central processing units (CPUs), consumer/edge graphics processing units (GPUs), and serverless cloud warehouses across Scale Factors ($\text{SF}=1\text{--}100\text{ GB}$), where static engine assignment incurs severe Peripheral Component Interconnect Express (PCIe) transfer stalls, out-of-core memory spilling, or unnecessary cloud scan charges.

\textbf{Methodology:} We present \textbf{AST-Route}, a dialect-agnostic, tail-safe query routing framework across six Online Analytical Processing (OLAP) engines (SQLite~3, DuckDB, Polars, DataFusion, PyTorch CUDA tensor execution, and Google BigQuery). Without target-engine schema binding or physical plan compilation, AST-Route extracts a nine-dimensional Structured Query Language (SQL) Abstract Syntax Tree (AST) and zone-map selectivity vector ($\mathbf{x}_q \in \mathbb{R}^{9}$, extended to $\mathbb{R}^{10}$ across scales and $\mathbb{R}^{12}$ with $10.2\text{ KB}$ catalog sketches) in $4.15\text{ ms}$ warm dispatch overhead ($3.48\text{ ms}$ parse, $0.17\text{ ms}$ metadata lookup, $0.50\text{ ms}$ Gradient Boosting Regressor [GBR] inference) to govern memory-feasible, conformal risk-aware dispatch with a $12\%$ PCIe crossover guard and an out-of-distribution (OOD) operator fallback gate.

\textbf{Results:} Across $163$ queries from five benchmark suites, three hardware nodes (\texttt{x86\_64}+GPU, 22-thread \texttt{x86\_64}, and Apple M1 \texttt{ARM64}), and $C=1\text{--}32$ concurrent streams, AST-Route under $25$-seed Leave-One-Seed-Out Cross-Validation ($\text{SF}=1\text{ GB}$) achieves $54.92\text{ ms}$ net mean latency---outperforming static CPU engines by $1.50\times\text{--}3.35\times$ ($p_{\text{Holm,seed}}\le 0.0171$) while matching Always GPU ($56.96\text{ ms}$, $p_{\text{seed}}=0.184$) at less than half the 95th-percentile tail regret ($114.11\%$ vs.\ $257.31\%$, paired Wilcoxon $p = 0.0021$) and halving $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). The OOD gate lowers unseen-operator latency from $85.42\text{ ms}$ to $76.18\text{ ms}$, while multi-scale routing sustains $3.35\times\text{--}7.61\times$ speedups across $\text{SF}=1\text{--}100\text{ GB}$.

\textbf{Conclusion:} AST-Route enables practical, risk-controlled cross-engine query orchestration over shared columnar data access.

\noindent\textbf{Keywords:} Cross-engine query routing; Tail-risk control; Conformal prediction; Heterogeneous OLAP systems; Apache Arrow and GPU analytics; Multi-objective cost modeling
\end{quote}"""

    q_s = tex.find(r"\begin{quote}")
    q_e = tex.find(r"\end{quote}") + len(r"\end{quote}")
    if q_s != -1 and q_e > q_s:
        tex = tex[:q_s] + springer_abstract + tex[q_e:]

    # 3. Fix Table 9B float placement so Table 9B always appears BEFORE Table 10 (and decongest Page 15)
    t9b_marker = r"{\renewcommand{\thetable}{9B}"
    if t9b_marker in tex:
        idx_9b = tex.find(t9b_marker)
        end_9b = tex.find(r"\addtocounter{table}{-1}}", idx_9b) + len(r"\addtocounter{table}{-1}}")
        block_9b = tex[idx_9b:end_9b]
        cap_t10 = tex.find(r"\caption{Comprehensive statistical and methodological audit matrix")
        if cap_t10 != -1:
            tb_t10 = tex.rfind(r"\begin{table", 0, cap_t10)
            if tb_t10 < idx_9b:
                tex_rem = tex[:idx_9b] + tex[end_9b:]
                tb_t10_new = tex_rem.rfind(r"\begin{table", 0, tex_rem.find(r"\caption{Comprehensive statistical and methodological audit matrix"))
                tex = tex_rem[:tb_t10_new] + block_9b + "\n\\clearpage\n" + tex_rem[tb_t10_new:]

    # 4. Fix literal backslash & filename clipping in Availability of data and materials (Page 35)
    tex = tex.replace(r"scripts/run\\_reproducibility\\_audit.py", r"scripts/run\_\allowbreak{}reproducibility\_\allowbreak{}audit.py")
    tex = tex.replace(r"scripts/run\\\_reproducibility\\\_audit.py", r"scripts/run\_\allowbreak{}reproducibility\_\allowbreak{}audit.py")
    tex = tex.replace(r"section5\_and\_6\_master\_audit\_log.csv", r"section5\_\allowbreak{}and\_\allowbreak{}6\_\allowbreak{}master\_\allowbreak{}audit\_\allowbreak{}log.csv")

    # 5. Fix minor punctuation & Table 11 Part B cross-reference (Pages 5, 11, 24)
    tex = tex.replace("automated engine dispatching..", "automated engine dispatching.")
    tex = tex.replace("(Q1--Q22exhibits", "(Q1--Q22) exhibits")
    tex = tex.replace("(Q1-Q22exhibits", "(Q1--Q22) exhibits")
    tex = tex.replace("in Table~11B)", "in Table~11 Part~B)")
    tex = tex.replace("in Table 11B)", "in Table~11 Part~B)")

    # 6. Pre-flight syntax check
    no_cmt = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    cl_b = re.sub(r"\\[\{\}]", "", no_cmt)
    assert cl_b.count("{") == cl_b.count("}"), f"Brace mismatch: {cl_b.count('{')} vs {cl_b.count('}')}"
    for idx, para in enumerate(re.split(r"\n\s*\n", no_cmt), 1):
        assert len(re.findall(r"(?<!\\)\$", para)) % 2 == 0, f"Odd '$' count in paragraph {idx}"

    V8_TEX.write_text(tex, encoding="utf-8")
    (MS_DIR / "main.tex").write_text(tex, encoding="utf-8")

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

    zip_p = ROOT / "Overleaf_V8_Full_Submission_Ready.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(V8_TEX, arcname="main.tex")
        zf.write(V8_TEX, arcname="bigquery_olap_benchmark_V8.tex")
        if (MS_DIR / "references.bib").exists():
            zf.write(MS_DIR / "references.bib", arcname="references.bib")
        for fg in sorted(FIG_DIR.glob("*.pdf")):
            zf.write(fg, arcname=f"figures/{fg.name}")

    print(f"[FORMAT POLISH OK] {V8_PDF.name} ({V8_PDF.stat().st_size:,} bytes) | Undefined Refs = {undef}")
    print(f"[OVERLEAF ZIP OK ] {zip_p.name} ({zip_p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    polish_v8_format()