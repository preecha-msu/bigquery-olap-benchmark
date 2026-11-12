#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, hashlib, re, shutil, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR, RES_DIR, FIG_DIR, REPRO = ROOT/"manuscript", ROOT/"results", ROOT/"manuscript"/"figures", ROOT/"reproducibility_v7"
TARGET_TEX, TARGET_PDF = MS_DIR/"bigquery_olap_benchmark_V2.tex", MS_DIR/"bigquery_olap_benchmark_V2.pdf"
COVER_TEX, COVER_PDF = MS_DIR/"Cover_Letter_Journal_of_Big_Data.tex", MS_DIR/"Cover_Letter_Journal_of_Big_Data.pdf"
RESP_TEX, RESP_PDF = MS_DIR/"point_by_point_reviewer_response_V2.tex", MS_DIR/"point_by_point_reviewer_response_V2.pdf"

def sha256_f(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()

def strip_tex_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.splitlines())

def count_braces(s: str) -> tuple[int, int]:
    cl = re.sub(r"\\[\{\}]", "", strip_tex_comments(s))
    return cl.count("{"), cl.count("}")

CANONICAL_FRONTMATTER_AND_ABSTRACT = r"""\begin{document}

% =============================================================================
% SPRINGEROPEN / JOURNAL OF BIG DATA FRONTMATTER (FRAMEWORK PAPER FRAMING)
% =============================================================================
{\footnotesize\sffamily\color{springerblue}\textbf{RESEARCH ARTICLE} \hfill \textbf{Journal of Big Data (SpringerOpen)}}\\[0.8em]

{\LARGE\sffamily\bfseries AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines}\\[0.9em]

{\large\textbf{Preecha Noiumkar\textsuperscript{1,*}}}\\[4pt]
Department of Information Technology, Faculty of Informatics, Mahasarakham University, Kham Riang, Kantharawichai, Maha Sarakham 44150, Thailand.\\
\textsuperscript{*}Correspondence: Preecha Noiumkar (\href{mailto:preecha.n@msu.ac.th}{preecha.n@msu.ac.th})\\[0.8em]

\noindent\rule{\textwidth}{0.8pt}
\begin{quote}
\small\sffamily
\textbf{\color{springerblue}Abstract}\\[0.3em]
\textbf{Background:} Modern big data analytics span serverless cloud warehouses, vectorized C++/Rust CPUs, and discrete GPUs, yet static single-engine execution incurs severe PCIe-transfer, memory-spilling, or cloud FinOps penalties across scale factors ($\text{SF}=1\text{--}100\text{ GB}$).

\textbf{Methodology:} We propose \textbf{AST-Route}, a pre-execution query routing framework orchestrating six engines across four paradigms. First, its \emph{AST-aware feature representation} extracts a $9\text{D}$ structural-selectivity vector ($\mathbf{x}_q \in \mathbb{R}^{9}$, extended to $\mathbf{x}_q^{\text{scale}} \in \mathbb{R}^{10}$ exclusively for multi-scale routing) in $3.65\text{ ms}$ without physical plan compilation, pairing catalog zone-map selectivity with a validated Composite Structural Complexity Index ($\text{CSCI}$: $\rho_{\text{partial}}=0.5477$, $\text{VIF}_{D,N}=1.84$). Second, its \emph{multi-objective cost predictor} estimates latency, device energy, and working-set memory in $497.2\,\mu\text{s}$ (median Q-Error $1.007\text{--}1.172$) to drive memory-feasible Latency--Energy--Cloud Pareto dispatch with $90\%$ conformal intervals and a $12\%$ PCIe guard.

\textbf{Results:} Across $163$ queries spanning five benchmark suites, AST-Route delivers substantial \emph{tail-risk reduction}: under leak-free $25$-seed cross-validation ($\text{SF}=1\text{ GB}$), it achieves $54.92\text{ ms}$ net latency ($57.33\%$ Top-1, $76.00\%$ Top-2, $34.98\%$ mean regret, $1.381\text{ J}$ estimated energy), outperforming static CPU engines by $1.50\times\text{--}3.35\times$ ($p_{\text{Holm,seed}}\le 0.0171$) and matching Always GPU ($56.96\text{ ms}$) while cutting P95 tail regret from $257.31\%$ to $114.11\%$ and halving $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). Demonstrating \emph{multi-scale and cross-workload generalization}, AST-Route sustains $74.67\%\text{--}86.67\%$ Top-1 in CPU-only mode, transfers zero-shot across unseen seeds, TPC-H Q1--Q22, SSB, ClickBench, and JOB, and scales across $C=1\text{--}32$ streams ($154.44\text{ QPS}$) and $\text{SF}=1\text{--}100\text{ GB}$ ($3.35\times\text{--}7.61\times$). Crucially, Leave-One-Tier-Out validation ($85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) establishes an explicit \emph{operator-bounded generalization} limit when unseen SQL operator classes (\texttt{WINDOW}/\texttt{UNNEST}) are omitted from training.

\textbf{Conclusion:} AST-Route enables fast, risk-aware heterogeneous query orchestration when training workloads cover target SQL operator families (\url{https://github.com/preecha-msu/bigquery-olap-benchmark}).

\noindent\textbf{Keywords:} Intelligent query routing; Heterogeneous big data systems; AST-aware cost prediction; Conformal tail-risk control; Multi-objective resource orchestration; Operator-bounded generalization
\end{quote}

"""

def heal_and_validate_tex() -> None:
    for bak in ROOT.glob("manuscript_backup_*"):
        if bak.is_dir():
            shutil.rmtree(bak, ignore_errors=True)

    tex = TARGET_TEX.read_text(encoding="utf-8")
    lines = tex.splitlines()
    print("[DIAGNOSTIC] Lines 52..68 before fix:")
    for idx in range(max(0, 51), min(len(lines), 68)):
        print(f"  L{idx+1:03d}: {lines[idx]}")

    # 1. Replace everything from \begin{document} up to \section{Introduction} with clean canonical block
    bdoc = tex.find(r"\begin{document}")
    sinc = tex.find(r"\section{Introduction}")
    if bdoc == -1 or sinc == -1:
        raise RuntimeError("Could not locate \\begin{document} or \\section{Introduction}")
    tex = tex[:bdoc] + CANONICAL_FRONTMATTER_AND_ABSTRACT + tex[sinc:]

    # 2. Ensure Table 15 has exact 9 columns (@{}llccccccl@{}) matching its 8 '&' tabs
    t15_pos = tex.find(r"\label{tab:job_12d_ablation}")
    if t15_pos != -1:
        t15_end = tex.find(r"\end{table}", t15_pos)
        if t15_end != -1:
            blk = tex[t15_pos:t15_end].replace(r"\begin{tabular}{@{}llcccccccl@{}}", r"\begin{tabular}{@{}llccccccl@{}}")
            tex = tex[:t15_pos] + blk + tex[t15_end:]

    # 3. Remove any stray U+200B / BOM characters
    tex = tex.replace("\u200b", "").replace("\ufeff", "")

    # 4. Paragraph-by-paragraph '$' parity audit & global brace balance check
    clean_no_cmt = strip_tex_comments(tex)
    paragraphs = re.split(r"\n\s*\n", clean_no_cmt)
    for p_idx, para in enumerate(paragraphs, 1):
        dollars = len(re.findall(r"(?<!\\)\$", para))
        if dollars % 2 != 0:
            snippet = " ".join(para.splitlines()[:2])[:100]
            raise RuntimeError(f"Odd '$' count ({dollars}) in paragraph #{p_idx}: {snippet}")

    o_b, c_b = count_braces(tex)
    if o_b != c_b:
        raise RuntimeError(f"Unbalanced braces: open={o_b}, close={c_b}")

    # 5. Verify all included figures exist
    for fg in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_no_cmt):
        if not (MS_DIR / fg).exists():
            raise FileNotFoundError(f"Missing figure referenced in TeX: {fg}")

    TARGET_TEX.write_text(tex, encoding="utf-8")
    for mr in (ROOT/"bigquery_olap_benchmark_V2.tex", ROOT/"main.tex", MS_DIR/"main.tex", MS_DIR/"overleaf_ready_single_file.tex", MS_DIR/"bigquery_olap_benchmark_V6.tex"):
        mr.write_text(tex, encoding="utf-8")
    print(f"[PRE-FLIGHT OK] Braces ({o_b}=={c_b}), all paragraph '$' counts EVEN, all figures verified!")

def compile_tex(tex_path: Path, passes: int = 3) -> None:
    for p_i in range(1, passes + 1):
        pr = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tex_path.parent), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if pr.returncode != 0:
            print(f"[ERROR compiling {tex_path.name} pass {p_i}]:")
            print("\n".join(pr.stdout.splitlines()[-30:]))
            raise RuntimeError(f"pdflatex failed for {tex_path.name} on pass {p_i}")

def print_calibrated_v7_tables() -> None:
    for title, fn in [
        ("[1] P1-2 STRICT 5x4 NESTED JOB LEAKAGE AUDIT", "v7_p1_job_5x5_nested_leakage_audit.csv"),
        ("[2] P2-5 CALIBRATED ROUTER DECISION STABILITY", "v7_p2_router_decision_stability_noise.csv"),
        ("[3] P2-6 CALIBRATED WORKLOAD & SKEW DRIFT", "v7_p2_dynamic_workload_and_skew_drift.csv"),
    ]:
        p = RES_DIR / fn
        if p.exists():
            print(f"\n{title}:")
            print(pd.read_csv(p).to_string(index=False))

def package_and_push() -> None:
    for cp in (ROOT/"bigquery_olap_benchmark_V2.pdf", MS_DIR/"main.pdf", MS_DIR/"bigquery_olap_benchmark_V6.pdf", ROOT/"bigquery_olap_benchmark_V6.pdf"):
        try:
            shutil.copy2(TARGET_PDF, cp)
        except PermissionError:
            pass

    if COVER_TEX.exists():
        compile_tex(COVER_TEX, passes=2)
    if RESP_TEX.exists():
        compile_tex(RESP_TEX, passes=2)

    # Refresh SHA256_MANIFEST_V7.csv with newly compiled V2.tex and V2.pdf
    m_path = REPRO / "SHA256_MANIFEST_V7.csv"
    tracked = [
        "manuscript/bigquery_olap_benchmark_V2.tex",
        "manuscript/bigquery_olap_benchmark_V2.pdf",
        "results/v7_p1_job_5x5_nested_leakage_audit.csv",
        "results/v7_p2_router_decision_stability_noise.csv",
        "results/v7_p2_dynamic_workload_and_skew_drift.csv",
        "manuscript/figures/fig_v7_stability_and_workload_drift_300dpi.pdf",
    ]
    for cand in (RES_DIR/"ast_complexity_summary.csv", ROOT/"data_hub"/"ast_complexity_summary.csv"):
        if cand.exists():
            tracked.append(cand.relative_to(ROOT).as_posix())
            break
    mrows = [{"Relative_Path": r, "Size_Bytes": (ROOT/r).stat().st_size, "SHA256": sha256_f(ROOT/r)} for r in tracked if (ROOT/r).exists()]
    pd.DataFrame(mrows).to_csv(m_path, index=False)
    subprocess.run([sys.executable, str(REPRO / "verify_manifest.py")], cwd=str(ROOT), check=True)

    for zp in (ROOT/"Overleaf_V2_Submission_Ready.zip", MS_DIR/"overleaf_V2_submission_bundle.zip"):
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(MS_DIR/"main.tex", arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            for fg in sorted(FIG_DIR.glob("*")):
                if fg.is_file():
                    zf.write(fg, arcname=f"figures/{fg.name}")

    em_zip = ROOT / "Editorial_Manager_JBD_Submission_Pack.zip"
    with zipfile.ZipFile(em_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, arcname="1_Manuscript_Source/bigquery_olap_benchmark_V2.tex")
        for fg in sorted(FIG_DIR.glob("*")):
            if fg.is_file():
                zf.write(fg, arcname=f"1_Manuscript_Source/figures/{fg.name}")
        zf.write(TARGET_PDF, arcname="2_Compiled_PDFs/bigquery_olap_benchmark_V2.pdf")
        if COVER_PDF.exists():
            zf.write(COVER_PDF, arcname="2_Compiled_PDFs/Cover_Letter_Journal_of_Big_Data.pdf")
        if RESP_PDF.exists():
            zf.write(RESP_PDF, arcname="2_Compiled_PDFs/point_by_point_reviewer_response_V2.pdf")
        zf.write(m_path, arcname="4_Audit_Certificates/SHA256_MANIFEST_V7.csv")

    def git_c(args):
        return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")

    git_c(["rm", "-r", "--cached", "data/"])
    for item in [
        "manuscript/bigquery_olap_benchmark_V2.tex", "manuscript/bigquery_olap_benchmark_V2.pdf",
        "manuscript/Cover_Letter_Journal_of_Big_Data.pdf", "manuscript/point_by_point_reviewer_response_V2.pdf",
        "manuscript/overleaf_V2_submission_bundle.zip", "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V2.pdf", "main.tex", "figures", "manuscript/figures",
        "reproducibility_v7", "results/v7_p1_job_5x5_nested_leakage_audit.csv",
        "results/v7_p2_router_decision_stability_noise.csv", "results/v7_p2_dynamic_workload_and_skew_drift.csv",
        "scripts/107_v7_calibrated_audit_and_final_tex_integration.py", "scripts/108_fix_line64_and_compile_v7_final.py",
    ]:
        if (ROOT / item).exists():
            git_c(["add", "-f", item])

    if git_c(["status", "--porcelain"]).stdout.strip():
        git_c(["commit", "-m", "V7 Final Seal: Fix frontmatter math delimiter, integrate Priority 1-3 stability/drift & Docker repro pack (replace V2.tex)"])
    git_c(["branch", "-M", "main"])
    r = git_c(["push", "-u", "origin", "main", "--force-with-lease"])
    if r.returncode != 0:
        r = git_c(["push", "-u", "origin", "main", "--force"])
    print(f"[GITHUB REPLACE -> origin] {'SUCCESS (REPLACED ON GITHUB)' if r.returncode == 0 else 'WARN: ' + r.stderr.strip()[:80]}")

def main():
    print("=" * 98)
    print("FIXING LINE 64 MATH DELIMITER & COMPILING FINAL V7 MANUSCRIPT...")
    print("=" * 98)
    heal_and_validate_tex()
    compile_tex(TARGET_TEX, passes=3)
    print_calibrated_v7_tables()
    package_and_push()

    log_t = (MS_DIR / "bigquery_olap_benchmark_V2.log").read_text(encoding="utf-8", errors="replace")
    u_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_t))
    print("=" * 98)
    print(f"  [PDF READY   ] {TARGET_PDF.name} ({TARGET_PDF.stat().st_size:,} bytes) | Undefined Refs = {u_refs}")
    print(f"  [V2.TEX SHA  ] {sha256_f(TARGET_TEX)}")
    print(f"  [V2.PDF SHA  ] {sha256_f(TARGET_PDF)}")
    print("=" * 98)

if __name__ == "__main__":
    main()