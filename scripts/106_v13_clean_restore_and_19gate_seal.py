#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/106_v13_clean_restore_and_19gate_seal.py
Restores intact V13.tex from Git HEAD, applies all reviewer-verified patches safely
(zero cross-table DOTALL deletion), enforces 19/19 Gates (including G19 table preservation),
compiles PDF, builds Clean-Room Flat13.zip, and writes SHA-256 audit log.
"""

from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
MS_FIG_DIR = MS_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
TEX_PATH = MS_DIR / "bigquery_olap_benchmark_V13.tex"
PDF_PATH = MS_DIR / "bigquery_olap_benchmark_V13.pdf"
MACROS_PATH = MS_DIR / "v13_verified_macros.tex"
ZIP_PATH = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat13.zip"
AUDIT_JSON = RESULTS_DIR / "v13_deep_audit_verification_sha256.json"
OFFICIAL_REPO_URL = "https://github.com/preecha-msu/bigquery-olap-benchmark"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(txt: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())


def restore_and_patch() -> tuple[str, int]:
    # 1. Restore intact V13.tex from Git HEAD (recovering Tables 8-14)
    subprocess.run(
        ["git", "checkout", "--", "manuscript/bigquery_olap_benchmark_V13.tex"],
        cwd=str(ROOT),
        check=True,
    )
    tex = TEX_PATH.read_text(encoding="utf-8")
    n_tables_orig = tex.count(r"\begin{table")
    print(f"[RESTORE] Restored intact {TEX_PATH.name} from Git HEAD ({n_tables_orig} tables)")

    # 2. Ensure v13_verified_macros.tex is included once
    if "v13_verified_macros.tex" not in tex and MACROS_PATH.exists():
        tex = re.sub(
            r"(\\documentclass(?:\[[^\]]*\])?\{[^\}]+\})",
            lambda m: m.group(1) + "\n\\input{v13_verified_macros.tex}",
            tex,
            count=1,
        )

    # 3. Fix corrupted LaTeX artifacts & unify official GitHub URL (no nested \url, no raw markdown)
    tex = re.sub(r"\\?ln\(\\1\+T_?e\)", lambda _: r"\ln(1 + T_e)", tex)
    tex = tex.replace(r"(\1+Te)", r"(1 + T_e)").replace(r"\1+Te", r"1 + T_e")
    tex = re.sub(r"RAM\[e,\s*VRAM\s*\\e\]?", lambda _: r"[\text{RAM}_e, \text{VRAM}_e]", tex)
    tex = re.sub(
        r"https?://github\.com/[A-Za-z0-9_.-]+/(?:db_cluster_distributed|my_research_files|bigquery-olap-benchmark)(?:\.git)?",
        lambda _: OFFICIAL_REPO_URL,
        tex,
    )
    tex = tex.replace("preechan-glitch/my_research_files", "preecha-msu/bigquery-olap-benchmark")
    tex = tex.replace("preecha-msu/db_cluster_distributed", "preecha-msu/bigquery-olap-benchmark")
    tex = re.sub(r"\\url\{\[https?://[^\]]+\]\((https?://[^\)]+)\)\}", lambda m: f"\\url{{{m.group(1)}}}", tex)
    tex = re.sub(r"(?<!\\url\{)(?<!\\href\{)\[([^\]]+)\]\((https?://[^\)]+)\)", lambda m: f"{m.group(1)} (\\url{{{m.group(2)}}})", tex)
    while r"\url{\url{" in tex:
        tex = re.sub(r"\\url\{\\url\{([^\}]+)\}\}", lambda m: f"\\url{{{m.group(1)}}}", tex)

    # 4. Eliminate holdout 0.0018 contradictions (keep 0.0018 ONLY on Mondrian drift lines)
    fixed_lines = []
    for ln in tex.splitlines():
        if "0.0018" in ln and "mondrian" not in ln.lower():
            ln = re.sub(r"p\s*=\s*0\.0018", "p = 0.125", ln)
            ln = re.sub(r"0\.0018", "0.125", ln)
        fixed_lines.append(ln)
    tex = "\n".join(fixed_lines)

    # 5. Single Structured Abstract (Background/Methods/Results/Conclusions, <= 270 words) & delete duplicate L133
    structured_quote = (
        "\\begin{quote}\n"
        "\\small\\sffamily\n"
        "\\textbf{\\color{springerblue}Abstract}\\\\[0.3em]\n"
        "\\textbf{Background:} Hybrid analytical query dispatch across in-process CPU engines (DuckDB, Polars, DataFusion, SQLite), local GPUs, and serverless cloud warehouses (Google BigQuery) suffers severe tail-latency regressions when small or selective queries incur PCIe transfer or cloud orchestration overheads.\n"
        "\\textbf{Methods:} We evaluate a pre-execution dispatch framework combining engine-agnostic SQL AST parsing and Parquet footer metadata sketching ($172\\,\\mu\\text{s}$ cached; $4.15\\text{ ms}$ cold parse+sketch) with a post-hoc harmonized PCIe margin guard across a 400-query benchmark spanning four hardware nodes (75 TPC-H development queries across $K=25$ seeds, 55 cross-schema calibration queries, 115 extended/public-trace queries, and 155 standard holdout queries including 99 TPC-DS, 43 ClickBench [20 native SQL + 23 dialect-transpiled], and 13 SSB; plus 30 author-synthesized \\texttt{Redset-Synth} telemetry queries audited separately).\n"
        "\\textbf{Results:} Three-way ablation shows that PCIe margin guarding drives 81.3\\% of P95 tail-regret reduction, while a two-engine portfolio (\\{Polars, GPU\\}) retains 98.7\\% of full six-engine utility; AST/GBR/CSCI enable cross-dialect dispatch without engine-specific planner lock-in (though native DuckDB \\texttt{EXPLAIN-DAG} is faster at $1.41\\text{ ms}$ on single-dialect queries). On development LOSO-CV ($N=75$), P95 regret drops from 257.31\\% (Always GPU) and 358.40\\% (Always Polars) to 114.11\\%. On external holdout clusters ($K=38$), tail-regret reduction is directional but not statistically significant (cluster permutation $p = 0.125$ vs.\\ CPU, $p = 0.198$ vs.\\ GPU). On out-of-distribution prospective holdouts (\\texttt{Seed S7/DSB/JCC-H}, $n=25$), zero-shot accuracy drops to 12.0\\% Top-1 (83.98\\% mean regret, 346.0\\% P95 regret), trailing Always Polars (66.75\\% mean regret).\n"
        "\\textbf{Conclusions:} Tail-safe hybrid OLAP dispatch is governed primarily by transfer-margin abstention and compact engine pools, and requires schema-local calibration before deployment on skewed or out-of-distribution workloads.\n\n"
        "\\noindent\\textbf{Keywords:} Cross-engine query routing; Tail-regret control; Abstention guards; Heterogeneous OLAP systems; Apache Arrow and GPU analytics; Pre-execution metadata\n"
        "\\end{quote}"
    )

    sec1 = tex.find(r"\section{Introduction}")
    if sec1 == -1:
        sec1 = tex.find(r"\section{INTRODUCTION}")
    head, tail = tex[:sec1], tex[sec1:]
    head = re.sub(r"\n*Hybrid analytical query routers must choose across vectorized CPU[^\n]*\n*", "\n\n", head)
    head = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", head, flags=re.DOTALL)
    q_matches = list(re.finditer(r"\\begin\{quote\}.*?\\end\{quote\}", head, flags=re.DOTALL))
    if q_matches:
        s0, e0 = q_matches[0].span()
        head = head[:s0] + structured_quote + head[e0:]
        for extra in reversed(list(re.finditer(r"\\begin\{quote\}.*?\\end\{quote\}", head, flags=re.DOTALL))[1:]):
            head = head[: extra.start()] + head[extra.end() :]
    else:
        head = head.rstrip() + "\n\n" + structured_quote + "\n\n"
    tex = head + tail

    # 6. ClickBench split, VIF 3.84, Dialect-Agnostic EXPLAIN-DAG note, Table 16 claim
    tex = re.sub(r"\$?43\$?\s+(?:verbatim\s+|untouched\s+)?ClickBench(?!\s*[\[\(]20\s+native)", lambda _: "43 ClickBench (20 native SQL + 23 dialect-transpiled)", tex, flags=re.I)
    tex = tex.replace(r"$\max\text{VIF}=1.86$ across the six raw operators", r"$\text{VIF}_{\max}=3.84$ across the six raw operators (matching Table~\ref{tab:v12_csci_stat_energy})")
    tex = tex.replace(r"($3.6\times$ faster, zero catalog coupling)", r"(dialect-agnostic with zero catalog coupling; note that for single-dialect DuckDB-only queries, native \texttt{EXPLAIN-DAG} is faster at $1.41\text{ ms}$, Table~3 footnote~$^\ddagger$)")
    tex = re.sub(r"100(?:\.0)?\\%\s*routing\s*accuracy", lambda _: r"4/4 illustrative cross-node oracle crossover regimes (not a per-node router accuracy evaluation)", tex, flags=re.I)

    # 7. Retire LME (-62.06 pp, SE=1.08, z=-57.37/-16.8/-3.12) cleanly
    tex = re.sub(
        r"Fitting a Restricted Maximum Likelihood \(REML\) \\textbf\{Hierarchical Linear Mixed-Effects Model\}[^\n]*?(?:\$p\s*=\s*0\.125\$|10\^\{-15\})[^\n]*?\)\.",
        lambda _: r"Because random-intercept-only LME (\texttt{1 | Family}) omits random policy slopes and underestimates SEs when pooled with development queries, we retire LME $z$-scores from primary inference and rely strictly on holdout-only cluster permutation ($K=38$ external clusters: $p_{\text{perm}} = 0.125$ vs.\ CPU, $p_{\text{perm}} = 0.198$ vs.\ GPU; directional but non-significant).",
        tex,
    )
    tex = re.sub(
        r"\\textbf\{LME Model \(\$G_\{\\text\{total\}\}=42\$, \$N=312\$Q\)\}[^\n]*\\\\",
        lambda _: r"\textbf{Holdout Cluster Perm.\ ($K=38$, $N=68$Q)} & Sign-Flip Perm.\ (LME Retired) & $\mathbf{-38.32\text{ pp}}$ ($\text{SE}_{\text{CR1}}=33.30$) & $95\%\text{ CI }[-30.0,-0.3]$ & $p_{\text{perm}}=0.125$ & $\mathbf{p_{\text{perm}}=0.125}$ (NS) & \textbf{Directional; LME retired (no random slope)} \\",
        tex,
    )
    tex = re.sub(
        r"Across \$\\mathbf\{G_\{\\text\{total\}\}=42\}\$ \\textbf\{independent template families \(\$N=312\$\)\}, REML linear mixed-effects modeling confirms a \$-62\.06\\text\{ pp\}\$ regret reduction[^\n]*?\)",
        lambda _: r"On strictly external holdout clusters ($K=38$, retiring random-intercept LME), cluster-level sign-flip permutation shows directional but non-significant regret reduction ($p_{\text{perm}}=0.125$ vs.\ CPU, $p_{\text{perm}}=0.198$ vs.\ GPU)",
        tex,
    )
    tex = re.sub(r"z\s*=\s*[-−](?:57\.37|16\.8|3\.12)", lambda _: r"p_{\text{perm}} = 0.125", tex)

    # 8. Line-by-line G08 (Panel A), G09 (Table 13 Cold n=1), G10 (Table 14 Warm R=5)
    out_lines = []
    for ln in tex.splitlines():
        if "Panel A:" in ln and "Chronology" in ln:
            ln = r"\multicolumn{3}{@{}l}{\textbf{Panel A: True Git Endpoint \& Guard Chronology --- Locked Single-Shot Holdout (Not Preregistered; Post-Hoc Harmonization Disclosed)}} \\"
        if ("3578.44" in ln or "9882.64" in ln) and "(Cold $n=1$)" not in ln:
            ln = ln.replace("GB", "GB (Cold $n=1$)", 1)
        if "DuckDB" in ln and ("850.32" in ln or "3297" in ln or "3,297" in ln) and "(Warm $R=5$)" not in ln:
            ln = ln.replace("GB", "GB (Warm $R=5$)", 1)
        out_lines.append(ln)
    tex = "\n".join(out_lines) + "\n"

    # 9. Insert §5.1 Prospective Failure Analysis + 3-Suite Table with Mahalanobis OOD Gate column (if not already present)
    new_3suite_tbl = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Empirical per-suite failure breakdown of the frozen zero-shot router on the 25-query prospective out-of-distribution suite (\\texttt{Seed\\_S7}, \\texttt{DSB\\_Subset}, and \\texttt{JCC\\_H\\_Skew}; SHA-256 lock \\texttt{2a8acab7...}), including Mahalanobis OOD gate warning/abstention rates.}\n"
        "\\label{tab:prospective_3suite_failure}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Prospective Suite} & \\textbf{$n$} & \\textbf{Top-1} & \\textbf{Mean Regret} & \\textbf{P95 Regret} & \\textbf{Routed to DataFusion} & \\textbf{OOD Gate (Mahalanobis) Flag/Abstain} \\\\\n"
        "\\midrule\n"
        "\\texttt{Seed\\_S7} (Unseen TPC-H Seed) & 15 & 6.7\\% & 114.23\\% & 421.08\\% & 66.7\\% & 0.0\\% (Missed shift) \\\\\n"
        "\\texttt{DSB\\_Subset} (Decision Support) & 5 & 40.0\\% & 26.22\\% & 52.22\\% & 80.0\\% & 0.0\\% (Missed shift) \\\\\n"
        "\\texttt{JCC\\_H\\_Skew} (Zipfian Skew Joins) & 5 & 0.0\\% & 51.12\\% & 92.17\\% & 100.0\\% & 0.0\\% (Missed skew) \\\\\n"
        "\\midrule\n"
        "\\textbf{Overall Prospective} & \\textbf{25} & \\textbf{12.0\\%} & \\textbf{83.98\\%} & \\textbf{346.00\\%} & \\textbf{76.0\\%} & \\textbf{0.0\\% (Primary Limitation)} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\vspace{2pt}\n"
        "\\parbox{\\textwidth}{\\scriptsize \\textit{Primary Limitation:} Because the Mahalanobis OOD gate operates on static 9D AST+selectivity coordinates, it failed to flag or abstain on any of the 25 prospective queries (0.0\\% OOD trigger rate) despite severe Zipfian join-skew and schema shifts, confirming that static AST-space OOD gating cannot detect runtime cardinality skew without schema-local calibration.}\n"
        "\\end{table}"
    )
    if "tab:prospective_3suite_failure" in tex:
        # Non-crossing per-table replacement: each match stops at its own \end{table}
        tex = re.sub(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
            lambda m: new_3suite_tbl if "tab:prospective_3suite_failure" in m.group(0) else m.group(0),
            tex,
            flags=re.DOTALL,
        )
    else:
        prosp_block = (
            "\n\n"
            r"\paragraph{Prospective Out-of-Distribution Holdout Failure Analysis (\texttt{Seed S7 / DSB / JCC-H}).} "
            r"To test strict zero-shot transfer beyond our development distribution, we evaluated the frozen router once on 25 prospective out-of-distribution queries (\texttt{Seed\_S7}, \texttt{DSB\_Subset}, and \texttt{JCC\_H\_Skew}). "
            r"Zero-shot dispatch degraded to $12.0\%$ Top-1 accuracy, $83.98\%$ mean regret, and $346.00\%$ P95 regret—underperforming both Always Polars ($66.75\%$ mean regret) and Always DuckDB ($186.74\%$ P95 regret). "
            r"Table~\ref{tab:prospective_3suite_failure} reports the empirical per-suite breakdown and Mahalanobis OOD gate trigger rates."
            "\n"
            "\\begin{table}[!htbp]\n"
            "\\centering\n"
            "\\caption{Empirical per-suite failure breakdown of the frozen zero-shot router on the 25-query prospective out-of-distribution suite (\\texttt{Seed\\_S7}, \\texttt{DSB\\_Subset}, and \\texttt{JCC\\_H\\_Skew}; SHA-256 lock \\texttt{2a8acab7...}), including Mahalanobis OOD gate warning/abstention rates.}\n"
            "\\label{tab:prospective_3suite_failure}\n"
            "\\scriptsize\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{@{}lcccccc@{}}\n"
            "\\toprule\n"
            "\\textbf{Prospective Suite} & \\textbf{$n$} & \\textbf{Top-1} & \\textbf{Mean Regret} & \\textbf{P95 Regret} & \\textbf{Routed to DataFusion} & \\textbf{OOD Gate (Mahalanobis) Flag/Abstain} \\\\\n"
            "\\midrule\n"
            "\\texttt{Seed\\_S7} (Unseen TPC-H Seed) & 15 & 6.7\\% & 114.23\\% & 421.08\\% & 66.7\\% & 0.0\\% (Missed shift) \\\\\n"
            "\\texttt{DSB\\_Subset} (Decision Support) & 5 & 40.0\\% & 26.22\\% & 52.22\\% & 80.0\\% & 0.0\\% (Missed shift) \\\\\n"
            "\\texttt{JCC\\_H\\_Skew} (Zipfian Skew Joins) & 5 & 0.0\\% & 51.12\\% & 92.17\\% & 100.0\\% & 0.0\\% (Missed skew) \\\\\n"
            "\\midrule\n"
            "\\textbf{Overall Prospective} & \\textbf{25} & \\textbf{12.0\\%} & \\textbf{83.98\\%} & \\textbf{346.00\\%} & \\textbf{76.0\\%} & \\textbf{0.0\\% (Primary Limitation)} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\vspace{2pt}\n"
            "\\parbox{\\textwidth}{\\scriptsize \\textit{Primary Limitation:} Because the Mahalanobis OOD gate operates on static 9D AST+selectivity coordinates, it failed to flag or abstain on any of the 25 prospective queries (0.0\\% OOD trigger rate) despite severe Zipfian join-skew and schema shifts, confirming that static AST-space OOD gating cannot detect runtime cardinality skew without schema-local calibration.}\n"
            "\\end{table}\n\n"
        )
        for anchor in (r"\\section\{Discussion", r"\\subsection\{Threats to Validity", r"\\section\{Threats to Validity", r"\\section\{Conclusion"):
            if re.search(anchor, tex):
                tex = re.sub(anchor, lambda m: prosp_block + m.group(0), tex, count=1)
                break

    # Verify brace balance
    cleaned = re.sub(r"\\[\{\}]", "", strip_comments(tex))
    if cleaned.count("{") != cleaned.count("}"):
        raise RuntimeError(f"Brace imbalance: {{={cleaned.count('{')} vs }}={cleaned.count('}')}")

    TEX_PATH.write_text(tex, encoding="utf-8")
    return tex, n_tables_orig


def run_19_gate_audit(tex: str, n_tables_orig: int) -> dict:
    no_comm = strip_comments(tex)
    bad_0018 = [ln.strip() for ln in no_comm.splitlines() if "0.0018" in ln and "mondrian" not in ln.lower()]
    gates = {
        "G01_No_Holdout_0018": len(bad_0018) == 0,
        "G02_No_Duplicate_Abstract_L133": "Hybrid analytical query routers must choose across vectorized CPU" not in no_comm,
        "G03_No_LME_Beta_6206": "-62.06" not in no_comm,
        "G04_No_LME_Z_Scores": not bool(re.search(r"z\s*=\s*[-−](?:57\.37|16\.8|3\.12)", no_comm)),
        "G05_No_Broken_Dollar_Holdout": "$holdout cluster-permutation" not in no_comm,
        "G06_VIF_384_Reconciled_L150": r"\max\text{VIF}=1.86" not in no_comm,
        "G07_Dialect_Agnostic_EXPLAIN_Note": "dialect-agnostic with zero catalog coupling" in no_comm,
        "G08_Locked_Single_Shot_Label": "Locked Single-Shot Holdout (Not Preregistered" in no_comm,
        "G09_Table13_Cold_n1_Annotated": "(Cold $n=1$)" in no_comm,
        "G10_Table14_Warm_R5_Annotated": "(Warm $R=5$)" in no_comm,
        "G11_No_Raw_Markdown_In_URL": r"\url{[https://" not in no_comm,
        "G12_Mahalanobis_OOD_Col_In_3Suite_Table": "OOD Gate (Mahalanobis) Flag/Abstain" in no_comm,
        "G13_No_Corrupted_LaTeX_Artifacts": (r"\1+Te" not in no_comm) and ("RAM[e,VRAM" not in no_comm),
        "G14_Single_Official_GitHub_URL": ("preechan-glitch" not in no_comm) and ("db_cluster_distributed" not in no_comm),
        "G15_No_Nested_URL": r"\url{\url{" not in no_comm,
        "G16_ClickBench_20_Native_23_Transpiled": "20 native SQL + 23 dialect-transpiled" in no_comm,
        "G17_No_Table16_100pct_Router_Claim": not bool(re.search(r"100(?:\.0)?\\%\s*routing\s*accuracy", no_comm, flags=re.I)),
        "G18_Prospective_3Suite_Table_Present": "tab:prospective_3suite_failure" in no_comm,
        "G19_All_Tables_Preserved": tex.count(r"\begin{table") >= n_tables_orig,
    }
    print("\n================ 19-GATE AUTOMATED CONSISTENCY AUDIT ================")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"  [INFO] Tables preserved: {tex.count(chr(92)+'begin{table')} >= {n_tables_orig}")
    print("=====================================================================\n")
    if not all(gates.values()):
        raise RuntimeError("Not all 19 gates passed!")
    return gates


def compile_and_package_flat13(tex: str) -> dict:
    stem = TEX_PATH.stem
    for ext in (".aux", ".out", ".toc", ".bbl", ".blg"):
        (MS_DIR / f"{stem}{ext}").unlink(missing_ok=True)

    print("[BUILD] Compiling bigquery_olap_benchmark_V13.tex (pdflatex -> bibtex -> pdflatex x2)...")
    for step_idx, cmd in enumerate(
        [
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX_PATH.name],
            ["bibtex", stem],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX_PATH.name],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX_PATH.name],
        ],
        start=1,
    ):
        res = subprocess.run(cmd, cwd=str(MS_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 and cmd[0] == "pdflatex":
            print("\n".join(res.stdout.splitlines()[-35:]))
            raise RuntimeError(f"pdflatex failed at step {step_idx}")

    log_txt = (MS_DIR / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    undef_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    undef_cites = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    m_pg = re.search(rf"Output written on {re.escape(stem)}\.pdf \((\d+) pages", log_txt)
    pages = int(m_pg.group(1)) if m_pg else -1
    print(f"[PDF PASS] {PDF_PATH.name} ({pages} pages | Undefined Refs={undef_refs} | Undefined Cites={undef_cites})")

    refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(tex))))
    resolved = []
    for ref in refs:
        cands = [MS_DIR / ref, ROOT / ref, MS_FIG_DIR / Path(ref).name, ROOT_FIG_DIR / Path(ref).name, MS_DIR / Path(ref).name]
        if not Path(ref).suffix:
            for ext in (".jpg", ".pdf", ".png", ".jpeg"):
                cands.extend([MS_DIR / (ref + ext), MS_FIG_DIR / (Path(ref).name + ext), ROOT_FIG_DIR / (Path(ref).name + ext)])
        fnd = next((c for c in cands if c.exists() and c.is_file()), None)
        if fnd is not None:
            resolved.append((ref, fnd))

    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        lambda m: m.group(1) + m.group(2),
        tex,
    )

    ZIP_PATH.unlink(missing_ok=True)
    support = ["references.bib", "sn-jnl.cls", "sn-mathphys-num.bst", "sn-basic.bst", "sn-apacite.bst", "sn-chicago.bst", "sn-mps.bst", "sn-vancouver.bst"]
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V13.tex", flat_tex.encode("utf-8"))
        if PDF_PATH.exists():
            zf.write(PDF_PATH, arcname="bigquery_olap_benchmark_V13.pdf")
        if MACROS_PATH.exists():
            zf.write(MACROS_PATH, arcname=MACROS_PATH.name)
        for s in support:
            for bd in (MS_DIR, ROOT):
                if (bd / s).exists():
                    zf.write(bd / s, arcname=s)
                    break
        seen = set()
        for _, ip in resolved:
            if ip.name not in seen:
                zf.write(ip, arcname=ip.name)
                seen.add(ip.name)

    cdir = ROOT / "_cleanroom_flat13_verify"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            if any("/" in n or "\\" in n for n in zf.namelist()):
                raise AssertionError("Nested folder found inside Flat13.zip!")
            zf.extractall(cdir)
        (cdir / "bigquery_olap_benchmark_V13.pdf").unlink(missing_ok=True)
        for cmd in [
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V13.tex"],
            ["bibtex", "bigquery_olap_benchmark_V13"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V13.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V13.tex"],
        ]:
            subprocess.run(cmd, cwd=str(cdir), capture_output=True, text=True, check=(cmd[0] == "pdflatex"))
        print(f"[CLEAN-ROOM PASS] {ZIP_PATH.name} verified ({ZIP_PATH.stat().st_size / 1024:.1f} KB)")
    finally:
        shutil.rmtree(cdir, ignore_errors=True)

    return {"pages": pages, "undef_refs": undef_refs, "undef_cites": undef_cites}


def main() -> None:
    tex, n_tables_orig = restore_and_patch()
    gates = run_19_gate_audit(tex, n_tables_orig)
    build = compile_and_package_flat13(tex)

    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    payload["gates_19"] = gates
    payload["build"] = build
    payload["artifacts_sha256"] = {
        TEX_PATH.name: sha256_file(TEX_PATH),
        PDF_PATH.name: sha256_file(PDF_PATH),
        MACROS_PATH.name: sha256_file(MACROS_PATH) if MACROS_PATH.exists() else "N/A",
        ZIP_PATH.name: sha256_file(ZIP_PATH),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[AUDIT] 19/19 GATES PASSED | SHA-256 updated in:", AUDIT_JSON)


if __name__ == "__main__":
    main()
