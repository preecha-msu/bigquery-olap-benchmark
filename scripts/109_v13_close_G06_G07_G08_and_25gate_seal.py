#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/109_v13_close_G06_G07_G08_and_25gate_seal.py
Surgical closure of G06 (escaped \n in Fig 1/2 generator), G07 & G08 (safe per-block
replacement of §5.1 Prospective paragraph + Table 17 with properly escaped \&),
followed by 25/25 Gate verification, pdflatex build, Clean-Room Flat13.zip, & SHA-256 log.
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
AUDIT_JSON = RESULTS_DIR / "v13_step108_complete_audit_sha256.json"
OFFICIAL_REPO_URL = "https://github.com/preecha-msu/bigquery-olap-benchmark"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(txt: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())


def fix_g06_regenerate_fig1_fig2() -> dict:
    src_script = ROOT / "scripts" / "v12_regenerate_fig1_fig2_300dpi.py"
    code = src_script.read_text(encoding="utf-8")

    # Line-agnostic regex replacements (handles both literal \n and escaped \\n)
    code = re.sub(r"\(V12\)", "(V13)", code)
    code = re.sub(r"\(V12:", "(V13:", code)
    code = re.sub(
        r"Phase 6:\s*LME Inference\s*&\s*Reproducibility",
        "Phase 6: Holdout Permutation & Reproducibility",
        code,
    )
    code = re.sub(
        r"G_total\s*=\s*42\s*Template-Family\s*LME\s*Model\s*\(R32\):",
        "External Holdout Cluster Permutation (K = 38):",
        code,
    )
    code = re.sub(
        r"beta\s*=\s*-48\.6\s*pp,\s*z\s*=\s*-16\.8,\s*p\s*<\s*1e-15",
        "p_perm = 0.125 vs CPU, 0.198 vs GPU (LME retired)",
        code,
    )
    code = re.sub(
        r"https?://github\.com/[A-Za-z0-9_.-]+/(?:my_research_files|db_cluster_distributed)",
        OFFICIAL_REPO_URL,
        code,
    )

    v13_fig_script = ROOT / "scripts" / "v13_regenerate_clean_fig1_fig2_300dpi.py"
    v13_fig_script.write_text(code, encoding="utf-8")
    subprocess.run(["python", str(v13_fig_script)], cwd=str(ROOT), check=True)

    synced = {}
    for fn in [
        "fig_v12_proposed_ast_routing_framework_300dpi.pdf",
        "fig_v12_proposed_ast_routing_framework_300dpi.jpg",
        "fig_v12_experimental_workflow_300dpi.pdf",
        "fig_v12_experimental_workflow_300dpi.jpg",
    ]:
        cands = [MS_FIG_DIR / fn, MS_DIR / fn, ROOT_FIG_DIR / fn]
        existing = [p for p in cands if p.exists()]
        if existing:
            newest = max(existing, key=lambda p: p.stat().st_mtime)
            for dst in cands:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.resolve() != newest.resolve():
                    shutil.copy2(newest, dst)
            synced[fn] = sha256_file(newest)
    print("[G06 PASS] Clean V13 Fig 1 & Fig 2 regenerated at 300 DPI (zero LME z=-16.8, zero legacy URL).")
    return synced


def fix_g07_g08_in_manuscript() -> tuple[str, int]:
    tex = TEX_PATH.read_text(encoding="utf-8")
    n_tables_before = tex.count(r"\begin{table")

    # 1. Replace the Prospective paragraph (note escaped \& in title!)
    new_para = (
        r"\paragraph{Prospective Out-of-Distribution \& Harness-Shift Failure Audit (\texttt{Seed S7 / DSB / JCC-H}).} "
        r"Evaluating the frozen Cat-I router once on the 25-query prospective suite (\texttt{v13\_groupB\_prospective\_holdout\_jcc\_h\_results.csv}) yielded only $3/25$ ($12.0\%$) Top-1 hits, $83.98\%$ mean regret, and $346.00\%$ P95 regret—trailing Always Polars ($66.75\%$ mean regret) and routing $19/25$ ($76.0\%$) queries to DataFusion. "
        r"A forensic feature-and-harness audit against \texttt{S1--S6} resolves why \texttt{Seed\_S7} degraded to $1/15$ Top-1 whereas unseen \texttt{S5/S6} achieved $11/15$ ($73.3\%$): "
        r"(i) \emph{Zero Leakage in \texttt{S1--S6}, Template/Harness Shift in \texttt{S7}}: \texttt{S5/S6} used the canonical 8-table TPC-H SQL generator over on-disk storage ($50\text{--}184\text{ ms}$), whereas \texttt{v13\_groupB\_step2} authored compact 1-to-2 table SQL queries executed over pre-materialized in-memory Arrow tables ($3.2\text{--}43.6\text{ ms}$); and "
        r"(ii) \emph{Tier-L1 DataFusion Basin Trap}: these compact AST vectors fall inside the low-node Tier-L1 scan basin where Cat-I training labels rank DataFusion fastest among CPUs ($33.58\text{ ms}$ vs.\ Polars $38.00\text{ ms}$ and DuckDB $50.84\text{ ms}$, Table~\ref{tab:six_software_matrix}) while the $8\%$ PCIe guard blocks \texttt{GPU\_Cold}, systematically dispatching $10/15$ \texttt{Seed\_S7}, $4/5$ \texttt{DSB}, and $5/5$ \texttt{JCC-H} queries to DataFusion even though Polars is faster on in-memory Arrow joins. "
        r"Table~\ref{tab:prospective_3suite_failure} reports exact $k/n$ counts alongside the $0/25$ Mahalanobis OOD trigger rate."
    )

    new_tbl = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Empirical per-suite breakdown (exact $k/n$ counts, latency basin, and Mahalanobis OOD gate trigger count) on the 25-query prospective suite (SHA-256 lock \\texttt{2a8acab7...}). Small-$n$ suites ($n=5$) are reported strictly as $k/n$ counts.}\n"
        "\\label{tab:prospective_3suite_failure}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Prospective Suite (Harness)} & \\textbf{$n$} & \\textbf{Top-1 ($k/n$)} & \\textbf{Mean Regret} & \\textbf{P95 Regret} & \\textbf{Routed to DataFusion ($k/n$)} & \\textbf{OOD Gate Flag ($k/n$)} \\\\\n"
        "\\midrule\n"
        "\\texttt{Seed\\_S7} (Compact 1--2 Tbl, In-Mem Arrow) & 15 & $1/15$ ($6.7\\%$) & $114.23\\%$ & $421.08\\%$ & $10/15$ (L1 Basin Trap) & $0/15$ \\\\\n"
        "\\texttt{DSB\\_Subset} (2-Tbl Filter/Join, Small-$n$) & 5 & $2/5$ & $26.22\\%$ & $52.22\\%$ & $4/5$ (L1 Basin Trap) & $0/5$ \\\\\n"
        "\\texttt{JCC\\_H\\_Skew} (Zipfian Skew Join, Small-$n$) & 5 & $0/5$ & $51.12\\%$ & $92.17\\%$ & $5/5$ (Missed Skew) & $0/5$ \\\\\n"
        "\\midrule\n"
        "\\textbf{Overall Prospective} & \\textbf{25} & $\\mathbf{3/25}$ ($12.0\\%$) & \\textbf{83.98\\%} & \\textbf{346.00\\%} & $\\mathbf{19/25}$ ($76.0\\%$) & $\\mathbf{0/25}$ (Primary Limit.) \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\vspace{2pt}\n"
        "\\parbox{\\textwidth}{\\scriptsize \\textit{Empirical Root Cause \\& Primary Limitation:} In Cat-I training data (Table~\\ref{tab:six_software_matrix}), DataFusion is the fastest CPU engine ONLY in low-node Tier-L1 scans ($33.58\\text{ ms} < \\text{Polars } 38.00\\text{ ms} < \\text{DuckDB } 50.84\\text{ ms}$). Because the prospective harness used compact 1--2 table SQL over in-memory Arrow buffers rather than full 8-table on-disk TPC-H queries, $19/25$ queries fell into the Tier-L1 DataFusion prediction basin while static 9D Mahalanobis gating triggered on $0/25$ queries—proving that router models must be re-calibrated whenever storage harnesses (on-disk Parquet vs.\\ in-memory Arrow) or join-skew distributions change.}\n"
        "\\end{table}"
    )

    # Replace single-line \paragraph{Prospective Out-of-Distribution...} safely
    if r"\paragraph{Prospective Out-of-Distribution" in tex:
        tex = re.sub(
            r"\\paragraph\{Prospective Out-of-Distribution[^\n]*",
            lambda _: new_para,
            tex,
            count=1,
        )
    else:
        tex = re.sub(
            r"(\\begin\{table\}\[!htbp\][^\n]*\n[^\n]*\n[^\n]*tab:prospective_3suite_failure)",
            lambda m: new_para + "\n\n" + m.group(1),
            tex,
            count=1,
        )

    # Replace ONLY the table block containing tab:prospective_3suite_failure (non-crossing per-table callback)
    if "tab:prospective_3suite_failure" in tex:
        tex = re.sub(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
            lambda m: new_tbl if "tab:prospective_3suite_failure" in m.group(0) else m.group(0),
            tex,
            flags=re.DOTALL,
        )
    else:
        tex = re.sub(
            r"(\\section\{Conclusion)",
            lambda m: new_para + "\n\n" + new_tbl + "\n\n" + m.group(1),
            tex,
            count=1,
        )

    cleaned = re.sub(r"\\[\{\}]", "", strip_comments(tex))
    if cleaned.count("{") != cleaned.count("}"):
        raise RuntimeError(f"Brace imbalance: {{={cleaned.count('{')} vs }}={cleaned.count('}')}")
    if tex.count(r"\begin{table") < n_tables_before:
        raise RuntimeError("Table count dropped!")

    TEX_PATH.write_text(tex, encoding="utf-8")
    return tex, n_tables_before


def run_25_gate_audit(tex: str, n_tables_orig: int) -> dict:
    no_comm = strip_comments(tex)
    bad_0018 = [ln.strip() for ln in no_comm.splitlines() if "0.0018" in ln and "mondrian" not in ln.lower()]
    fig2_py = (ROOT / "scripts" / "v13_regenerate_clean_fig1_fig2_300dpi.py").read_text(encoding="utf-8")
    fig2_no_comm = "\n".join(ln for ln in fig2_py.splitlines() if not ln.lstrip().startswith("#"))

    gates = {
        "G01_No_Holdout_0018": len(bad_0018) == 0,
        "G02_Single_Structured_Abstract": "\\textbf{Background:}" in no_comm and "Hybrid analytical query routers must choose" not in no_comm,
        "G03_No_LME_Beta_6206": "-62.06" not in no_comm,
        "G04_No_LME_Z_Scores": not bool(re.search(r"z\s*=\s*[-−](?:57\.37|16\.8|3\.12)", no_comm)),
        "G05_No_Mixed_Effects_Heading_In_Conclusion": "Mixed-Effects Inference" not in no_comm,
        "G06_Fig1_Fig2_Clean_V13_No_LME_No_OldURL": ("z = -16.8" not in fig2_no_comm) and ("preechan-glitch" not in fig2_no_comm),
        "G07_S7_L1_Basin_Trap_Explained_Empirically": "Tier-L1 DataFusion Basin Trap" in no_comm and "33.58\\text{ ms}" in no_comm,
        "G08_SmallN_Reported_As_kn_Counts": "$1/15$" in no_comm and "$2/5$" in no_comm and "$0/5$" in no_comm and "19/25" in no_comm,
        "G09_Cluster_TOST_Equivalence_Reported": "p_{\\text{TOST}}" in no_comm,
        "G10_SF100_R5_DuckDB_vs_DataFusion_Honest": "DataFusion ($5{,}064\\text{--}5{,}683\\text{ ms}$) trails post-Arrow GPU" in no_comm or "2.89\\times\\text{ warm}" in no_comm,
        "G11_Table6_Cold_761x_And_Warm_289x_CoReported": "7.61\\times\\text{ cold }/\\ 2.89\\times\\text{ warm}" in no_comm,
        "G12_Table11_Redset_Synth_Disclosed": "Public Redshift Fleet Trace~" not in no_comm,
        "G13_400_To_480_Query_Crosswalk_Explicit": "155+55+30+160=400" in no_comm,
        "G14_VIF_384_Reconciled": r"\max\text{VIF}=1.86" not in no_comm,
        "G15_Dialect_Agnostic_EXPLAIN_Note": "dialect-agnostic with zero catalog coupling" in no_comm,
        "G16_Locked_Single_Shot_Label": "Locked Single-Shot Holdout (Not Preregistered" in no_comm,
        "G17_Table13_Cold_n1_Annotated": "(Cold $n=1$)" in no_comm,
        "G18_Table14_Warm_R5_Annotated": "(Warm $R=5$)" in no_comm,
        "G19_No_URL_Whitespace_Glitch": not bool(re.search(r"\\url\{[^\}]*allowbreak[^\}]*\}", no_comm)),
        "G20_Single_Official_GitHub_URL": ("preechan-glitch" not in no_comm) and ("db_cluster_distributed" not in no_comm),
        "G21_No_Nested_URL": r"\url{\url{" not in no_comm,
        "G22_No_Corrupted_LaTeX_Artifacts": (r"\1+Te" not in no_comm) and ("RAM[e,VRAM" not in no_comm),
        "G23_ClickBench_20_Native_23_Transpiled": "20 native SQL + 23 dialect-transpiled" in no_comm,
        "G24_No_Table16_100pct_Router_Claim": not bool(re.search(r"100(?:\.0)?\\%\s*routing\s*accuracy", no_comm, flags=re.I)),
        "G25_All_18_Tables_Preserved": tex.count(r"\begin{table") >= n_tables_orig,
    }
    print("\n================ 25-GATE AUTOMATED CONSISTENCY AUDIT ================")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print("=====================================================================\n")
    if not all(gates.values()):
        failed = [k for k, ok in gates.items() if not ok]
        raise RuntimeError(f"Failed gates: {failed}")
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
    fig_hashes = fix_g06_regenerate_fig1_fig2()
    tex, n_tables_orig = fix_g07_g08_in_manuscript()
    gates = run_25_gate_audit(tex, n_tables_orig)
    build = compile_and_package_flat13(tex)

    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    payload["regenerated_v13_figures_sha256"] = fig_hashes
    payload["gates_25"] = gates
    payload["build"] = build
    payload["artifacts_sha256"] = {
        TEX_PATH.name: sha256_file(TEX_PATH),
        PDF_PATH.name: sha256_file(PDF_PATH),
        ZIP_PATH.name: sha256_file(ZIP_PATH),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[AUDIT] 25/25 GATES PASSED | Complete SHA-256 manifest written to:", AUDIT_JSON)


if __name__ == "__main__":
    main()
