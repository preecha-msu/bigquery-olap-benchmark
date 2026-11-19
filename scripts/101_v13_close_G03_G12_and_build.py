#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/101_v13_close_G03_G12_and_build.py
Surgical closure of G03 (Single Structured Abstract inside \begin{quote}...\end{quote})
and G12 (ClickBench 20 native + 23 dialect-transpiled split), followed by 12-Gate Audit,
pdflatex build, Clean-Room Flat13.zip verification, and SHA-256 logging.
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(txt: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())


def close_g03_and_g12() -> str:
    tex = TEX_PATH.read_text(encoding="utf-8")

    # 1. Close G03: Replace the frontmatter \begin{quote}...\end{quote} (before \section{Introduction})
    #    with ONE Structured Abstract (Background / Methods / Results / Conclusions) + Keywords
    structured_quote_abstract = (
        "\\begin{quote}\n"
        "\\small\\sffamily\n"
        "\\textbf{\\color{springerblue}Abstract}\\\\[0.3em]\n"
        "\\textbf{Background:} Hybrid analytical query dispatch across in-process CPU engines (DuckDB, Polars, DataFusion), discrete local GPUs, and serverless cloud warehouses (Google BigQuery) suffers from severe tail-latency regressions whenever small or selective queries incur PCIe host-to-device transfer or cloud orchestration overheads.\n"
        "\\textbf{Methods:} We evaluate a pre-execution dispatch framework combining zero-compile SQL Abstract Syntax Tree (AST) parsing and Parquet footer metadata sketching ($\\sim 172\\,\\mu\\text{s}$) with a post-hoc harmonized PCIe margin guard across a 400-query benchmark (75 factorial TPC-H development queries across $K=25$ seed clusters, 55 cross-schema calibration queries, 115 public/trace-profiled queries including 30 author-synthesized \\texttt{Redset-Synth} queries, and 155 standard holdout queries comprising 99 TPC-DS, 43 ClickBench [20 native SQL + 23 dialect-transpiled], and 13 SSB) across four hardware nodes.\n"
        "\\textbf{Results:} Three-way ablation shows that PCIe margin guarding drives 81.3\\% of P95 tail-regret reduction, while a compact two-engine portfolio (\\{Polars, GPU\\}) retains 98.7\\% of full five-engine utility; AST/GBR/CSCI provide lightweight overhead reduction rather than primary accuracy gains. On the 75-query development suite (25-seed LOSO-CV), P95 regret drops from 257.31\\% (Always GPU) and 358.40\\% (Always Polars) to 114.11\\% (mean $54.92\\text{ ms}$ vs.\\ $56.96\\text{ ms}$ Always GPU). On external holdout clusters ($K=38$), tail-regret reduction is directional but not statistically significant (cluster permutation $p = 0.125$ vs.\\ CPU, $p = 0.198$ vs.\\ GPU). Crucially, on out-of-distribution prospective holdouts (\\texttt{Seed S7/DSB/JCC-H}), zero-shot accuracy drops to 12.0\\% Top-1 (83.98\\% mean regret, 346.0\\% P95 regret), trailing Always Polars (66.75\\% mean regret).\n"
        "\\textbf{Conclusions:} Tail-safe hybrid OLAP routing is governed primarily by transfer-margin abstention and compact engine pools, but requires schema-local calibration before deployment on skewed or out-of-distribution workloads.\n\n"
        "\\noindent\\textbf{Keywords:} Cross-engine query routing; Tail-regret control; Abstention guards; Heterogeneous OLAP systems; Apache Arrow and GPU analytics; Pre-execution metadata\n"
        "\\end{quote}"
    )

    sec1_idx = tex.find(r"\section{Introduction}")
    if sec1_idx == -1:
        sec1_idx = tex.find(r"\section{INTRODUCTION}")
    head = tex[:sec1_idx] if sec1_idx != -1 else tex[:8000]
    tail = tex[sec1_idx:] if sec1_idx != -1 else tex[8000:]

    # Remove any duplicate \begin{abstract}...\end{abstract} in head
    head = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", head, flags=re.DOTALL)

    # Replace ALL \begin{quote}...\end{quote} blocks in frontmatter with the single structured abstract
    q_matches = list(re.finditer(r"\\begin\{quote\}.*?\\end\{quote\}", head, flags=re.DOTALL))
    if q_matches:
        s0, e0 = q_matches[0].span()
        head = head[:s0] + structured_quote_abstract + head[e0:]
        for extra in reversed(list(re.finditer(r"\\begin\{quote\}.*?\\end\{quote\}", head, flags=re.DOTALL))[1:]):
            head = head[: extra.start()] + head[extra.end() :]
    else:
        head = head.rstrip() + "\n\n" + structured_quote_abstract + "\n\n"

    tex = head + tail

    # 2. Close G12: Clarify ClickBench 20 native + 23 dialect-transpiled split everywhere in body
    tex = re.sub(
        r"\$?43\$?\s+(?:verbatim\s+|untouched\s+)?ClickBench(?!\s*[\[\(]20\s+native)",
        lambda _: "43 ClickBench (20 native SQL + 23 dialect-transpiled)",
        tex,
        flags=re.IGNORECASE,
    )

    # Verify brace balance
    cleaned = re.sub(r"\\[\{\}]", "", strip_comments(tex))
    if cleaned.count("{") != cleaned.count("}"):
        raise RuntimeError(f"Brace imbalance: {{={cleaned.count('{')} vs }}={cleaned.count('}')}")

    TEX_PATH.write_text(tex, encoding="utf-8")
    return tex


def run_12_gate_audit(tex: str) -> dict:
    no_comm = strip_comments(tex)
    bad_0018 = [ln.strip() for ln in no_comm.splitlines() if "0.0018" in ln and "mondrian" not in ln.lower()]
    sec1_idx = no_comm.find(r"\section{Introduction}")
    head_only = no_comm[:sec1_idx] if sec1_idx != -1 else no_comm[:8000]

    gates = {
        "G01_No_Holdout_0018_Contradiction": len(bad_0018) == 0,
        "G02_Holdout_p0125_Present": "0.125" in no_comm,
        "G03_Single_Structured_Abstract": (
            "\\textbf{Background:}" in head_only
            and "\\textbf{Methods:}" in head_only
            and "\\textbf{Results:}" in head_only
            and "\\textbf{Conclusions:}" in head_only
            and len(re.findall(r"\\begin\{quote\}", head_only)) == 1
            and len(re.findall(r"\\begin\{abstract\}", head_only)) == 0
        ),
        "G04_Prospective_S7_DSB_JCCH_In_Main_Results": "Prospective Out-of-Distribution Holdout Failure Analysis" in no_comm,
        "G05_No_Conflicting_LME_Z_Scores": not bool(re.search(r"z\s*=\s*[-−](?:57\.37|16\.8|3\.12)", no_comm)),
        "G06_No_False_OSF_Prospective_Claim": "prospective osf protocol provided" not in no_comm.lower(),
        "G07_No_Corrupted_LaTeX_Artifacts": (r"\1+Te" not in no_comm) and ("RAM[e,VRAM" not in no_comm),
        "G08_Single_Official_GitHub_URL": ("preechan-glitch" not in no_comm) and ("db_cluster_distributed" not in no_comm),
        "G09_No_Nested_URL_Recursion": r"\url{\url{" not in no_comm,
        "G10_No_Raw_Markdown_Links": not bool(re.search(r"(?<!\\url\{)(?<!\\href\{)\[[^\]]+\]\(https?://[^\)]+\)", no_comm)),
        "G11_No_Table16_100pct_Router_Claim": not bool(re.search(r"100(?:\.0)?\\%\s*routing\s*accuracy", no_comm, flags=re.I)),
        "G12_ClickBench_Native_Transpiled_Split": "20 native SQL + 23 dialect-transpiled" in no_comm,
    }
    print("\n================ 12-GATE AUTOMATED CONSISTENCY AUDIT ================")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print("=====================================================================\n")
    if not all(gates.values()):
        raise RuntimeError("Not all 12 gates passed!")
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
    tex = close_g03_and_g12()
    gates = run_12_gate_audit(tex)
    build = compile_and_package_flat13(tex)

    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    payload["gates"] = gates
    payload["build"] = build
    payload["artifacts_sha256"] = {
        TEX_PATH.name: sha256_file(TEX_PATH),
        PDF_PATH.name: sha256_file(PDF_PATH),
        MACROS_PATH.name: sha256_file(MACROS_PATH),
        ZIP_PATH.name: sha256_file(ZIP_PATH),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[AUDIT] 12/12 GATES PASSED | SHA-256 updated in:", AUDIT_JSON)


if __name__ == "__main__":
    main()
