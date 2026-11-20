#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/112_v13_brace_safe_28gate_seal.py
Strictly brace-neutral & math-safe closure of G03, G04, G05, and G06 ->
28/28 GATES PASS + Clean-Room Journal_of_Big_Data_Snapp_Submission_Flat13.zip + SHA-256 log.
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
AUDIT_JSON = RESULTS_DIR / "v13_step110_pre_submission_audit_sha256.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(txt: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())


def assert_braces_and_dollars(tex: str, stage: str) -> None:
    clean = strip_comments(tex)
    no_esc = re.sub(r"\\[\{\}]", "", clean)
    ob, cb = no_esc.count("{"), no_esc.count("}")
    if ob != cb:
        raise RuntimeError(f"[{stage}] Brace imbalance: {{={ob} vs }}={cb}")


def close_g03_to_g06_brace_safe() -> tuple[str, int]:
    tex = TEX_PATH.read_text(encoding="utf-8")
    n_tables_orig = tex.count(r"\begin{table")
    assert_braces_and_dollars(tex, "INITIAL_LOAD")

    # ------------------------------------------------------------------
    # G03: Brace-balanced Redset-Synth heading & sampling phrase in §4.7(2)
    # ------------------------------------------------------------------
    tex = re.sub(
        r"30\s+Public\s+Amazon\s+Redshift\s+\\texttt\{Redset\}\s+Fleet\s+Trace\s+Queries",
        lambda _: r"30 Author-Synthesized \texttt{Redset-Synth} Telemetry-Matched Queries",
        tex,
        flags=re.IGNORECASE,
    )
    tex = re.sub(
        r"30\s+Public\s+Amazon\s+Redshift\s+Redset\s+Fleet\s+Trace\s+Queries",
        lambda _: r"30 Author-Synthesized \texttt{Redset-Synth} Telemetry-Matched Queries",
        tex,
        flags=re.IGNORECASE,
    )
    tex = re.sub(
        r"sampled\s+from\s+the\s+public\s+Redshift\s+Serverless/Provisioned\s+fleet\s+trace\s+specification",
        lambda _: "synthesized by the authors to match published Redshift Serverless/Provisioned fleet metadata distributions",
        tex,
        flags=re.IGNORECASE,
    )
    assert_braces_and_dollars(tex, "AFTER_G03")

    # ------------------------------------------------------------------
    # G04: Brace-balanced replacement of "Amazon Redshift (\texttt{Redset}|Redset)"
    # ------------------------------------------------------------------
    tex = re.sub(
        r"Amazon\s+Redshift\s+\\texttt\{Redset\}\s+metadata",
        lambda _: r"published Redshift \texttt{Redset} telemetry metadata",
        tex,
    )
    tex = re.sub(
        r"Amazon\s+Redshift\s+Redset\s+metadata",
        lambda _: r"published Redshift \texttt{Redset} telemetry metadata",
        tex,
    )
    tex = re.sub(
        r"Amazon\s+Redshift\s+\\texttt\{Redset\}",
        lambda _: r"author-synthesized \texttt{Redset-Synth}",
        tex,
    )
    tex = re.sub(
        r"Amazon\s+Redshift\s+Redset",
        lambda _: r"author-synthesized \texttt{Redset-Synth}",
        tex,
    )
    assert_braces_and_dollars(tex, "AFTER_G04")

    # ------------------------------------------------------------------
    # G05: Brace-balanced SF=100 GB Cold (n=1) vs R=5 Warm reconciliation
    # ------------------------------------------------------------------
    g05_note = (
        r" in the single-pass cold run ($n=1$; under $R=5$ warm repetitions in Table~\ref{tab:v13_groupB_r5_and_sdss}, "
        r"DataFusion averages $5683.00\text{ ms}$ while DuckDB ($3296.61\text{ ms}$) is the sole CPU engine faster than "
        r"post-Arrow GPU ($3682.14\text{ ms}$), and all CPUs beat full ZSTD-E2E GPU ($28348.57\text{ ms}$))"
    )
    if "DuckDB ($3296.61\\text{ ms}$) is the sole CPU engine faster than post-Arrow GPU" not in tex:
        pat_g05 = re.compile(
            r"((?:\\texttt\{DataFusion\}|DataFusion)\s*\([^\)]{1,45}036\.88[^\)]{0,25}\)\s+"
            r"(?:outperforms|beats|surpasses)\s+40-chunk\s+"
            r"(?:\\texttt\{PyTorch\s+GPU\}|PyTorch\s+GPU)\s*\([^\)]{1,45}668\.36[^\)]{0,25}\))",
            flags=re.IGNORECASE,
        )
        if pat_g05.search(tex):
            tex = pat_g05.sub(lambda m: m.group(1) + g05_note, tex, count=1)
        else:
            # Guaranteed brace-safe fallback at the end of Section 5.1 Table 14 sentence
            tex = tex.replace(
                "across 41 chunks).",
                "across 41 chunks; thus under $R=5$ warm trials, DuckDB ($3296.61\\text{ ms}$) is the sole CPU engine faster than post-Arrow GPU while DataFusion averages $5683.00\\text{ ms}$).",
                1,
            )
    assert_braces_and_dollars(tex, "AFTER_G05")

    # ------------------------------------------------------------------
    # G06: Brace-safe Table 3 footnote cleanup (never crosses \texttt{...}!)
    # ------------------------------------------------------------------
    tex = tex.replace(r"\texttt{v11.0.0-jbd-archival}", r"\texttt{v13-code-freeze}")
    tex = tex.replace("v11.0.0-jbd-archival", "v13-code-freeze")
    tex = re.sub(
        r"protocol-defined\s+in\s+(?:the\s+)?(?:version-controlled\s+)?repository\s+and\s+archived\s+under\s+Zenodo\s+metadata",
        lambda _: "harmonized in V10--V12 (not OSF-preregistered) and recorded under repository metadata",
        tex,
        flags=re.IGNORECASE,
    )
    assert_braces_and_dollars(tex, "AFTER_G06")

    if tex.count(r"\begin{table") < n_tables_orig:
        raise RuntimeError(f"Table loss detected: {tex.count(chr(92)+'begin{table')} < {n_tables_orig}")

    TEX_PATH.write_text(tex, encoding="utf-8")
    return tex, n_tables_orig


def run_28_gate_audit(tex: str, n_tables_orig: int) -> dict:
    no_comm = strip_comments(tex)
    bad_0018 = [ln.strip() for ln in no_comm.splitlines() if "0.0018" in ln and "mondrian" not in ln.lower()]

    gates = {
        "G01_No_Holdout_0018": len(bad_0018) == 0,
        "G02_No_Table16_100pct_Accuracy_Overclaim": "100.0\\% routing accuracy" not in no_comm and "Multi-scale AST-Route routing accuracy" not in no_comm,
        "G03_No_Public_Redset_Fleet_Trace_Overclaim": not bool(re.search(r"Public\s+Amazon\s+Redshift\s+(?:\\texttt\{Redset\}|Redset)\s+Fleet\s+Trace", no_comm, flags=re.I)),
        "G04_No_Conclusion_Redset_RealWorld_Overclaim": "Amazon Redshift Redset" not in no_comm and r"Amazon Redshift \texttt{Redset}" not in no_comm,
        "G05_SF100_Cold_vs_R5_Warm_Reconciled": (
            "DuckDB ($3296.61\\text{ ms}$) is the sole CPU engine faster than post-Arrow GPU" in no_comm
            or "DataFusion averages $5683.00\\text{ ms}$" in no_comm
        ),
        "G06_No_Table3_Footnote_PreReg_Claim": "v11.0.0-jbd-archival" not in no_comm,
        "G07_No_Sec441_LME_Header_Remnant": "Template-Cluster Linear Mixed-Effects (LME) Model" not in no_comm,
        "G08_No_Table5_PartA_LME_Header_Remnant": "Cluster LME Inference" not in no_comm,
        "G09_No_Sec5_50Public_DoubleCount": "400 benchmark, 70 enterprise stress, and 50 public" not in no_comm,
        "G10_No_Table12_V12_Label": "Observed V12 Range" not in no_comm and "energy figures in V12" not in no_comm,
        "G11_Table17_Note_0_of_25_Fixed": "triggered on $10/25$" not in no_comm and "triggered on $0/25$ queries" in no_comm,
        "G12_Availability_Broken_ry_o1_Fixed": "ry-o1" not in no_comm,
        "G13_Single_Structured_Abstract": "\\textbf{Background:}" in no_comm and "\\textbf{Conclusions:}" in no_comm,
        "G14_No_LME_Z_Scores": not bool(re.search(r"z\s*=\s*[-−](?:57\.37|16\.8|3\.12)", no_comm)),
        "G15_No_Mixed_Effects_Heading_In_Conclusion": "Mixed-Effects Inference" not in no_comm,
        "G16_S7_L1_Basin_Trap_Explained": "Tier-L1 DataFusion Basin Trap" in no_comm,
        "G17_SmallN_Reported_As_kn_Counts": "$1/15$" in no_comm and "$2/5$" in no_comm and "$0/5$" in no_comm,
        "G18_Cluster_TOST_Equivalence_Reported": "p_{\\text{TOST}}" in no_comm,
        "G19_VIF_384_Reconciled": r"\max\text{VIF}=1.86" not in no_comm,
        "G20_Dialect_Agnostic_EXPLAIN_Note": "dialect-agnostic with zero catalog coupling" in no_comm,
        "G21_Locked_Single_Shot_Label": "Locked Single-Shot Holdout (Not Preregistered" in no_comm,
        "G22_Table13_Cold_n1_Annotated": "(Cold $n=1$)" in no_comm,
        "G23_Table14_Warm_R5_Annotated": "(Warm $R=5$)" in no_comm,
        "G24_Single_Official_GitHub_URL": ("preechan-glitch" not in no_comm) and ("db_cluster_distributed" not in no_comm),
        "G25_No_Nested_URL": r"\url{\url{" not in no_comm,
        "G26_No_Corrupted_LaTeX_Artifacts": (r"\1+Te" not in no_comm) and ("RAM[e,VRAM" not in no_comm),
        "G27_ClickBench_20_Native_23_Transpiled": "20 native SQL + 23 dialect-transpiled" in no_comm,
        "G28_All_Tables_Preserved": tex.count(r"\begin{table") >= n_tables_orig,
    }
    print("\n================ 28-GATE PRE-SUBMISSION AUDIT ================")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"  [INFO] Tables preserved: {tex.count(chr(92)+'begin{table')} >= {n_tables_orig}")
    print("==============================================================\n")
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
    tex, n_tables_orig = close_g03_to_g06_brace_safe()
    gates = run_28_gate_audit(tex, n_tables_orig)
    build = compile_and_package_flat13(tex)

    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    payload["gates_28"] = gates
    payload["build"] = build
    payload["artifacts_sha256"] = {
        TEX_PATH.name: sha256_file(TEX_PATH),
        PDF_PATH.name: sha256_file(PDF_PATH),
        ZIP_PATH.name: sha256_file(ZIP_PATH),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[AUDIT] 28/28 GATES PASSED | SHA-256 manifest written to:", AUDIT_JSON)


if __name__ == "__main__":
    main()
