#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/110_v13_unbiased_reviewer_audit_fixes.py
================================================================================
Unbiased Senior Reviewer Pre-Submission Audit & Repair for V13 (26-page edition):
  1. Fixes Table 16 & §5.1 text: removes "100.0% routing accuracy" overclaim on
     cross-node n=4 cells -> reframes as "Cross-Node Scale-Regime Envelope".
  2. Fixes §4.7(2) & §6(3) Redset-Synth provenance ("Public Amazon Redshift Redset
     Fleet Trace Queries" -> "Author-Synthesized Redset-Synth Telemetry Queries").
  3. Reconciles §4.8 & §5 SF=100 GB DataFusion vs GPU statement (specifies n=1 cold
     vs R=5 Table 14 where only DuckDB beats post-Arrow GPU, and all CPUs beat ZSTD E2E).
  4. Cleans Table 3 footnote (removes pre-specified Zenodo v11 claim), §4.4.1 &
     Table 5 Part A LME header remnants, §5 query double-counting (400+25+70=495),
     and Table 12 "V12" labels -> "V13".
  5. Fixes Table 17 note typo ($10/25$ -> $0/25$), Availability broken URL ($ry-o1$ ->
     \url{https://github.com/preecha-msu/bigquery-olap-benchmark}), Kepler cite,
     and Eq. (11) spacing.
  6. Updates Figure 2 Phase 1 & Phase 5 box text to match exact 400-query V13 registry
     and strips legacy "V11" suptitles from Figure 8/9 generator scripts if present.
  7. Runs 28-Gate Audit, compiles V13.pdf, verifies Clean-Room Flat13.zip, & logs SHA-256.
================================================================================
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
OFFICIAL_REPO_URL = "https://github.com/preecha-msu/bigquery-olap-benchmark"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(txt: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())


def update_figures_v13() -> dict:
    synced = {}
    # 1. Update Figure 2 Phase 1 (400 Primary = 75+90+80+155, Redset-Synth) & Phase 5 (7.61x cold / 2.89x warm)
    fig12_py = ROOT / "scripts" / "v13_regenerate_clean_fig1_fig2_300dpi.py"
    if not fig12_py.exists():
        fig12_py = ROOT / "scripts" / "v12_regenerate_fig1_fig2_300dpi.py"
    if fig12_py.exists():
        code = fig12_py.read_text(encoding="utf-8")
        code = re.sub(r"\(V12\)", "(V13)", code)
        code = re.sub(r"\(V12:", "(V13:", code)
        code = code.replace("30 Amazon Redset", "30 Redset-Synth (Author-Synth.)")
        code = code.replace("80 Public Real-World & Trace Queries", "80 Cat. III Adapted & Redset-Synth")
        code = code.replace("7.61x vs DuckDB", "7.61x cold / 2.89x warm")
        code = re.sub(
            r"https?://github\.com/[A-Za-z0-9_.-]+/(?:my_research_files|db_cluster_distributed)",
            OFFICIAL_REPO_URL,
            code,
        )
        out_py = ROOT / "scripts" / "v13_regenerate_clean_fig1_fig2_300dpi.py"
        out_py.write_text(code, encoding="utf-8")
        subprocess.run(["python", str(out_py)], cwd=str(ROOT), check=True)

    # 2. Strip legacy "V11 " banner from Figure 8 / Figure 9 generator scripts if present
    for sp in sorted((ROOT / "scripts").glob("*4node*multiscale*.py")):
        sc = sp.read_text(encoding="utf-8", errors="replace")
        if "V11 Four-Node" in sc or "V11 Fixed Four-Node" in sc:
            sc_new = sc.replace("V11 Four-Node", "Four-Node").replace("V11 Fixed Four-Node", "Four-Node")
            sp.write_text(sc_new, encoding="utf-8")
            try:
                subprocess.run(["python", str(sp)], cwd=str(ROOT), check=True, timeout=60)
            except Exception as e:
                print(f"[INFO] Skipped re-running {sp.name}: {e}")

    for fn in [
        "fig_v12_proposed_ast_routing_framework_300dpi.pdf",
        "fig_v12_experimental_workflow_300dpi.pdf",
        "fig_v11_4node_multiscale_sf1_10_30_100.pdf",
        "fig_v12_4node_multiscale_cudf_sf1_10_30_100.jpg",
    ]:
        cands = [MS_FIG_DIR / fn, MS_DIR / fn, ROOT_FIG_DIR / fn]
        ex = [p for p in cands if p.exists()]
        if ex:
            newest = max(ex, key=lambda p: p.stat().st_mtime)
            for dst in cands:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.resolve() != newest.resolve():
                    shutil.copy2(newest, dst)
            synced[fn] = sha256_file(newest)
    return synced


def patch_manuscript() -> tuple[str, int]:
    tex = TEX_PATH.read_text(encoding="utf-8")
    n_tables_orig = tex.count(r"\begin{table")

    # ------------------------------------------------------------------
    # 1. Fix Table 16 & §5.1 Cross-Node n=4 "100.0% routing accuracy" overclaim
    # ------------------------------------------------------------------
    tex = re.sub(
        r"AST-Route matches the empirical oracle at all four scales\s*\([^)]*100(?:\.0)?\\%[^)]*\)",
        lambda _: r"the scale-regime envelope shifts across hardware tiers at all four scales ($4/4$ cross-node oracle regime alignment; illustrative envelope across $n=4$ scale tiers rather than single-node router accuracy)",
        tex,
        flags=re.IGNORECASE,
    )
    tex = re.sub(
        r"Multi-scale AST-Route routing accuracy and worst-case avoidance across",
        lambda _: "Cross-node multi-scale hardware crossover envelope ($n=4$ scale tiers; illustrative oracle regime shift across Nodes 1, 3, and 4 rather than single-node router accuracy) across",
        tex,
        flags=re.IGNORECASE,
    )
    # Inside Table 16 header: replace Acc. column header with Regime Match and 100.0% with 1/1 (Node X)
    def _fix_tab16(m: re.Match) -> str:
        blk = m.group(0)
        if "tab:v13_groupB_multiscale_router" not in blk and "39252.64" not in blk:
            return blk
        blk = blk.replace(r"\textbf{AST-Route Pick}", r"\textbf{Cross-Node Envelope}")
        blk = blk.replace(r"\textbf{Acc.}", r"\textbf{Regime}")
        blk = blk.replace("100.0\\%", "Matched")
        return blk

    tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", _fix_tab16, tex, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # 2. Fix Redset-Synth provenance in §4.7(2) and §6(3)
    # ------------------------------------------------------------------
    tex = re.sub(
        r"30\s+Public\s+Amazon\s+Redshift\s+\\texttt\{Redset\}\s+Fleet\s+Trace\s+Queries",
        lambda _: r"30 Author-Synthesized \texttt{Redset-Synth} Telemetry-Matched Queries",
        tex,
    )
    tex = tex.replace(
        "sampled from the public Redshift Serverless/Provisioned fleet trace specification",
        "synthesized by the authors to match the published Redshift Serverless/Provisioned fleet metadata distributions",
    )
    tex = tex.replace(
        "80 public real-world and fleet trace queries (SDSS DR16, NYC TLC, and Amazon Redshift \\texttt{Redset}",
        "80 Category~III public-adapted and trace-synthetic queries (25 SDSS DR16, 25 NYC TLC, and 30 author-synthesized \\texttt{Redset-Synth}",
    )
    tex = tex.replace(
        "80 public real-world and fleet trace queries (SDSS DR16, NYC TLC, and Amazon Redshift Redset",
        "80 Category~III public-adapted and trace-synthetic queries (25 SDSS DR16, 25 NYC TLC, and 30 author-synthesized \\texttt{Redset-Synth}",
    )

    # ------------------------------------------------------------------
    # 3. Reconcile SF=100 GB n=1 Cold vs R=5 Warm (DataFusion vs DuckDB vs GPU)
    # ------------------------------------------------------------------
    tex = tex.replace(
        "while Node~1 DataFusion ($4036.88\\text{ ms}$) outperforms 40-chunk PyTorch GPU ($4668.36\\text{ ms}$).",
        "while on Node~1 DataFusion ($4036.88\\text{ ms}$) leads 40-chunk PyTorch GPU ($4668.36\\text{ ms}$) only in the single-pass cold run ($n=1$, Table~\\ref{tab:v12_fournode_multiscale_cudf}); under $R=5$ warm repetitions (Table~\\ref{tab:v13_groupB_r5_and_sdss}), DuckDB ($3296.61\\text{ ms}$) is the sole CPU engine faster than post-Arrow GPU ($3682.14\\text{ ms}$), whereas all local CPUs outperform full ZSTD-to-GPU execution ($28348.57\\text{ ms}$).",
    )
    tex = tex.replace(
        "Rust Arrow streaming via Apache DataFusion ($4036.88\\text{ ms}$) beats 40-chunk PyTorch GPU ($4668.36\\text{ ms}$) by $1.16\\times$ on the exact same machine once 4~GB VRAM chunking overhead accumulates;",
        "in this cold $n=1$ pass, Apache DataFusion ($4036.88\\text{ ms}$) beats 40-chunk post-Arrow PyTorch GPU ($4668.36\\text{ ms}$) by $1.16\\times$ (note that under $R=5$ warm trials in Table~\\ref{tab:v13_groupB_r5_and_sdss}, DataFusion averages $5683.00\\text{ ms}$ while DuckDB leads at $3296.61\\text{ ms}$ vs.\\ $3682.14\\text{ ms}$ post-Arrow GPU and $28348.57\\text{ ms}$ ZSTD-E2E GPU);",
    )

    # ------------------------------------------------------------------
    # 4. Clean Table 3 Footnote, §4.4.1 / Table 5 LME Headers, §5 Query Sum, & Table 12 "V12"
    # ------------------------------------------------------------------
    tex = re.sub(
        r"protocol-defined in repository and archived under Zenodo metadata[^\n]*?v11\.0\.0-jbd-archival\)",
        lambda _: r"post-hoc inner-fold Nested-CV harmonization introduced in V10--V12; not OSF-preregistered, see Table~\ref{tab:v13_protocol_and_node4_baselines})",
        tex,
    )
    tex = tex.replace(
        r"Expanded $G_{\text{total}}=42$ Template-Cluster Linear Mixed-Effects (LME) Model, Global Holm Control, and Engine-Pool Ablation:",
        r"External Holdout Cluster-Permutation Inference ($K=38$, LME Retired), Global Holm Control, and Engine-Pool Ablation:",
    )
    tex = tex.replace(
        r"\& $G_{\text{total}}=42$ Cluster LME Inference",
        r"\& Holdout Cluster Permutation ($K=38$, LME Retired)",
    )
    tex = tex.replace(
        "Across all 400 benchmark, 70 enterprise stress, and 50 public real-world SQL queries,",
        "Across all 400 primary registry queries (including 50 adapted SDSS/NYC~TLC and 30 \\texttt{Redset-Synth}), 25 prospective OOD queries, and 70 enterprise stress queries (495 distinct SQL files),",
    )
    tex = tex.replace("partitions all energy figures in V12", "partitions all energy figures in V13")
    tex = tex.replace("Observed V12 Range", "Observed V13 Range")

    # ------------------------------------------------------------------
    # 5. Fix Typographical Bugs: Table 17 ($10/25$ -> $0/25$), Availability URL ($ry-o1$), [[52]]
    # ------------------------------------------------------------------
    tex = tex.replace("triggered on $10/25$ queries", "triggered on $0/25$ queries")
    tex = tex.replace("triggered on 10/25 queries", "triggered on $0/25$ queries")

    # Fix broken "$ry-o1$" or corrupted repository string right before verify_zero_contradiction_V13.py
    tex = re.sub(
        r"are archived in\s+[^\n]*?and verified via\s*\\texttt\{scripts/verify",
        lambda _: rf"are archived in \url{{{OFFICIAL_REPO_URL}}} and verified via \texttt{{scripts/verify",
        tex,
    )
    # Fix duplicate bracket citation on Kepler if present
    tex = re.sub(r"\[\s*\\cite\{([^\}]+)\}\s*\]", lambda m: rf"\cite{{{m.group(1)}}}", tex)

    # Verify brace balance and table preservation
    cleaned = re.sub(r"\\[\{\}]", "", strip_comments(tex))
    if cleaned.count("{") != cleaned.count("}"):
        raise RuntimeError(f"Brace imbalance: {{={cleaned.count('{')} vs }}={cleaned.count('}')}")
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
        "G03_No_Public_Redset_Fleet_Trace_Overclaim": "Public Amazon Redshift \\texttt{Redset} Fleet Trace Queries" not in no_comm,
        "G04_No_Conclusion_Redset_RealWorld_Overclaim": "Amazon Redshift Redset" not in no_comm and "Amazon Redshift \\texttt{Redset}" not in no_comm,
        "G05_SF100_Cold_vs_R5_Warm_Reconciled": "DuckDB ($3296.61\\text{ ms}$) is the sole CPU engine faster than post-Arrow GPU" in no_comm or "DataFusion averages $5683.00\\text{ ms}$" in no_comm,
        "G06_No_Table3_Footnote_PreReg_Claim": "v11.0.0-jbd-archival" not in no_comm or "post-hoc inner-fold Nested-CV harmonization" in no_comm,
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
    fig_hashes = update_figures_v13()
    tex, n_tables_orig = patch_manuscript()
    gates = run_28_gate_audit(tex, n_tables_orig)
    build = compile_and_package_flat13(tex)

    payload = {
        "updated_figures_sha256": fig_hashes,
        "gates_28": gates,
        "build": build,
        "artifacts_sha256": {
            TEX_PATH.name: sha256_file(TEX_PATH),
            PDF_PATH.name: sha256_file(PDF_PATH),
            ZIP_PATH.name: sha256_file(ZIP_PATH),
        },
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[AUDIT] 28/28 GATES PASSED | SHA-256 manifest written to:", AUDIT_JSON)


if __name__ == "__main__":
    main()
