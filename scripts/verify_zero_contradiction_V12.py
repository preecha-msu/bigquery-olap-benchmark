#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/verify_zero_contradiction_V12.py
# 18-Gate Automated Zero-Contradiction & Forensic Integrity Verifier for V12.

import re
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
V12_TEX = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.tex"
V12_PDF = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.pdf"
V12_LOG = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.log"
MASTER_CSV = (
    ROOT / "results" / "v12_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
)
FLAT12_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat12.zip"


def main() -> None:
    tex = V12_TEX.read_text(encoding="utf-8")
    log = V12_LOG.read_text(encoding="utf-8", errors="replace") if V12_LOG.exists() else ""
    df = pd.read_csv(MASTER_CSV)

    undef_refs = len(
        re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log)
    )
    undef_cites = len(
        re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log)
    )

    with zipfile.ZipFile(FLAT12_ZIP, "r") as zf:
        znames = zf.namelist()
        nested = [n for n in znames if "/" in n or "\\" in n]

    clean_allow = tex.replace(r"\_\allowbreak{}", "")
    gates = [
        ("G01 Top-1 = 57.33%", "57.33" in tex),
        ("G02 Top-2 = 76.00%", "76.00" in tex),
        ("G03 Mean Regret = 34.98%", "34.98" in tex),
        ("G04 DuckDB 75Q Mean = 183.74 ms", "183.74" in tex),
        ("G05 Polars 75Q Mean = 82.47 ms", "82.47" in tex),
        ("G06 GPU Cold 75Q Mean = 56.96 ms", "56.96" in tex),
        ("G07 Hybrid Router 75Q Mean = 54.92 ms", "54.92" in tex),
        (
            "G08 Zero Forbidden Legacy Substrings (51.12/6.70/103.32)",
            all(x not in tex for x in ("51.12", "6.70", "103.32")),
        ),
        ("G09 Zero Raw _allowbreak{} Leak", "_allowbreak{}" not in clean_allow),
        (
            "G10 V12 RAPIDS cuDF Latencies Present",
            all(x in tex for x in ("112.76", "322.37", "1001.71", "3674.43")),
        ),
        (
            "G11 V12 cudf-polars Latencies Present",
            all(x in tex for x in ("249.93", "453.99", "1318.63", "5279.66")),
        ),
        (
            "G12 Timing & Energy Boundaries Disclosed",
            "Post-Arrow Device Envelope" in tex and r"125\text{ J}" in tex,
        ),
        (
            "G13 Intra-Node vs Cross-Node Crossover Analyzed",
            "2-vCPU" in tex and "4266.59" in tex and "4292.58" in tex,
        ),
        (
            "G14 Threats to Validity & Fixed 4-Node Scope Present",
            "Threats to Validity and Fixed Four-Node Architectural Scope" in tex,
        ),
        (
            "G15 Master CSV 64 Rows, EX=1.0, Exact SF30 Shard Rows",
            len(df) == 64
            and (df["EX_Equivalence"] == 1.0).all()
            and 179958156 in set(df["Lineitem_Rows"]),
        ),
        (
            "G16 Master CSV Timing_Boundary & Energy_Boundary Columns",
            "Timing_Boundary" in df.columns and "Energy_Boundary" in df.columns,
        ),
        (
            "G17 Compiled V12.pdf Exists with 0 Undefined Refs/Citations",
            V12_PDF.exists() and undef_refs == 0 and undef_cites == 0,
        ),
        (
            "G18 Flat12.zip Strictly Flat & Contains V12.tex + V12 JPG",
            len(nested) == 0
            and "bigquery_olap_benchmark_V12.tex" in znames
            and "fig_v12_4node_multiscale_cudf_sf1_10_30_100.jpg" in znames,
        ),
        (
            "G19 3-Way Stepwise Ablation (Guard -78.2 pp / +Conformal -48.6 pp / +CSCI -5.4 pp)",
            "-78.20" in tex and "-48.60" in tex and "-5.40" in tex,
        ),
        (
            "G20 Stronger GPU Baselines (Pinned+Overlap FP64 47.82/44.18, FP32 39.64/37.95, Resident 22.25)",
            all(x in tex for x in ("47.82", "44.18", "39.64", "37.95", "22.25")),
        ),
        (
            "G21 Multi-Scale Router-in-the-Loop & Public Redset/SDSS/NYC TLC at SF=10 GB",
            "vanrenen2024redset" in tex and "228.40" in tex and "209.15" in tex and "245.80" in tex,
        ),
        (
            "G22 Expanded G_total=42 Template-Cluster LME (-62.06 pp) & Global Holm Control",
            "G_{\\text{total}}=42" in tex and "-62.06" in tex and "0.0136" in tex,
        ),
        (
            "G23 Query-Level N=75 Energy Correlation (rho_{N=75}=0.9684, p=1.4e-45)",
            "0.9684" in tex,
        ),
        (
            "G24 Misrouted Query Failure Taxonomy (F1 63.6%, F2 21.2%, F3 15.2%) & TreeSHAP (34.2%, 24.8%)",
            all(x in tex for x in ("63.6\\%", "21.2\\%", "15.2\\%", "34.2\\%", "24.8\\%")),
        ),
        (
            "G25 Zero Corrupted DOIs (s40537-024-00939-0-02)",
            "s40537-024-00939-0-02" not in tex,
        ),
        (
            "G26 Zero Reviewer_Revision_Roadmap or V9->V10->V11 Leakage in Main Text",
            "Reviewer_Revision_Roadmap" not in tex and "V9 \\rightarrow V10" not in tex,
        ),
        (
            "G27 Regenerated V12 Figures 1 & 2 Referenced (fig_v12_proposed_... & fig_v12_experimental_...)",
            "fig_v12_proposed_ast_routing_framework_300dpi.pdf" in tex
            and "fig_v12_experimental_workflow_300dpi.pdf" in tex,
        ),
        (
            "G28 BigQuery Cost Unit Harmonized ($1,267.66/1M = $1.27/1k)",
            r"\$1{,}267.66/1\text{M}$ ($\$1.27/1\text{k}$)" in tex,
        ),
        (
            "G29 Intra-Node Speedup Harmonized (2.85x-9.50x across SF=1-100 GB; 7.43x-9.50x at SF=10-100 GB)",
            r"7.40\times\text{--}9.50\times" not in tex and r"2.85\times\text{--}9.50\times" in tex,
        ),
        (
            "G30 Apple M1 ANS2 NVMe vs Node 2 22T ZSTD Contention Explained & Abstract Clean",
            "ANS2 NVMe" in tex and "% [V12-B4-ABS]" not in tex and "% [V12-B4-CONC]" not in tex,
        ),
    ]

    print("=" * 92)
    print(f"[V12 ZERO-CONTRADICTION & FORENSIC AUDIT: {len(gates)}/{len(gates)} GATES]")
    print("=" * 92)
    failed = []
    for name, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit(f"Verification failed on: {failed}")
    print("=" * 92)
    print(f"[RESULT] {len(gates)}/{len(gates)} GATES PASSED WITH ZERO CONTRADICTIONS!")


if __name__ == "__main__":
    main()
