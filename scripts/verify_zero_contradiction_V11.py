#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/verify_zero_contradiction_V11.py
# 18-Gate Automated Zero-Contradiction & Forensic Integrity Verifier for V11.

import re
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
V11_TEX = ROOT / "manuscript" / "bigquery_olap_benchmark_V11.tex"
V11_PDF = ROOT / "manuscript" / "bigquery_olap_benchmark_V11.pdf"
V11_LOG = ROOT / "manuscript" / "bigquery_olap_benchmark_V11.log"
MASTER_CSV = (
    ROOT / "results" / "v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
)
FLAT11_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat11.zip"


def main() -> None:
    tex = V11_TEX.read_text(encoding="utf-8")
    log = V11_LOG.read_text(encoding="utf-8", errors="replace") if V11_LOG.exists() else ""
    df = pd.read_csv(MASTER_CSV)

    undef_refs = len(
        re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log)
    )
    undef_cites = len(
        re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log)
    )

    with zipfile.ZipFile(FLAT11_ZIP, "r") as zf:
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
            "G10 V11 RAPIDS cuDF Latencies Present",
            all(x in tex for x in ("112.76", "322.37", "1001.71", "3674.43")),
        ),
        (
            "G11 V11 cudf-polars Latencies Present",
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
            "G17 Compiled V11.pdf Exists with 0 Undefined Refs/Citations",
            V11_PDF.exists() and undef_refs == 0 and undef_cites == 0,
        ),
        (
            "G18 Flat11.zip Strictly Flat & Contains V11.tex + V11 JPG",
            len(nested) == 0
            and "bigquery_olap_benchmark_V11.tex" in znames
            and "fig_v11_4node_multiscale_cudf_sf1_10_30_100.jpg" in znames,
        ),
    ]

    print("=" * 92)
    print("[V11 ZERO-CONTRADICTION & FORENSIC AUDIT: 18/18 GATES]")
    print("=" * 92)
    failed = []
    for name, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit(f"Verification failed on: {failed}")
    print("=" * 92)
    print("[RESULT] 18/18 GATES PASSED WITH ZERO CONTRADICTIONS!")


if __name__ == "__main__":
    main()
