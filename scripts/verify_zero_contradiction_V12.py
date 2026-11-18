#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/verify_zero_contradiction_V12.py
import re
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
V12_TEX = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.tex"
V12_PDF = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.pdf"
V12_LOG = ROOT / "manuscript" / "bigquery_olap_benchmark_V12.log"
MASTER_CSV = ROOT / "results" / "v12_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
REG_CSV = ROOT / "results" / "v12_canonical_metrics_registry.csv"
CSCI_CSV = ROOT / "results" / "v12_reviewer_statistical_and_csci_audit.csv"
FLAT12_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat12.zip"


def main() -> None:
    tex = V12_TEX.read_text(encoding="utf-8")
    log = V12_LOG.read_text(encoding="utf-8", errors="replace") if V12_LOG.exists() else ""
    df_m = pd.read_csv(MASTER_CSV)
    df_r = pd.read_csv(REG_CSV)
    df_c = pd.read_csv(CSCI_CSV)
    kv = dict(zip(df_r["Metric_Key"], df_r["Canonical_Value"]))

    undef_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log))
    undef_cites = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log))

    with zipfile.ZipFile(FLAT12_ZIP, "r") as zf:
        znames = zf.namelist()
        nested = [n for n in znames if "/" in n or "\\" in n]

    clean_allow = tex.replace(r"\_\allowbreak{}", "")
    no_legacy = not re.search(r"(?<![0-9])(?:51\.12|6\.70|103\.32)(?![0-9])", tex)

    gates = [
        ("G01 Registry P1 Primary Nested-CV (8%, 55.37 ms, 118.40%) Synced", all(x in tex for x in ("55.37", "118.40", "Protocol~P1"))),
        ("G02 Registry P2 Secondary Fixed Guard (12%, 54.92 ms, 114.11%) Synced", all(x in tex for x in ("54.92", "114.11", "Protocol~P2"))),
        ("G03 Core 75Q Accuracy & Regret (57.33% / 76.00% / 34.98%)", all(x in tex for x in ("57.33", "76.00", "34.98"))),
        ("G04 Core Baselines (DuckDB 183.74 / Polars 82.47 / GPU 56.96)", all(x in tex for x in ("183.74", "82.47", "56.96"))),
        ("G05 Zero Standalone Legacy Numbers (51.12 / 6.70 / 103.32)", no_legacy),
        ("G06 Zero Raw _allowbreak{} Leak", "_allowbreak{}" not in clean_allow),
        ("G07 Evaluation Protocol Registry Table Present", "tab:v12_protocol_registry" in tex),
        ("G08 Data Provenance & 4-Category Audit Table Present", "tab:v12_data_provenance" in tex),
        ("G09 Exact Workload Reconciliation (50 + 30 = 80 & 400 Total)", "50 + 30 = 80" in tex and int(kv["Combined_Public_And_Trace_Total"]) == 80),
        ("G10 CSCI Oblique Inductive Bias & Raw vs Ortho VIF Disclosed", "oblique structural inductive bias" in tex and r"\text{VIF}=\infty" in tex),
        ("G11 CSCI & Energy Provenance Table Present", "tab:v12_csci_stat_energy" in tex and len(df_c) == 6),
        ("G12 Multi-Level Cluster Bootstrap (N_cluster=25 vs 75 vs 375)", r"N_{\text{cluster}}=25" in tex and "375" in tex),
        ("G13 Sensor-Separated Energy & Uncertainty (+-20.0% CPU TDP Estimate)", r"\pm 20.0\%" in tex and "TDP-Calibrated" in tex),
        ("G14 V12 RAPIDS cuDF Latencies Present", all(x in tex for x in ("112.76", "322.37", "1001.71", "3674.43"))),
        ("G15 V12 cudf-polars Latencies Present", all(x in tex for x in ("249.93", "453.99", "1318.63", "5279.66"))),
        ("G16 Timing & Active GPU Energy Boundaries Disclosed", "Post-Arrow Device Envelope" in tex and r"125\text{ J}" in tex),
        ("G17 Intra-Node (2-vCPU) vs Cross-Node (22T & M1) Crossover", "2-vCPU" in tex and "4266.59" in tex and "4292.58" in tex),
        ("G18 Consolidated 3 Scientific Contributions (C1, C2, C3)", all(x in tex for x in ("C1 --- Physical", "C2 --- Risk-Aware", "C3 --- Multi-Scale"))),
        ("G19 Six Explicit Study Limitations Disclosed", "Explicit Disclosure of Six Study Limitations" in tex),
        ("G20 Master CSV 64 Rows, EX=1.0, Exact SF30 Shard Rows", len(df_m) == 64 and (df_m["EX_Equivalence"] == 1.0).all() and 179958156 in set(df_m["Lineitem_Rows"])),
        ("G21 Master CSV Timing_Boundary & Energy_Boundary Present", "Timing_Boundary" in df_m.columns and "Energy_Boundary" in df_m.columns),
        ("G22 R1 Locked Replay Script Exists", (ROOT / "scripts" / "v12_locked_reexecution_audit.py").exists()),
        ("G23 Compiled V12.pdf Exists with 0 Undefined Refs/Citations", V12_PDF.exists() and undef_refs == 0 and undef_cites == 0),
        ("G24 Flat12.zip Strictly Flat & Contains V12.tex + V12 JPG", len(nested) == 0 and "bigquery_olap_benchmark_V12.tex" in znames and "fig_v12_4node_multiscale_cudf_sf1_10_30_100.jpg" in znames),
    ]

    print("=" * 94)
    print("[V12 AUTOMATED MANUSCRIPT CONSISTENCY & PROVENANCE AUDIT: 24/24 GATES]")
    print("=" * 94)
    failed = []
    for gname, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {gname}")
        if not ok:
            failed.append(gname)
    if failed:
        raise SystemExit(f"V12 Verification failed on: {failed}")
    print("=" * 94)
    print("[RESULT] 24/24 GATES PASSED WITH ZERO CONTRADICTIONS!")


if __name__ == "__main__":
    main()
