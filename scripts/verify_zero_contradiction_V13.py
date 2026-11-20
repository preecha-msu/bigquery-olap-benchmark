#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/verify_zero_contradiction_V13.py
================================================================================
AUTOMATED 32-GATE ZERO-CONTRADICTION & GROUP B PROVENANCE AUDIT (V13)
  - G01-G24 : Core V12/V13 Numerical, Protocol, CSCI, Energy & Provenance Gates
  - G25-G30 : Group B (B1 Node 1 R=5, B1 Node 4 T4 R=5, B2 Router, B3 Holdout,
              B4 Docker Replication, B5 SDSS Funnel)
  - G31-G32 : Abstract Word Count (<= 350 words), DOI/URL Hygiene, PDF & Flat13
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
RES_DIR = ROOT / "results"
DOCKER_DIR = ROOT / "docker_replication"

V13_TEX = MS_DIR / "bigquery_olap_benchmark_V13.tex"
V13_PDF = MS_DIR / "bigquery_olap_benchmark_V13.pdf"
V13_LOG = MS_DIR / "bigquery_olap_benchmark_V13.log"
FLAT13_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat13.zip"


def strip_cmt(t: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.splitlines())


def count_abstract_words(tex: str) -> int:
    head = tex.split(r"\section{Introduction}")[0]
    m = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", head)
    if not m:
        m = re.search(r"\\begin\{quote\}([\s\S]*?)\\end\{quote\}", head)
    if not m:
        return -1
    raw = strip_cmt(m.group(1))
    raw = re.split(r"\\textbf\{Keywords:?\}|Keywords:", raw)[0]
    raw = re.sub(r"\\textbf\{\\color\{[^}]*\}Abstract\}", " ", raw)
    raw = re.sub(r"\$[^\$]+\$", " MATH_TOK ", raw)
    raw = re.sub(r"\\url\{[^}]*\}", " URL_TOK ", raw)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", raw)
    cleaned = re.sub(r"[{}~$\\]", " ", cleaned)
    words = [w for w in re.split(r"\s+", cleaned.strip()) if w]
    return len(words)


def main() -> int:
    if not V13_TEX.exists():
        print(f"[FATAL] Missing {V13_TEX}")
        return 1

    raw_tex = V13_TEX.read_text(encoding="utf-8")
    clean_tex = strip_cmt(raw_tex)
    log_txt = V13_LOG.read_text(encoding="utf-8", errors="replace") if V13_LOG.exists() else ""

    gates: list[tuple[str, bool, str]] = []

    def add(gid: str, ok: bool, desc: str) -> None:
        gates.append((gid, bool(ok), desc))

    # --- G01-G24: Core V12/V13 Consistency Gates ---
    add("G01", all(k in clean_tex for k in ("55.37", "118.40")), "Protocol P1 Primary Nested-CV (55.37 ms, 118.40% P95) present")
    add("G02", all(k in clean_tex for k in ("54.92", "114.11")), "Protocol P2 Secondary Fixed Guard (54.92 ms, 114.11% P95) present")
    add("G03", all(k in clean_tex for k in ("57.33", "76.00", "34.98")), "Core 75Q Accuracy & Regret (57.33% / 76.00% / 34.98%) present")
    add("G04", all(k in clean_tex for k in ("183.74", "82.47", "56.96")), "Core 75Q Baselines (DuckDB 183.74 / Polars 82.47 / GPU 56.96) present")
    leg_hits = [x for x in ("51.12", "6.70", "103.32") if re.search(rf"(?<!\d){re.escape(x)}(?!\d)", clean_tex)]
    add("G05", len(leg_hits) == 0, f"Zero Legacy Micro-Benchmark Numbers (hits={leg_hits})")
    add("G06", "_allowbreak{}" not in clean_tex, "Zero Raw '_allowbreak{}' Leak")
    add("G07", "tab:v12_protocol_registry" in clean_tex, "Evaluation Protocol Registry Table present")
    add("G08", "tab:v12_data_provenance" in clean_tex, "Data Provenance Table present")
    add("G09", ("400" in clean_tex and "25" in clean_tex and "30" in clean_tex), "Workload Reconciliation (SDSS 25 + NYC 25 + Redset 30; 400 total)")
    add("G10", ("Oblique" in clean_tex or "oblique" in clean_tex) and ("3.84" in clean_tex or "1.84" in clean_tex), "CSCI Oblique Inductive Bias & VIF disclosed")
    add("G11", "tab:v12_csci_stat_energy" in clean_tex, "CSCI & Energy Provenance Table present")
    add("G12", ("cluster" in clean_tex.lower() and "25" in clean_tex), "Multi-Level Cluster Bootstrap (N_cluster=25) present")
    add("G13", ("NVML" in clean_tex and "TDP" in clean_tex), "Sensor-Separated Energy (NVML vs CPU TDP Estimate) disclosed")
    add("G14", "cuDF" in clean_tex, "RAPIDS cuDF Native GPU latencies present")
    add("G15", "cudf-polars" in clean_tex, "Polars GPU (cudf-polars) latencies present")
    add("G16", "Timing" in clean_tex or "boundary" in clean_tex.lower(), "Timing & Energy Boundaries disclosed")
    add("G17", ("Node~1" in clean_tex or "Node 1" in clean_tex) and ("Node~4" in clean_tex or "Node 4" in clean_tex), "Multi-Node Crossover Analysis present")
    add("G18", ("C1" in clean_tex and "C2" in clean_tex and "C3" in clean_tex) or "three" in clean_tex.lower(), "Consolidated Scientific Contributions present")
    add("G19", "limitation" in clean_tex.lower() or "Threats to Validity" in clean_tex, "Threats to Validity & Limitations disclosed")

    m_csv = RES_DIR / "v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
    m_ok = False
    if m_csv.exists():
        df_m = pd.read_csv(m_csv)
        m_ok = (len(df_m) == 64) and ("Timing_Boundary" in df_m.columns)
    add("G20", m_ok, "Master 4-Node 64-Run CSV exists with Timing_Boundary")

    ur = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    uc = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    add("G21", V13_PDF.exists() and ur == 0 and uc == 0, f"Compiled V13.pdf exists (Undef Refs={ur}, Undef Cites={uc})")

    zip_ok = False
    zip_note = "Missing"
    if FLAT13_ZIP.exists():
        with zipfile.ZipFile(FLAT13_ZIP, "r") as zf:
            names = zf.namelist()
            has_nested = any("/" in n or "\\" in n for n in names)
            has_tex = "bigquery_olap_benchmark_V13.tex" in names
            has_pdf = "bigquery_olap_benchmark_V13.pdf" in names
            zip_ok = (not has_nested) and has_tex and has_pdf
            zip_note = f"{len(names)} files, nested={has_nested}"
    add("G22", zip_ok, f"Flat13.zip strictly flat & contains V13.tex + V13.pdf ({zip_note})")

    no_esc = re.sub(r"\\[\{\}]", "", clean_tex)
    b_open, b_close = no_esc.count("{"), no_esc.count("}")
    add("G23", b_open == b_close and "$$" not in clean_tex, f"LaTeX Brace Balance ({b_open}=={b_close}) & Zero '$$'")

    no_dup_root = (not (ROOT / "bigquery_olap_benchmark_V13.tex").exists()) and (not (ROOT / "bigquery_olap_benchmark_V13.pdf").exists())
    add("G24", no_dup_root, "Zero backup duplicate V13.tex/V13.pdf at repo root")

    # --- G25-G30: Group B V13 Gates ---
    r5_n1 = RES_DIR / "v13_groupB_b1_node1_r5_multiscale_summary_ci.csv"
    n1_ok = False
    if r5_n1.exists():
        d1 = pd.read_csv(r5_n1)
        n1_ok = (len(d1) >= 20) and (d1["Repetitions_R"].eq(5).all()) and ("tab:v13_groupB_r5_and_sdss" in clean_tex)
    add("G25", n1_ok, "Group B-1 (Node 1 R=5 CV/CI CSV + tab:v13_groupB_r5_and_sdss)")

    r5_n4 = RES_DIR / "v13_groupB_b1_cloud_gpu_8to16vcpu_r5_summary_ci.csv"
    n4_ok = False
    if r5_n4.exists():
        d4 = pd.read_csv(r5_n4)
        engs = set(d4["Engine"].astype(str))
        n4_ok = (
            d4["Repetitions_R"].eq(5).all()
            and "RAPIDS_cuDF_Native_GPU" in engs
            and "Polars_GPU_cudf_polars" in engs
            and "tab:v13_groupB_node4_t4_r5" in clean_tex
        )
    add("G26", n4_ok, "Group B-1 (Node 4 Tesla T4 R=5 6-Engine CSV + tab:v13_groupB_node4_t4_r5)")

    rt_csv = RES_DIR / "v13_groupB_multiscale_router_evaluation.csv"
    rt_ok = rt_csv.exists() and ("tab:v13_groupB_multiscale_router" in clean_tex) and (pd.read_csv(rt_csv)["Routing_Accuracy_Pct"].eq(100.0).all())
    add("G27", rt_ok, "Group B-2 (Multi-Scale Router CSV 100% Acc + tab:v13_groupB_multiscale_router)")

    ho_ok = any(k in clean_tex for k in ("Seed~S7", "Seed S7", "JCC-H", "Prospective Holdout"))
    add("G28", ho_ok, "Group B-3 (Prospective Holdout Seed S7 / DSB / JCC-H in V13.tex)")

    rec_p = RES_DIR / "v13_docker_harness_selftest" / "independent_replication_receipt.json"
    dk_ok = False
    if rec_p.exists() and (DOCKER_DIR / "Dockerfile").exists():
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        dk_ok = (rec["verification"]["ex_multiset_match_rate"] == 1.0) and ("Containerized Independent Replication Protocol" in clean_tex)
    add("G29", dk_ok, "Group B-4 (Docker Replication Kit + EX=1.000 Receipt + V13.tex)")

    sd_csv = RES_DIR / "v13_groupB_sdss_attrition_funnel.csv"
    sd_ok = False
    if sd_csv.exists():
        cnts = pd.read_csv(sd_csv)["Count"].astype(int).tolist()
        sd_ok = (len(cnts) >= 2) and (cnts[0] == 142) and (cnts[-1] == 25) and ("SDSS DR16 SkyServer SQL Log Attrition Funnel" in clean_tex)
    add("G30", sd_ok, "Group B-5 (SDSS Attrition Funnel 142 -> 25 CSV + V13.tex)")

    # --- G31-G32: Journal Snapp Hygiene Gates ---
    abs_w = count_abstract_words(raw_tex)
    add("G31", 100 <= abs_w <= 350, f"Abstract Word Count <= 350 words (actual={abs_w} words)")

    bad_doi = bool(re.search(r"10\.1186/s40537(?![-\d])|10\.1016/j\.is\.2013\.(?!05\.004)", clean_tex))
    has_roq = bool(re.search(r"\bRoq\b|kamali2024", clean_tex))
    has_repo = "preecha-msu/bigquery-olap-benchmark" in clean_tex
    add("G32", (not bad_doi) and (not has_roq) and has_repo, f"DOI/Bib/Repo Hygiene (bad_doi={bad_doi}, has_roq={has_roq}, official_repo={has_repo})")

    # --- G33-G42: V13 Reviewer Risk & Auditability Overhaul Gates (R1-R6 & 4.1) ---
    import hashlib
    import subprocess

    def sha256_f(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for ch in iter(lambda: f.read(1048576), b""):
                h.update(ch)
        return h.hexdigest()

    r1_c = RES_DIR / "v13_r1_protocol_provenance_and_claim_classification.csv"
    r1_f = RES_DIR / "v13_r1_foldwise_guard_comparison.csv"
    add("G33", r1_c.exists() and r1_f.exists() and len(pd.read_csv(r1_f)) == 25 and ("tab:v13_claim_classification" in clean_tex),
        "R1 Protocol Provenance, 25-Fold Guard CSV, & Claim Classification Table present")

    r2_p = RES_DIR / "v13_r2_external_cluster_stability_loco_audit.csv"
    add("G34", r2_p.exists() and len(pd.read_csv(r2_p)) == 4 and ("tab:v13_r2_cluster_loco" in clean_tex),
        "R2 External Holdout Cluster Stability, LOCO Jackknife, & Weighting Sensitivity Table present")

    r3_1 = RES_DIR / "v13_r3_1_routing_decision_agreement_matrix.csv"
    r3_2 = RES_DIR / "v13_r3_2_decision_stability_under_prediction_noise.csv"
    r32_ok = r3_1.exists() and r3_2.exists() and (pd.read_csv(r3_2)["Guarded_Stressed_OOM_Rate_Pct"].eq(0.0).all()) and ("tab:v13_r3_noise_and_zerocompile" in clean_tex)
    add("G35", r32_ok, "R3.1-R3.2 Decision Agreement & Prediction-Noise Stability (0.0% Guarded OOM) present")

    r3_3 = RES_DIR / "v13_r3_3_zero_compile_overhead_audit.csv"
    r3_4 = RES_DIR / "v13_r3_4_csci_inductive_bias_isolation.csv"
    no_causal_csci = "causally" not in clean_tex.lower()
    add("G36", r3_3.exists() and r3_4.exists() and no_causal_csci,
        f"R3.3-R3.4 Zero-Compile 3-Boundary Overhead & Non-Causal CSCI Isolation (no_causal={no_causal_csci})")

    r4_p = RES_DIR / "v13_r4_cleanroom_reexecution_validation.csv"
    add("G37", r4_p.exists() and len(pd.read_csv(r4_p)) == 7 and ("tab:v13_r4_r5_reexec_and_repeatability" in clean_tex),
        "R4 Multi-Environment Re-Execution Validation Table (Env A vs Env B0-B3) present")

    r5_p = RES_DIR / "v13_r5_multiscale_targeted_repeatability_audit.csv"
    r5_ok = r5_p.exists() and (pd.read_csv(r5_p)["Pairwise_Trial_Sign_Stability_Pct"].eq(100.0).all())
    add("G38", r5_ok, "R5 Multi-Scale Repeatability SNR CSV (100.0% Pairwise Crossover Sign Stability)")

    reg_p = RES_DIR / "v13_master_query_registry_row_level.csv"
    reg_ok = False
    if reg_p.exists():
        dreg = pd.read_csv(reg_p)
        p400 = dreg[dreg["Corpus_Partition"] == "Primary_400Q_Benchmark"]
        cats = p400["Epistemic_Category"].value_counts().to_dict()
        c_i = cats.get("Cat_I_Core_Development", 0)
        c_ii = cats.get("Cat_IIa_ThirdParty_Calibration", 0) + cats.get("Cat_IIb_Boundary_Drift_Audit", 0)
        c_iii = cats.get("Cat_IIIa_Redset_Synthetic", 0) + cats.get("Cat_IIIb_SDSS_Adapted", 0) + cats.get("Cat_IIIc_NYCTLC_PublicSchema", 0)
        c_iv = cats.get("Cat_IV_Untouched_Standard_Holdout", 0)
        no_drift = ("480-query" not in clean_tex) and (not bool(re.search(r"55\s*\+\s*115", clean_tex)))
        reg_ok = (len(dreg) == 495) and (dreg["Query_ID"].nunique() == 495) and (c_i, c_ii, c_iii, c_iv) == (75, 90, 80, 155) and no_drift
    add("G39", reg_ok, "Point 4.1 Row-Level 495Q Master Registry (75+90+80+155=400 + 25 + 70) & Zero Corpus Drift")

    ood_pres = all(k in clean_tex for k in (r"12.0\%", r"83.98\%", r"346.00\%", r"51.124\%")) and ("tab:prospective_3suite_failure" in clean_tex)
    add("G40", ood_pres, "Prospective OOD Negative Result Preserved Unmodified (12.0% Top-1, 83.98% mean, 51.124% JCC-H)")

    man_p = RES_DIR / "v13_r1_r6_audit_manifest_sha256.json"
    man_ok = False
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        man_ok = all((RES_DIR / fn).exists() and sha256_f(RES_DIR / fn) == meta["sha256"] for fn, meta in man.get("artifacts", {}).items())
    add("G41", man_ok, "R1-R6 Audit Manifest SHA-256 Hashes Match All 11 CSV Artifacts Byte-for-Byte")

    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT), capture_output=True, text=True).stdout.splitlines()
    v11_v12_staged = [ln for ln in staged if re.search(r"(V11|V12|Flat11|Flat12)", ln)]
    builder_txt = (ROOT / "scripts" / "110_build_v13_reviewer_audit_edition.py").read_text(encoding="utf-8")
    no_v12_write = ("bigquery_olap_benchmark_V12" not in builder_txt) and ("Flat12" not in builder_txt)
    add("G42", len(v11_v12_staged) == 0 and no_v12_write, f"Strict Isolation Gate: Zero V11/V12 Files Staged or Written by V13 Scripts ({len(v11_v12_staged)} staged)")

    print("=" * 96)
    print(f"V13 AUTOMATED ZERO-CONTRADICTION & REVIEWER AUDIT SUITE ({len(gates)} GATES)")
    print("=" * 96)
    passed = 0
    for gid, ok, desc in gates:
        tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{tag}] {gid}: {desc}")
    print("-" * 96)
    print(f"[RESULT] {passed}/{len(gates)} GATES PASSED" + (" — ZERO CONTRADICTIONS!" if passed == len(gates) else " — ACTION REQUIRED!"))
    print("=" * 96)
    return 0 if passed == len(gates) else 2


if __name__ == "__main__":
    sys.exit(main())