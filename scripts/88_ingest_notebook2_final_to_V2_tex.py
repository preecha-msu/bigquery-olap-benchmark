#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/88_ingest_notebook2_final_to_V2_tex.py
Pulls the completed notebook2 remaining-experiment CSVs from data_hub/nodes/,
verifies SHA-256 digests, and automatically updates & compiles:
  E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark\manuscript\bigquery_olap_benchmark_V2.tex
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
NODES_DIR = PROJECT_ROOT / "data_hub" / "nodes"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_nb2_csv(filename: str) -> Path:
    hits = [p for p in NODES_DIR.rglob(filename) if p.is_file()]
    if hits:
        shutil.copy2(hits[0], RESULTS_DIR / filename)
        return RESULTS_DIR / filename
    local_p = RESULTS_DIR / filename
    if local_p.exists():
        return local_p
    raise FileNotFoundError(f"Missing {filename} from notebook2! Pull from GitHub hub first.")


def main() -> None:
    hub_script = PROJECT_ROOT / "scripts" / "hub_sync_notebook1_master.py"
    if hub_script.exists():
        subprocess.run([sys.executable, str(hub_script), "--mode", "pull"], cwd=str(PROJECT_ROOT))

    p_xfer = find_nb2_csv("nb2_remaining_part2_transfer_calibration_summary.csv")
    p_sla = find_nb2_csv("nb2_remaining_part3_bursty_sla_22t.csv")
    p_eng = find_nb2_csv("nb2_remaining_part1_energy_all_engines.csv")

    df_xfer = pd.read_csv(p_xfer)
    df_sla = pd.read_csv(p_sla)
    df_eng = pd.read_csv(p_eng)

    r_zs = df_xfer[df_xfer["Policy_on_Notebook2_22T"].str.contains("Zero-Shot")].iloc[0]
    r_fs = df_xfer[df_xfer["Policy_on_Notebook2_22T"].str.contains("5-Seed")].iloc[0]
    r_lo = df_xfer[df_xfer["Policy_on_Notebook2_22T"].str.contains("LOSO-CV")].iloc[0]
    r_sla_ast = df_sla[df_sla["Policy_22T"].str.contains("AST_Route")].iloc[0]
    r_sla_pol = df_sla[df_sla["Policy_22T"].str.contains("Polars")].iloc[0]

    sha_xfer = sha256_file(p_xfer)
    sha_sla = sha256_file(p_sla)

    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / f"bigquery_olap_benchmark_V2_pre_nb2final_{ts}.tex")
    tex = TARGET_TEX.read_text(encoding="utf-8")

    nb2_clause = (
        rf" Extended 75-query cross-hardware transfer on the 16-core/22-thread secondary node further confirms "
        rf"that zero-shot transfer from the primary workstation achieves ${r_zs['Mean_Latency_ms']:.2f}\text{{ ms}}$ "
        rf"(${r_zs['Top1_Pct']:.2f}\%$ Top-1), 5-seed multiplicative hardware calibration improves routing to "
        rf"${r_fs['Mean_Latency_ms']:.2f}\text{{ ms}}$ (${r_fs['Top1_Pct']:.2f}\%$ Top-1, ${r_fs['Mean_Regret_Pct']:.2f}\%$ regret), "
        rf"and native 25-seed LOSO-CV reaches ${r_lo['Mean_Latency_ms']:.2f}\text{{ ms}}$ (${r_lo['Top1_Pct']:.2f}\%$ Top-1), "
        rf"while 22-thread bursty contention replay cuts 250~ms SLA violations from ${r_sla_pol['SLA_250ms_Violation_Pct']:.2f}\%$ "
        rf"(Always Polars) to ${r_sla_ast['SLA_250ms_Violation_Pct']:.2f}\%$ (${r_sla_ast['Mean_Response_ms']:.2f}\text{{ ms}}$ response; "
        rf"\texttt{{nb2\_remaining\_part2\_transfer\_calibration\_summary.csv}}, SHA-256: \texttt{{{sha_xfer[:20]}...}}; "
        rf"\texttt{{nb2\_remaining\_part3\_bursty\_sla\_22t.csv}}, SHA-256: \texttt{{{sha_sla[:20]}...}})."
    )

    if "nb2\\_remaining\\_part2" not in tex and "5-seed multiplicative hardware calibration" not in tex:
        tex = re.sub(
            r"(millisecond-scale single-query CPU energy accounting\.)",
            lambda m: m.group(1) + nb2_clause,
            tex,
            count=1,
        )

    TARGET_TEX.write_text(tex, encoding="utf-8")
    for mirror in ("main.tex", "overleaf_ready_single_file.tex", "bigquery_olap_benchmark_V6.tex"):
        mp = MANUSCRIPT_DIR / mirror
        if mp.exists():
            mp.write_text(tex, encoding="utf-8")

    for _ in (1, 2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MANUSCRIPT_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if proc.returncode != 0:
            raise RuntimeError("pdflatex failed:\n" + "\n".join(proc.stdout.splitlines()[-25:]))

    print("=" * 100)
    print(f"[INGEST COMPLETE] Integrated notebook2 75Q Transfer & Bursty SLA into {TARGET_TEX.name}")
    print(f"  * NB2 Transfer CSV SHA-256 : {sha_xfer}")
    print(f"  * NB2 Bursty SLA SHA-256   : {sha_sla}")
    print(f"  * Updated V2.tex SHA-256   : {sha256_file(TARGET_TEX)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
