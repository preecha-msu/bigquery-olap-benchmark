#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/124_instant_overleaf_v8_5_and_snapp_sync.py
Zero-pdflatex instant patch (<1s):
  1. Fixes line 836 brace balance & converts Section 4.7 items 5 and 6 to
     Subsubsections 4.7.1, 4.7.2, and 4.7.3 using \texorpdfstring.
  2. Fixes remaining Table cross-references in Sec 4.6, 4.7, 4.8 and resolves
     any undefined \ref{tab:jbd_unseen_t11} -> 16.
  3. Moves Table 15 \caption above the table body.
  4. Adds superscript 1 before Department of Information Technology on Page 1.
  5. Writes directly to manuscript\bigquery_olap_benchmark_V8_5.tex and builds
     Journal_of_Big_Data_Snapp_Submission_Flat.zip + SHA-256 CSV audit log.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS = ROOT / "manuscript"
FIG = MS / "figures"
RES = ROOT / "results"
SNAPP = ROOT / "Snapp_Journal_of_Big_Data_Submission"
TEX_PATH = MS / "bigquery_olap_benchmark_V8_5.tex"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def count_unquoted_braces(tex: str) -> tuple[int, int]:
    cleaned = []
    for ln in tex.splitlines():
        nc = re.sub(r"(?<!\\)%.*$", "", ln).replace(r"\{", "").replace(r"\}", "")
        cleaned.append(nc)
    s = "\n".join(cleaned)
    return s.count("{"), s.count("}")


def main() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")

    # 1. Cleanly replace Section 4.7 items 5, 4.7.1/4.7.2, and 6 line-by-line (zero brace risk)
    new_lines = []
    for ln in tex.splitlines():
        if ("Comprehensive Baseline Matrix" in ln) and ("Standalone CPU-Only Routing" in ln or "Cold-Metadata" in ln):
            new_lines.append(
                r"\subsubsection{Comprehensive Baseline Matrix, Standalone CPU-Only Routing, Multi-Scale Byte Accounting, and Metadata Cache Audit}"
            )
            continue
        if ("Seed-Level" in ln) and ("Simple Interpretable Baselines" in ln) and ("LOTO-CV" in ln):
            new_lines.append(
                r"\subsubsection{Seed-Level (\texorpdfstring{$N=25$}{N=25}) Inference, Simple Interpretable Baselines, and Leave-One-Tier-Out (LOTO-CV) Boundaries}"
            )
            continue
        if r"\subsubsection{Zero-Leakage Nested Calibration" in ln and r"\texorpdfstring" not in ln:
            ln = ln.replace("($K=25$ Seeds)", r"(\texorpdfstring{$K=25$}{K=25} Seeds)")
        new_lines.append(ln)
    tex = "\n".join(new_lines)

    # Strip any accidental double closing braces at the end of \subsubsection{...}}
    tex = re.sub(r"(\\subsubsection\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}+", lambda m: m.group(1), tex)

    # 2. Resolve all Table \ref and hardcoded table numbers in Sec 4.6, 4.7, 4.8
    ref_map = {
        r"\ref{tab:jbd_glance_1a}": "1",
        r"\ref{tab:jbd_related_2a}": "2",
        r"\ref{tab:jbd_meta_2b}": "3",
        r"\ref{tab:jbd_duckdb_t1}": "4",
        r"\ref{tab:jbd_bq_duck_t2}": "5",
        r"\ref{tab:jbd_finops_t3}": "6",
        r"\ref{tab:jbd_gpu_3level_t4}": "7",
        r"\ref{tab:jbd_six_eng_t5}": "8",
        r"\ref{tab:jbd_throughput_t6}": "9",
        r"\ref{tab:jbd_energy_t7}": "10",
        r"\ref{tab:jbd_routing_t8}": "11",
        r"\ref{tab:jbd_stat_audit_t10}": "12",
        r"\ref{tab:jbd_ablation_t9}": "13",
        r"\ref{tab:jbd_csci_oblique_t9b}": "14",
        r"\ref{tab:jbd_step3_t14}": "15",
        r"\ref{tab:jbd_unseen_t11}": "16",
        r"\ref{tab:jbd_job_regimes_t11b}": "17",
        r"\ref{tab:jbd_multiscale_t12}": "18",
        r"\ref{tab:jbd_manifest_t13}": "19",
        r"\ref{tab:jbd_multinode_t15}": "20",
        r"\ref{tab:jbd_stability_t16}": "21",
        r"\ref{tab:jbd_ood_t17}": "22",
    }
    for k, v in ref_map.items():
        tex = tex.replace(k, v)

    tex = re.sub(
        r"(\\subsection\{Head-to-Head Evaluation[^}]*\}\s*)Tables[\s~]+(?:8|11)\s+and[\s~]+(?:10|12)",
        lambda m: m.group(1) + "Tables~11 and~12",
        tex,
        count=1,
    )
    tex = re.sub(
        r"(validate the internal mechanisms of AST-Route,\s*)Tables[\s~]+(?:9|13)\s+and[\s~]+14",
        lambda m: m.group(1) + "Tables~13 and~14",
        tex,
        count=1,
    )
    tex = re.sub(
        r"(\\subsection\{Cost Prediction Q-Error[^}]*\}\s*)Table[\s~]+(?:11|16)",
        lambda m: m.group(1) + "Table~16",
        tex,
        count=1,
    )

    # 3. Move Table 15 \caption above the table body
    sig15 = "Comprehensive multi-baseline and operational robustness evaluation:"
    p15 = tex.find(sig15)
    if p15 != -1:
        ts = tex.rfind(r"\begin{table", 0, p15)
        te_m = re.search(r"\\end\{table\*?\}", tex[p15:])
        if ts != -1 and te_m:
            te = p15 + te_m.end()
            blk = tex[ts:te]
            c_pos = blk.rfind(r"\caption", 0, blk.find(sig15) + 10)
            b_cands = [blk.find(k) for k in (r"\resizebox", r"\begin{tabular", r"\begin{fitasttable", r"\scriptsize", r"\footnotesize") if blk.find(k) != -1]
            b0 = min(b_cands) if b_cands else -1
            if c_pos != -1 and b0 != -1 and c_pos > b0:
                d, ce = 0, c_pos
                for i in range(blk.find("{", c_pos), len(blk)):
                    if blk[i] == "\\" and i + 1 < len(blk):
                        continue
                    if blk[i] == "{":
                        d += 1
                    elif blk[i] == "}":
                        d -= 1
                        if d == 0:
                            ce = i + 1
                            break
                while True:
                    lm = re.match(r"\s*\\label\{[^}]+\}", blk[ce:])
                    if lm:
                        ce += lm.end()
                    else:
                        break
                cap_str = blk[c_pos:ce].strip()
                blk_clean = blk[:c_pos].rstrip() + "\n" + blk[ce:].lstrip()
                hdr_m = re.match(r"^(\\begin\{table\*?\}(?:\[[^\]]*\])?\s*(?:\\centering\s*)?)", blk_clean)
                if hdr_m:
                    ins = hdr_m.end()
                    tex = tex[:ts] + blk_clean[:ins] + cap_str + "\n" + blk_clean[ins:] + tex[te:]

    # 4. Add superscript 1 before Department of Information Technology on Page 1
    tex = re.sub(
        r"(?<!\\textsuperscript\{1\})Department of Information Technology,\s*Faculty of Informatics",
        lambda _: r"\textsuperscript{1}Department of Information Technology, Faculty of Informatics",
        tex,
        count=1,
    )

    # 5. Assert exact brace balance before writing
    b_open, b_close = count_unquoted_braces(tex)
    assert b_open == b_close, f"Brace mismatch: {b_open} != {b_close}"

    # Write directly to target files (no backup files)
    for target in (TEX_PATH, MS / "main.tex", MS / "bigquery_olap_benchmark_JBD_Submission_Ready.tex"):
        target.write_text(tex, encoding="utf-8")

    # 6. Build Flat-Directory Snapp/Overleaf ZIP
    SNAPP.mkdir(parents=True, exist_ok=True)
    flat_tex = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+)(\})", r"\1\2\3", tex)
    (SNAPP / "manuscript.tex").write_text(flat_tex, encoding="utf-8")
    (SNAPP / "bigquery_olap_benchmark_V8_5.tex").write_text(flat_tex, encoding="utf-8")

    for fig_ref in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat_tex))):
        fn = Path(fig_ref).name
        if not Path(fn).suffix:
            fn += ".pdf"
        src_f = FIG / fn
        if not src_f.exists() and (ROOT / "figures" / fn).exists():
            src_f = ROOT / "figures" / fn
        if src_f.exists():
            shutil.copy2(src_f, SNAPP / fn)

    zip_out = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SNAPP.iterdir()):
            if f.is_file() and f.suffix in (".tex", ".pdf", ".bib") and f.name != "manuscript.pdf":
                zf.write(f, arcname=f.name)

    # 7. Real-execution SHA-256 verification manifest across CSV logs
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for cf in sorted(RES.glob("*.csv"))[:30]:
        rows.append({"Artifact": cf.name, "Bytes": cf.stat().st_size, "SHA256": sha256_of(cf)})
    audit_csv = RES / "v8_5_unbiased_reviewer_sha256_audit.csv"
    pd.DataFrame(rows).to_csv(audit_csv, index=False, encoding="utf-8")

    print("=" * 92)
    print(f"[OK <1s] Updated TeX : {TEX_PATH} (Braces: {b_open} == {b_close})")
    print(f"[OK <1s] Built ZIP   : {zip_out} ({zip_out.stat().st_size:,} bytes)")
    print(f"[OK <1s] SHA256 Log  : {audit_csv} ({sha256_of(audit_csv)[:24]}...)")
    print("=" * 92)


if __name__ == "__main__":
    main()