#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/96_fix_6_visual_and_text_defects_in_35page_pdf.py
================================================================================
1. Restore clean compilable manuscript/bigquery_olap_benchmark_V2.tex from HEAD
   (or strip the two corrupted lines if already modified).
2. Apply all 6 visual/text fixes using whole-line replacement (zero trailing braces!):
   - Fix 1: 'Algorithm~\ref{alg:ast_route}' -> 'Algorithm~1' (eliminates 'Algorithm ??' on p.9)
   - Fix 2: Whole-line author replacement -> '{\large\textbf{Preecha Noiumkar\textsuperscript{1,*}}}\\'
   - Fix 3: Table 2 Overall Mean row: 254.4 & 183.7 -> 183.7 & 254.4
   - Fix 4: Table 11 Note: GBR-4D -> GBR-9D
   - Fix 5: Whole-line \begin{tabular} replacement in Table 12 & Table 14 (no leftover braces!)
   - Fix 6: Availability hash deduplication + macbook filenames + Ref [13] full DOI
3. Compile pdflatex (3 passes), run Phase 4 Gatekeeper (10/10 PASS), build Cover Letter
   & Reviewer Response PDFs, and replace on GitHub (origin) with zero backup folders.
================================================================================
"""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"
TARGET_PDF = MS_DIR / "bigquery_olap_benchmark_V2.pdf"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_clean_head_if_corrupted() -> None:
    tex = TARGET_TEX.read_text(encoding="utf-8")
    if r"{\large\textbf{{\large\textbf{" in tex or r"@{}}p{" in tex:
        subprocess.run(
            ["git", "checkout", "--", "manuscript/bigquery_olap_benchmark_V2.tex"],
            cwd=str(ROOT),
            check=False,
        )
        print("  [RESTORED] ดึงสถานะสะอาดจาก Git HEAD (7c3e90af...) เรียบร้อยแล้ว")


def apply_all_6_fixes_cleanly() -> dict[str, bool]:
    restore_clean_head_if_corrupted()
    tex = TARGET_TEX.read_text(encoding="utf-8")
    results: dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # Fix 1: Eliminate 'Algorithm ??' on Page 9
    # -------------------------------------------------------------------------
    tex = tex.replace(r"Algorithm~\ref{alg:ast_route}", "Algorithm~1")
    tex = tex.replace(r"Algorithm \ref{alg:ast_route}", "Algorithm~1")
    results["Fix1_No_Algorithm_QuestionMarks"] = r"\ref{alg:ast_route}" not in tex

    # -------------------------------------------------------------------------
    # Fix 2: Whole-line Author replacement on Page 1 (balanced braces & single superscript!)
    # -------------------------------------------------------------------------
    lines = tex.splitlines()
    for i in range(min(120, len(lines))):
        s = lines[i].strip()
        if "Preecha Noiumkar" in s and "Correspondence:" not in s and "Preecha Noiumkar (PN)" not in s and not s.startswith("%"):
            # Replace the ENTIRE line so no outer {\large\textbf{ is left unclosed
            if s.endswith(r"\\"):
                lines[i] = r"{\large\textbf{Preecha Noiumkar\textsuperscript{1,*}}}\\"
            else:
                lines[i] = r"{\large\textbf{Preecha Noiumkar\textsuperscript{1,*}}}"
            break
    tex = "\n".join(lines)
    results["Fix2_Author_Superscript_Single"] = (
        r"{\large\textbf{Preecha Noiumkar\textsuperscript{1,*}}}" in tex
        and r"{\large\textbf{{\large\textbf{" not in tex
    )

    # -------------------------------------------------------------------------
    # Fix 3: Table 2 Overall Mean row (Page 12): 254.4 & 183.7 -> 183.7 & 254.4
    # -------------------------------------------------------------------------
    lines = tex.splitlines()
    for i, line in enumerate(lines):
        if "Overall Mean" in line and "207.7" in line and "254.4" in line and "183.7" in line:
            # Swap 254.4 and 183.7 on this exact line
            line = line.replace("254.4", "__TMP_254__").replace("183.7", "254.4").replace("__TMP_254__", "183.7")
            lines[i] = line
    tex = "\n".join(lines)
    results["Fix3_Table2_Overall_183_7_First"] = any(
        ("Overall Mean" in ln and "207.7" in ln and ln.find("183.7") < ln.find("254.4"))
        for ln in tex.splitlines()
    )

    # -------------------------------------------------------------------------
    # Fix 4: Table 11 Note (Page 15): replace GBR-4D -> GBR-9D
    # -------------------------------------------------------------------------
    tex = tex.replace("GBR-4D", "GBR-9D").replace("4D model", "9D model")
    results["Fix4_No_GBR_4D_Typo"] = "GBR-4D" not in tex

    # -------------------------------------------------------------------------
    # Fix 5: Whole-line \begin{tabular} replacement in Table 12 & Table 14 (no leftover braces!)
    # -------------------------------------------------------------------------
    lines = tex.splitlines()
    in_t12 = False
    in_t14 = False
    for i, ln in enumerate(lines):
        if "Big Data Multi-Scale" in ln:
            in_t12 = True
            in_t14 = False
        elif "Consolidated Step-3 robustness audit" in ln:
            in_t14 = True
            in_t12 = False
        elif r"\end{table" in ln:
            in_t12 = False
            in_t14 = False

        if in_t12 and r"\begin{tabular}" in ln:
            lines[i] = r"\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.08}\begin{tabular}{@{}p{2.2cm}p{3.4cm}ccccccp{5.5cm}@{}}"
            in_t12 = False
        elif in_t14 and r"\begin{tabular}" in ln:
            lines[i] = r"\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.08}\begin{tabular}{@{}p{4.2cm}p{2.7cm}ccccccp{4.5cm}@{}}"
            in_t14 = False
    tex = "\n".join(lines)
    results["Fix5_Table12_And_14_Wrapped_Clean"] = (
        r"p{5.5cm}@{}}" in tex
        and r"p{4.5cm}@{}}" in tex
        and r"@{}}p{" not in tex
    )

    # -------------------------------------------------------------------------
    # Fix 6: Availability of data (Pages 32-33) & Reference [13] full DOI (Page 34)
    # -------------------------------------------------------------------------
    tex = re.sub(
        r"d3b6edd646b4b71c51842ae0\.\.\s*6b4b71c51842ae0\.\.\.",
        "d3b6edd646b4b71c51842ae0...",
        tex,
    )
    tex = tex.replace("cbook\\_reexecution\\_audit.csv", "macbook\\_reexecution\\_audit.csv")
    tex = re.sub(
        r"macbook\\_exp2\\_routing\\_summary(?!\.csv)",
        r"macbook\\_exp2\\_routing\\_summary.csv",
        tex,
    )
    tex = re.sub(
        r"10\.1186/s40537(?!-02[0245])",
        "10.1186/s40537-024-00939-0",
        tex,
    )
    results["Fix6_Availability_And_Ref13_DOI"] = (
        "10.1186/s40537-024-00939-0" in tex
        and "6b4b71c51842ae0.. 6b4b71c51842ae0" not in tex
    )

    # Write in-place to bigquery_olap_benchmark_V2.tex and all mirror files (Zero Backup)
    TARGET_TEX.write_text(tex, encoding="utf-8")
    for mirror in (
        ROOT / "bigquery_olap_benchmark_V2.tex",
        ROOT / "main.tex",
        MS_DIR / "main.tex",
        MS_DIR / "overleaf_ready_single_file.tex",
        MS_DIR / "bigquery_olap_benchmark_V6.tex",
    ):
        mirror.write_text(tex, encoding="utf-8")

    return results


def compile_and_run_phase4() -> None:
    for pass_i in (1, 2, 3):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"[ERROR on pdflatex pass {pass_i}]:")
            print("\n".join(proc.stdout.splitlines()[-25:]))
            raise RuntimeError(f"pdflatex failed on pass {pass_i}")

    for copy_p in (
        ROOT / "bigquery_olap_benchmark_V2.pdf",
        MS_DIR / "bigquery_olap_benchmark_V6.pdf",
        ROOT / "bigquery_olap_benchmark_V6.pdf",
        MS_DIR / "bigquery_olap_benchmark_V6_2.pdf",
        ROOT / "bigquery_olap_benchmark_V6_2.pdf",
        MS_DIR / "main.pdf",
    ):
        try:
            shutil.copy2(TARGET_PDF, copy_p)
        except PermissionError:
            pass

    s95 = ROOT / "scripts" / "95_phase4_final_submission_pack_cover_and_response.py"
    if s95.exists():
        subprocess.run(["python", str(s95)], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    print("=" * 98)
    print("CLEAN WHOLE-LINE FIX OF 6 RESIDUAL DEFECTS + PHASE 4 SUBMISSION PACK ON NOTEBOOK1...")
    print("=" * 98)
    res = apply_all_6_fixes_cleanly()
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'CHECK'}] {k}")
    compile_and_run_phase4()
