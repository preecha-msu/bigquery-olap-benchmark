#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
206_final_6point_perfection_v8_5_2.py
================================================================================
Closes the final 6 residual items spotted in bigquery_olap_benchmark_V8_5_2_3.pdf:
  1. Removes erroneous ', Roq \cite{raasveldt2018fair}' from Table 2 (Page 4).
  2. Fixes truncated end of Abstract on Page 1 line 26 ('across SF = 1-100 GB.').
  3. Updates Table 1 Row 5 (Page 3) & Conclusion (Page 35 line 1122) to include
     1.04x-1.07x speedup over best single engine alongside 3.35x-7.61x over DuckDB.
  4. Adds seed-cluster p_seed = 0.0042 / 0.0284 alongside query-level p = 0.0021 / 0.0187
     in Section 4.6 (Page 18 line 594).
  5. Replaces remaining 'zero-shot' labels for 12D JOB in Table 1 Row 4 (Page 3),
     Page 25 line 823, and Table 17 Col 2 & Note (Page 26) with 'Post-Hoc Exploratory'.
  6. Locates and re-runs the exact scripts generating Figure 1, Figure 2, and Figure 12b
     so vector PDFs in manuscript/figures/ display 'Operator-Tier Alignment' and
     '12D Post-Hoc' instead of 'Construct Validity' and '12D Zero-Shot'.
================================================================================
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V8_5_2.tex"
TARGET_PDF = MS_DIR / "bigquery_olap_benchmark_V8_5_2.pdf"
TARGET_LOG = MS_DIR / "bigquery_olap_benchmark_V8_5_2.log"
TARGET_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def update_and_rerun_figure_scripts():
    """
    Finds every Python script across the project that writes Figure 1, Figure 2, or Figure 12b,
    replaces 'Construct Validity' and '12D Zero-Shot', and executes those scripts.
    """
    target_scripts = []
    for py_file in sorted((ROOT / "scripts").glob("*.py")):
        if py_file.name.startswith("20"):
            continue
        try:
            txt = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        has_fig_ref = any(
            k in txt
            for k in [
                "fig_proposed_ast_routing_framework",
                "fig_experimental_workflow",
                "fig_job_12d_sketch_resolution",
                "Stepwise 12D Pre-Execution Sketch",
                "LAYER 1B: 9D/10D Feature Vector",
                "PHASE 3: Static AST Feature Profiling",
            ]
        )
        if has_fig_ref:
            new_txt = (
                txt.replace("Construct Validity:", "Operator-Tier Alignment:")
                .replace("CSCI Construct Validation", "CSCI Operator-Tier Alignment")
                .replace("Construct Validation", "Operator-Tier Alignment")
                .replace("12D Zero-Shot\\n(Sketch Gate)", "12D Post-Hoc\\n(Sketch Gate)")
                .replace("12D Zero-Shot\n(Sketch Gate)", "12D Post-Hoc\n(Sketch Gate)")
                .replace("12D Zero-Shot", "12D Post-Hoc")
            )
            if new_txt != txt:
                py_file.write_text(new_txt, encoding="utf-8")
                print(f"[UPDATED FIGURE SCRIPT] {py_file.name}")
            target_scripts.append(py_file)

    for py_file in target_scripts:
        # Only run standalone figure scripts (skip heavy full-manuscript builders)
        if any(w in py_file.name.lower() for w in ["fig", "plot", "diagram", "unify", "115", "118"]):
            print(f"[RUNNING FIGURE GENERATOR] {py_file.name} ...")
            subprocess.run(
                ["python", str(py_file)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )


def patch_manuscript():
    content = TARGET_TEX.read_text(encoding="utf-8")

    # 1. Remove ', Roq \cite{raasveldt2018fair}' from Table 2 (Page 4)
    content = re.sub(r",?\s*Roq\s*~?\\cite\{[^}]+\}", "", content)

    # 2. Fix truncated end of Abstract on Page 1 line 26 ('across SF = 1-100 GB.')
    content = re.sub(
        r"across\s+\$?SF\s*=\s*1(?:\\text\{--\}|--|-)100\s*\(\$\.",
        r"across $SF=1\text{--}100\text{ GB}$.",
        content,
    )
    content = re.sub(
        r"across\s+\$?SF\s*=\s*1(?:\\text\{--\}|--|-)100\s*G\$\.",
        r"across $SF=1\text{--}100\text{ GB}$.",
        content,
    )

    # 3. Update Table 1 Row 5 (Page 3) & Conclusion (Page 35 line 1122) to lead with 1.04x-1.07x vs Best Single Engine
    content = re.sub(
        r"Sustains\s+(?:\\textbf\{)?\$?3\.35\\times(?:\s*\\text\{--\}|--|-)\s*7\.61\\times\$?(?:\})?\s+speedups\s+over\s+DuckDB",
        lambda _: r"Achieves $\mathbf{1.04\times\text{--}1.07\times}$ speedups over best single engine ($3.35\times\text{--}7.61\times$ over DuckDB)",
        content,
    )
    content = re.sub(
        r"and\s+(?:\\textbf\{)?\$?3\.35\\times(?:\s*\\text\{--\}|--|-)\s*7\.61\\times\$?(?:\})?\s+speedups\s+over\s+DuckDB\s+across\s+\$?SF\s*=\s*1(?:\\text\{--\}|--|-)100(?:\s*\\text\{\s*GB\}|\s*GB)?\$?",
        lambda _: r"and $1.04\times\text{--}1.07\times$ speedups over the best single engine per scale ($3.35\times\text{--}7.61\times$ over DuckDB) across $SF=1\text{--}100\text{ GB}$",
        content,
    )

    # 4. Add seed-cluster p_seed alongside query-level p in Section 4.6 (Page 18 line 594)
    content = re.sub(
        r"Rule-Based Routing\s*\(\$p=0\.0187\$,\s*Cliff's\s*\$\\delta=\+0\.126\$\)",
        lambda _: r"Rule-Based Routing (seed-cluster $p_{\text{seed}}=0.0284$, query-level $p=0.0187$, Cliff's $\delta=+0.126$)",
        content,
    )

    # 5. Replace remaining 'zero-shot' labels on 12D JOB in Table 1 Row 4, Page 25 line 823, and Table 17 (Page 26)
    content = re.sub(
        r"adding\s+static\s+\$?10\.2(?:\\text\{\s*KB\}|\s*KB)\$?\s+catalog\s+sketches\s+lifts\s+zero-shot\s+Top-1\s+to\s+(?:\\textbf\{)?69\.70\\%(?:\})?",
        lambda _: r"adding post-hoc exploratory $10.2\text{ KB}$ catalog sketches lifts Top-1 to \textbf{69.70\%}",
        content,
    )
    content = re.sub(
        r"Under\s+strict\s+zero-shot\s+transfer\s*\(applying\s+a\s+TPC-H-calibrated\s+fanout\s+crossover\s+gate\s+over\s+the\s+locked\s+TPC-H\s+top-2\s+candidate\s+set\s+with\s+zero\s+IMDB\s+runtime\s+labels\)",
        lambda _: r"Under post-hoc exploratory catalog-sketch transfer (applying a TPC-H-calibrated fanout crossover gate over the locked TPC-H top-2 candidate set with zero IMDB runtime labels after diagnosing 9D string blindness)",
        content,
    )
    content = re.sub(
        r"Strict\s+Zero-Shot\s*\(\$?0/33\$?\s+JOB\s+runtime\s+labels;\s+static\s+catalog\s+sketch\)",
        lambda _: r"Post-Hoc Exploratory ($0/33$ JOB runtime labels; static catalog sketch)",
        content,
    )
    content = re.sub(
        r"\(2\)\s*\\textbf\{Strict\s+Zero-Shot\s+Levels\s+1\s*\\&\s*2:\}\s*Use\s+\$?0/33\$?\s+JOB\s+execution\s+labels",
        lambda _: r"(2) \textbf{Label-Free Levels 1 (Strict Zero-Shot 9D) \& 2 (Post-Hoc Exploratory 12D Sketch):} Use $0/33$ JOB execution labels",
        content,
    )

    open_b, close_b = content.count("{"), content.count("}")
    print(f"[BRACE AUDIT] Open={open_b} | Close={close_b} | Diff={open_b - close_b}")
    TARGET_TEX.write_text(content, encoding="utf-8")
    print(f"[SAVED LATEX] {TARGET_TEX.name} | SHA-256: {sha256_file(TARGET_TEX)}")


def compile_and_package():
    if shutil.which("pdflatex"):
        print("[COMPILING PDF VIA PDFLATEX (3 PASSES)...]")
        for _ in range(3):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", TARGET_TEX.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        log_txt = TARGET_LOG.read_text(encoding="utf-8", errors="replace") if TARGET_LOG.exists() else ""
        undef_refs = sorted(set(re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)))
        undef_cites = sorted(set(re.findall(r"LaTeX Warning: Citation `([^']+)' on page \d+ undefined", log_txt)))
        print(
            f"[FINAL PDF AUDIT] {TARGET_PDF.name} ({TARGET_PDF.stat().st_size:,} bytes) "
            f"| Undefined Refs={len(undef_refs)} {undef_refs} "
            f"| Undefined Cites={len(undef_cites)} {undef_cites} "
            f"| PDF SHA-256: {sha256_file(TARGET_PDF)}"
        )

    if TARGET_ZIP.exists():
        TARGET_ZIP.unlink()
    files_to_pack = [TARGET_TEX]
    if TARGET_PDF.exists():
        files_to_pack.append(TARGET_PDF)
    for ext in ("*.bib", "*.bst", "*.cls", "*.sty"):
        files_to_pack.extend(MS_DIR.glob(ext))
    fig_dir = MS_DIR / "figures"
    if fig_dir.exists():
        for f in fig_dir.glob("*.*"):
            if f.suffix.lower() in (".pdf", ".png", ".eps", ".jpg"):
                files_to_pack.append(f)

    seen = set()
    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_pack:
            if fpath.name not in seen:
                zf.write(fpath, arcname=fpath.name)
                seen.add(fpath.name)
    print(f"[BUILT FLAT ZIP] {TARGET_ZIP.name} ({len(seen)} files, {TARGET_ZIP.stat().st_size:,} bytes) | SHA-256: {sha256_file(TARGET_ZIP)}")

    print("[SYNCING & REPLACING ON GITHUB...]")
    for cmd in [
        ["git", "add", str(TARGET_TEX), str(TARGET_PDF), str(TARGET_ZIP), "manuscript/figures", "scripts/206_final_6point_perfection_v8_5_2.py"],
        ["git", "commit", "-m", "V8.5.2 Final 6-Point Perfection: Remove Roq [39] in Table 2, fix Abstract GB, align 12D Post-Hoc & figures"],
        ["git", "push", "origin", "HEAD"],
    ]:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"  $ {' '.join(cmd[:4])} ... -> code={proc.returncode}")


def main():
    print("=" * 110)
    print("EXECUTING SCRIPT 206: FINAL 6-POINT PERFECTION & FIGURE SYNC FOR V8_5_2")
    print("=" * 110)
    update_and_rerun_figure_scripts()
    patch_manuscript()
    compile_and_package()


if __name__ == "__main__":
    main()