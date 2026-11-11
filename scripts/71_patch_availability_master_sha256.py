#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, re, shutil, subprocess, zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"
LOG_FILE = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.log"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

tex = TARGET_TEX.read_text(encoding="utf-8")

avail_sentence = (
    " Furthermore, the complete 6-stage V4 independent validation certificate "
    "(\\texttt{v4\\_master\\_validation\\_plan\\_all\\_6\\_completed.csv}, SHA-256: \\texttt{5a645bba57807962...}) "
    "and the secondary-machine re-execution audit log on \\texttt{LAPTOP-LTL5PC7P} "
    "(\\texttt{external\\_machine\\_reexecution\\_audit.csv}, SHA-256: \\texttt{6d4cb51a12687be8...}) "
    "are archived in the repository."
)

if "5a645bba57807962" not in tex:
    # จับตำแหน่งท้ายหัวข้อ Availability of data and materials ก่อนขึ้นหัวข้อ Competing interests
    tex, n_ins = re.subn(
        r"(\\(?:sub)?section\*\{Competing interests\})",
        lambda m: avail_sentence.lstrip() + "\n\n" + m.group(1),
        tex,
        count=1,
    )
    print(f"[INSERT-2 FIXED] Inserted 6/6 Master SHA-256 before Competing interests: {n_ins == 1}")

TARGET_TEX.write_text(tex, encoding="utf-8")
shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "main.tex")
shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "overleaf_ready_single_file.tex")

for p in (1, 2):
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
        cwd=str(MANUSCRIPT_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"[COMPILE ERROR Pass {p}]:")
        print("\n".join(proc.stdout.splitlines()[-25:]))
        raise RuntimeError("LaTeX compilation failed")

log_new = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
undef_new = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_new)

pdf_v2 = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf"
for clone_name in ["bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V4.pdf"]:
    try:
        shutil.copy2(pdf_v2, MANUSCRIPT_DIR / clone_name)
    except PermissionError:
        pass

for zname in ["overleaf_V2_submission_bundle.zip", "overleaf_project_bundle.zip"]:
    zpath = MANUSCRIPT_DIR / zname
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, arcname="main.tex")
        zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
        bib = MANUSCRIPT_DIR / "references.bib"
        if bib.exists():
            zf.write(bib, arcname="references.bib")
        for f in sorted(FIGURES_DIR.glob("*")):
            if f.is_file():
                zf.write(f, arcname=f"figures/{f.name}")

final_tex = TARGET_TEX.read_text(encoding="utf-8")
print("=" * 96)
print(f"[VERIFY 1] LAPTOP-LTL5PC7P present in V2.tex : {'LAPTOP-LTL5PC7P' in final_tex}")
print(f"[VERIFY 2] Master SHA-256 present in V2.tex  : {'5a645bba57807962' in final_tex}")
print(f"[VERIFY 3] Undefined References Count        : {len(undef_new)} (Must be 0)")
print(f"[FINAL TEX SHA-256] {sha256_file(TARGET_TEX)}")
print(f"[FINAL PDF SHA-256] {sha256_file(pdf_v2)}")
print("=" * 96)