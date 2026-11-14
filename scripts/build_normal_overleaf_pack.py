#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, shutil, subprocess, zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. ซิงค์ไฟล์ .tex หลักไปยัง main.tex และ overleaf_ready_single_file.tex สำหรับใช้งานแบบปกติ
tex_content = TARGET_TEX.read_text(encoding="utf-8")
for name in ("main.tex", "overleaf_ready_single_file.tex", "bigquery_olap_benchmark_V6.tex"):
    (MS_DIR / name).write_text(tex_content, encoding="utf-8")

# 2. สร้างไฟล์ .zip แบบปกติ (Self-Contained: มี main.tex, references.bib และโฟลเดอร์ figures/ ครบทุกรูป)
zip_targets = [
    ROOT / "Overleaf_V6_Project.zip",
    MS_DIR / "Overleaf_V6_Project.zip",
    MS_DIR / "overleaf_V2_submission_bundle.zip",
    MS_DIR / "overleaf_project_bundle.zip",
]

for zpath in zip_targets:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MS_DIR / "main.tex", arcname="main.tex")
        zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
        bib = MS_DIR / "references.bib"
        if bib.exists():
            zf.write(bib, arcname="references.bib")
        if FIG_DIR.exists():
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

# 3. อัปเดตไฟล์ point_by_point_reviewer_response_V2.tex (ถ้ามี) ให้รวมผล Node 3 (Apple M1 ARM64)
resp_tex = MS_DIR / "point_by_point_reviewer_response_V2.tex"
if resp_tex.exists():
    rt = resp_tex.read_text(encoding="utf-8")
    mac_resp_note = (
        " In addition, we executed cross-ISA verification on a third independent node "
        "(\\texttt{Pannapats-MacBook-Air.local}, Apple M1 8C/8T, 8.0~GB unified RAM, \\texttt{macOS-15.6.1-arm64}, "
        "audit CSV SHA-256: \\texttt{92b7772319ed09ca...}, concurrency CSV SHA-256: \\texttt{8f87f22044dda65d...}), "
        "confirming 100\\% multiset equivalence ($\\text{EX}=1.000$, max FP64 relative error $5.74\\times 10^{-16} < 10^{-9}$), "
        "reproducing the Tier~L4 Polars vs.\\ DuckDB window crossover ($66.41\\text{ ms}$ vs.\\ $70.75\\text{ ms}$), "
        "and achieving $22.29\\text{ QPS}$ at $C=4$ and a $3.19\\times$ P50 tail-latency reduction at $C=32$ ($523.75\\text{ ms}$ vs.\\ $1{,}672.49\\text{ ms}$)."
    )
    if "Pannapats-MacBook-Air.local" not in rt and "\\end{document}" in rt:
        rt = rt.replace("\\end{document}", "\n\\medskip\\noindent\\textbf{Cross-ISA Apple Silicon ARM64 Verification Update:}" + mac_resp_note + "\n\\end{document}")
        resp_tex.write_text(rt, encoding="utf-8")
        for _ in (1, 2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", resp_tex.name], cwd=str(MS_DIR), capture_output=True)

print("=" * 95)
print(f"[1] ไฟล์ .tex เดี่ยวพร้อม Copy & Paste : {MS_DIR / 'overleaf_ready_single_file.tex'}")
print(f"    SHA-256                          : {sha256_file(TARGET_TEX)}")
print(f"[2] ไฟล์ .zip พร้อม Upload ขึ้น Overleaf: {ROOT / 'Overleaf_V6_Project.zip'}")
print(f"    SHA-256                          : {sha256_file(ROOT / 'Overleaf_V6_Project.zip')}")
if (MS_DIR / "point_by_point_reviewer_response_V2.pdf").exists():
    print(f"[3] ไฟล์ Response Letter PDF          : {MS_DIR / 'point_by_point_reviewer_response_V2.pdf'}")
print("=" * 95)
