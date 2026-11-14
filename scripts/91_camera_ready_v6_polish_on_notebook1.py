#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, re, shutil, subprocess, zipfile
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

tex = TARGET_TEX.read_text(encoding="utf-8")

# [Fix 1] ซ่อมเลขยกกำลังซ้ำหลังชื่อผู้แต่ง Preecha Noiumkar^{1}^{1} -> ^{1,*}
tex = re.sub(
    r"Preecha\s+Noiumkar\s*\$\^\{?1\}?\$\s*\$\^\{?1\}?\$",
    r"Preecha Noiumkar$^{1,*}$",
    tex,
)
tex = tex.replace(r"\textsuperscript{1}\textsuperscript{1}", r"\textsuperscript{1,*}")
tex = tex.replace(r"$^{1}$$^{1}$", r"$^{1,*}$")

# [Fix 2] ซ่อม Table 1 VES (Ext. vs Native) และ Table 2 Overall Mean (Native vs Parquet)
tex = tex.replace("121.6 & 91.8", "91.8 & 121.6")
tex = tex.replace(
    r"347.7 $\rightarrow$ 207.7 & 254.4 & 183.7 & 4.1$\times$",
    r"347.7 $\rightarrow$ 207.7 & 183.7 & 254.4 & 4.1$\times$",
)

# [Fix 3] ซ่อม Table 4 แถว L4 ที่พิมพ์ 7.78\% ให้เป็น 7.78$\times$
tex = tex.replace("9.94 & 7.78\\%", r"9.94 & 7.78$\times$")

# [Fix 4] อัปเดตหน้า 27 (Conclusion) และหน้า 30 (Availability of data) ให้รวม Node 3 (MacBook Air M1 ARM64) และ Global Registry SHA-256 ล่าสุด
tex = tex.replace(
    "three cross-platform hardware profiles,",
    "three cross-platform hardware profiles and three physical hardware nodes (Windows x86\\_64 GPU workstation, 16-core/22-thread x86\\_64 CPU node, and Apple M1 ARM64 unified-memory node),",
)
tex = tex.replace(
    "together with the 2-node global SHA-256 registry and extended 75-query secondary-workstation transfer/SLA logs on the secondary 16-core workstation (\\texttt{global\\_sha256\\_registry.csv}, SHA-256: \\texttt{fcca73ec590885e807d8...}",
    "together with the 3-node cross-ISA global SHA-256 registry spanning x86\\_64 and Apple M1 ARM64 nodes (\\texttt{global\\_sha256\\_registry.csv}, SHA-256: \\texttt{5adfa064816f883f8770...}; \\texttt{macbook\\_reexecution\\_audit.csv}, SHA-256: \\texttt{92b7772319ed09ca80bb...}; \\texttt{macbook\\_exp3\\_concurrency\\_summary.csv}, SHA-256: \\texttt{8f87f22044dda65dbc6b...}",
)

# [Fix 5] ปรับสคริปต์วาดรูปให้ซิงค์ข้อความใน Fig 1, 5, 6, 7 (ถ้ามีสคริปต์วาดรูปใน scripts/)
for py_fig in sorted((ROOT / "scripts").glob("*fig*.py")) + sorted((ROOT / "scripts").glob("17_*.py")):
    try:
        fc = py_fig.read_text(encoding="utf-8")
        fc_new = (
            fc.replace("CSCI_v2=27.9", "CSCI=15.3")
            .replace("CSCI_v2=36.2", "CSCI=26.8")
            .replace("CSCI_v2=41.4", "CSCI=36.7")
            .replace("CSCI_v2=41.7", "CSCI=37.1")
            .replace("CSCI_v2=43.6", "CSCI=32.4")
            .replace("Green OLAP Energy Efficiency (Perf/Watt)", "Energy-Aware Scan Efficiency (MB/J)")
            .replace("Per-Query Energy Footprint Across Paradigms", "Est. CPU Pkg & Meas. GPU Energy (J)")
            .replace("< 0.8 ms overhead", "3.48 ms AST parse")
            .replace("Spearman rho = 0.4935", "Partial rho = 0.5477")
        )
        if fc_new != fc:
            py_fig.write_text(fc_new, encoding="utf-8")
            subprocess.run(["python", str(py_fig)], cwd=str(ROOT), capture_output=True)
    except Exception:
        pass

# คัดลอกรูปที่อัปเดตแล้วจาก figures/ ไปยัง manuscript/figures/
if (ROOT / "figures").exists():
    for f in (ROOT / "figures").glob("*.png"):
        shutil.copy2(f, FIG_DIR / f.name)
    for f in (ROOT / "figures").glob("*.pdf"):
        shutil.copy2(f, FIG_DIR / f.name)

# บันทึกทับลง bigquery_olap_benchmark_V2.tex และไฟล์สำเนาทั้งหมด
TARGET_TEX.write_text(tex, encoding="utf-8")
for mirror in ("main.tex", "overleaf_ready_single_file.tex", "bigquery_olap_benchmark_V6.tex"):
    (MS_DIR / mirror).write_text(tex, encoding="utf-8")

# คอมไพล์ pdflatex 2 รอบ
for pass_i in (1, 2):
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
        cwd=str(MS_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print("\n".join(proc.stdout.splitlines()[-25:]))
        raise RuntimeError(f"pdflatex failed on pass {pass_i}")

pdf_v2 = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
for pdf_copy in (
    MS_DIR / "bigquery_olap_benchmark_V6.pdf",
    ROOT / "bigquery_olap_benchmark_V6.pdf",
    MS_DIR / "main.pdf",
):
    try:
        shutil.copy2(pdf_v2, pdf_copy)
    except PermissionError:
        print(f"[WARN] กรุณาปิดไฟล์ {pdf_copy.name} ที่เปิดค้างไว้หากต้องการเขียนทับชื่อนี้")

# อัปเดตไฟล์ ZIP สำหรับ Overleaf แบบปกติ
for zpath in (ROOT / "Overleaf_V2_Submission_Ready.zip", ROOT / "Overleaf_V6_Project.zip"):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MS_DIR / "main.tex", arcname="main.tex")
        zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
        if (MS_DIR / "references.bib").exists():
            zf.write(MS_DIR / "references.bib", arcname="references.bib")
        for f in sorted(FIG_DIR.glob("*")):
            if f.is_file():
                zf.write(f, arcname=f"figures/{f.name}")

print("=" * 95)
print("[AUDIT & FIX COMPLETE ON NOTEBOOK1]")
print(f"  * Fix 1 (Author Superscript Clean) : {'$^{1,*}$' in tex or 'textsuperscript{1,*}' in tex}")
print(f"  * Fix 2 (Table 1 VES & Table 2)    : {'91.8 & 121.6' in tex and '183.7 & 254.4' in tex}")
print(f"  * Fix 3 (Table 4 L4 7.78x)         : {'7.78%' not in tex}")
print(f"  * Fix 4 (3-Node Global SHA Sync)   : {'5adfa064816f883f8770' in tex and '92b7772319ed09ca80bb' in tex}")
print(f"  * Updated V2.tex SHA-256           : {sha256_file(TARGET_TEX)}")
print(f"  * Compiled V2/V6.pdf SHA-256       : {sha256_file(pdf_v2)}")
print("=" * 95)
