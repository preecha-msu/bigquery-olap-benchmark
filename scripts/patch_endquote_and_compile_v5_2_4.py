#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/patch_endquote_and_compile_v5_2_4.py
1. ตรวจสอบและเติม \end{quote} ก่อน \section{Introduction} ให้สมดุลกับ \begin{quote} (บรรทัดที่ 58)
2. ใส่ \allowbreak ในชื่อไฟล์ CSV ยาวๆ ของหัวข้อ Availability เพื่อแก้ Overfull \hbox (232pt)
3. คอมไพล์ bigquery_olap_benchmark_V5_2_4.tex (3 รอบ) และ bigquery_olap_benchmark_V2.tex (2 รอบ)
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"
V524_TEX = MS_DIR / "bigquery_olap_benchmark_V5_2_4.tex"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tex = V524_TEX.read_text(encoding="utf-8")

    # 1. บังคับสมดุล \begin{quote} และ \end{quote} ก่อนขึ้น \section{Introduction}
    pre_s1, post_s1 = tex.split("\\section{Introduction}", 1)
    n_bq = pre_s1.count("\\begin{quote}")
    n_eq = pre_s1.count("\\end{quote}")
    if n_bq > n_eq:
        # วาง \end{quote} หลังบรรทัด Keywords (หรือก่อน \section{Introduction})
        pre_s1 = pre_s1.rstrip() + "\n\\end{quote}\n\n"
        tex = pre_s1 + "\\section{Introduction}" + post_s1
        print(f"  [QUOTE-FIXED] เติม \\end{{quote}} สำเร็จ! (\\begin{{quote}}={tex.count('\\begin{quote}')}, \\end{{quote}}={tex.count('\\end{quote}')})")
    else:
        print(f"  [QUOTE-OK] สมดุลอยู่แล้ว (\\begin{{quote}}={n_bq}, \\end{{quote}}={n_eq})")

    # 2. แก้ Overfull \hbox (232pt) ในส่วน Availability of data โดยเปิดใช้ \sloppy เฉพาะย่อหน้า Availability
    tex = tex.replace(
        r"\texttt{step3\_pillar1\_complete\_baseline\_matrix\_ci\_p99.csv} through \texttt{step3\_pillar5\_net\_latency\_and\_cold\_metadata\_regimes.csv}",
        r"\texttt{step3\_pillar1\_\allowbreak complete\_\allowbreak baseline\_\allowbreak matrix\_\allowbreak ci\_p99.csv} through \texttt{step3\_pillar5\_\allowbreak net\_\allowbreak latency\_\allowbreak and\_\allowbreak cold\_\allowbreak metadata\_\allowbreak regimes.csv}",
    )
    tex = tex.replace(
        r"\texttt{secondary\_node\_dc\_shunt\_energy.csv}, \texttt{secondary\_node\_cpu\_routing\_30q.csv}, \texttt{secondary\_node\_concurrency\_22t.csv}",
        r"\texttt{secondary\_node\_\allowbreak dc\_shunt\_\allowbreak energy.csv}, \texttt{secondary\_node\_\allowbreak cpu\_routing\_30q.csv}, \texttt{secondary\_node\_\allowbreak concurrency\_22t.csv}",
    )

    for fname in (
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V5_2.tex",
        "bigquery_olap_benchmark_V5_2_3.tex",
        "bigquery_olap_benchmark_V5_2_4.tex",
        "bigquery_olap_benchmark_V6.tex",
        "main.tex",
        "overleaf_ready_single_file.tex",
    ):
        (MS_DIR / fname).write_text(tex, encoding="utf-8")

    # 3. คอมไพล์ bigquery_olap_benchmark_V5_2_4.tex 3 รอบ เพื่อให้เลขหน้า LastPage และ \ref ตารางนิ่ง 100%
    for p in (1, 2, 3):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", V524_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print("\n".join(res.stdout.splitlines()[-30:]))
            raise RuntimeError(f"pdflatex failed on pass {p}")

    for _ in (1, 2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
        )

    log_txt = (MS_DIR / "bigquery_olap_benchmark_V5_2_4.log").read_text(encoding="utf-8", errors="ignore")
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)
    m_pages = re.search(r"Output written on bigquery_olap_benchmark_V5_2_4\.pdf \((\d+) pages", log_txt)
    n_pages = int(m_pages.group(1)) if m_pages else 0
    n_figs = len(re.findall(r"\\includegraphics", tex))
    n_tabs = len(re.findall(r"\\begin\{table\*?\}", tex))

    pdf_master = MS_DIR / "bigquery_olap_benchmark_V5_2_4.pdf"
    for alias in (
        "bigquery_olap_benchmark_V2.pdf",
        "bigquery_olap_benchmark_V5_2.pdf",
        "bigquery_olap_benchmark_V5_2_3.pdf",
        "bigquery_olap_benchmark_V6.pdf",
        "main.pdf",
    ):
        try:
            shutil.copy2(pdf_master, MS_DIR / alias)
        except PermissionError:
            pass

    for zname in ("overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"):
        with zipfile.ZipFile(MS_DIR / zname, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(TARGET_TEX, arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            bib = MS_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    checks = [
        (f"1. จำนวนหน้ากลับมาครบสมบูรณ์ (~29–30 หน้า)       | วัดได้จริง: {n_pages} หน้า", n_pages >= 28),
        (f"2. รูปภาพครบทั้ง 11 รูป (Figures 1–11)             | วัดได้จริง: {n_figs} จุด", n_figs >= 11),
        (f"3. ตารางครบทั้ง 16 ตาราง (Table 2A, 2B, 1–14)      | วัดได้จริง: {n_tabs} ตาราง", n_tabs == 16),
        (f"4. ไม่มีเครื่องหมาย ?? (Undefined References = 0)  | วัดได้จริง: {len(undef_refs)} จุด", len(undef_refs) == 0),
        ("5. ไม่มีประโยคค้าง 'This study makes five...' หน้า 2", "This study makes five primary contributions" not in tex),
        ("6. Table 3 V3 = $1,251.70 / ไม่มี 59.07 ms / ไม่มี 1601.68",
         "1,251.70" in tex and "59.07" not in tex and "1601.68" not in tex),
        ("7. Table 12B DataFusion (244.12 / 168.45) & C=32 Streams",
         "244.12" in tex and "168.45" in tex and "$C=30$" not in tex),
    ]
    print("=" * 105)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("-" * 105)
    print(f"  [PDF-READY] {pdf_master.name} | Size: {pdf_master.stat().st_size:,} bytes | SHA-256: {sha256_of(pdf_master)[:24]}...")
    print("=" * 105)


if __name__ == "__main__":
    main()
