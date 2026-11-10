#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import subprocess
import hashlib
from pathlib import Path

root = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
scripts_dir = root / "scripts"
ms_dir = root / "manuscript"
tex_path = ms_dir / "bigquery_olap_benchmark_V2.tex"

# 1. ตรวจสอบและอัปเดต 17_unify_all_figures_style.py ให้ใช้ 10,000-Trial และ 0.9880 ทั้งใน Figure 1 และ Figure 2
fig_script = scripts_dir / "17_unify_all_figures_style.py"
if fig_script.exists():
    code = fig_script.read_text(encoding="utf-8")
    new_code = (
        code.replace("1,000-Trial", "10,000-Trial")
        .replace("1000-Trial", "10,000-Trial")
        .replace("0.8189", "0.9880")
    )
    if new_code != code:
        fig_script.write_text(new_code, encoding="utf-8")
        print("[FIG-SCRIPT-UPDATED] อัปเดต 10,000-Trial / 0.9880 ใน 17_unify_all_figures_style.py")
    subprocess.run(["python", str(fig_script)], cwd=str(root), check=False)

# 2. ตรวจสอบ bigquery_olap_benchmark_V2.tex อีกครั้งว่าไม่มีคำเก่าหลงเหลือและบันทึกทับ
tex = tex_path.read_text(encoding="utf-8")
tex_new = tex.replace("1,000-Trial", "10,000-Trial").replace("1,000-trial", "10,000-trial")
tex_new = re.sub(r"on the 5-query rapid micro-benchmark subset[^\n]+", "", tex_new)
if tex_new != tex:
    tex_path.write_text(tex_new, encoding="utf-8")
    print("[TEX-UPDATED] ล้างข้อความเก่าใน bigquery_olap_benchmark_V2.tex เรียบร้อยแล้ว")

# 3. คอมไพล์ bigquery_olap_benchmark_V2.tex -> bigquery_olap_benchmark_V2.pdf (2 รอบ)
for _ in (1, 2):
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=str(ms_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# 4. รายงานผล Verification Audit
tex_final = tex_path.read_text(encoding="utf-8")
pdf_path = ms_dir / "bigquery_olap_benchmark_V2.pdf"

print("=" * 85)
print("[FINAL AUDIT REPORT FOR REVIEWER ISSUE 3.1A & 3.1B]")
print("  * Count of '0.9880' in TeX      :", tex_final.count("0.9880"))
print("  * Count of '10,000' in TeX      :", tex_final.count("10,000"))
print("  * Count of '54.92' in TeX       :", tex_final.count("54.92"))
print("  * Count of '183.74' in TeX      :", tex_final.count("183.74"))
print("  * Residual '51.12' or '6.70'    :", tex_final.count("51.12") + tex_final.count("6.70"))
print("  * TeX SHA-256                   :", hashlib.sha256(tex_path.read_bytes()).hexdigest())
if pdf_path.exists():
    print("  * PDF SHA-256                   :", hashlib.sha256(pdf_path.read_bytes()).hexdigest())
print("=" * 85)
