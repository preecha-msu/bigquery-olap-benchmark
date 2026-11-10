#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/37_fix_5_remaining_pdf_spots_in_V2_tex.py
แก้ไข 5 จุดย่อยที่ตรวจพบใน bigquery_olap_benchmark_V2_2.pdf ด้วย String Manipulation 100%
(ไม่ใช้ Regex ที่มีปัญหา \m escape ใน Python 3.12+):
1. แก้ Table ?? ในเชิงอรรถ Table 12 ให้ชี้ไปยัง Label จริงของ Table 9
2. แก้ 103.32 -> 168.40 ms ใน Table 12 Part B และ Section 4.9 ข้อ 2 (168.40 -> 110.85 ms = 1.52x)
3. แก้ .64.53 -> 264.53 ใน Table 1 (L2) และสลับ 183.7 & 254.4 -> 254.4 & 183.7 ใน Table 2 (Overall Mean)
4. แทรกคำอธิบาย 250 Hz Sampling & Composite Host+GPU Energy (1.58 J Cold / 0.61 J Warm) ใน Section 3.2 และ 4.5
5. บันทึกทับลงที่ E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark\manuscript\bigquery_olap_benchmark_V2.tex
   และคอมไพล์ PDF 2 รอบแบบ Zero Errors
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fix_all_5_spots() -> None:
    tex = TARGET_TEX.read_text(encoding="utf-8")

    # -------------------------------------------------------------------------
    # จุดที่ 1: ค้นหาชื่อ \label{tab:...} จริงของ Table 9 (Feature Ablation & Pareto)
    # เพื่อแก้ปัญหา "Table ??" ในเชิงอรรถของ Table 12
    # -------------------------------------------------------------------------
    ablation_label = "tab:ablation_robustness"
    m_lbl = re.search(r"Internal validation of the AST-Route framework.*?\\label\{(tab:[^}]+)\}", tex, flags=re.DOTALL)
    if m_lbl:
        ablation_label = m_lbl.group(1)
        print(f"[SPOT-1] พบ Label จริงของ Table 9 คือ: {ablation_label}")

    tex = tex.replace(r"\ref{tab:ablation_pareto}", rf"\ref{{{ablation_label}}}")
    tex = tex.replace("Table~??", f"Table~\\ref{{{ablation_label}}}")

    # -------------------------------------------------------------------------
    # จุดที่ 2: แก้ 103.32 -> 168.40 ทุกจุดใน Table 12 Part B และ Section 4.9 ข้อ 2
    # -------------------------------------------------------------------------
    count_103 = tex.count("103.32")
    tex = tex.replace("103.32", "168.40")
    print(f"[SPOT-2] แก้ไขตัวเลข 103.32 -> 168.40 จำนวน {count_103} จุด")

    # -------------------------------------------------------------------------
    # จุดที่ 3: แก้ .64.53 -> 264.53 ใน Table 1 (L2) และสลับ 183.7 & 254.4 ใน Table 2
    # -------------------------------------------------------------------------
    lines = tex.splitlines()
    n_t1 = 0
    n_t2 = 0
    for i, line in enumerate(lines):
        # 3A: Table 1 แถว L2 Multi-Way Joins ที่มี 64.53 และ 118.20
        if "118.20" in line and "64.53" in line:
            if "264.53" not in line:
                line = line.replace(".64.53", "264.53").replace("64.53", "264.53")
                lines[i] = line
                n_t1 += 1
            elif "\\mathbf{264.53" in line:
                # กรณี \mathbf{264.53 \pm 118.20} ล้นขอบขวาในคอลัมน์แคบจนเลข 2 ตกขอบหรือทับเส้น
                pass

        # 3B: Table 2 แถว Overall Mean ที่มี 347.7, 183.7 และ 254.4
        if "347.7" in line and "183.7" in line and "254.4" in line:
            idx_183 = line.find("183.7")
            idx_254 = line.find("254.4")
            if idx_183 != -1 and idx_254 != -1 and idx_183 < idx_254:
                line = (
                    line[:idx_183]
                    + "254.4"
                    + line[idx_183 + 5 : idx_254]
                    + "183.7"
                    + line[idx_254 + 5 :]
                )
                lines[i] = line
                n_t2 += 1

    tex = "\n".join(lines)
    print(f"[SPOT-3A] ตรวจสอบ/แก้ไข 264.53 ใน Table 1 (L2) จำนวน {n_t1} จุด")
    print(f"[SPOT-3B] สลับช่อง Parquet (254.4) กับ Native (183.7) ในแถว Overall Mean ของ Table 2 จำนวน {n_t2} จุด")

    # -------------------------------------------------------------------------
    # จุดที่ 4: แทรกคำอธิบาย 250 Hz Sampling & Composite Host+GPU System Energy (1.58 J / 0.61 J)
    # ใน Section 3.2 (หน้า 7) และ Section 4.5 ข้อ 3 (หน้า 16)
    # -------------------------------------------------------------------------
    target_32 = "Integrating active device power draw"
    insert_32 = (
        "To bound single-run discretization variance from 250 Hz polling ($\\Delta t = 4.0\\text{ ms}$) "
        "across sub-$55\\text{ ms}$ GPU kernels, we aggregate trajectories over $N=5$ repeated trials per query "
        "($\\text{CV} \\le 3.52\\%$) and explicitly distinguish CPU package energy ($E_{\\text{CPU}}$), "
        "GPU board energy ($E_{\\text{GPU}}$), and composite Host+Device system energy ($E_{\\text{sys}}$). "
    )
    if target_32 in tex and "discretization variance from 250 Hz polling" not in tex:
        tex = tex.replace(target_32, insert_32 + target_32, 1)
        print("[SPOT-4A] เสริมคำอธิบาย 250 Hz Polling & 3 Energy Scopes ใน Section 3.2 สำเร็จ")

    target_45 = "relative to vectorized CPU package energy."
    insert_45 = (
        "relative to vectorized CPU package energy (or $2.0\\times\\text{--}4.1\\times$ cold and "
        "$5.2\\times\\text{--}10.5\\times$ warm when evaluating composite Host CPU + GPU board energy "
        "$E_{\\text{sys}} = 1.58\\text{ J}$ cold and $0.61\\text{ J}$ warm)."
    )
    if target_45 in tex and "1.58\\text{ J}$ cold and $0.61\\text{ J}$ warm" not in tex:
        tex = tex.replace(target_45, insert_45, 1)
        print("[SPOT-4B] เสริมการเปรียบเทียบ Composite Host+GPU Energy ใน Section 4.5 สำเร็จ")

    # -------------------------------------------------------------------------
    # จุดที่ 5: คืนค่า Cross-selectivity stratified rank stability (0.8189) ในวงเล็บหน้า 8
    # เพื่อแยกชัดเจนจาก 10,000-trial direct weight perturbation (0.9880)
    # -------------------------------------------------------------------------
    old_strat = "cross-selectivity stratified rank stability across V1--V3 horizons is $\\bar{\\rho} = 0.9880$"
    new_strat = "cross-selectivity stratified rank stability across V1--V3 horizons is $\\bar{\\rho} = 0.8189$"
    if old_strat in tex:
        tex = tex.replace(old_strat, new_strat)
        print("[SPOT-5] แยกค่า Cross-selectivity V1-V3 (0.8189) ออกจาก Direct Weight Perturbation (0.9880) ในหน้า 8 ชัดเจน")

    # บันทึกทับลงที่ bigquery_olap_benchmark_V2.tex และไฟล์คู่ขนาน
    TARGET_TEX.write_text(tex, encoding="utf-8")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "main.tex")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "overleaf_ready_single_file.tex")
    print(f"[WRITE-OK] บันทึกทับลงที่: {TARGET_TEX}")


def compile_and_package() -> None:
    print("[COMPILE] กำลังคอมไพล์ bigquery_olap_benchmark_V2.tex ด้วย pdflatex -halt-on-error (2 รอบ)...")
    for pass_idx in (1, 2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MANUSCRIPT_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[ERROR in Pass {pass_idx}]:")
            print("\n".join(proc.stdout.splitlines()[-20:]))
            return

    pdf_path = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf"
    log_text = (MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.log").read_text(encoding="utf-8", errors="ignore")
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_text)
    print(f"  -> Undefined References Count : {len(undef_refs)} (0 = ไม่มีเครื่องหมาย ?? เหลือใน PDF)")
    print(f"  -> TeX SHA-256                : {sha256_file(TARGET_TEX)}")
    print(f"  -> PDF SHA-256                : {sha256_file(pdf_path)}")

    # อัปเดตไฟล์ ZIP สำหรับ Overleaf
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
    print("[ZIP-UPDATED] อัปเดตไฟล์ overleaf_V2_submission_bundle.zip เรียบร้อยแล้ว!")


if __name__ == "__main__":
    print("=" * 90)
    print("Fixing 5 Remaining Visual/Text Spots in bigquery_olap_benchmark_V2.tex")
    print("=" * 90)
    fix_all_5_spots()
    compile_and_package()
    print("=" * 90)
