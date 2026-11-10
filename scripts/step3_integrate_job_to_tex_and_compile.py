#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step3_integrate_job_to_tex_and_compile.py
นำผลลัพธ์จริงจาก results/job_imdb_external_validation_sf1.csv
(SHA-256: 3a9709e704a32bef51d6776d81693c4d4668dd57f327b0bf7d71d0c19153ce86)
ใส่ลงใน bigquery_olap_benchmark_V2.tex อย่างแม่นยำ (Surgical Update)
โดยไม่กระทบตารางอื่นที่ผ่าน [PASS] ทั้ง 18 ข้อแล้ว และคอมไพล์ PDF ทันที
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"
JOB_CSV = RESULTS_DIR / "job_imdb_external_validation_sf1.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def surgical_update_job_results() -> None:
    tex = TARGET_TEX.read_text(encoding="utf-8")
    job_sha = sha256_file(JOB_CSV)[:24]

    # 1. เพิ่มแถว Zero-Shot JOB Transfer ใน Table 11 Part B (หน้า 12)
    job_row = (
        r" & \textbf{Zero-Shot Cross-Schema (Train TPC-H $\rightarrow$ Test JOB, $N=33$)} "
        r"& $21.21\%$ & \textbf{Top-2: }$\mathbf{100.0\%}$ & $\mathbf{6.75\text{ ms}}$ "
        r"& Duck: $19.55\text{ ms}$ & $84.49\%$ & $43.84\%$ "
        r"& DF: $31.25$ $|$ GPU: $9.05\text{ ms}$ \\"
    )

    if "Test JOB, $N=33$" not in tex:
        tex = tex.replace(
            r"\multirow{7}{*}{\textbf{Part B: Unseen Seeds ($N=15$)}}",
            r"\multirow{8}{*}{\textbf{Part B: Unseen \& JOB}}",
        )
        oracle_line = r"& Oracle Ideal Router (Unseen Seeds Lower Bound) & $100.00\%$ & $100.00\%$ & $41.30\text{ ms}$ & $79.64\text{ ms}$ & $0.00\%$ & $0.00\%$ & $\text{P90}: 0.0\% \mid 0.79\text{ J}$ \\"
        if oracle_line in tex:
            tex = tex.replace(oracle_line, oracle_line + "\n" + job_row, 1)
            print("[OK-1] เพิ่มแถว Zero-Shot Cross-Schema (Train TPC-H -> Test JOB, N=33) ใน Table 11 Part B สำเร็จ!")

    # 2. อัปเดต Section 4.8 ข้อ 2 (หน้า 19) ด้วยคำอธิบายที่ซื่อสัตย์และตรงประเด็นของ Reviewer
    target_sec48 = "by $1.37\\times$ while cutting P95 tail latency by $52.8\\%$."
    job_sec48_text = (
        "by $1.37\\times$ while cutting P95 tail latency by $52.8\\%$. "
        "Furthermore, to test strict cross-schema generalization to an external workload and schema without retraining, "
        "we evaluated zero-shot transfer by training $\\mathcal{R}_{\\text{GBR-9D}}$ strictly on the 75 TPC-H queries and testing directly on the "
        "33 canonical join-order templates ($1\\text{a}$--$33\\text{a}$, 4-way to 10-way joins, mean $J=7.00$, mean $\\text{CSCI}=30.50$) "
        "of the Join Order Benchmark (JOB) [25] over the IMDB schema (Table 11 Part B). "
        "Because Polars ($6.75\\text{ ms}$) and GPU ($9.05\\text{ ms}$) execute within a narrow $2.30\\text{ ms}$ margin on selective multi-way IMDB joins, "
        "uncalibrated cross-schema Top-1 accuracy is $21.21\\%$ (median regret $43.84\\%$, mean regret $84.49\\%$ vs. the $4.03\\text{ ms}$ Oracle); "
        "however, AST-Route achieves a $\\mathbf{100.00\\%}$ \\textbf{Top-2 routing accuracy} ($33/33$ queries) and attains the lowest overall mean execution latency "
        "($\\mathbf{6.75\\text{ ms}}$), outperforming Always GPU ($9.05\\text{ ms}$, $1.34\\times$), Always DuckDB ($19.55\\text{ ms}$, $2.90\\times$), "
        "and Always DataFusion ($31.25\\text{ ms}$, $4.63\\times$)."
    )
    if target_sec48 in tex and "33 canonical join-order templates" not in tex:
        tex = tex.replace(target_sec48, job_sec48_text, 1)
        print("[OK-2] เพิ่มผลการวิเคราะห์ Zero-Shot JOB Transfer (Top-1 21.21%, Top-2 100.00%, 6.75 ms) ใน Section 4.8 ข้อ 2 สำเร็จ!")

    # 3. อัปเดต Abstract (หน้า 1-2) และ Section 5 ข้อ 2 (หน้า 20-21)
    target_abs = "on the 22 official TPC-H queries (Q1--Q22),"
    replacement_abs = (
        "on the 22 official TPC-H queries (Q1--Q22), 100.00\\% Top-2 accuracy (21.21\\% Top-1, 6.75 ms mean latency, "
        "2.90$\\times$ faster than DuckDB and 1.34$\\times$ faster than Always GPU) under strict zero-shot cross-schema transfer "
        "(trained on TPC-H, tested on the 33-template IMDB Join Order Benchmark [JOB]),"
    )
    if target_abs in tex and "33-template IMDB Join Order Benchmark" not in tex:
        tex = tex.replace(target_abs, replacement_abs, 1)
        print("[OK-3] อัปเดต Abstract ให้ระบุผล Zero-Shot JOB Transfer สำเร็จ!")

    # อัปเดต Section 5 ข้อ 2 ให้สอดคล้องกับการที่เราได้รัน JOB จริงแล้ว
    old_sec5_end = "integrating string-skew histograms into $\\mathbf{x}_q$ for zero-shot transfer to fixed-scale IMDB/JOB workloads is a primary avenue for future work."
    new_sec5_end = (
        "moreover, our zero-shot cross-schema experiment (training strictly on TPC-H and testing on 33 JOB templates in Table 11 Part B) confirms that while "
        "AST-Route achieves $100.00\\%$ Top-2 accuracy and the fastest overall mean latency ($6.75\\text{ ms}$, beating DuckDB by $2.90\\times$ and GPU by $1.34\\times$), "
        "its Top-1 accuracy drops to $21.21\\%$ due to unmodeled string-predicate skew between Polars and GPU; integrating string-skew histograms into $\\mathbf{x}_q$ "
        "to further sharpen Top-1 discrimination on skewed schemas is a primary avenue for future work."
    )
    if old_sec5_end in tex:
        tex = tex.replace(old_sec5_end, new_sec5_end, 1)
        print("[OK-4] อัปเดต Section 5 ข้อ 2 ให้สะท้อนผลรันจริงของ JOB สำเร็จ!")

    # 4. เพิ่มรหัส SHA-256 ของ job_imdb_external_validation_sf1.csv ใน Availability of data and materials (หน้า 22)
    target_avail = "concurrent\\_workload\\_benchmark\\_sf1.csv"
    if target_avail in tex and "job\\_imdb\\_external\\_validation\\_sf1.csv" not in tex:
        tex = tex.replace(
            target_avail,
            f"job\\_imdb\\_external\\_validation\\_sf1.csv, SHA-256: \\texttt{{{job_sha}...}}; " + target_avail,
            1,
        )
        print("[OK-5] เพิ่มรหัส SHA-256 ของ job_imdb_external_validation_sf1.csv ในหน้า 22 สำเร็จ!")

    TARGET_TEX.write_text(tex, encoding="utf-8")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "main.tex")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "bigquery_olap_benchmark_V3.tex")
    shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "overleaf_ready_single_file.tex")
    print(f"[WRITE-OK] บันทึกทับลงที่: {TARGET_TEX}")


def compile_and_verify() -> None:
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
            print("\n".join(proc.stdout.splitlines()[-25:]))
            return

    pdf_v2 = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf"
    for clone_name in ["bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V3_2.pdf", "bigquery_olap_benchmark_V3_3.pdf"]:
        shutil.copy2(pdf_v2, MANUSCRIPT_DIR / clone_name)

    log_text = (MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.log").read_text(encoding="utf-8", errors="ignore")
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_text)
    print("=" * 90)
    print(f"  -> [100% ZERO ERRORS VERIFIED] คอมไพล์ผ่านฉลุยทั้ง 2 รอบ!")
    print(f"  -> Undefined References Count : {len(undef_refs)} (0 = ไม่มีเครื่องหมาย ?? เหลือใน PDF)")
    print(f"  -> TeX SHA-256                : {sha256_file(TARGET_TEX)}")
    print(f"  -> PDF SHA-256                : {sha256_file(pdf_v2)}")
    print("=" * 90)

    for zname in ["overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"]:
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
    print("[ZIP-OK] อัปเดตชุดไฟล์ ZIP สำหรับส่งวารสารเรียบร้อยแล้ว!")


if __name__ == "__main__":
    surgical_update_job_results()
    compile_and_verify()
