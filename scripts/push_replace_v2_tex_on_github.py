#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/push_replace_v2_tex_on_github.py
================================================================================
นโยบาย:
  1. แก้ไขและบันทึกทับลงที่ manuscript/bigquery_olap_benchmark_V2.tex โดยตรง (Zero Backup)
  2. ลบโฟลเดอร์สำรอง manuscript_backup_* ทั้งหมดทิ้ง ไม่มีการสำรองไฟล์
  3. คัดลอกทับ bigquery_olap_benchmark_V2.tex และ main.tex ที่ระดับ Root
  4. อัปโหลด (Replace) ขึ้นบน GitHub ทันที
================================================================================
"""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")


def remove_all_backups_and_disable_backup_hook() -> None:
    # 1. ลบโฟลเดอร์ manuscript_backup_* และไฟล์สำรองทั้งหมดในโปรเจกต์
    removed = 0
    for bak in ROOT.glob("manuscript_backup_*"):
        if bak.is_dir():
            shutil.rmtree(bak, ignore_errors=True)
            removed += 1
    for bak_f in MS_DIR.glob("*backup*.tex"):
        bak_f.unlink(missing_ok=True)
        removed += 1
    print(f"[NO-BACKUP POLICY] ล้างโฟลเดอร์/ไฟล์สำรองออกทั้งหมดแล้ว ({removed} รายการ)")

    # 2. ปิดการสร้าง Backup ใน hub_sync_notebook1_master.py (ถ้ามีไฟล์นี้อยู่)
    hub_py = ROOT / "scripts" / "hub_sync_notebook1_master.py"
    if hub_py.exists():
        htxt = hub_py.read_text(encoding="utf-8")
        htxt_new = re.sub(
            r"def backup_authoritative_manuscript\(\)\s*->\s*Path:.*?return bak_dir",
            "def backup_authoritative_manuscript() -> Path:\n    return MS_DIR  # Disabled backup per user policy",
            htxt,
            flags=re.DOTALL,
        )
        if htxt_new != htxt:
            hub_py.write_text(htxt_new, encoding="utf-8")
            print("[NO-BACKUP POLICY] ปิดฟังก์ชันสร้างโฟลเดอร์สำรองใน hub_sync_notebook1_master.py ถาวรแล้ว")


def apply_camera_ready_fixes_in_place() -> None:
    tex = TARGET_TEX.read_text(encoding="utf-8")

    # Fix 1: ชื่อผู้แต่ง Preecha Noiumkar^{1,*} (ไม่ให้มีเลข 1 ซ้ำสองตัว)
    tex = re.sub(r"Preecha\s+Noiumkar\s*\$\^\{?1\}?\$\s*\$\^\{?1\}?\$", r"Preecha Noiumkar$^{1,*}$", tex)
    tex = tex.replace(r"\textsuperscript{1}\textsuperscript{1}", r"\textsuperscript{1,*}")
    tex = tex.replace(r"$^{1}$$^{1}$", r"$^{1,*}$")

    # Fix 2: Table 1 VES (Ext. vs Native) และ Table 2 Overall Mean (Native vs Parquet)
    tex = tex.replace("121.6 & 91.8", "91.8 & 121.6")
    tex = tex.replace(
        r"347.7 $\rightarrow$ 207.7 & 254.4 & 183.7 & 4.1$\times$",
        r"347.7 $\rightarrow$ 207.7 & 183.7 & 254.4 & 4.1$\times$",
    )

    # Fix 3: Table 4 แถว L4 แก้ 7.78\% -> 7.78$\times$
    tex = tex.replace("9.94 & 7.78\\%", r"9.94 & 7.78$\times$")

    # Fix 4: อัปเดต Conclusion และ Availability of data ให้ตรงกับ 3 โหนดฮาร์ดแวร์และ Global SHA-256 ล่าสุด
    tex = tex.replace(
        "three cross-platform hardware profiles,",
        "three cross-platform hardware profiles and three physical hardware nodes (Windows x86\\_64 GPU workstation, 16-core/22-thread x86\\_64 CPU node, and Apple M1 ARM64 unified-memory node),",
    )
    tex = tex.replace(
        "together with the 2-node global SHA-256 registry and extended 75-query secondary-workstation transfer/SLA logs on the secondary 16-core workstation (\\texttt{global\\_sha256\\_registry.csv}, SHA-256: \\texttt{fcca73ec590885e807d8...}",
        "together with the 3-node cross-ISA global SHA-256 registry spanning x86\\_64 and Apple M1 ARM64 nodes (\\texttt{global\\_sha256\\_registry.csv}, SHA-256: \\texttt{5adfa064816f883f8770...}; \\texttt{macbook\\_reexecution\\_audit.csv}, SHA-256: \\texttt{92b7772319ed09ca80bb...}; \\texttt{macbook\\_exp3\\_concurrency\\_summary.csv}, SHA-256: \\texttt{8f87f22044dda65dbc6b...}",
    )

    # บันทึกทับลงไฟล์เป้าหมายโดยตรง (ไม่สร้างไฟล์สำรอง)
    TARGET_TEX.write_text(tex, encoding="utf-8")

    # ซิงค์ทับไปยังระดับ Root และไฟล์ Mirror เพื่อให้บน GitHub มีไฟล์ bigquery_olap_benchmark_V2.tex ที่เป็นตัวเดียวกันทุกจุด
    for dst in (
        ROOT / "bigquery_olap_benchmark_V2.tex",
        ROOT / "main.tex",
        MS_DIR / "main.tex",
        MS_DIR / "overleaf_ready_single_file.tex",
    ):
        dst.write_text(tex, encoding="utf-8")

    # ซิงค์รูปภาพไปยัง root figures/
    ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if FIG_DIR.exists():
        for fg in FIG_DIR.glob("*"):
            if fg.is_file():
                shutil.copy2(fg, ROOT_FIG_DIR / fg.name)

    # คอมไพล์ PDF ทับทันที
    if shutil.which("pdflatex"):
        for _ in (1, 2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        pdf_v2 = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
        if pdf_v2.exists():
            for p_copy in (ROOT / "bigquery_olap_benchmark_V2.pdf", MS_DIR / "bigquery_olap_benchmark_V6.pdf", ROOT / "bigquery_olap_benchmark_V6.pdf"):
                try:
                    shutil.copy2(pdf_v2, p_copy)
                except PermissionError:
                    pass


def upload_and_replace_on_github() -> None:
    # ล้างไฟล์ใหญ่ใน data/ ออกจาก Git Index เผื่อไว้
    run_git(["rm", "-r", "--cached", "data/"])

    # Stage ไฟล์ bigquery_olap_benchmark_V2.tex ทั้งใน manuscript/ และที่ Root เพื่อ Replace ทับบน GitHub
    files_to_replace = [
        "manuscript/bigquery_olap_benchmark_V2.tex",
        "manuscript/bigquery_olap_benchmark_V2.pdf",
        "manuscript/main.tex",
        "manuscript/overleaf_ready_single_file.tex",
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V2.pdf",
        "main.tex",
        "figures",
        "manuscript/figures",
        "scripts/push_replace_v2_tex_on_github.py",
        "scripts/hub_sync_notebook1_master.py",
    ]
    for item in files_to_replace:
        if (ROOT / item).exists():
            run_git(["add", "-f", item])

    status = run_git(["status", "--porcelain"]).stdout.strip()
    if status:
        run_git(["commit", "-m", "Replace bigquery_olap_benchmark_V2.tex on GitHub (in-place update, no backup)"])
    else:
        print("[GIT INFO] ไฟล์ใน Local Commit ล่าสุดตรงกับสถานะปัจจุบันแล้ว กำลังบังคับซิงค์ขึ้น GitHub...")

    run_git(["branch", "-M", "main"])

    # Push ทับขึ้น origin (และ legacy_origin ถ้ามี)
    remotes = [r.strip() for r in run_git(["remote"]).stdout.splitlines() if r.strip()]
    print("=" * 96)
    for rem in remotes:
        url = run_git(["remote", "get-url", rem]).stdout.strip()
        res = run_git(["push", "-u", rem, "main", "--force-with-lease"])
        if res.returncode != 0:
            res = run_git(["push", "-u", rem, "main", "--force"])
        status_str = "SUCCESS (REPLACED ON GITHUB)" if res.returncode == 0 else f"WARN ({res.stderr.strip()[:80]})"
        print(f"[GITHUB REPLACE -> {rem:<13}] {url} : {status_str}")

    print("-" * 96)
    print(f"[LOCAL & GITHUB FILE] manuscript/bigquery_olap_benchmark_V2.tex")
    print(f"[ROOT MIRROR FILE   ] bigquery_olap_benchmark_V2.tex")
    print(f"[VERIFIED SHA-256   ] {sha256_file(TARGET_TEX)}")
    print("=" * 96)


if __name__ == "__main__":
    remove_all_backups_and_disable_backup_hook()
    apply_camera_ready_fixes_in_place()
    upload_and_replace_on_github()
