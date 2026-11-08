#!/usr/bin/env python3
"""
scripts/00_audit_existing_workspace.py
ตรวจสอบไฟล์ CSV เดิมใน E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark
เทียบกับค่าที่ปรากฏในบทความ PDF เพื่อยืนยันจุดที่ต้องอัปเดตก่อนรัน SF1, SF10, SF100
"""

import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")

EXPECTED_CSV_FILES = [
    "ast_complexity_summary.csv",
    "raw_benchmark_sf1.csv",
    "raw_bigquery_cloud_sf1.csv",
    "raw_gpu_cuda_sf1.csv",
    "raw_multi_software_sf1.csv",
    "six_software_comparison_sf1.csv",
]

def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:24]

def audit_workspace() -> None:
    print(f"=== ตรวจสอบโฟลเดอร์วิจัย: {PROJECT_ROOT} ===\n")
    if not PROJECT_ROOT.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์ {PROJECT_ROOT} กรุณาตรวจสอบชื่อ Path")
        return

    # 1. ค้นหาไฟล์ .sql ใน workloads/
    sql_files = list(PROJECT_ROOT.rglob("*.sql"))
    print(f"[1] จำนวนไฟล์ SQL Workload ที่พบทั้งหมด: {len(sql_files)} ไฟล์")

    # 2. ค้นหาไฟล์ CSV ผลลัพธ์การทดลองเดิม
    print("\n[2] ตรวจสอบไฟล์ CSV ข้อมูลดิบและ SHA-256:")
    for fname in EXPECTED_CSV_FILES:
        matches = list(PROJECT_ROOT.rglob(fname))
        if matches:
            fpath = matches[0]
            df = pd.read_csv(fpath)
            print(f"  [FOUND] {fname:<32} | Rows: {len(df):<5} | Cols: {len(df.columns):<3} | SHA-256: {sha256_short(fpath)}...")
            print(f"          Columns: {list(df.columns)}")
        else:
            print(f"  [MISSING] ไม่พบไฟล์ {fname}")

if __name__ == "__main__":
    audit_workspace()