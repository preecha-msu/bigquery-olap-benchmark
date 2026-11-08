#!/usr/bin/env python3
"""
scripts/config_paths.py
กำหนด Path มาตรฐานของโปรเจกต์วิจัยบนเครื่อง Local (Windows Drive E:)
"""

from pathlib import Path

# Path หลักของงานวิจัย
PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")

# โฟลเดอร์ย่อยตามโครงสร้าง Repository
WORKLOADS_DIR = PROJECT_ROOT / "workloads"          # เก็บ 75 SQL queries (L1-L5) + 22 Official TPC-H
DATA_DIR = PROJECT_ROOT / "data"                    # เก็บข้อมูลจริง sf1 (1GB), sf10 (10GB), sf100 (100GB)
RESULTS_DIR = PROJECT_ROOT / "results"              # เก็บไฟล์ CSV Telemetry จริงและ SHA-256 logs
SCRIPTS_DIR = PROJECT_ROOT / "scripts"              # เก็บสคริปต์รันการทดลองและตรวจสถิติ
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"        # เก็บไฟล์ LaTeX Tables (.tex) และกราฟ 300 DPI

def ensure_project_dirs() -> None:
    """สร้างโฟลเดอร์ทั้งหมดอัตโนมัติหากยังไม่มี"""
    for d in [WORKLOADS_DIR, DATA_DIR, RESULTS_DIR, SCRIPTS_DIR, MANUSCRIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_project_dirs()
    print(f"[OK] Project Root ถูกตั้งค่าที่: {PROJECT_ROOT}")
    for sub in ["workloads", "data", "results", "scripts", "manuscript"]:
        p = PROJECT_ROOT / sub
        file_count = len(list(p.rglob("*"))) if p.exists() else 0
        print(f"  - {sub:<12}: {p} (พบ {file_count} รายการ)")