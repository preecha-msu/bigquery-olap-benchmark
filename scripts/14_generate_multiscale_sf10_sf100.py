#!/usr/bin/env python3
r"""
scripts/14_generate_multiscale_sf10_sf100.py
สร้างข้อมูลจริง TPC-H ที่ระดับ SF=10 (10 GB) และ SF=100 (100 GB)
พร้อมจัดเก็บเป็น ZSTD Parquet และ DuckDB Native ตลอดจนบันทึก SHA-256 Checksum
"""

import hashlib
import time
from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TPCH_TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier"
]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provision_scale_factor(sf: int, threads: int = 8, memory_limit_gb: int = 28) -> dict:
    sf_dir = DATA_DIR / f"sf{sf}"
    parquet_dir = sf_dir / "parquet_zstd"
    sf_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    duckdb_path = sf_dir / f"tpch_sf{sf}.duckdb"

    print(f"\n" + "=" * 80)
    print(f"[START] กำลังสร้างข้อมูลจริง TPC-H ที่ Scale Factor SF={sf} (~{sf} GB)...")
    print("=" * 80)

    t0 = time.perf_counter()
    con = duckdb.connect(str(duckdb_path))
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB';")

    con.execute("INSTALL tpch; LOAD tpch;")
    existing = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "lineitem" not in existing:
        print(f" -> กำลังรัน CALL dbgen(sf={sf}) ... (ใช้เวลาสักครู่ตามขนาดข้อมูล)")
        con.execute(f"CALL dbgen(sf={sf});")
    else:
        print(f" -> พบตาราง TPC-H ใน {duckdb_path} แล้ว ข้ามขั้นตอน dbgen()")

    row_counts = {}
    parquet_bytes = 0
    for tbl in TPCH_TABLES:
        cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        row_counts[tbl] = cnt
        pq_file = parquet_dir / f"{tbl}.parquet"
        if not pq_file.exists():
            print(f" -> Export ตาราง {tbl} ({cnt:,} แถว) เป็น ZSTD Parquet...")
            con.execute(
                f"COPY {tbl} TO '{pq_file.as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);"
            )
        parquet_bytes += pq_file.stat().st_size

    con.close()
    elapsed = time.perf_counter() - t0
    total_rows = sum(row_counts.values())
    size_gb = parquet_bytes / (1024 ** 3)

    manifest = {
        "scale_factor": sf,
        "total_rows": total_rows,
        "lineitem_rows": row_counts["lineitem"],
        "parquet_size_gb": round(size_gb, 3),
        "generation_time_sec": round(elapsed, 2),
        "lineitem_sha256": compute_sha256(parquet_dir / "lineitem.parquet"),
    }
    print(f"[DONE] SF={sf}: รวม {total_rows:,} แถว | ขนาด Parquet={size_gb:.2f} GB | เวลาใช้ไป {elapsed/60:.1f} นาที")
    return manifest


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []
    # สร้างข้อมูลทั้ง SF10 และ SF100
    for sf_val in [10, 100]:
        info = provision_scale_factor(sf=sf_val, threads=8, memory_limit_gb=28)
        manifests.append(info)

    df_manifest = pd.DataFrame(manifests)
    df_manifest.to_csv(RESULTS_DIR / "multiscale_manifest_sf10_sf100.csv", index=False)
    print(f"\n[OK] บันทึก Manifest ของ SF10 และ SF100 ลงใน {RESULTS_DIR}")