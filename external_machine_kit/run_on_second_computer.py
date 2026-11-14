#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_on_second_computer.py
================================================================================
Standalone External Computer Re-Execution & SHA-256 Verification Script
(For Reviewer Section 5, Process 6: Independent Environment Re-execution)
================================================================================
วิธีใช้งานบนคอมพิวเตอร์เครื่องที่ 2:
  1. ติดตั้งไลบรารีพื้นฐาน: pip install duckdb polars pyarrow pandas numpy psutil
  2. รันคำสั่ง: python run_on_second_computer.py
  3. นำไฟล์ผลลัพธ์ "external_machine_reexecution_audit.csv" และ
     "external_machine_environment_manifest.json" กลับมาวางที่โฟลเดอร์ results/ ของเครื่องหลัก
"""

import hashlib
import json
import platform
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import psutil

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=" * 90)
    print("STARTING INDEPENDENT ENVIRONMENT RE-EXECUTION ON EXTERNAL COMPUTER")
    print("=" * 90)

    env_info = {
        "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "os_platform": platform.platform(),
        "cpu_processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "duckdb_version": duckdb.__version__,
        "polars_version": pl.__version__,
        "sqlite3_version": sqlite3.sqlite_version,
    }
    print("[1] Detected External Machine Environment:")
    for k, v in env_info.items():
        print(f"    - {k}: {v}")

    pq_file = HERE / "canonical_tpch_reexec_dataset.parquet"
    con_duck = duckdb.connect(":memory:")
    con_duck.execute("PRAGMA threads=8;")
    con_duck.execute("SELECT setseed(0.42);")
    con_duck.execute("""
        CREATE TABLE lineitem AS
        SELECT
            range AS l_orderkey,
            (range % 50000) AS l_partkey,
            (range % 10000) AS l_suppkey,
            CAST((range % 50) + 1 AS DOUBLE) AS l_quantity,
            CAST(((range % 1000) + 100) * 1.25 AS DOUBLE) AS l_extendedprice,
            CAST((range % 10) * 0.01 AS DOUBLE) AS l_discount,
            DATE '1992-01-02' + CAST((range % 2525) AS INTEGER) AS l_shipdate,
            CASE WHEN range % 3 = 0 THEN 'A' WHEN range % 3 = 1 THEN 'N' ELSE 'R' END AS l_returnflag
        FROM range(500000);
    """)
    con_duck.execute("""
        CREATE TABLE orders AS
        SELECT
            range AS o_orderkey,
            (range % 25000) AS o_custkey,
            CAST(((range % 800) + 100) * 2.5 AS DOUBLE) AS o_totalprice,
            range % 5 AS o_orderpriority
        FROM range(250000);
    """)
    con_duck.execute(f"COPY lineitem TO '{pq_file.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    dataset_sha = sha256_file(pq_file)
    env_info["dataset_parquet_sha256"] = dataset_sha
    print(f"[2] Generated Canonical Parquet Dataset | SHA-256: {dataset_sha}")

    pl_lineitem = pl.read_parquet(pq_file)
    pl_orders = pl.from_pandas(con_duck.execute("SELECT * FROM orders").df())

    con_sq = sqlite3.connect(":memory:")
    pl_lineitem.to_pandas().to_sql("lineitem", con_sq, index=False)
    pl_orders.to_pandas().to_sql("orders", con_sq, index=False)

    benchmark_queries = [
        ("L1_Scan_Filter", "L1", "SELECT COUNT(*) AS cnt, SUM(l_extendedprice * (1.0 - l_discount)) AS checksum FROM lineitem WHERE l_quantity > 20.0"),
        ("L2_Multi_Join", "L2", "SELECT COUNT(*) AS cnt, SUM(l.l_extendedprice) AS checksum FROM lineitem l INNER JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE o.o_totalprice > 800.0"),
        ("L3_Nested_CTE", "L3", "WITH cte AS (SELECT l_partkey, SUM(l_extendedprice) AS rev FROM lineitem GROUP BY l_partkey) SELECT COUNT(*) AS cnt, SUM(rev) AS checksum FROM cte WHERE rev > 5000.0"),
        ("L4_Window_Frame", "L4", "SELECT COUNT(*) AS cnt, SUM(val) AS checksum FROM (SELECT l_extendedprice AS val, ROW_NUMBER() OVER (PARTITION BY l_returnflag ORDER BY l_extendedprice DESC) AS rn FROM lineitem) t WHERE rn <= 1000"),
        ("L5_Set_Unnest_In", "L5", "SELECT COUNT(*) AS cnt, SUM(l_quantity * l_extendedprice) AS checksum FROM lineitem WHERE l_discount IN (0.01, 0.03, 0.05, 0.07)"),
    ]

    rows = []
    for q_id, tier, sql in benchmark_queries:
        duck_times, pl_times = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            r_duck = con_duck.execute(sql).fetchone()
            duck_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            ctx = pl.SQLContext(lineitem=pl_lineitem, orders=pl_orders)
            df_res = ctx.execute(sql).collect()
            pl_times.append((time.perf_counter() - t0) * 1000.0)
            r_pl = (int(df_res.row(0)[0]), float(df_res.row(0)[1]))

        t0 = time.perf_counter()
        r_sq = con_sq.execute(sql).fetchone()
        sq_ms = (time.perf_counter() - t0) * 1000.0

        row_match = (int(r_duck[0]) == int(r_pl[0]) == int(r_sq[0]))
        rel_err = max(
            abs(float(r_duck[1]) - float(r_pl[1])) / max(abs(float(r_duck[1])), 1e-9),
            abs(float(r_duck[1]) - float(r_sq[1])) / max(abs(float(r_duck[1])), 1e-9),
        )

        rows.append({
            "External_Hostname": env_info["hostname"],
            "External_OS": env_info["os_platform"],
            "External_CPU_Threads": env_info["logical_threads"],
            "External_RAM_GB": env_info["ram_total_gb"],
            "Dataset_SHA256_Prefix16": dataset_sha[:16],
            "Query_ID": q_id,
            "Complexity_Tier": tier,
            "DuckDB_Mean_ms": round(float(np.mean(duck_times)), 3),
            "Polars_Mean_ms": round(float(np.mean(pl_times)), 3),
            "SQLite3_Mean_ms": round(sq_ms, 3),
            "Verified_Row_Count": int(r_duck[0]),
            "FP64_Aggregate_Checksum": round(float(r_duck[1]), 4),
            "Max_FP64_Rel_Error": f"{rel_err:.2e}",
            "Result_Equivalence": "PASS (EX=1.000)" if (row_match and rel_err < 1e-9) else "CHECK",
        })

    con_duck.close()
    con_sq.close()
    if pq_file.exists():
        pq_file.unlink()

    out_csv = HERE / "external_machine_reexecution_audit.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    out_csv_sha = sha256_file(out_csv)
    env_info["output_audit_csv_sha256"] = out_csv_sha

    out_json = HERE / "external_machine_environment_manifest.json"
    out_json.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    print("\n[3] External Computer Re-Execution Results:")
    print(df_out.to_string(index=False))
    print("=" * 90)
    print(f"[SUCCESS] Saved: {out_csv.name} | SHA-256: {out_csv_sha}")
    print(f"[SUCCESS] Saved: {out_json.name}")
    print("ก๊อปปี้ไฟล์ทั้ง 2 นี้กลับไปยังโฟลเดอร์ results/ ของเครื่องหลักได้เลยครับ!")
    print("=" * 90)


if __name__ == "__main__":
    main()
