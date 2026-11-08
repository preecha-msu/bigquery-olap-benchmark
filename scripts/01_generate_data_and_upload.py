# ไฟล์: scripts/01_generate_data_and_upload.py
import os
import time
import hashlib
import yaml
import argparse
from pathlib import Path
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TPCH_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_tpch_parquet_and_duckdb(root: Path, sf: int):
    folder_name = f"sf{sf}_{sf}gb"
    parquet_dir = root / "data" / "raw_parquet" / folder_name
    parquet_dir.mkdir(parents=True, exist_ok=True)

    duck_db_path = root / "data" / "engine_dbs" / "duckdb" / f"tpch_sf{sf}.duckdb"
    if duck_db_path.exists():
        duck_db_path.unlink()

    print(f"\n[STEP 1] Generating TPC-H Dataset (Scale Factor = {sf}) via DuckDB")
    print(f"  -> Target Parquet Dir : {parquet_dir}")
    print(f"  -> Target DuckDB File : {duck_db_path}")

    con = duckdb.connect(str(duck_db_path))
    con.execute("SET threads TO 8; SET memory_limit = '11GB';")
    con.execute("INSTALL tpch; LOAD tpch;")

    t0 = time.perf_counter()
    con.execute(f"CALL dbgen(sf={sf});")
    print(f"  -> In-database TPC-H tables generated in {time.perf_counter() - t0:.2f} s")

    manifest_rows = []
    for tbl in TPCH_TABLES:
        out_file = parquet_dir / f"{tbl}.parquet"
        out_posix = str(out_file).replace("\\", "/")
        
        t_start = time.perf_counter()
        if not out_file.exists():
            con.execute(f"COPY {tbl} TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        export_s = time.perf_counter() - t_start

        row_cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        size_mb = out_file.stat().st_size / (1024 ** 2)
        sha256_hex = compute_sha256(out_file)

        # สร้าง View สำหรับโหมด External Parquet บน DuckDB เพื่อเปรียบเทียบกับ Native Table
        con.execute(f"CREATE OR REPLACE VIEW ext_{tbl} AS SELECT * FROM read_parquet('{out_posix}');")

        manifest_rows.append({
            "table_name": tbl,
            "scale_factor": sf,
            "row_count": row_cnt,
            "parquet_size_mb": round(size_mb, 2),
            "export_time_s": round(export_s, 2),
            "sha256": sha256_hex
        })
        print(f"  -> [{tbl:<8}] Rows: {row_cnt:>10,} | Size: {size_mb:>7.2f} MB | SHA256: {sha256_hex[:12]}...")

    con.close()

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = root / "results" / "summary_tables" / f"dataset_manifest_sf{sf}.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\n[OK] Saved Dataset Verification Manifest (with SHA-256) to: {manifest_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TPC-H Parquet & Native DuckDB Database")
    parser.add_argument("--sf", type=int, default=None, help="Override TPC-H Scale Factor (e.g., 1 or 10)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "benchmark_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_sf = args.sf if args.sf is not None else cfg["experiment_controls"]["scale_factor"]
    generate_tpch_parquet_and_duckdb(PROJECT_ROOT, target_sf)