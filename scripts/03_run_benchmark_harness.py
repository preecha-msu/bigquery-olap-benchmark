# ไฟล์: scripts/03_run_benchmark_harness.py (75-Query DuckDB Local Execution Harness)
import time
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_duckdb_75_benchmark(root: Path, trials_override: int = None):
    with open(root / "config" / "benchmark_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sf = cfg["experiment_controls"]["scale_factor"]
    warmup = cfg["experiment_controls"]["warmup_runs"]
    trials = trials_override if trials_override is not None else cfg["experiment_controls"]["measurement_trials"]

    parquet_dir = root / "data" / "raw_parquet" / f"sf{sf}_{sf}gb"
    native_db_path = root / "data" / "engine_dbs" / f"tpch_sf{sf}.duckdb"

    df_ast = pd.read_csv(root / "workloads" / "ast_complexity_summary.csv").set_index("Query_File")
    sql_files = sorted((root / "workloads").rglob("*.sql"))

    # เตรียมการเชื่อมต่อโหมดที่ 1: External Parquet Views
    con_ext = duckdb.connect(database=":memory:")
    con_ext.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB';")
    for pq_file in parquet_dir.glob("*.parquet"):
        tbl_name = pq_file.stem
        con_ext.execute(f"CREATE OR REPLACE VIEW {tbl_name} AS SELECT * FROM read_parquet('{pq_file.as_posix()}');")

    # เตรียมการเชื่อมต่อโหมดที่ 2: Native DuckDB Columnar Storage
    con_nat = duckdb.connect(database=str(native_db_path), read_only=True)
    con_nat.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB';")

    print("=" * 105)
    print(f" [DUCKDB HARNESS] Running {len(sql_files)} Queries (25 Seeds x 3 Variations) | Warmup={warmup} | Trials={trials}")
    print("=" * 105)

    records = []
    for idx, fpath in enumerate(sql_files, start=1):
        q_name = fpath.name
        tier = fpath.parent.name
        sql_text = fpath.read_text(encoding="utf-8")
        ast_row = df_ast.loc[q_name]

        for _ in range(warmup):
            con_ext.execute(sql_text).fetchall()
            con_nat.execute(sql_text).fetchall()

        ext_times, nat_times = [], []
        for t_idx in range(1, trials + 1):
            t0 = time.perf_counter()
            res_ext = con_ext.execute(sql_text).fetchall()
            ext_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            res_nat = con_nat.execute(sql_text).fetchall()
            nat_ms = (time.perf_counter() - t1) * 1000.0

            ex_acc = int(len(res_ext) == len(res_nat))
            ext_times.append(ext_ms)
            nat_times.append(nat_ms)

            records.append({
                "query_file": q_name,
                "complexity_tier": tier,
                "seed_id": ast_row["Seed_ID"],
                "seed_short": ast_row["Seed_Short"],
                "variation_id": ast_row["Variation_ID"],
                "trial": t_idx,
                "ast_depth": int(ast_row["AST_Depth"]),
                "ast_nodes": int(ast_row["AST_Nodes"]),
                "joins": int(ast_row["Joins"]),
                "csci_score": float(ast_row["CSCI_Score"]),
                "external_parquet_ms": round(ext_ms, 3),
                "native_storage_ms": round(nat_ms, 3),
                "row_count": len(res_nat),
                "ex_accuracy": ex_acc
            })

        if idx % 5 == 0 or idx == len(sql_files):
            print(
                f"  -> [{idx:02d}/{len(sql_files):02d}] {q_name:<42} | "
                f"Parquet: {np.mean(ext_times):>7.2f} ms | Native: {np.mean(nat_times):>7.2f} ms"
            )

    con_ext.close()
    con_nat.close()

    df_out = pd.DataFrame(records)
    out_csv = root / "results" / "raw_logs" / f"raw_benchmark_sf{sf}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved {len(df_out)} DuckDB Trial Records to: {out_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Measurement trials per query (default: 5)")
    args = parser.parse_args()
    run_duckdb_75_benchmark(PROJECT_ROOT, args.trials)
