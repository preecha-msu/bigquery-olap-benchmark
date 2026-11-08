# ไฟล์: scripts/06_run_bigquery_cloud_benchmark.py (75-Query BigQuery Cloud Benchmark with Auto-Resume & 1GB Cap)
import time
import yaml
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sqlglot
from google.cloud import bigquery
from google.oauth2 import service_account

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ขยับเพดานความปลอดภัยต่อคิวรีเป็น 1,000 MB (1 GB) เพื่อรองรับคิวรี 8-Way Join (546 MB) ในโหมด SF=1GB
MAX_BYTES_PER_QUERY = 1000 * 1024 * 1024

def transpile_for_bigquery(sql_text: str) -> str:
    import re
    bq_sql = sqlglot.transpile(sql_text, read="postgres", write="bigquery")[0]
    for m in re.finditer(r"CROSS\s+JOIN\s+UNNEST\s*\(([^)]+)\)\s+AS\s+(\w+)", sql_text, flags=re.IGNORECASE):
        col_expr, alias = m.group(1).strip(), m.group(2).strip()
        bq_sql = re.sub(
            r"CROSS\s+JOIN\s+UNNEST\s*\(\s*" + re.escape(col_expr) + r"\s*\)(?:\s+AS\s+\w+)?",
            f"CROSS JOIN UNNEST({col_expr}) AS {alias}",
            bq_sql,
            flags=re.IGNORECASE
        )
    return bq_sql

def execute_bq_single_query(client: bigquery.Client, bq_sql: str, default_dataset: str) -> dict:
    job_config = bigquery.QueryJobConfig(
        use_query_cache=False,
        default_dataset=default_dataset,
        maximum_bytes_billed=MAX_BYTES_PER_QUERY
    )
    t0 = time.perf_counter()
    query_job = client.query(bq_sql, job_config=job_config)
    rows = list(query_job.result())
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if query_job.started and query_job.ended:
        server_ms = (query_job.ended - query_job.started).total_seconds() * 1000.0
    else:
        server_ms = wall_ms

    shuffle_bytes = 0
    if query_job.query_plan:
        for stage in query_job.query_plan:
            shuffle_bytes += getattr(stage, "shuffle_output_bytes", 0) or 0

    return {
        "wall_ms": wall_ms,
        "server_ms": server_ms,
        "slot_ms": float(query_job.slot_millis or 0),
        "scanned_mb": float(query_job.total_bytes_processed or 0) / (1024 ** 2),
        "shuffle_mb": float(shuffle_bytes) / (1024 ** 2),
        "row_count": len(rows)
    }

def run_bigquery_75_benchmark(root: Path, trials_override: int = 2, warmup_override: int = 0, reset: bool = False):
    with open(root / "config" / "benchmark_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cred_path = root / cfg["project_settings"]["credentials_path"]
    credentials = service_account.Credentials.from_service_account_file(str(cred_path))
    project_id = credentials.project_id
    location = cfg["project_settings"].get("gcp_location", "asia-southeast1")

    client = bigquery.Client(credentials=credentials, project=project_id, location=location)

    sf = cfg["experiment_controls"]["scale_factor"]
    trials = trials_override

    ds_unopt = f"{project_id}.tpch_unopt_sf{sf}"
    ds_opt = f"{project_id}.tpch_opt_sf{sf}"

    df_ast = pd.read_csv(root / "workloads" / "ast_complexity_summary.csv").set_index("Query_File")
    sql_files = sorted((root / "workloads").rglob("*.sql"))

    raw_out = root / "results" / "raw_logs" / f"raw_bigquery_cloud_sf{sf}.csv"
    existing_records = []
    completed_queries = set()

    # ตรวจสอบไฟล์เดิมเพื่อทำ Auto-Resume เฉพาะชุด 75 คิวรี
    if raw_out.exists() and not reset:
        try:
            df_prev = pd.read_csv(raw_out)
            if "seed_id" in df_prev.columns and "variation_id" in df_prev.columns:
                valid_qnames = set(df_ast.index)
                df_prev = df_prev[df_prev["query_file"].isin(valid_qnames)]
                q_counts = df_prev.groupby("query_file")["trial"].count()
                completed_queries = set(q_counts[q_counts >= trials].index)
                df_prev = df_prev[df_prev["query_file"].isin(completed_queries)]
                existing_records = df_prev.to_dict("records")
        except Exception:
            existing_records = []
            completed_queries = set()

    print("=" * 108)
    print(f" [BIGQUERY CLOUD HARNESS] Running {len(sql_files)} Queries (25 Seeds x 3 Variations) | Trials={trials}")
    print(f" Project: {project_id} | Cache=OFF | Safety Cap=1,000 MB/Query | Auto-Resumed: {len(completed_queries)} queries")
    print("=" * 108)

    records = list(existing_records)
    for idx, fpath in enumerate(sql_files, start=1):
        q_name = fpath.name
        if q_name in completed_queries:
            print(f"  -> [{idx:02d}/{len(sql_files):02d}] [SKIP-DONE] {q_name:<38} (Already recorded)")
            continue

        tier = fpath.parent.name
        raw_sql = fpath.read_text(encoding="utf-8")
        bq_sql = transpile_for_bigquery(raw_sql)
        ast_row = df_ast.loc[q_name]

        for _ in range(warmup_override):
            _ = execute_bq_single_query(client, bq_sql, ds_unopt)
            _ = execute_bq_single_query(client, bq_sql, ds_opt)

        u_times, o_times, o_slots, u_scans, o_scans = [], [], [], [], []
        for t_idx in range(1, trials + 1):
            res_u = execute_bq_single_query(client, bq_sql, ds_unopt)
            res_o = execute_bq_single_query(client, bq_sql, ds_opt)

            ex_acc = int(res_u["row_count"] == res_o["row_count"])
            u_times.append(res_u["server_ms"])
            o_times.append(res_o["server_ms"])
            o_slots.append(res_o["slot_ms"])
            u_scans.append(res_u["scanned_mb"])
            o_scans.append(res_o["scanned_mb"])

            records.append({
                "query_file": q_name,
                "complexity_tier": tier,
                "seed_id": ast_row.get("Seed_ID", q_name),
                "seed_short": ast_row.get("Seed_Short", q_name[:6]),
                "variation_id": ast_row.get("Variation_ID", "V1"),
                "trial": t_idx,
                "ast_depth": int(ast_row["AST_Depth"]),
                "ast_nodes": int(ast_row["AST_Nodes"]),
                "joins": int(ast_row["Joins"]),
                "csci_score": float(ast_row.get("CSCI_Score", 0.0)),
                "unopt_server_ms": round(res_u["server_ms"], 2),
                "unopt_slot_ms": round(res_u["slot_ms"], 2),
                "unopt_scanned_mb": round(res_u["scanned_mb"], 3),
                "unopt_shuffle_mb": round(res_u["shuffle_mb"], 3),
                "opt_server_ms": round(res_o["server_ms"], 2),
                "opt_slot_ms": round(res_o["slot_ms"], 2),
                "opt_scanned_mb": round(res_o["scanned_mb"], 3),
                "opt_shuffle_mb": round(res_o["shuffle_mb"], 3),
                "row_count": res_o["row_count"],
                "ex_accuracy": ex_acc
            })

        # บันทึก Checkpoint ลง CSV ทันทีทุกครั้งที่จบ 1 คิวรี
        pd.DataFrame(records).to_csv(raw_out, index=False)

        v_id = ast_row.get("Variation_ID", "V1")
        print(
            f"  -> [{idx:02d}/{len(sql_files):02d}] [{v_id}] {q_name:<38} | "
            f"Unopt: {np.mean(u_times):>5.0f}ms ({np.mean(u_scans):>5.1f}MB) | "
            f"Clust: {np.mean(o_times):>5.0f}ms ({np.mean(o_scans):>5.1f}MB) | "
            f"Slot: {np.mean(o_slots):>5.0f}ms"
        )

    print(f"\n[OK] Completed all 75 queries ({len(records)} total rows) saved to: {raw_out}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2, help="Measurement trials per query on BigQuery (default: 2)")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per query on BigQuery (default: 0)")
    parser.add_argument("--reset", action="store_true", help="Reset and run from query 01")
    args = parser.parse_args()
    run_bigquery_75_benchmark(PROJECT_ROOT, args.trials, args.warmup, args.reset)

