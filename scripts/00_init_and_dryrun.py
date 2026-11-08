# ไฟล์: scripts/00_init_and_dryrun.py
import os
import sys
import yaml
import argparse
from pathlib import Path
import pandas as pd
import duckdb
import sqlglot
from google.cloud import bigquery
from google.oauth2 import service_account

# ตั้งค่าการแสดงผลตาราง Pandas บนหน้าจอ PowerShell ให้ครบทุกคอลัมน์
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")

DIRECTORIES = [
    "config",
    "data/raw_parquet/sf1_1gb",
    "data/raw_parquet/sf10_10gb",
    "data/engine_dbs/duckdb",
    "data/engine_dbs/clickhouse_data",
    "workloads/L1_scan_filter",
    "workloads/L2_multi_joins",
    "workloads/L3_nested_cte",
    "workloads/L4_window_analytical",
    "workloads/L5_array_unnest",
    "scripts",
    "src",
    "results/raw_logs",
    "results/checkpoints_npz",
    "results/summary_tables",
    "results/figures_300dpi",
    "manuscript/figures",
]

DEFAULT_CONFIG = """project_settings:
  gcp_project_id: "your-gcp-project-id"
  gcp_location: "asia-southeast1"
  gcs_bucket: "gs://msu-sql-benchmark-sf10"
  credentials_path: "config/gcp_credentials.json"

experiment_controls:
  scale_factor: 10
  warmup_runs: 1
  measurement_trials: 10
  disable_query_cache: true
  alpha_significance: 0.05

datasets:
  bq_external_dataset: "tpch_external"
  bq_native_dataset: "tpch_native"
  local_parquet_dir: "data/raw_parquet/sf10_10gb"
"""

SAMPLE_DRYRUN_WORKLOAD = {
    "Q_L1_Scan": {
        "level": "L1: Scan & Filter",
        "sql": "SELECT repo_name, watch_count FROM `bigquery-public-data.github_repos.sample_repos` WHERE watch_count > 1000"
    },
    "Q_L2_Join": {
        "level": "L2: Multi-Way Join",
        "sql": """
            SELECT r.repo_name, COUNT(c.commit) AS commit_cnt
            FROM `bigquery-public-data.github_repos.sample_repos` r
            JOIN `bigquery-public-data.github_repos.sample_commits` c ON r.repo_name = c.repo_name
            GROUP BY r.repo_name
        """
    },
    "Q_L3_Nested_CTE": {
        "level": "L3: Nested CTE & Subquery",
        "sql": """
            WITH author_stats AS (
                SELECT author.name AS author_name, repo_name, COUNT(commit) AS c_cnt
                FROM `bigquery-public-data.github_repos.sample_commits`
                GROUP BY author_name, repo_name
            )
            SELECT repo_name, AVG(c_cnt) AS avg_commits_per_author
            FROM author_stats
            WHERE c_cnt > (SELECT AVG(c_cnt) FROM author_stats)
            GROUP BY repo_name
        """
    },
    "Q_L4_Window": {
        "level": "L4: Window & Analytical",
        "sql": """
            SELECT repo_name, author.name AS author_name, committer.date,
                   ROW_NUMBER() OVER (PARTITION BY repo_name ORDER BY committer.date DESC) AS rn
            FROM `bigquery-public-data.github_repos.sample_commits`
        """
    },
    "Q_L5_Unnest": {
        "level": "L5: Semi-Structured / UNNEST",
        "sql": """
            SELECT c.commit, d.new_path
            FROM `bigquery-public-data.github_repos.sample_commits` c,
            UNNEST(c.difference) AS d
            WHERE d.new_path LIKE '%.py'
        """
    }
}

def setup_workspace(root: Path) -> dict:
    print(f"\n[STEP 1] Initializing Workspace at: {root}")
    for d in DIRECTORIES:
        (root / d).mkdir(parents=True, exist_ok=True)
    
    cfg_file = root / "config" / "benchmark_config.yaml"
    if not cfg_file.exists():
        cfg_file.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print(f"  -> Created default config: {cfg_file}")
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def verify_local_engines(root: Path):
    print("\n[STEP 2] Verifying Local DuckDB & SQLGlot AST Parser")
    db_file = root / "data" / "engine_dbs" / "duckdb" / "env_check.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("SET threads TO 8; SET memory_limit = '12GB';")
    info = con.execute("SELECT current_setting('threads') AS t, current_setting('memory_limit') AS m;").fetchone()
    con.close()
    print(f"  -> DuckDB v{duckdb.__version__} Ready | Threads: {info[0]} | Memory Limit: {info[1]}")

    # ทดสอบแปลง SQL 4 เอนจิน
    test_sql = "SELECT c_nationkey, COUNT(*) FROM customer GROUP BY 1"
    for dialect in ["bigquery", "duckdb", "clickhouse", "trino"]:
        _ = sqlglot.transpile(test_sql, read="postgres", write=dialect)[0]
    print(f"  -> SQLGlot v{sqlglot.__version__} Multi-Dialect Transpiler Ready (bigquery, duckdb, clickhouse, trino)")

def run_bigquery_dryrun(root: Path, cfg: dict):
    print("\n[STEP 3] Verifying Google BigQuery Authentication & Dry-Run Quota Estimation")
    cred_path = root / cfg["project_settings"]["credentials_path"]
    
    if not cred_path.exists():
        print(f"  -> [SKIP] ไม่พบไฟล์ Service Account Key ที่: {cred_path}")
        print("     (เมื่อนำไฟล์ gcp_credentials.json มาวางแล้ว ให้รันสคริปต์นี้อีกครั้งเพื่อเช็คโควตา Dry Run)")
        return

    credentials = service_account.Credentials.from_service_account_file(str(cred_path))
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    print(f"  -> Connected to GCP Project: {client.project}")

    warmup = cfg["experiment_controls"]["warmup_runs"]
    trials = cfg["experiment_controls"]["measurement_trials"]
    total_runs = warmup + trials

    rows = []
    for q_id, q_info in SAMPLE_DRYRUN_WORKLOAD.items():
        job_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = client.query(q_info["sql"], job_config=job_cfg)
            gb_per_run = (job.total_bytes_processed or 0) / (1024 ** 3)
            total_gb = gb_per_run * total_runs
            rows.append({
                "Query_ID": q_id,
                "Complexity": q_info["level"],
                "Single_Run_GB": round(gb_per_run, 4),
                f"Runs({warmup}W+{trials}T)": total_runs,
                "Total_Scanned_GB": round(total_gb, 4),
                "Est_Cost_USD": round((total_gb / 1024.0) * 5.0, 4)
            })
        except Exception as e:
            rows.append({
                "Query_ID": q_id,
                "Complexity": q_info["level"],
                "Single_Run_GB": -1.0,
                f"Runs({warmup}W+{trials}T)": total_runs,
                "Total_Scanned_GB": -1.0,
                "Est_Cost_USD": str(e)[:45]
            })

    df_est = pd.DataFrame(rows)
    print("\n" + "=" * 85)
    print(df_est.to_string(index=False))
    print("=" * 85)

    out_csv = root / "results" / "summary_tables" / "dryrun_quota_estimation.csv"
    df_est.to_csv(out_csv, index=False)
    print(f"  -> Saved Dry-Run Quota Report to: {out_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize SQL Benchmark Workspace & Run BigQuery Dry-Run")
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT), help="Path to project root directory")
    args = parser.parse_args()

    root_path = Path(args.root)
    config_data = setup_workspace(root_path)
    verify_local_engines(root_path)
    run_bigquery_dryrun(root_path, config_data)