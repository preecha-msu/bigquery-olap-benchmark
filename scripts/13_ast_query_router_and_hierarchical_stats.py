#!/usr/bin/env python3
r"""
scripts/13_ast_query_router_and_hierarchical_stats.py
ดำเนินการตาม Checklist ข้อ 15 และ ข้อ 16 จากข้อมูลการรันจริง (100% Empirical):
1. คำนวณสถิติเชิงลำดับชั้น (Hierarchical Seed-Cluster Analysis, N_seeds=25 vs N_queries=75)
   พร้อม 10,000-Iteration Seed-Cluster Bootstrap 95% CI, Median [IQR], Wilcoxon Signed-Rank Test, และ Cliff's Delta
2. สร้างและประเมินผล AST-Aware Query Routing Framework ด้วย 5-Fold GroupCrossValidation (จัดกลุ่มตาม Seed_ID)
   เพื่อป้องกัน Data Leakage และวัดค่า Routing Accuracy, Total Workload Time, และ Routing Regret (ms)
"""

from pathlib import Path
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """คำนวณขนาดอิทธิพลแบบ Non-parametric Cliff's Delta จริงทุกคู่ข้อมูล"""
    n_x, n_y = len(x), len(y)
    diff_matrix = np.subtract.outer(x, y)
    more = np.sum(diff_matrix > 0)
    less = np.sum(diff_matrix < 0)
    delta = float((more - less) / (n_x * n_y))
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "Negligible"
    elif abs_d < 0.33:
        mag = "Small"
    elif abs_d < 0.474:
        mag = "Medium"
    else:
        mag = "Large"
    return round(delta, 4), mag


def seed_cluster_bootstrap_ci(
    df: pd.DataFrame,
    seed_col: str,
    val_col: str,
    n_boot: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    คำนวณ 95% Confidence Interval ด้วยวิธี Seed-Cluster Bootstrap (10,000 รอบ)
    โดยสุ่มตัวอย่างระดับ Seed_ID (25 กลุ่มอิสระ) เพื่อรักษาโครงสร้างความสัมพันธ์ภายใน Seed
    """
    rng = np.random.default_rng(random_state)
    seed_means = df.groupby(seed_col)[val_col].mean().to_numpy()
    n_seeds = len(seed_means)
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    boot_samples = seed_means[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_samples, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def run_hierarchical_statistics(merged: pd.DataFrame) -> pd.DataFrame:
    """วิเคราะห์สถิติทั้งระดับ Query (N=75) และระดับ Seed Cluster (N=25)"""
    engines = {
        "SQLite3_RowStore": "sqlite_rowstore_ms",
        "BigQuery_Unopt": "unopt_server_ms",
        "BigQuery_Clustered": "opt_server_ms",
        "DuckDB_Parquet": "external_parquet_ms",
        "DuckDB_Native": "native_storage_ms",
        "Polars_Rust": "polars_rust_ms",
        "DataFusion_Rust": "datafusion_rust_ms",
        "GPU_RTX2050_Warm": "gpu_kernel_ms",
        "GPU_RTX2050_Cold": "gpu_total_cold_ms",
    }

    stat_rows = []
    ref_q = merged["native_storage_ms"].to_numpy()
    ref_seed = merged.groupby("Seed_ID")["native_storage_ms"].mean().to_numpy()

    for eng_name, col in engines.items():
        vals_q = merged[col].to_numpy()
        vals_seed = merged.groupby("Seed_ID")[col].mean().to_numpy()

        mean_val = float(np.mean(vals_q))
        sd_val = float(np.std(vals_q, ddof=1))
        q25, med_val, q75 = np.percentile(vals_q, [25, 50, 75])
        ci_l, ci_u = seed_cluster_bootstrap_ci(merged, "Seed_ID", col, n_boot=10000)

        if eng_name == "DuckDB_Native":
            p_query, p_seed, delta, mag = 1.0, 1.0, 0.0, "Reference"
        else:
            _, p_query = wilcoxon(vals_q, ref_q, alternative="two-sided")
            _, p_seed = wilcoxon(vals_seed, ref_seed, alternative="two-sided")
            delta, mag = cliffs_delta(vals_q, ref_q)

        stat_rows.append({
            "Engine": eng_name,
            "Mean±SD (ms)": f"{mean_val:.2f} ± {sd_val:.2f}",
            "95% Seed-Bootstrap CI": f"[{ci_l:.2f}, {ci_u:.2f}]",
            "Median [IQR] (ms)": f"{med_val:.2f} [{q25:.2f}–{q75:.2f}]",
            "Wilcoxon p (N=75 Queries)": float(p_query),
            "Wilcoxon p (N=25 Seeds)": float(p_seed),
            "Cliff's Delta (vs DuckDB)": f"{delta:+.3f} ({mag})",
        })

    df_stats = pd.DataFrame(stat_rows)
    non_ref_mask = df_stats["Engine"] != "DuckDB_Native"
    p_seeds = df_stats.loc[non_ref_mask, "Wilcoxon p (N=25 Seeds)"].to_numpy()
    m = len(p_seeds)
    order = np.argsort(p_seeds)
    holm_p = np.empty_like(p_seeds)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        adj = min(1.0, p_seeds[orig_idx] * (m - rank_idx))
        running_max = max(running_max, adj)
        holm_p[orig_idx] = running_max
    df_stats.loc[non_ref_mask, "Holm-Adj p (N=25 Seeds)"] = holm_p
    df_stats.loc[~non_ref_mask, "Holm-Adj p (N=25 Seeds)"] = 1.0
    return df_stats


def evaluate_ast_query_router(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_engines = {
        "DuckDB_Native": "native_storage_ms",
        "Polars_Rust": "polars_rust_ms",
        "DataFusion_Rust": "datafusion_rust_ms",
        "GPU_RTX2050_Cold": "gpu_total_cold_ms",
        "BigQuery_Clustered": "opt_server_ms",
    }
    eng_names = list(candidate_engines.keys())
    eng_cols = list(candidate_engines.values())

    merged = merged.copy()
    merged["Selectivity_Level"] = merged["Variation_ID"].map({"V1": 1.0, "V2": 2.0, "V3": 3.0}).fillna(2.0)

    feature_cols = [
        "AST_Depth", "log2_AST_Nodes", "Joins", "CTEs_Subqueries",
        "Window_Funcs", "Has_UNNEST", "Case_Branches", "CSCI_v2_Revised", "Selectivity_Level"
    ]

    unique_seeds = sorted(merged["Seed_ID"].unique())
    seed_to_fold = {s: idx % 5 for idx, s in enumerate(unique_seeds)}
    merged["cv_fold"] = merged["Seed_ID"].map(seed_to_fold)

    routed_records = []

    for fold in range(5):
        train_df = merged[merged["cv_fold"] != fold].copy()
        test_df = merged[merged["cv_fold"] == fold].copy()

        X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
        mu = X_train.mean(axis=0)
        sigma = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))
        X_train_std = np.column_stack([np.ones(len(X_train)), (X_train - mu) / sigma])

        Y_train_log = np.log1p(train_df[eng_cols].to_numpy(dtype=np.float64))
        reg_eye = 1.0 * np.eye(X_train_std.shape[1])
        reg_eye[0, 0] = 0.0
        W_ridge = np.linalg.solve(X_train_std.T @ X_train_std + reg_eye, X_train_std.T @ Y_train_log)

        X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
        X_test_std = np.column_stack([np.ones(len(X_test)), (X_test - mu) / sigma])
        Y_pred_ms = np.expm1(X_test_std @ W_ridge)

        for idx_row, (_, row) in enumerate(test_df.iterrows()):
            actual_latencies = np.array([row[c] for c in eng_cols], dtype=np.float64)
            oracle_idx = int(np.argmin(actual_latencies))
            oracle_ms = float(actual_latencies[oracle_idx])

            if row["Has_UNNEST"] > 0:
                rule_eng = "DataFusion_Rust"
            elif row["Window_Funcs"] > 0 or row["Joins"] >= 3 or row["CTEs_Subqueries"] >= 2:
                rule_eng = "GPU_RTX2050_Cold"
            else:
                rule_eng = "DataFusion_Rust"
            rule_ms = float(row[candidate_engines[rule_eng]])

            pred_idx = int(np.argmin(Y_pred_ms[idx_row]))
            learned_eng = eng_names[pred_idx]
            learned_ms = float(actual_latencies[pred_idx])

            routed_records.append({
                "query_file": row["query_file"],
                "Complexity_Tier": row["Complexity_Tier"],
                "Seed_ID": row["Seed_ID"],
                "Variation_ID": row["Variation_ID"],
                "Oracle_Best_Engine": eng_names[oracle_idx],
                "Oracle_Best_ms": oracle_ms,
                "Rule_AST_Engine": rule_eng,
                "Rule_AST_ms": rule_ms,
                "Learned_AST_Engine": learned_eng,
                "Learned_AST_ms": learned_ms,
                **{name: float(row[col]) for name, col in candidate_engines.items()},
            })

    df_routed = pd.DataFrame(routed_records)

    strategies = {
        "Always BigQuery (Clustered Cloud)": df_routed["BigQuery_Clustered"],
        "Always DuckDB (C++ Vectorized CPU)": df_routed["DuckDB_Native"],
        "Always DataFusion (Rust Arrow CPU)": df_routed["DataFusion_Rust"],
        "Always Polars (Rust Arrow CPU)": df_routed["Polars_Rust"],
        "Always CUDA GPU (RTX 2050 Cold-Start)": df_routed["GPU_RTX2050_Cold"],
        "Proposed Rule-Based AST Router": df_routed["Rule_AST_ms"],
        "Proposed Learned AST Cost Router (5-Fold Seed CV)": df_routed["Learned_AST_ms"],
        "Oracle Optimal (Theoretical Lower Bound)": df_routed["Oracle_Best_ms"],
    }

    oracle_arr = df_routed["Oracle_Best_ms"].to_numpy()
    duck_total = float(df_routed["DuckDB_Native"].sum())

    summary_rows = []
    for strat_label, series_ms in strategies.items():
        arr = series_ms.to_numpy()
        regret_arr = arr - oracle_arr
        exact_hit = np.mean(np.isclose(arr, oracle_arr, atol=1e-3)) * 100.0
        near_optimal_hit = np.mean(regret_arr <= 5.0) * 100.0

        summary_rows.append({
            "Routing Strategy": strat_label,
            "Mean Latency (ms)": round(float(np.mean(arr)), 2),
            "Median Latency (ms)": round(float(np.median(arr)), 2),
            "P95 Tail Latency (ms)": round(float(np.percentile(arr, 95)), 2),
            "Total 75-Query Time (s)": round(float(np.sum(arr)) / 1000.0, 3),
            "Mean Regret vs Oracle (ms)": round(float(np.mean(regret_arr)), 2),
            "Exact Match (%)": round(float(exact_hit), 1),
            "Within-5ms Optimal (%)": round(float(near_optimal_hit), 1),
            "Speedup vs Always-DuckDB": f"{duck_total / float(np.sum(arr)):.2f}x",
        })

    return pd.DataFrame(summary_rows), df_routed


if __name__ == "__main__":
    merged_path = RESULTS_DIR / "merged_75queries_all_engines_sf1.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"กรุณารัน 01_reconcile_sf1_and_csci.py ก่อน เพื่อสร้าง {merged_path}")

    df_merged = pd.read_csv(merged_path)

    print("=" * 95)
    print("PART 1: HIERARCHICAL SEED-CLUSTER STATISTICAL ANALYSIS (N_seeds = 25 vs N_queries = 75)")
    print("=" * 95)
    df_hier_stats = run_hierarchical_statistics(df_merged)
    print(df_hier_stats.to_string(index=False))
    df_hier_stats.to_csv(RESULTS_DIR / "hierarchical_seed_cluster_stats_sf1.csv", index=False)

    print("\n" + "=" * 95)
    print("PART 2: AST-AWARE QUERY ROUTING EVALUATION (5-Fold Out-of-Seed Cross-Validation, N = 75)")
    print("=" * 95)
    df_router_summary, df_router_details = evaluate_ast_query_router(df_merged)
    print(df_router_summary.to_string(index=False))
    df_router_summary.to_csv(RESULTS_DIR / "ast_query_router_evaluation_sf1.csv", index=False)
    df_router_details.to_csv(RESULTS_DIR / "ast_query_router_per_query_sf1.csv", index=False)
    print(f"\n[OK] บันทึกผลสถิติเชิงลำดับชั้นและผลประเมิน AST Query Router ลงใน {RESULTS_DIR}")