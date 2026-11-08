# ไฟล์: scripts/16_statistical_ci_and_csci_validation.py
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def bootstrap_ci_seed_clustered(values: np.ndarray, seeds: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple:
    """คำนวณ 95% Confidence Interval ด้วย Seed-Clustered Bootstrap (สุ่มระดับกลุ่ม Seed ทั้ง 25 กลุ่ม)"""
    rng = np.random.default_rng(seed)
    unique_seeds = np.unique(seeds)
    n_groups = len(unique_seeds)
    seed_to_vals = {s: values[seeds == s] for s in unique_seeds}
    group_means = np.array([np.mean(seed_to_vals[s]) for s in unique_seeds])

    boot_idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    boot_means = np.mean(group_means[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(values)), float(ci_low), float(ci_high)

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr[:, None] - y_arr[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x_arr) * len(y_arr)))

def run_validation_experiments(root: Path, sf: int = 1):
    df_ast = pd.read_csv(root / "workloads" / "ast_complexity_summary.csv")
    df_duck = pd.read_csv(root / "results" / "raw_logs" / f"raw_benchmark_sf{sf}.csv")
    df_bq = pd.read_csv(root / "results" / "raw_logs" / f"raw_bigquery_cloud_sf{sf}.csv")
    df_gpu = pd.read_csv(root / "results" / "raw_logs" / f"raw_gpu_cuda_sf{sf}.csv")
    df_multi = pd.read_csv(root / "results" / "raw_logs" / f"raw_multi_software_sf{sf}.csv")
    df_hw = pd.read_csv(root / "results" / "raw_logs" / f"raw_hardware_energy_sf{sf}.csv")
    df_route = pd.read_csv(root / "results" / "raw_logs" / f"raw_query_routing_predictions_sf{sf}.csv")

    if "Query_File" in df_ast.columns:
        df_ast = df_ast.rename(columns={"Query_File": "query_file"})
    if "Seed_Short" not in df_ast.columns:
        df_ast["Seed_Short"] = df_ast["query_file"].apply(lambda s: "_".join(s.split("_")[1:3]))

    dk_q = df_duck.groupby("query_file", as_index=False).agg(
        DuckDB_Parquet_ms=("external_parquet_ms", "mean"),
        DuckDB_Native_ms=("native_storage_ms", "mean")
    )
    bq_q = df_bq.groupby("query_file", as_index=False).agg(
        BQ_Unopt_ms=("unopt_server_ms", "mean"),
        BQ_Clust_ms=("opt_server_ms", "mean")
    )
    gp_q = df_gpu.groupby("query_file", as_index=False).agg(
        GPU_H2D_ms=("gpu_pcie_h2d_ms", "mean"),
        GPU_Warm_ms=("gpu_kernel_ms", "mean"),
        GPU_Cold_ms=("gpu_total_cold_ms", "mean"),
        GPU_Warm_SD=("gpu_kernel_ms", "std")
    )
    mu_q = df_multi.groupby("query_file", as_index=False).agg(
        Polars_ms=("polars_rust_ms", "mean"),
        DataFusion_ms=("datafusion_rust_ms", "mean"),
        SQLite_ms=("sqlite_rowstore_ms", "mean")
    )
    hw_q = df_hw.groupby("query_file", as_index=False).agg(
        SQLite_J=("sqlite_energy_j", "mean"),
        DuckDB_J=("duckdb_energy_j", "mean"),
        Polars_J=("polars_energy_j", "mean"),
        DataFusion_J=("datafusion_energy_j", "mean"),
        GPU_Warm_J=("gpu_warm_energy_j", "mean"),
        GPU_Cold_J=("gpu_cold_energy_j", "mean")
    )

    m = (
        df_ast.merge(dk_q, on="query_file")
        .merge(bq_q, on="query_file")
        .merge(gp_q, on="query_file")
        .merge(mu_q, on="query_file")
        .merge(hw_q, on="query_file")
        .merge(df_route[["query_file", "Proposed9_GBR_Pred", "Rule_Based_Pred"]], on="query_file")
    )

    seeds = m["Seed_Short"].values

    # คำนวณเวลาจริงของ Proposed 9D Router และ Rule-Based Router รายคิวรี
    eng_col_map = {
        "DuckDB": "DuckDB_Native_ms",
        "Polars": "Polars_ms",
        "DataFusion": "DataFusion_ms",
        "GPU_Cold": "GPU_Cold_ms",
        "BigQuery": "BQ_Clust_ms"
    }
    m["Proposed9_Router_ms"] = [m.loc[i, eng_col_map[m.loc[i, "Proposed9_GBR_Pred"]]] for i in range(len(m))]
    m["Rule_Router_ms"] = [m.loc[i, eng_col_map[m.loc[i, "Rule_Based_Pred"]]] for i in range(len(m))]

    # 1. คำนวณ 95% Seed-Cluster Bootstrap CIs สำหรับทุก Engine และ Router
    ci_metrics = [
        ("SQLite 3 (Full Row-Store)", m["SQLite_ms"].values),
        ("Google BigQuery (Unoptimized)", m["BQ_Unopt_ms"].values),
        ("Google BigQuery (Clustered)", m["BQ_Clust_ms"].values),
        ("DuckDB (External Parquet)", m["DuckDB_Parquet_ms"].values),
        ("DuckDB (Native Columnar)", m["DuckDB_Native_ms"].values),
        ("Apache DataFusion (Rust Arrow)", m["DataFusion_ms"].values),
        ("Polars Lazy SQL (Rust Arrow)", m["Polars_ms"].values),
        ("RTX 2050 GPU (Cold Incl. PCIe)", m["GPU_Cold_ms"].values),
        ("RTX 2050 GPU (Warm In-VRAM)", m["GPU_Warm_ms"].values),
        ("Rule-Based Heuristic Router", m["Rule_Router_ms"].values),
        ("Proposed 9D AST+Sel. Router", m["Proposed9_Router_ms"].values),
    ]

    print("=" * 115)
    print(" [1. SEED-CLUSTERED 10,000-ITERATION BOOTSTRAP 95% CONFIDENCE INTERVALS (N=75 QUERIES, 25 SEEDS)]")
    print("=" * 115)
    ci_rows = []
    for label, arr in ci_metrics:
        mean_v, lo_v, hi_v = bootstrap_ci_seed_clustered(arr, seeds)
        print(f"  • {label:<34}: Mean = {mean_v:>8.2f} ms (95% CI: [{lo_v:>8.2f}, {hi_v:>8.2f}] ms)")
        ci_rows.append({"System_or_Policy": label, "Mean_ms": round(mean_v, 2), "CI95_Low_ms": round(lo_v, 2), "CI95_High_ms": round(hi_v, 2)})

    # 2. คำนวณ Cliff's Delta Effect Size (\delta) และ Paired Wilcoxon p-values สำหรับคู่เปรียบเทียบสำคัญใน Abstract
    effect_pairs = [
        ("Columnar CPU (Polars) vs. Row-Store (SQLite 3)", m["SQLite_ms"].values, m["Polars_ms"].values),
        ("BigQuery Unoptimized vs. BigQuery Clustered", m["BQ_Unopt_ms"].values, m["BQ_Clust_ms"].values),
        ("DuckDB Parquet vs. DuckDB Native Columnar", m["DuckDB_Parquet_ms"].values, m["DuckDB_Native_ms"].values),
        ("DuckDB CPU vs. RTX 2050 GPU (Cold-Start)", m["DuckDB_Native_ms"].values, m["GPU_Cold_ms"].values),
        ("DuckDB CPU vs. RTX 2050 GPU (Warm In-VRAM)", m["DuckDB_Native_ms"].values, m["GPU_Warm_ms"].values),
        ("Rule-Based Router vs. Proposed 9D AST Router", m["Rule_Router_ms"].values, m["Proposed9_Router_ms"].values),
    ]
    print("\n" + "=" * 115)
    print(" [2. NON-PARAMETRIC CLIFF'S DELTA EFFECT SIZES & PAIRED WILCOXON SIGNED-RANK TESTS]")
    print("=" * 115)
    for pair_name, a1, a2 in effect_pairs:
        d_val = cliffs_delta(a1, a2)
        _, p_w = stats.wilcoxon(a1, a2)
        print(f"  • {pair_name:<48}: Cliff's delta = {d_val:>6.3f}, Wilcoxon p = {p_w:.3e}")

    # 3. CSCI Construct Validity & Monte Carlo Weight Perturbation Sensitivity (\pm 20% Uniform Perturbation across 1,000 trials)
    # ตรวจสอบความสัมพันธ์ระหว่าง CSCI กับเวลาประมวลผล CPU/Cloud และการเร่งความเร็วของ GPU (GPU Speedup Ratio)
    gpu_speedup = m["DuckDB_Native_ms"].values / np.maximum(m["GPU_Warm_ms"].values, 1e-3)
    csci_vals = m["CSCI_Score"].values
    rho_sp, p_sp = stats.spearmanr(csci_vals, gpu_speedup)
    tau_kd, p_kd = stats.kendalltau(csci_vals, gpu_speedup)

    # ทดสอบความไวของค่าน้ำหนัก CSCI (Weight Perturbation Ablation: สุ่มแกว่งน้ำหนัก \pm 20% จำนวน 1,000 รอบ)
    rng = np.random.default_rng(42)
    base_weights = np.array([1.5, 2.5, 3.0, 3.5, 4.0, 1.5])
    # ดึงคอลัมน์โครงสร้าง AST
    D_v = m["AST_Depth"].values
    J_v = m["Joins"].values
    S_v = m["Subqueries_CTEs"].values if "Subqueries_CTEs" in m.columns else np.ones(len(m))
    W_v = m["Window_Funcs"].values if "Window_Funcs" in m.columns else np.zeros(len(m))
    U_v = m["Unnest_Ops"].values if "Unnest_Ops" in m.columns else np.zeros(len(m))
    C_v = m["Case_Branches"].values if "Case_Branches" in m.columns else np.ones(len(m))
    F_mat = np.column_stack([D_v, J_v, S_v, W_v, U_v, C_v])
    csci_recomputed = F_mat @ base_weights

    rank_corrs = []
    for _ in range(1000):
        perturbed_w = base_weights * rng.uniform(0.80, 1.20, size=len(base_weights))
        csci_p = F_mat @ perturbed_w
        r_c, _ = stats.spearmanr(csci_vals, csci_p)
        rank_corrs.append(r_c)

    # 4. GPU Timer Precision (Coefficient of Variation CV% Across 5 Hardware-Timed Trials per Query)
    df_gpu["cv_warm"] = df_gpu.groupby("query_file")["gpu_kernel_ms"].transform(lambda s: (s.std(ddof=1) / max(s.mean(), 1e-6)) * 100.0)
    mean_gpu_cv = float(df_gpu.groupby("query_file")["cv_warm"].first().mean())

    print("\n" + "=" * 115)
    print(" [3. CSCI CONSTRUCT VALIDATION & GPU HARDWARE TIMER RELIABILITY]")
    print("=" * 115)
    print(f"  • CSCI vs. GPU Kernel Speedup Spearman Correlation : rho = {rho_sp:.4f} (p = {p_sp:.3e}), Kendall's tau = {tau_kd:.4f}")
    print(f"  • CSCI Weight Sensitivity (1,000x +-20% Perturb.)  : Mean Rank Stability rho = {np.mean(rank_corrs):.4f} (Min = {np.min(rank_corrs):.4f}, SD = {np.std(rank_corrs):.4f})")
    print(f"  • GPU cudaEvent Hardware Timer Repeatability (CV%) : Mean Intra-Query CV = {mean_gpu_cv:.2f}% across N=375 trials")
    print("=" * 115)

    out_ci_csv = root / "results" / "summary_tables" / f"bootstrap_ci_and_validation_sf{sf}.csv"
    pd.DataFrame(ci_rows).to_csv(out_ci_csv, index=False)

if __name__ == "__main__":
    run_validation_experiments(PROJECT_ROOT, sf=1)
