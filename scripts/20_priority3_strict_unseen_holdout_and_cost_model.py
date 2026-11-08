# ไฟล์: scripts/20_priority3_strict_unseen_holdout_and_cost_model.py
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def set_academic_serif_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.85,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

PALETTE = {
    "slate_grey": "#7F8C8D",
    "deep_blue":  "#4C72B0",
    "sage_green": "#55A868",
    "amber_gold": "#DD8452",
    "royal_purp": "#8172B2",
    "crimson":    "#C44E52",
    "teal_cyan":  "#64B5CD",
    "dark_navy":  "#1B263B"
}

def sha256_full(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.maximum(np.asarray(y_true, dtype=float), 1e-4)
    yp = np.maximum(np.asarray(y_pred, dtype=float), 1e-4)
    return np.maximum(yp / yt, yt / yp)

def load_full_telemetry(root: Path, sf: int = 1) -> pd.DataFrame:
    df_route = pd.read_csv(root / "results" / "raw_logs" / f"raw_query_routing_predictions_sf{sf}.csv")
    df_duck  = pd.read_csv(root / "results" / "raw_logs" / f"raw_benchmark_sf{sf}.csv")
    df_bq    = pd.read_csv(root / "results" / "raw_logs" / f"raw_bigquery_cloud_sf{sf}.csv")
    df_gpu   = pd.read_csv(root / "results" / "raw_logs" / f"raw_gpu_cuda_sf{sf}.csv")
    df_multi = pd.read_csv(root / "results" / "raw_logs" / f"raw_multi_software_sf{sf}.csv")
    df_hw    = pd.read_csv(root / "results" / "raw_logs" / f"raw_hardware_energy_sf{sf}.csv")

    dk_q = df_duck.groupby("query_file", as_index=False).agg(DuckDB_ms=("native_storage_ms", "mean"))
    bq_q = df_bq.groupby("query_file", as_index=False).agg(BigQuery_ms=("opt_server_ms", "mean"), BigQuery_Slot_ms=("opt_slot_ms", "mean"))
    gp_q = df_gpu.groupby("query_file", as_index=False).agg(GPU_Cold_ms=("gpu_total_cold_ms", "mean"), GPU_Warm_ms=("gpu_kernel_ms", "mean"))
    mu_q = df_mul = df_multi.groupby("query_file", as_index=False).agg(Polars_ms=("polars_rust_ms", "mean"), DataFusion_ms=("datafusion_rust_ms", "mean"))
    hw_q = df_hw.groupby("query_file", as_index=False).agg(
        DuckDB_J=("duckdb_energy_j", "mean"),
        Polars_J=("polars_energy_j", "mean"),
        DataFusion_J=("datafusion_energy_j", "mean"),
        GPU_Cold_J=("gpu_cold_energy_j", "mean"),
        Polars_RAM_GB=("polars_ram_gb", "mean"),
        GPU_VRAM_GB=("gpu_vram_gb", "mean")
    )

    m = (
        df_route.merge(dk_q, on="query_file")
        .merge(bq_q, on="query_file")
        .merge(gp_q, on="query_file")
        .merge(mu_q, on="query_file")
        .merge(hw_q, on="query_file")
    )
    m["BigQuery_J"] = np.round((m["BigQuery_Slot_ms"] / 1000.0) * 3.5, 3)
    return m

def evaluate_routing_subset(df_sub: pd.DataFrame, pred_engines: list, protocol_name: str, model_name: str) -> dict:
    eng_cols = {
        "DuckDB": "DuckDB_ms",
        "Polars": "Polars_ms",
        "DataFusion": "DataFusion_ms",
        "GPU_Cold": "GPU_Cold_ms",
        "BigQuery": "BigQuery_ms"
    }
    eng_j_cols = {
        "DuckDB": "DuckDB_J",
        "Polars": "Polars_J",
        "DataFusion": "DataFusion_J",
        "GPU_Cold": "GPU_Cold_J",
        "BigQuery": "BigQuery_J"
    }
    lat_mat = df_sub[list(eng_cols.values())].values
    eng_names = list(eng_cols.keys())
    oracle_idx = np.argmin(lat_mat, axis=1)
    oracle_engines = [eng_names[i] for i in oracle_idx]
    oracle_lat = np.min(lat_mat, axis=1)

    chosen_lat = np.array([df_sub.iloc[i][eng_cols[pred_engines[i]]] for i in range(len(df_sub))])
    chosen_j   = np.array([df_sub.iloc[i][eng_j_cols[pred_engines[i]]] for i in range(len(df_sub))])
    regrets    = ((chosen_lat - oracle_lat) / np.maximum(oracle_lat, 1e-6)) * 100.0

    top1_acc   = np.mean([pred_engines[i] == oracle_engines[i] for i in range(len(df_sub))]) * 100.0
    near10_hit = np.mean(chosen_lat <= oracle_lat * 1.10) * 100.0
    zero_reg   = np.mean(regrets < 1.0) * 100.0

    return {
        "Evaluation_Protocol": protocol_name,
        "Routing_Policy": model_name,
        "Queries_N": len(df_sub),
        "Top1_Accuracy_Pct": round(float(top1_acc), 2),
        "Near_Optimal_10Pct_Hit": round(float(near10_hit), 2),
        "Zero_Regret_Hit_Pct": round(float(zero_reg), 2),
        "Mean_Latency_ms": round(float(np.mean(chosen_lat)), 2),
        "P95_Latency_ms": round(float(np.percentile(chosen_lat, 95)), 2),
        "Mean_Regret_Pct": round(float(np.mean(regrets)), 2),
        "Median_P50_Regret_Pct": round(float(np.median(regrets)), 2),
        "P90_Regret_Pct": round(float(np.percentile(regrets, 90)), 2),
        "Mean_Energy_J": round(float(np.mean(chosen_j)), 3)
    }

def run_priority3_experiments(root: Path, sf: int = 1):
    df = load_full_telemetry(root, sf=sf)
    feature_cols = [
        "AST_Depth", "Total_AST_Nodes", "Joins", "Subqueries_CTEs",
        "Window_Funcs", "Unnest_Ops", "Case_Branches", "CSCI_Score", "Selectivity"
    ]
    X_all = df[feature_cols].values.astype(float)
    seeds = df["Seed_Short"].astype(str).values

    eng_cols = ["DuckDB_ms", "Polars_ms", "DataFusion_ms", "GPU_Cold_ms", "BigQuery_ms"]
    eng_names = ["DuckDB", "Polars", "DataFusion", "GPU_Cold", "BigQuery"]
    lat_mat = df[eng_cols].values

    # =========================================================================
    # 1. COST & RESOURCE PREDICTION MODEL EVALUATION (R^2, MAPE %, Q-ERROR P50/P90)
    # =========================================================================
    target_definitions = [
        ("DuckDB Latency (ms)",             df["DuckDB_ms"].values),
        ("Polars Latency (ms)",             df["Polars_ms"].values),
        ("DataFusion Latency (ms)",         df["DataFusion_ms"].values),
        ("CUDA GPU Cold Latency (ms)",      df["GPU_Cold_ms"].values),
        ("BigQuery Clustered Latency (ms)", df["BigQuery_ms"].values),
        ("Per-Query Energy Footprint (J)",  df["Polars_J"].values),
        ("Peak Working-Set Memory (GB)",    df["Polars_RAM_GB"].values),
    ]
    Y_targets = np.column_stack([t[1] for t in target_definitions])
    Y_oof_pred = np.zeros_like(Y_targets)

    logo = LeaveOneGroupOut()
    for tr_idx, te_idx in logo.split(X_all, groups=seeds):
        reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=55, max_depth=3, learning_rate=0.08, random_state=42))
        reg.fit(X_all[tr_idx], np.log1p(Y_targets[tr_idx]))
        Y_oof_pred[te_idx] = np.expm1(reg.predict(X_all[te_idx]))

    cost_eval_rows = []
    for col_i, (t_name, y_true) in enumerate(target_definitions):
        y_pred = np.maximum(Y_oof_pred[:, col_i], 1e-3)
        r2_log = r2_score(np.log1p(y_true), np.log1p(y_pred))
        mape_v = mean_absolute_percentage_error(y_true, y_pred) * 100.0
        q_errs = compute_q_error(y_true, y_pred)
        cost_eval_rows.append({
            "Predicted_Cost_or_Resource_Target": t_name,
            "Log_R2_Score": round(float(r2_log), 3),
            "MAPE_Pct": round(float(mape_v), 2),
            "Median_Q_Error_P50": round(float(np.median(q_errs)), 3),
            "Tail_Q_Error_P90": round(float(np.percentile(q_errs, 90)), 3),
            "Mean_True": round(float(np.mean(y_true)), 2),
            "Mean_Predicted": round(float(np.mean(y_pred)), 2)
        })
    df_cost_eval = pd.DataFrame(cost_eval_rows)

    print("=" * 125)
    print(" [PRIORITY 3 - PART 1: COST & RESOURCE PREDICTION MODEL ACCURACY (R^2, MAPE, AND Q-ERROR ACROSS 25-SEED LOSO-CV)]")
    print("=" * 125)
    print(df_cost_eval.to_string(index=False))

    # =========================================================================
    # 2. STRICT UNSEEN HOLDOUT EVALUATION (20 DEV SEEDS -> 5 LOCKED UNSEEN SEEDS)
    # เลือก Seed ตัวที่ 5 ของแต่ละ Complexity Tier (L1..L5) เป็น Locked Unseen Seeds (5 Seeds = 15 Queries)
    # =========================================================================
    unseen_seed_list = []
    for tier_k, grp in df.groupby("Complexity_Tier"):
        tier_unique_seeds = sorted(grp["Seed_Short"].astype(str).unique())
        # เลือก Seed ตัวสุดท้ายของแต่ละเทียร์ (เช่น s5 / S5) เป็น Locked Unseen Seed
        unseen_seed_list.append(tier_unique_seeds[-1])

    print(f"\n  -> [UNSEEN HOLDOUT SPLIT] Locked 5 Unseen Test Seeds (15 Queries): {unseen_seed_list}")
    unseen_mask = df["Seed_Short"].astype(str).isin(unseen_seed_list).values
    dev_mask    = ~unseen_mask

    df_dev    = df.iloc[np.where(dev_mask)[0]].reset_index(drop=True)
    df_unseen = df.iloc[np.where(unseen_mask)[0]].reset_index(drop=True)

    # Train AST-Route Cost-Regression Model ONLY on the 20 Development Seeds (60 Queries)
    reg_dev = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=55, max_depth=3, learning_rate=0.08, random_state=42))
    reg_dev.fit(X_all[dev_mask], np.log1p(lat_mat[dev_mask]))

    # Predict on the 5 Completely Unseen Seeds (15 Queries across V1-V3)
    pred_log_unseen = reg_dev.predict(X_all[unseen_mask])
    pred_eng_unseen = [eng_names[i] for i in np.argmin(pred_log_unseen, axis=1)]

    # Train RF Classifier ONLY on the 20 Development Seeds
    y_oracle_all = np.array([eng_names[i] for i in np.argmin(lat_mat, axis=1)])
    clf_dev = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)
    clf_dev.fit(X_all[dev_mask], y_oracle_all[dev_mask])
    pred_rf_unseen = list(clf_dev.predict(X_all[unseen_mask]))

    holdout_rows = []
    p_label = "Strict Unseen Holdout (5 Locked Seeds, N=15)"
    holdout_rows.append(evaluate_routing_subset(df_unseen, ["DuckDB"]*len(df_unseen), p_label, "1. Always DuckDB (CPU DBMS)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, ["GPU_Cold"]*len(df_unseen), p_label, "2. Always GPU (CUDA Cold-Start)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, ["BigQuery"]*len(df_unseen), p_label, "3. Always BigQuery (Cloud OLAP)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, ["Polars"]*len(df_unseen), p_label, "   Always Polars (Rust Arrow)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, list(df_unseen["Rule_Based_Pred"].values), p_label, "4. Rule-Based Routing (Heuristic)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, pred_rf_unseen, p_label, "5a. Proposed AST-Route RF (9D, Unseen Test)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, pred_eng_unseen, p_label, "5b. Proposed AST-Route Cost-Reg (9D, Unseen Test)"))
    holdout_rows.append(evaluate_routing_subset(df_unseen, list(y_oracle_all[unseen_mask]), p_label, "   Oracle Ideal Router (Lower Bound)"))

    p_full = "Full 25-Seed LOSO-CV (25 Disjoint Seeds, N=75)"
    holdout_rows.append(evaluate_routing_subset(df, ["DuckDB"]*len(df), p_full, "1. Always DuckDB (CPU DBMS)"))
    holdout_rows.append(evaluate_routing_subset(df, ["GPU_Cold"]*len(df), p_full, "2. Always GPU (CUDA Cold-Start)"))
    holdout_rows.append(evaluate_routing_subset(df, list(df["Rule_Based_Pred"].values), p_full, "4. Rule-Based Routing (Heuristic)"))
    holdout_rows.append(evaluate_routing_subset(df, list(df["Proposed9_GBR_Pred"].values), p_full, "5b. Proposed AST-Route Cost-Reg (9D, LOSO-CV)"))
    holdout_rows.append(evaluate_routing_subset(df, list(y_oracle_all), p_full, "   Oracle Ideal Router (Lower Bound)"))

    df_holdout = pd.DataFrame(holdout_rows)
    print("\n" + "=" * 125)
    print(" [PRIORITY 3 - PART 2: STRICT UNSEEN SEED HOLDOUT (5 LOCKED SEEDS, N=15) & REGRET DISTRIBUTION ANALYSIS]")
    print("=" * 125)
    print(df_holdout.to_string(index=False))

    # =========================================================================
    # 3. TIER-BY-TIER BREAKDOWN: PROPOSED AST-ROUTE vs. RULE-BASED ROUTING (L1-L5)
    # =========================================================================
    tier_order = ["L1_scan_filter", "L2_multi_joins", "L3_nested_cte", "L4_window_analytical", "L5_array_unnest"]
    tier_labels = {
        "L1_scan_filter": "L1: Scan & Filter",
        "L2_multi_joins": "L2: Multi-Way Joins",
        "L3_nested_cte": "L3: Nested CTEs",
        "L4_window_analytical": "L4: Window Func.",
        "L5_array_unnest": "L5: Array / UNNEST"
    }
    tier_comp_rows = []
    for tk in tier_order:
        sub_t = df[df["Complexity_Tier"] == tk].reset_index(drop=True)
        r_rule = evaluate_routing_subset(sub_t, list(sub_t["Rule_Based_Pred"].values), tk, "Rule-Based")
        r_prop = evaluate_routing_subset(sub_t, list(sub_t["Proposed9_GBR_Pred"].values), tk, "AST-Route (9D)")
        r_orac = evaluate_routing_subset(sub_t, list(sub_t["Oracle_Engine"].values), tk, "Oracle")
        tier_comp_rows.append({
            "Complexity_Tier": tier_labels[tk],
            "Rule_Based_Latency_ms": r_rule["Mean_Latency_ms"],
            "AST_Route_Latency_ms": r_prop["Mean_Latency_ms"],
            "Oracle_Latency_ms": r_orac["Mean_Latency_ms"],
            "Rule_Based_Regret_Pct": r_rule["Mean_Regret_Pct"],
            "AST_Route_Regret_Pct": r_prop["Mean_Regret_Pct"],
            "Rule_Based_Acc_Pct": r_rule["Top1_Accuracy_Pct"],
            "AST_Route_Acc_Pct": r_prop["Top1_Accuracy_Pct"],
            "AST_Route_Speedup_vs_Rule": f"{r_rule['Mean_Latency_ms']/r_prop['Mean_Latency_ms']:.2f}x"
        })
    df_tier_comp = pd.DataFrame(tier_comp_rows)

    print("\n" + "=" * 125)
    print(" [PRIORITY 3 - PART 3: TIER-BY-TIER HEAD-TO-HEAD BREAKDOWN (AST-ROUTE vs. RULE-BASED ROUTING ACROSS L1-L5)]")
    print("=" * 125)
    print(df_tier_comp.to_string(index=False))
    print("=" * 125)

    out_p3_csv = root / "results" / "summary_tables" / f"priority3_unseen_holdout_and_cost_model_sf{sf}.csv"
    df_holdout.to_csv(out_p3_csv, index=False)

    # วาดกราฟ 300 DPI สไตล์ Figure 3 (3-Panel: (a) Strict Unseen Seeds Holdout, (b) Tier-by-Tier vs Rule-Based, (c) Cost Model Q-Error)
    set_academic_serif_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=300)

    sub_u = df_holdout[df_holdout["Evaluation_Protocol"].str.startswith("Strict Unseen")].iloc[[0, 3, 4, 1, 5, 6, 7]]
    x_u = np.arange(len(sub_u))
    lbl_u = ["Always\nDuckDB", "Always\nPolars", "Rule-Based\nRouting", "Always\nGPU", "AST-Route\nRF (9D)", "AST-Route\nGBR (9D)", "Oracle\nIdeal"]
    cols_u = [PALETTE["slate_grey"], PALETTE["amber_gold"], PALETTE["royal_purp"], PALETTE["crimson"], PALETTE["teal_cyan"], PALETTE["sage_green"], PALETTE["dark_navy"]]
    bars_u = ax1.bar(x_u, sub_u["Mean_Latency_ms"], width=0.56, color=cols_u, edgecolor="black", lw=0.75)
    ax1.set_ylabel("Unseen Holdout Mean Latency (ms)")
    ax1.set_title("(a) Strict Unseen Seed Holdout (5 Seeds, N=15)")
    ax1.set_xticks(x_u); ax1.set_xticklabels(lbl_u, fontsize=7.8)
    ax1.set_ylim(0, max(sub_u["Mean_Latency_ms"]) * 1.22)
    ax1.grid(axis="y", linestyle="--")
    for b, reg_v in zip(bars_u, sub_u["Mean_Regret_Pct"]):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 3.0, f"{h:.1f}ms\n(+{reg_v:.1f}%)", ha="center", va="bottom", fontsize=7.0)

    x_t = np.arange(5)
    w_t = 0.27
    xt_t = ["L1\nScan/Filter", "L2\nMulti-Join", "L3\nNested CTE", "L4\nWindow", "L5\nUNNEST"]
    ax2.bar(x_t - w_t, df_tier_comp["Rule_Based_Latency_ms"], w_t, label="Rule-Based Routing (Heuristic)", color=PALETTE["royal_purp"], edgecolor="black", lw=0.75)
    ax2.bar(x_t,       df_tier_comp["AST_Route_Latency_ms"],  w_t, label="Proposed AST-Route (9D Cost-Reg)", color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax2.bar(x_t + w_t, df_tier_comp["Oracle_Latency_ms"],     w_t, label="Oracle Ideal Lower Bound", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    ax2.set_ylabel("Mean Execution Latency (ms)")
    ax2.set_title("(b) Tier-by-Tier Latency: AST-Route vs. Rule-Based")
    ax2.set_xticks(x_t); ax2.set_xticklabels(xt_t, fontsize=8.2)
    ax2.set_ylim(0, max(df_tier_comp["Rule_Based_Latency_ms"]) * 1.25)
    ax2.grid(axis="y", linestyle="--")
    ax2.legend(loc="upper left", frameon=True, fontsize=7.6)

    sub_c = df_cost_eval.iloc[[1, 2, 3, 4, 5, 6]]
    x_c = np.arange(len(sub_c))
    w_c = 0.34
    lbl_c = ["Polars\nLatency", "DataFusion\nLatency", "GPU Cold\nLatency", "BigQuery\nLatency", "Energy\n(Joules)", "Memory\n(RAM GB)"]
    b_q50 = ax3.bar(x_c - w_c/2, sub_c["Median_Q_Error_P50"], w_c, label="Median Q-Error (P50, Ideal = 1.00)", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    b_q90 = ax3.bar(x_c + w_c/2, sub_c["Tail_Q_Error_P90"],   w_c, label="90th Percentile Q-Error (P90)", color=PALETTE["amber_gold"], edgecolor="black", lw=0.75)
    ax3.axhline(1.0, color="black", linestyle=":", lw=1.1)
    ax3.set_ylabel("Multiplicative Q-Error (max(y_hat/y, y/y_hat))")
    ax3.set_title("(c) Cost & Resource Predictor Q-Error (LOSO-CV)")
    ax3.set_xticks(x_c); ax3.set_xticklabels(lbl_c, fontsize=7.8)
    ax3.set_ylim(0.8, max(sub_c["Tail_Q_Error_P90"]) * 1.18)
    ax3.grid(axis="y", linestyle="--")
    ax3.legend(loc="upper right", frameon=True, fontsize=7.6)
    for rect in b_q50:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2, h + 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=6.9)
    for rect in b_q90:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2, h + 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=6.9)

    plt.tight_layout(w_pad=1.8)
    pdf_name = f"fig_priority3_unseen_and_cost_qerror_sf{sf}_300dpi.pdf"
    fig.savefig(root / "results" / "figures_300dpi" / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(root / "manuscript" / "figures" / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    h_p3 = sha256_full(out_p3_csv)
    print(f"\n[OK] Saved Priority 3 Unseen Holdout & Q-Error Figure (300 DPI): {root / 'manuscript/figures' / pdf_name}")
    print(f"  • priority3_unseen_holdout_and_cost_model_sf1.csv -> SHA-256: {h_p3[:24]}...")

if __name__ == "__main__":
    run_priority3_experiments(PROJECT_ROOT, sf=1)
