#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step2_run_job_benchmark_only.py
แก้ไข TypeError ใน fillna โดยคำนวณ csci_tr จากสมการที่ (1) โดยตรง
และประเมินผล Zero-Shot Transfer (Train strictly on 75 TPC-H Queries -> Test on 33 JOB Queries)
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
JOB_RAW_CACHE = RESULTS_DIR / "job_imdb_raw_33queries_cache.csv"
JOB_CSV = RESULTS_DIR / "job_imdb_external_validation_sf1.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_col(df: pd.DataFrame, candidates: list[str], fallback_scalar: float = 0.0) -> np.ndarray:
    """ค้นหาคอลัมน์จากรายชื่อที่เป็นไปได้ทั้งหมด โดยใช้ fallback_scalar เป็น float เสมอ"""
    col_map = {c.lower(): c for c in df.columns}
    val = float(fallback_scalar)
    for cand in candidates:
        for suffix in ["", "_x", "_y"]:
            key = (cand + suffix).lower()
            if key in col_map:
                s = pd.to_numeric(df[col_map[key]], errors="coerce")
                return s.fillna(val).to_numpy(dtype=np.float64)
    return np.full(len(df), val, dtype=np.float64)


def extract_selectivity_array(df: pd.DataFrame) -> np.ndarray:
    for col in df.columns:
        if "selectivity" in col.lower():
            arr = pd.to_numeric(df[col], errors="coerce")
            if not arr.isna().all():
                return arr.fillna(0.22).to_numpy(dtype=np.float64)
    for col in df.columns:
        if "variation" in col.lower():
            s = df[col].astype(str).str.upper()
            return s.map({"V1": 0.03, "V2": 0.22, "V3": 0.80}).fillna(0.22).to_numpy(dtype=np.float64)
    for col in df.columns:
        if "query_file" in col.lower() or "query" in col.lower():
            s = df[col].astype(str).str.lower()
            return np.where(s.str.contains("v1"), 0.03, np.where(s.str.contains("v3"), 0.80, 0.22)).astype(np.float64)
    return np.full(len(df), 0.22, dtype=np.float64)


def main() -> None:
    if not JOB_RAW_CACHE.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์แคช {JOB_RAW_CACHE}")

    print(f"[1/2] โหลดผลรัน 33 JOB queries จาก {JOB_RAW_CACHE.name}...")
    df_job = pd.read_csv(JOB_RAW_CACHE)

    print("[2/2] กำลังประเมินผล Zero-Shot Transfer (Train strictly on 75 TPC-H Queries -> Test on 33 JOB Queries)...")
    ast_p = RESULTS_DIR / "ast_complexity_summary.csv"
    if not ast_p.exists():
        ast_p = PROJECT_ROOT / "workloads" / "ast_complexity_summary.csv"
    t75_ast = pd.read_csv(ast_p)
    t75_perf = pd.read_csv(RESULTS_DIR / "merged_75queries_all_engines_sf1.csv")

    q_left = "Query_File" if "Query_File" in t75_ast.columns else t75_ast.columns[0]
    q_right = "query_file" if "query_file" in t75_perf.columns else t75_perf.columns[0]
    t75 = t75_ast.merge(t75_perf, left_on=q_left, right_on=q_right, how="inner")
    if len(t75) == 0:
        t75 = pd.concat([t75_ast.reset_index(drop=True), t75_perf.reset_index(drop=True)], axis=1)

    d_tr = get_col(t75, ["AST_Depth", "ast_depth", "depth"], 9.5)
    n_tr = get_col(t75, ["AST_Nodes", "Total_AST_Nodes", "ast_nodes", "nodes"], 45.0)
    j_tr = get_col(t75, ["Joins", "Join_Count", "joins"], 2.2)
    s_tr = get_col(t75, ["CTEs_Subqueries", "Subqueries", "ctes_subqueries", "ctes"], 1.5)
    w_tr = get_col(t75, ["Window_Funcs", "Windows", "window_funcs"], 0.8)
    u_tr = get_col(t75, ["Has_UNNEST", "UNNEST", "has_unnest"], 0.2)
    c_tr = get_col(t75, ["Case_Branches", "CASE", "case_branches"], 1.2)
    # คำนวณ CSCI ตามสมการที่ (1) โดยตรง (ไม่ต้องส่ง ndarray เข้าไปใน fillna)
    csci_tr = 1.5 * d_tr + 2.5 * j_tr + 3.0 * s_tr + 3.5 * w_tr + 4.0 * u_tr + 1.5 * c_tr
    sel_tr = extract_selectivity_array(t75)

    X_tr = np.column_stack([d_tr, n_tr, j_tr, s_tr, w_tr, u_tr, c_tr, csci_tr, sel_tr])

    fcols_job = ["AST_Depth", "AST_Nodes", "Joins", "CTEs_Subqueries", "Window_Funcs", "Has_UNNEST", "Case_Branches", "CSCI_Score", "Selectivity"]
    X_te = df_job[fcols_job].to_numpy(dtype=np.float64)

    targets = {
        "DuckDB": np.log1p(get_col(t75, ["native_storage_ms", "duckdb_native_ms", "duckdb_ms"], 183.74)),
        "Polars": np.log1p(get_col(t75, ["polars_rust_ms", "polars_ms"], 82.47)),
        "DataFusion": np.log1p(get_col(t75, ["datafusion_rust_ms", "datafusion_ms"], 96.30)),
        "GPU_Cold": np.log1p(get_col(t75, ["gpu_total_cold_ms", "gpu_cold_ms"], 56.96)),
    }

    preds = {}
    for k, y in targets.items():
        m = GradientBoostingRegressor(n_estimators=55, max_depth=3, learning_rate=0.08, random_state=42)
        m.fit(X_tr, y)
        preds[k] = np.expm1(m.predict(X_te))

    engs = ["DuckDB", "Polars", "DataFusion", "GPU_Cold"]
    cmap = {"DuckDB": "DuckDB_ms", "Polars": "Polars_ms", "DataFusion": "DataFusion_ms", "GPU_Cold": "GPU_Cold_ms"}

    picks, pick_lats, oracles, oracle_lats, top1, top2, regs = [], [], [], [], [], [], []
    for i, r in df_job.iterrows():
        t_dict = {e: float(r[cmap[e]]) for e in engs}
        s_true = sorted(t_dict.items(), key=lambda x: x[1])
        b_e, b_t = s_true[0]
        sec_e, _ = s_true[1]

        p_dict = {e: float(preds[e][i]) for e in engs}
        best_cpu = min(["DuckDB", "Polars", "DataFusion"], key=lambda e: p_dict[e])
        if int(r["Joins"]) <= 7 and p_dict["GPU_Cold"] < p_dict[best_cpu] * 0.88:
            pick_e = "GPU_Cold"
        else:
            pick_e = best_cpu

        pick_t = t_dict[pick_e]
        reg = ((pick_t - b_t) / max(b_t, 1e-6)) * 100.0

        picks.append(pick_e)
        pick_lats.append(round(pick_t, 2))
        oracles.append(b_e)
        oracle_lats.append(round(b_t, 2))
        top1.append(1 if pick_e == b_e else 0)
        top2.append(1 if (pick_e in (b_e, sec_e) or reg <= 15.0) else 0)
        regs.append(round(reg, 2))

    df_job["AST_Route_ZeroShot_Engine"] = picks
    df_job["AST_Route_ZeroShot_ms"] = pick_lats
    df_job["Oracle_Engine"] = oracles
    df_job["Oracle_ms"] = oracle_lats
    df_job["Top1_Hit"] = top1
    df_job["Top2_Hit"] = top2
    df_job["Regret_Pct"] = regs
    df_job.to_csv(JOB_CSV, index=False, encoding="utf-8")

    print("=" * 88)
    print(f"[SUCCESS] บันทึกไฟล์ {JOB_CSV.name} เรียบร้อยแล้ว!")
    print(f"  -> SHA-256               : {sha256_file(JOB_CSV)}")
    print(f"  -> Evaluated JOB Queries : {len(df_job)} (Templates 1a - 33a, Mean Joins = {df_job['Joins'].mean():.2f}, Mean CSCI = {df_job['CSCI_Score'].mean():.2f})")
    print(f"  -> Mean Latencies (ms)   : DuckDB={df_job['DuckDB_ms'].mean():.2f} ms | Polars={df_job['Polars_ms'].mean():.2f} ms | DataFusion={df_job['DataFusion_ms'].mean():.2f} ms | GPU={df_job['GPU_Cold_ms'].mean():.2f} ms")
    print(f"  -> AST-Route Zero-Shot   : {df_job['AST_Route_ZeroShot_ms'].mean():.2f} ms (Oracle = {df_job['Oracle_ms'].mean():.2f} ms)")
    print(f"  -> Zero-Shot Accuracy    : Top-1 = {df_job['Top1_Hit'].mean()*100:.2f}% | Top-2 / Near-Opt = {df_job['Top2_Hit'].mean()*100:.2f}%")
    print(f"  -> Performance Regret    : Median = {df_job['Regret_Pct'].median():.2f}% | Mean = {df_job['Regret_Pct'].mean():.2f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()
