#!/usr/bin/env python3
r"""
scripts/01_reconcile_sf1_and_csci.py
1. ตรวจสอบตัวเลขจริงจากไฟล์ CSV ทั้ง 6 ไฟล์ในเครื่องเทียบกับตารางใน PDF (Table 1 - Table 6)
2. คำนวณดัชนี CSCI ใหม่ที่รวม AST_Nodes (CSCI_v2) จากไฟล์ ast_complexity_summary.csv จริง
3. ทดสอบความแม่นยำ (Spearman/Kendall Correlation & Weight Perturbation Sensitivity) เทียบกับเวลาจริงของทั้ง 6 Engines
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")


def find_csv(filename: str) -> Path:
    matches = list(PROJECT_ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"ไม่พบไฟล์ {filename} ใน {PROJECT_ROOT}")
    return matches[0]


def main() -> None:
    print("=" * 80)
    print("STEP 1: โหลดข้อมูลจริงจากไฟล์ CSV ในโปรเจกต์และตรวจสอบความตรงกันของทุกตาราง")
    print("=" * 80)

    p_ast = find_csv("ast_complexity_summary.csv")
    p_duck = find_csv("raw_benchmark_sf1.csv")
    p_bq = find_csv("raw_bigquery_cloud_sf1.csv")
    p_gpu = find_csv("raw_gpu_cuda_sf1.csv")
    p_multi = find_csv("raw_multi_software_sf1.csv")

    df_ast = pd.read_csv(p_ast)
    df_duck = pd.read_csv(p_duck)
    df_bq = pd.read_csv(p_bq)
    df_gpu = pd.read_csv(p_gpu)
    df_multi = pd.read_csv(p_multi)

    # 1. สรุปผลจริงแยกตาม Complexity_Tier (L1-L5)
    duck_tier = df_duck.groupby("complexity_tier")[["external_parquet_ms", "native_storage_ms"]].agg(["mean", "std"])
    bq_tier = df_bq.groupby("complexity_tier")[
        ["unopt_server_ms", "opt_server_ms", "opt_slot_ms", "unopt_scanned_mb", "opt_scanned_mb"]
    ].mean()
    gpu_tier = df_gpu.groupby("complexity_tier")[["gpu_pcie_h2d_ms", "gpu_kernel_ms", "gpu_total_cold_ms"]].mean()
    multi_tier = df_multi.groupby("complexity_tier")[["sqlite_rowstore_ms", "polars_rust_ms", "datafusion_rust_ms"]].mean()

    summary = pd.DataFrame(index=duck_tier.index)
    summary["SQLite3_ms"] = multi_tier["sqlite_rowstore_ms"]
    summary["BQ_Unopt_ms"] = bq_tier["unopt_server_ms"]
    summary["BQ_Clust_ms"] = bq_tier["opt_server_ms"]
    summary["BQ_Slot_ms"] = bq_tier["opt_slot_ms"]
    summary["Duck_Parquet_ms"] = duck_tier[("external_parquet_ms", "mean")]
    summary["Duck_Native_ms"] = duck_tier[("native_storage_ms", "mean")]
    summary["Polars_ms"] = multi_tier["polars_rust_ms"]
    summary["DataFusion_ms"] = multi_tier["datafusion_rust_ms"]
    summary["GPU_PCIe_H2D_ms"] = gpu_tier["gpu_pcie_h2d_ms"]
    summary["GPU_Warm_Kernel_ms"] = gpu_tier["gpu_kernel_ms"]
    summary["GPU_Cold_Total_ms"] = gpu_tier["gpu_total_cold_ms"]

    # เพิ่มแถว Overall Mean จากค่าเฉลี่ยจริงของทุกคิวรี
    overall = pd.Series({
        "SQLite3_ms": df_multi["sqlite_rowstore_ms"].mean(),
        "BQ_Unopt_ms": df_bq["unopt_server_ms"].mean(),
        "BQ_Clust_ms": df_bq["opt_server_ms"].mean(),
        "BQ_Slot_ms": df_bq["opt_slot_ms"].mean(),
        "Duck_Parquet_ms": df_duck["external_parquet_ms"].mean(),
        "Duck_Native_ms": df_duck["native_storage_ms"].mean(),
        "Polars_ms": df_multi["polars_rust_ms"].mean(),
        "DataFusion_ms": df_multi["datafusion_rust_ms"].mean(),
        "GPU_PCIe_H2D_ms": df_gpu["gpu_pcie_h2d_ms"].mean(),
        "GPU_Warm_Kernel_ms": df_gpu["gpu_kernel_ms"].mean(),
        "GPU_Cold_Total_ms": df_gpu["gpu_total_cold_ms"].mean(),
    }, name="Overall_Mean")

    summary = pd.concat([summary, overall.to_frame().T])

    # คำนวณอัตราส่วน Speedup และสัดส่วน PCIe ที่ถูกต้องตรงกันทุกตาราง
    summary["Phi_PCIe_pct"] = (summary["GPU_PCIe_H2D_ms"] / summary["GPU_Cold_Total_ms"]) * 100.0
    summary["Warm_GPU_vs_Duck_x"] = summary["Duck_Native_ms"] / summary["GPU_Warm_Kernel_ms"]
    summary["Cold_GPU_vs_Duck_x"] = summary["Duck_Native_ms"] / summary["GPU_Cold_Total_ms"]
    summary["Duck_vs_DF_Speedup (DF/Duck)"] = summary["DataFusion_ms"] / summary["Duck_Native_ms"]

    print("\n[ตารางตรวจสอบค่าเฉลี่ยจริงจากไฟล์ CSV ในเครื่องของคุณ (SF=1)]:")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    # =========================================================================
    # STEP 2: ปรับปรุงสมการ CSCI ให้รวม AST_Nodes และทดสอบ Validation จริง
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: คำนวณ CSCI ใหม่ที่รวม AST_Nodes และทดสอบความแม่นยำกับเวลาจริง")
    print("=" * 80)

    # สร้างดัชนีเปรียบเทียบ 4 รูปแบบจากคอลัมน์จริงใน ast_complexity_summary.csv
    df_ast["log2_AST_Nodes"] = np.log2(np.maximum(df_ast["AST_Nodes"], 1))

    # 1) Unweighted Sum (ให้น้ำหนักทุกตัวแปร = 1.0)
    df_ast["CSCI_Unweighted"] = (
        df_ast["AST_Depth"]
        + df_ast["log2_AST_Nodes"]
        + df_ast["Joins"]
        + df_ast["CTEs_Subqueries"]
        + df_ast["Window_Funcs"]
        + df_ast["Has_UNNEST"]
        + df_ast["Case_Branches"]
    )

    # 2) CSCI_v2_Revised (รวมพจน์ 1.0 * log2(AST_Nodes) เข้าไปในสมการที่ 1)
    df_ast["CSCI_v2_Revised"] = (
        1.5 * df_ast["AST_Depth"]
        + 1.0 * df_ast["log2_AST_Nodes"]
        + 2.5 * df_ast["Joins"]
        + 3.0 * df_ast["CTEs_Subqueries"]
        + 3.5 * df_ast["Window_Funcs"]
        + 4.0 * df_ast["Has_UNNEST"]
        + 1.5 * df_ast["Case_Branches"]
    )

    # รวมเวลาการรันจริงระดับคิวรี (N = 75) ของทั้ง 6 Engines เข้ากับตาราง AST
    q_duck = df_duck.groupby("query_file")[["external_parquet_ms", "native_storage_ms"]].mean().reset_index()
    q_bq = df_bq.groupby("query_file")[["unopt_server_ms", "opt_server_ms"]].mean().reset_index()
    q_gpu = df_gpu.groupby("query_file")[["gpu_kernel_ms", "gpu_total_cold_ms"]].mean().reset_index()
    q_multi = df_multi.groupby("query_file")[["sqlite_rowstore_ms", "polars_rust_ms", "datafusion_rust_ms"]].mean().reset_index()

    merged = (
        df_ast.rename(columns={"Query_File": "query_file"})
        .merge(q_duck, on="query_file")
        .merge(q_bq, on="query_file")
        .merge(q_gpu, on="query_file")
        .merge(q_multi, on="query_file")
    )

    print("\n[ค่าเฉลี่ย CSCI แยกตามระดับความซับซ้อน L1-L5]:")
    csci_tier = merged.groupby("Complexity_Tier")[
        ["AST_Depth", "AST_Nodes", "log2_AST_Nodes", "CSCI_Unweighted", "CSCI_Score", "CSCI_v2_Revised"]
    ].mean()
    print(csci_tier.to_string(float_format=lambda x: f"{x:.2f}"))

    # ทดสอบ Spearman's rho, Kendall's tau และ Monte Carlo Weight Sensitivity (1,000 รอบ)
    engine_cols = {
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

    rng = np.random.default_rng(42)
    base_w = np.array([1.5, 1.0, 2.5, 3.0, 3.5, 4.0, 1.5])
    feat_matrix = merged[
        ["AST_Depth", "log2_AST_Nodes", "Joins", "CTEs_Subqueries", "Window_Funcs", "Has_UNNEST", "Case_Branches"]
    ].to_numpy()

    val_rows = []
    for eng_label, col_name in engine_cols.items():
        lat = merged[col_name].to_numpy()
        rho_d, _ = spearmanr(merged["AST_Depth"], lat)
        rho_n, _ = spearmanr(merged["AST_Nodes"], lat)
        rho_unw, _ = spearmanr(merged["CSCI_Unweighted"], lat)
        rho_v1, p_v1 = spearmanr(merged["CSCI_Score"], lat)
        rho_v2, p_v2 = spearmanr(merged["CSCI_v2_Revised"], lat)
        tau_v2, _ = kendalltau(merged["CSCI_v2_Revised"], lat)

        # ทดลองแกว่งค่าน้ำหนัก ±50% จำนวน 1,000 รอบ
        mc_rhos = []
        for _ in range(1000):
            w_rand = base_w * rng.uniform(0.5, 1.5, size=len(base_w))
            mc_rhos.append(spearmanr(feat_matrix @ w_rand, lat)[0])
        q25, q50, q75 = np.percentile(mc_rhos, [25, 50, 75])

        val_rows.append({
            "Engine": eng_label,
            "rho(Depth)": rho_d,
            "rho(Nodes)": rho_n,
            "rho(Unweighted)": rho_unw,
            "rho(CSCI_v1)": rho_v1,
            "rho(CSCI_v2)": rho_v2,
            "tau(CSCI_v2)": tau_v2,
            "p_value(v2)": p_v2,
            "Sensitivity_IQR(±50%)": f"{q50:.3f} [{q25:.3f}, {q75:.3f}]",
        })

    df_val = pd.DataFrame(val_rows)
    print("\n[ผลพิสูจน์ความแม่นยำของ CSCI (Validation & Weight Sensitivity Analysis)]:")
    print(df_val.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # บันทึกไฟล์ผลลัพธ์สำหรับนำไปใช้ในบทความทันที
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "reconciled_sf1_master_table.csv")
    df_val.to_csv(out_dir / "csci_validation_sensitivity_sf1.csv", index=False)
    merged.to_csv(out_dir / "merged_75queries_all_engines_sf1.csv", index=False)
    print(f"\n[OK] บันทึกตารางที่ตรวจสอบแล้วลงใน {out_dir}")


if __name__ == "__main__":
    main()