#!/usr/bin/env python3
r"""
scripts/17_summarize_multiscale_scalability.py
รวบรวมผลการทดลองจริงจาก multiscale_telemetry_sf1.csv, sf10.csv, และ sf100.csv
1. สรุปตารางเปรียบเทียบทั้ง DuckDB_Native และ DuckDB_Parquet ข้ามสเกล (1GB, 10GB, 100GB)
2. แยกวิเคราะห์ตามระดับความซับซ้อน L1-L5 ที่สเกล SF=1, SF=10, SF=100
3. สร้างไฟล์ตาราง LaTeX (manuscript/table_multiscale_scalability.tex) อัตโนมัติ
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"


def summarize_multiscale() -> None:
    scales = [1, 10, 100]
    all_dfs = []
    summary_rows = []

    for sf in scales:
        csv_path = RESULTS_DIR / f"multiscale_telemetry_sf{sf}.csv"
        if not csv_path.exists():
            print(f"[WARN] ไม่พบไฟล์ {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        all_dfs.append(df)

        for eng in ["DuckDB_Native", "DuckDB_Parquet"]:
            sub = df[df["engine"] == eng]
            if sub.empty:
                continue

            total_time_s = float(sub["t_total_ms"].sum()) / 1000.0
            mean_ms = float(sub["t_total_ms"].mean())
            med_ms = float(sub["t_total_ms"].median())
            p95_ms = float(np.percentile(sub["t_total_ms"], 95))
            max_ram_gb = float(sub["peak_ram_gb"].max())
            mean_ram_gb = float(sub["peak_ram_gb"].mean())
            max_spill_gb = float(sub["disk_spill_gb"].max()) if "disk_spill_gb" in sub.columns else 0.0
            spill_queries = int((sub["disk_spill_gb"] > 0.01).sum()) if "disk_spill_gb" in sub.columns else 0
            oom_count = int(sub["oom_event"].sum())

            summary_rows.append({
                "Scale_Factor": f"SF={sf} ({sf} GB)",
                "Engine_Mode": eng,
                "Total_75Q_Time_s": round(total_time_s, 2),
                "Mean_Latency_ms": round(mean_ms, 2),
                "Median_Latency_ms": round(med_ms, 2),
                "P95_Latency_ms": round(p95_ms, 2),
                "Mean_RAM_GB": round(mean_ram_gb, 2),
                "Peak_RAM_GB": round(max_ram_gb, 2),
                "Peak_Disk_Spill_GB": round(max_spill_gb, 2),
                "Spilled_Queries": spill_queries,
                "OOM_Limit_Events": oom_count,
            })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 105)
    print("MULTISCALE EMPIRICAL SCALABILITY & MEMORY/SPILL SUMMARY (SF=1, SF=10, SF=100)")
    print("=" * 105)
    print(df_summary.to_string(index=False))

    out_csv = RESULTS_DIR / "multiscale_scalability_summary_table.csv"
    df_summary.to_csv(out_csv, index=False)

    # สรุปแยกตาม Complexity Tier (L1-L5) ข้ามสเกล SF=1, SF=10, SF=100
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        nat_all = df_all[df_all["engine"] == "DuckDB_Native"]
        tier_pivot = nat_all.pivot_table(
            index="complexity_tier",
            columns="scale_factor",
            values=["t_total_ms", "peak_ram_gb", "disk_spill_gb"],
            aggfunc={"t_total_ms": "mean", "peak_ram_gb": "max", "disk_spill_gb": "max"},
        )
        print("\n" + "=" * 105)
        print("TIER-BY-TIER SCALABILITY BREAKDOWN (DuckDB Native across SF=1, SF=10, SF=100)")
        print("=" * 105)
        print(tier_pivot.to_string(float_format=lambda x: f"{x:.2f}"))
        tier_pivot.to_csv(RESULTS_DIR / "multiscale_tier_breakdown.csv")

    # สร้างไฟล์ LaTeX Table อัตโนมัติ
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    tex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Multi-scale empirical scalability, memory footprint, and out-of-core disk spilling evaluation across TPC-H $SF=1$ GB, $SF=10$ GB, and $SF=100$ GB ($N=75$ queries per scale factor).}",
        r"\label{tab:multiscale_scalability}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"\textbf{Scale Factor} & \textbf{Storage Mode} & \textbf{Total (s)} & \textbf{Mean (ms)} & \textbf{Median (ms)} & \textbf{P95 (ms)} & \textbf{Peak RAM (GB)} & \textbf{Peak Spill (GB)} & \textbf{Spill Qs} & \textbf{OOM/Limit} \\",
        r"\midrule",
    ]
    for _, r in df_summary.iterrows():
        sf_lbl = str(r["Scale_Factor"])
        eng_lbl = str(r["Engine_Mode"]).replace("_", r"\_")
        tex_lines.append(
            f"{sf_lbl} & {eng_lbl} & {r['Total_75Q_Time_s']:.2f} & {r['Mean_Latency_ms']:.2f} & "
            f"{r['Median_Latency_ms']:.2f} & {r['P95_Latency_ms']:.2f} & {r['Peak_RAM_GB']:.2f} & "
            f"{r['Peak_Disk_Spill_GB']:.2f} & {r['Spilled_Queries']} & {r['OOM_Limit_Events']} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    tex_path = MANUSCRIPT_DIR / "table_multiscale_scalability.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"\n[OK] บันทึกตาราง LaTeX Multi-Scale ลงใน {tex_path}")


if __name__ == "__main__":
    summarize_multiscale()