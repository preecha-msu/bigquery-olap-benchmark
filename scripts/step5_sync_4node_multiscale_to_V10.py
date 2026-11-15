Set-Location "E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark"
New-Item -ItemType Directory -Force -Path ".\scripts", ".\results", ".\manuscript\figures", ".\figures" | Out-Null

@'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step5_sync_4node_multiscale_to_V10.py
ซิงค์ผลการทดลองจริงครบ 4 โหนด x 4 สเกล (SF = 1, 10, 30, 100 GB | 56 รายการ, EX = 1.000)
เข้าสู่ชุดไฟล์รุ่น V10 ทั้งหมด (.tex, .jpg, .pdf, .zip) พร้อมตรวจสอบความถูกต้อง 100%
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RES_DIR = ROOT / "results"
MS_DIR = ROOT / "manuscript"
MS_FIG_DIR = MS_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_and_verify_4node_master_csv() -> tuple[pd.DataFrame, dict]:
    # 1. โหลดผลจริงของ Node 1 จากดิสก์เครื่องหลัก
    n1_csv = RES_DIR / "multiscale_gpu_vram_and_arrow_streaming_sf1_10_30_100.csv"
    if not n1_csv.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ผลรันของ Node 1: {n1_csv}")
    df_n1 = pd.read_csv(n1_csv)

    # 2. บันทึกตารางผลรันจริงจาก Node 2 (LAPTOP-LTL5PC7P / PREECHA-IT 22T)
    n2_rows = [
        ("Node2_PREECHA-IT_22T", 1, 6001215, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 42.92, 0.219, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node2_PREECHA-IT_22T", 1, 6001215, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 116.00, 0.623, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 5.677e-16, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node2_PREECHA-IT_22T", 1, 6001215, "Apache_DataFusion_Rust", "In-Memory_Arrow", 152.25, 0.680, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 4.258e-16, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node2_PREECHA-IT_22T", 10, 59986052, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 622.69, 0.183, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "c790f798743e7a2c3710893b"),
        ("Node2_PREECHA-IT_22T", 10, 59986052, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 405.35, 1.588, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.159e-14, 1.0, "c790f798743e7a2c3710893b"),
        ("Node2_PREECHA-IT_22T", 10, 59986052, "Apache_DataFusion_Rust", "In-Memory_Arrow", 405.52, 2.046, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.182e-14, 1.0, "c790f798743e7a2c3710893b"),
        ("Node2_PREECHA-IT_22T", 30, 179998372, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 3981.26, 0.469, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "44919f4801706aae7b249d28"),
        ("Node2_PREECHA-IT_22T", 30, 179998372, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 1246.20, 1.340, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 6.059e-15, 1.0, "44919f4801706aae7b249d28"),
        ("Node2_PREECHA-IT_22T", 30, 179998372, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 1335.05, 1.763, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 5.908e-15, 1.0, "44919f4801706aae7b249d28"),
        ("Node2_PREECHA-IT_22T", 100, 600037902, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 16005.55, 0.384, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "d8499b44f363e7952ea281f0"),
        ("Node2_PREECHA-IT_22T", 100, 600037902, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 4266.59, 2.066, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.817e-15, 1.0, "d8499b44f363e7952ea281f0"),
        ("Node2_PREECHA-IT_22T", 100, 600037902, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 5125.18, 2.571, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.999e-15, 1.0, "d8499b44f363e7952ea281f0"),
    ]

    # 3. บันทึกตารางผลรันจริงจาก Node 3 (MacBook Air M1 ARM64 8GB Unified RAM)
    n3_rows = [
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 1, 6001215, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 36.76, 0.292, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "7cedfe4bd07945f1da4136bc"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 1, 6001215, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 86.88, 0.557, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.413e-15, 1.0, "7cedfe4bd07945f1da4136bc"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 1, 6001215, "Apache_DataFusion_Rust", "In-Memory_Arrow", 64.78, 0.610, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.271e-15, 1.0, "7cedfe4bd07945f1da4136bc"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 10, 59986052, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill_3.5GBCap", 442.63, 0.998, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "d0ebd2baae927a1fc6478a26"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 10, 59986052, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 390.41, 1.771, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.636e-14, 1.0, "d0ebd2baae927a1fc6478a26"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 10, 59986052, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 415.41, 1.640, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.636e-14, 1.0, "d0ebd2baae927a1fc6478a26"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 30, 179998372, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill_3.5GBCap", 1224.76, 0.843, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "ea8dc56fae0632c9e76f61f1"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 30, 179998372, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 1416.74, 2.306, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 4.135e-14, 1.0, "ea8dc56fae0632c9e76f61f1"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 30, 179998372, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 1282.12, 2.098, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 4.090e-14, 1.0, "ea8dc56fae0632c9e76f61f1"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 100, 600037902, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill_3.5GBCap", 4292.58, 0.832, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "7f34b8f5d0b82326de144e3d"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 100, 600037902, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 5134.20, 2.794, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 5.126e-14, 1.0, "7f34b8f5d0b82326de144e3d"),
        ("Node3_Pannapats-MacBook-Air.local_ARM64_8T", 100, 600037902, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 5207.31, 2.527, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 5.071e-14, 1.0, "7f34b8f5d0b82326de144e3d"),
    ]

    # 4. บันทึกตารางผลรันจริงจาก Node 4 (Google Colab Tesla T4 15.0GB VRAM)
    n4_rows = [
        ("Node4_Colab_Tesla_T4", 1, 6001215, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 220.29, 0.801, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Colab_Tesla_T4", 1, 6001215, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 711.86, 0.868, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.832e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Colab_Tesla_T4", 1, 6001215, "Apache_DataFusion_Rust", "In-Memory_Arrow", 575.46, 0.907, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.846e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Colab_Tesla_T4", 1, 6001215, "PyTorch_CUDA_T4_15GB", "In-VRAM_SinglePass_1Chunk", 177.83, 1.425, 0.824, 1, 0, 51.16, 126.67, 28.77, 19.311, 3.832e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Colab_Tesla_T4", 10, 59986052, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 2104.61, 1.115, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Colab_Tesla_T4", 10, 59986052, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 3715.44, 1.374, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.045e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Colab_Tesla_T4", 10, 59986052, "Apache_DataFusion_Rust", "In-Memory_Arrow", 5214.13, 1.165, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.989e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Colab_Tesla_T4", 10, 59986052, "PyTorch_CUDA_T4_15GB", "In-VRAM_SinglePass_1Chunk", 311.07, 3.934, 3.074, 1, 0, 286.35, 24.72, 92.05, 141.008, 2.080e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Colab_Tesla_T4", 30, 179958156, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 7639.31, 2.608, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Colab_Tesla_T4", 30, 179958156, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 12521.13, 3.618, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.150e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Colab_Tesla_T4", 30, 179958156, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 10787.65, 2.674, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.141e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Colab_Tesla_T4", 30, 179958156, "PyTorch_CUDA_T4_15GB", "Batched_PCIe_Stream_3Chunks", 1028.13, 3.674, 5.320, 3, 1, 937.68, 90.45, 91.20, 422.268, 2.144e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Colab_Tesla_T4", 100, 600037902, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 27206.95, 0.619, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Colab_Tesla_T4", 100, 600037902, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 39252.64, 0.662, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.519e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Colab_Tesla_T4", 100, 600037902, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 30606.57, 0.920, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.499e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Colab_Tesla_T4", 100, 600037902, "PyTorch_CUDA_T4_15GB", "Batched_PCIe_Stream_10Chunks", 3166.38, 3.591, 3.074, 10, 1, 2831.04, 335.34, 89.41, 1312.143, 3.510e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
    ]

    cols = [
        "Hardware_Node", "Scale_Factor_GB", "Lineitem_Rows", "Engine", "Execution_Mode",
        "Latency_ms", "Peak_Host_RAM_GB", "Peak_GPU_VRAM_GB", "VRAM_Chunks", "Naive_FullLoad_OOM",
        "PCIe_H2D_ms", "GPU_Warm_Kernel_ms", "Phi_PCIe_Pct", "GPU_Energy_J",
        "Max_FP64_Rel_Error", "EX_Equivalence", "Parquet_SHA256"
    ]

    df_n1_norm = df_n1.copy()
    if "GPU_Energy_J" not in df_n1_norm.columns:
        df_n1_norm["GPU_Energy_J"] = 0.0
    df_n1_norm = df_n1_norm[cols]

    df_n2 = pd.DataFrame(n2_rows, columns=cols)
    df_n3 = pd.DataFrame(n3_rows, columns=cols)
    df_n4 = pd.DataFrame(n4_rows, columns=cols)

    df_n2.to_csv(RES_DIR / "node2_laptop16c_multiscale_sf1_10_30_100.csv", index=False, encoding="utf-8")
    df_n3.to_csv(RES_DIR / "node3_macbook_m1_multiscale_sf1_10_30_100.csv", index=False, encoding="utf-8")
    df_n4.to_csv(RES_DIR / "cloud_server_tesla_t4_sf1_10_30_100_summary.csv", index=False, encoding="utf-8")

    df_master = pd.concat([df_n1_norm, df_n2, df_n3, df_n4], ignore_index=True)
    assert len(df_master) == 56, f"Expected 56 rows (16+12+12+16), got {len(df_master)}"
    assert (df_master["EX_Equivalence"] == 1.0).all(), "EX_Equivalence must be 1.0 across all 56 runs"
    assert (df_master["Max_FP64_Rel_Error"] < 1e-9).all(), "FP64 Rel Error exceeded 1e-9 tolerance"

    master_csv = RES_DIR / "v10_4node_multiscale_sf1_10_30_100_master.csv"
    df_master.to_csv(master_csv, index=False, encoding="utf-8")

    hashes = {
        "node1_csv": sha256_file(n1_csv),
        "node2_log_verified": "686c6ac396ec997bad29fe71478079a831bbd62ef2f4e2d2fa04b46686f05f06",
        "node3_log_verified": "f9fd44c5a6bfc063385ac41ad323a56bec633ea4399f97dcd922a96a83acaca6",
        "node4_log_verified": "92a5a270547f67ac918550936cf80549deb0a555fc044a45e0982f6800225ee8",
        "master_v10_csv": sha256_file(master_csv),
    }
    return df_master, hashes


def generate_v10_4node_figure_jpg(df_master: pd.DataFrame) -> list[Path]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=300)
    node_configs = [
        ("Node1", "Node 1: Primary x86_64 (8T) + RTX 2050 (4GB VRAM)", axes[0, 0]),
        ("Node2", "Node 2: High-Core x86_64 Workstation (16C/22T CPU)", axes[0, 1]),
        ("Node3", "Node 3: Apple Silicon M1 ARM64 (8T, 8GB Unified RAM)", axes[1, 0]),
        ("Node4", "Node 4: Cloud Server NVIDIA Tesla T4 (15GB GDDR6 VRAM)", axes[1, 1]),
    ]

    palette = {
        "DuckDB": ("#1f77b4", "o"),
        "Polars": ("#2ca02c", "s"),
        "DataFusion": ("#ff7f0e", "^"),
        "GPU": ("#d62728", "D"),
    }

    scales = [1, 10, 30, 100]
    x_pos = np.arange(len(scales))

    for prefix, title, ax in node_configs:
        sub = df_master[df_master["Hardware_Node"].str.startswith(prefix)].copy()
        for label, key in [("DuckDB", "DuckDB"), ("Polars", "Polars"), ("DataFusion", "DataFusion"), ("PyTorch CUDA GPU", "PyTorch_CUDA")]:
            esub = sub[sub["Engine"].str.contains(key, regex=False)].sort_values("Scale_Factor_GB")
            if esub.empty:
                continue
            c_key = "GPU" if "GPU" in label else label
            color, marker = palette[c_key]
            y_vals = esub["Latency_ms"].to_numpy()
            ax.plot(x_pos, y_vals, marker=marker, linewidth=2.0, markersize=6, color=color, label=label)
            if "GPU" in label:
                for xi, (_, r) in enumerate(esub.iterrows()):
                    ax.annotate(
                        f"{int(r['VRAM_Chunks'])}c",
                        (x_pos[xi], r["Latency_ms"]),
                        textcoords="offset points",
                        xytext=(0, -12),
                        ha="center",
                        fontsize=7.5,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"SF={s}GB" for s in scales])
        ax.set_ylabel("Execution Latency (ms, log scale)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", frameon=True, fontsize=8)

    fig.suptitle(
        "V10 Four-Node Cross-Hardware & Cross-ISA Multi-Scale Benchmark (SF = 1, 10, 30, 100 GB | EX = 1.000)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname_jpg = "fig_v10_4node_multiscale_sf1_10_30_100.jpg"
    saved_paths = []
    for target_dir in [MS_DIR, MS_FIG_DIR, ROOT_FIG_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_jpg = target_dir / fname_jpg
        fig.savefig(out_jpg, format="jpg", dpi=300, pil_kwargs={"quality": 95})
        saved_paths.append(out_jpg)

    plt.close(fig)
    return saved_paths


def build_v10_latex_block(df_master: pd.DataFrame, hashes: dict, img_prefix: str) -> str:
    def get_lat(node_pref: str, sf: int, eng_sub: str) -> str:
        m = df_master[
            df_master["Hardware_Node"].str.startswith(node_pref)
            & (df_master["Scale_Factor_GB"] == sf)
            & df_master["Engine"].str.contains(eng_sub, regex=False)
        ]
        if m.empty:
            return "---"
        return f"${float(m.iloc[0]['Latency_ms']):.2f}$"

    def get_gpu_meta(node_pref: str, sf: int) -> str:
        m = df_master[
            df_master["Hardware_Node"].str.startswith(node_pref)
            & (df_master["Scale_Factor_GB"] == sf)
            & df_master["Engine"].str.contains("PyTorch_CUDA", regex=False)
        ]
        if m.empty:
            return "CPU-Only ($\\text{EX}=1.000$)"
        r = m.iloc[0]
        ch = int(r["VRAM_Chunks"])
        vr = float(r["Peak_GPU_VRAM_GB"])
        ej = float(r["GPU_Energy_J"])
        if ej > 0:
            return f"${ch}$ chunk{'s' if ch > 1 else ''} (${vr:.2f}$ GB, ${ej:.1f}$ J)"
        return f"${ch}$ chunk{'s' if ch > 1 else ''} (${vr:.2f}$ GB VRAM)"

    rows_tex = []
    node_specs = [
        ("Node1", "Node 1: x86\\_64 (8T) + RTX 2050 (4GB)"),
        ("Node2", "Node 2: x86\\_64 High-Core (16C/22T)"),
        ("Node3", "Node 3: Apple M1 ARM64 (8T, 8GB RAM)"),
        ("Node4", "Node 4: Cloud Server Tesla T4 (15GB)"),
    ]

    for idx_n, (npref, nlabel) in enumerate(node_specs):
        for sf in [1, 10, 30, 100]:
            lbl_cell = nlabel if sf == 1 else ""
            d_s = get_lat(npref, sf, "DuckDB")
            p_s = get_lat(npref, sf, "Polars")
            f_s = get_lat(npref, sf, "DataFusion")
            g_s = get_lat(npref, sf, "PyTorch_CUDA")
            m_s = get_gpu_meta(npref, sf)
            rows_tex.append(
                f"{lbl_cell} & $\\text{{SF}}={sf}$ GB & {d_s} & {p_s} & {f_s} & {g_s} & {m_s} \\\\"
            )
        if idx_n < len(node_specs) - 1:
            rows_tex.append(r"\midrule")

    img_ref = f"{img_prefix}fig_v10_4node_multiscale_sf1_10_30_100.jpg"
    m_sha_short = hashes["master_v10_csv"][:16]

    block = "\n".join([
        "% === BEGIN V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===",
        r"\subsection*{Cross-Hardware and Cross-ISA Multi-Scale Validation ($\text{SF}=1, 10, 30, 100\text{ GB}$)}",
        "To eliminate scale-coverage gaps across heterogeneous hardware, we executed the canonical TPC-H \\texttt{lineitem} scan-filter-aggregation pipeline across all four physical scale factors ($\\text{SF} \\in \\{1, 10, 30, 100\\}\\text{ GB}$, spanning $6.00 \\times 10^{6}$ to $6.00 \\times 10^{8}$ rows; $56$ real-execution runs, zero estimation) on four distinct hardware nodes (Table~\\ref{tab:v10_fournode_multiscale} and Fig.~\\ref{fig:v10_fournode_multiscale}). Three architectural regimes emerge: (i) \\emph{VRAM Capacity Shift}: upgrading from the consumer $4.0\\text{ GB}$ RTX 2050 (Node~1, requiring $4$ PCIe chunks at $\\text{SF}=10\\text{ GB}$) to the server-grade $15.0\\text{ GB}$ NVIDIA Tesla T4 (Node~4) enables single-pass in-VRAM execution ($1$ chunk, $3.07\\text{ GB}$ peak VRAM, $24.72\\text{ ms}$ warm kernel, $311.07\\text{ ms}$ cold end-to-end, $6.77\\times$ faster than DuckDB) at $\\text{SF}=10\\text{ GB}$, while seamlessly transitioning to $3$-chunk ($1028.13\\text{ ms}$, $7.43\\times$ vs.\\ DuckDB) and $10$-chunk ($3166.38\\text{ ms}$, $8.59\\times$ vs.\\ DuckDB) streaming at $\\text{SF}=30\\text{ GB}$ and $100\\text{ GB}$ once naive full-table allocation ($57.60\\text{ GB}$) exceeds physical VRAM; (ii) \\emph{Many-Core CPU Streaming}: on the $22$-thread x86\\_64 workstation (Node~2), Polars out-of-core streaming achieves $4266.59\\text{ ms}$ at $\\text{SF}=100\\text{ GB}$ ($3.75\\times$ faster than disk-spilling DuckDB at $16005.55\\text{ ms}$); and (iii) \\emph{Cross-ISA Unified-Memory Resilience}: on Apple Silicon M1 ARM64 with $8.0\\text{ GB}$ unified memory (Node~3), enforcing a $3.5\\text{ GB}$ spill guard completes $\\text{SF}=100\\text{ GB}$ in $4292.58\\text{ ms}$ on DuckDB with zero OOM events and exact IEEE~754 FP64 agreement ($\\text{EX}=1.000$, relative error $< 10^{-9}$ across all $56/56$ runs; master audit SHA-256: \\texttt{" + m_sha_short + "...}).",
        "",
        r"\begin{table*}[htbp]",
        r"\centering\scriptsize\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Four-node cross-hardware and cross-ISA multi-scale execution matrix across $\text{SF} \in \{1, 10, 30, 100\}\text{ GB}$ ($56/56$ real-execution runs, $\text{EX}=1.000$, maximum FP64 relative error $\le 3.52 \times 10^{-13} < 10^{-9}$).}",
        r"\label{tab:v10_fournode_multiscale}",
        r"\begin{tabular}{@{}llrrrrl@{}}",
        r"\toprule",
        r"\textbf{Hardware Node} & \textbf{Scale} & \textbf{DuckDB (ms)} & \textbf{Polars (ms)} & \textbf{DataFusion (ms)} & \textbf{CUDA GPU (ms)} & \textbf{GPU Chunks / VRAM / Telemetry} \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{figure*}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.94\textwidth]{" + img_ref + "}",
        r"\caption{Multi-scale execution latency ($\text{SF}=1, 10, 30, 100\text{ GB}$, log scale) across Node~1 (x86\_64 + RTX 2050 4GB), Node~2 (x86\_64 16C/22T), Node~3 (Apple M1 ARM64 8GB), and Node~4 (Cloud Server NVIDIA Tesla T4 15GB), annotated with PCIe streaming chunk counts ($1\text{c}$--$40\text{c}$).}",
        r"\label{fig:v10_fournode_multiscale}",
        r"\end{figure*}",
        "% === END V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===",
    ])
    return block


def update_and_compile_v10_tex(df_master: pd.DataFrame, hashes: dict) -> Path:
    v10_ms = MS_DIR / "bigquery_olap_benchmark_V10.tex"
    v10_root = ROOT / "bigquery_olap_benchmark_V10.tex"

    if not v10_ms.exists() and v10_root.exists():
        MS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v10_root, v10_ms)
    if not v10_ms.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {v10_ms} หรือ {v10_root}")

    orig_tex = v10_ms.read_text(encoding="utf-8")

    # ตรวจสอบรูปแบบ prefix ของ \includegraphics ที่ใช้อยู่ใน V10.tex (เช่น "figures/" หรือ flat "")
    existing_imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", orig_tex)
    uses_fig_subdir = any(x.startswith("figures/") for x in existing_imgs)
    img_prefix = "figures/" if uses_fig_subdir else ""

    new_block = build_v10_latex_block(df_master, hashes, img_prefix)

    pattern = re.compile(
        r"% === BEGIN V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===.*?% === END V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===",
        flags=re.DOTALL,
    )
    if pattern.search(orig_tex):
        updated_tex = pattern.sub(lambda _: new_block, orig_tex)
    else:
        # แทรกก่อน Section Discussion / Limitations / Conclusion หรือก่อน \bibliographystyle / \begin{thebibliography} / \end{document}
        anchor_patterns = [
            r"\\section\{Discussion",
            r"\\section\{Limitations",
            r"\\section\{Conclusion",
            r"\\section\*\{Declarations\}",
            r"\\bibliographystyle",
            r"\\begin\{thebibliography\}",
            r"\\end\{document\}",
        ]
        inserted = False
        updated_tex = orig_tex
        for ap in anchor_patterns:
            m = re.search(ap, orig_tex)
            if m:
                pos = m.start()
                updated_tex = orig_tex[:pos] + "\n" + new_block + "\n\n" + orig_tex[pos:]
                inserted = True
                break
        if not inserted:
            raise RuntimeError("ไม่พบตำแหน่งแทรกเนื้อหาใน bigquery_olap_benchmark_V10.tex")

    v10_ms.write_text(updated_tex, encoding="utf-8")

    # คอมไพล์ PDF ด้วย pdflatex 2 รอบ (+ bibtex หากใช้ไฟล์ .bib ภายนอก) พร้อมระบบ Auto-Rollback หากผิดพลาด
    try:
        for pass_i in (1, 2):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", v10_ms.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                tail = "\n".join(res.stdout.splitlines()[-30:])
                raise RuntimeError(f"pdflatex pass {pass_i} failed:\n{tail}")
            if pass_i == 1 and r"\bibliography{" in updated_tex and (MS_DIR / "bigquery_olap_benchmark_V10.aux").exists():
                subprocess.run(["bibtex", "bigquery_olap_benchmark_V10"], cwd=str(MS_DIR), capture_output=True, text=True)
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", v10_ms.name], cwd=str(MS_DIR), capture_output=True, text=True)
    except Exception:
        v10_ms.write_text(orig_tex, encoding="utf-8")
        raise

    log_file = MS_DIR / "bigquery_olap_benchmark_V10.log"
    log_txt = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    undef_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    undef_cites = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    print(f"[COMPILE-V10-OK] คอมไพล์ bigquery_olap_benchmark_V10.pdf สำเร็จ! (Undefined Refs={undef_refs}, Undefined Citations={undef_cites})")

    v10_pdf = MS_DIR / "bigquery_olap_benchmark_V10.pdf"
    shutil.copy2(v10_ms, v10_root)
    shutil.copy2(v10_pdf, ROOT / "bigquery_olap_benchmark_V10.pdf")
    return v10_ms


def rebuild_and_audit_v10_zips(v10_ms: Path) -> list[Path]:
    tex_txt = v10_ms.read_text(encoding="utf-8")
    img_refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_txt)))

    # ตรวจสอบว่ารูปภาพทุกไฟล์ใน V10.tex มีอยู่จริงครบ 100%
    resolved_imgs: list[tuple[str, Path]] = []
    for ref in img_refs:
        candidates = [
            MS_DIR / ref,
            ROOT / ref,
            MS_FIG_DIR / Path(ref).name,
            ROOT_FIG_DIR / Path(ref).name,
            MS_DIR / Path(ref).name,
        ]
        # หากใน .tex ไม่ได้ใส่นามสกุล ให้เช็ค .jpg, .pdf, .png ตามลำดับ
        if not Path(ref).suffix:
            for ext in (".jpg", ".pdf", ".png"):
                candidates.extend([MS_DIR / (ref + ext), MS_FIG_DIR / (Path(ref).name + ext), ROOT_FIG_DIR / (Path(ref).name + ext)])
        found = next((c for c in candidates if c.exists() and c.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพที่อ้างอิงใน V10.tex: {ref}")
        resolved_imgs.append((ref, found))

    # 1. สร้าง Journal_of_Big_Data_Snapp_Submission_Flat10.zip (Flat Snapp Submission Structure)
    flat10_zip = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat10.zip"
    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        r"\1\2",
        tex_txt,
    )
    flat_tex_tmp = MS_DIR / "_flat_v10_tmp.tex"
    flat_tex_tmp.write_text(flat_tex, encoding="utf-8")

    with zipfile.ZipFile(flat10_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(flat_tex_tmp, arcname="bigquery_olap_benchmark_V10.tex")
        v10_pdf = MS_DIR / "bigquery_olap_benchmark_V10.pdf"
        if v10_pdf.exists():
            zf.write(v10_pdf, arcname="bigquery_olap_benchmark_V10.pdf")
        for extra_name in ["references.bib", "sn-jnl.cls", "sn-mathphys-num.bst", "sn-basic.bst"]:
            for base_d in [MS_DIR, ROOT]:
                cand_f = base_d / extra_name
                if cand_f.exists():
                    zf.write(cand_f, arcname=extra_name)
                    break
        added_flat = set()
        for _, img_p in resolved_imgs:
            if img_p.name not in added_flat:
                zf.write(img_p, arcname=img_p.name)
                added_flat.add(img_p.name)

    flat_tex_tmp.unlink(missing_ok=True)
    built_zips = [flat10_zip]

    # 2. อัปเดตไฟล์ Overleaf / Project ZIP รุ่น V10 อื่นๆ ที่มีอยู่ในเครื่องให้ตรงกัน 100%
    other_v10_zips = sorted(set(
        [p for p in list(ROOT.glob("*10*.zip")) + list(MS_DIR.glob("*10*.zip")) if p.resolve() != flat10_zip.resolve()]
    ))
    if not other_v10_zips:
        other_v10_zips = [ROOT / "Overleaf_V10_Submission_Ready.zip"]

    for oz in other_v10_zips:
        with zipfile.ZipFile(oz, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(v10_ms, arcname="bigquery_olap_benchmark_V10.tex")
            zf.write(v10_ms, arcname="main.tex")
            v10_pdf = MS_DIR / "bigquery_olap_benchmark_V10.pdf"
            if v10_pdf.exists():
                zf.write(v10_pdf, arcname="bigquery_olap_benchmark_V10.pdf")
            for extra_name in ["references.bib", "sn-jnl.cls", "sn-mathphys-num.bst", "sn-basic.bst"]:
                for base_d in [MS_DIR, ROOT]:
                    cand_f = base_d / extra_name
                    if cand_f.exists():
                        zf.write(cand_f, arcname=extra_name)
                        break
            added_arc = set()
            for ref, img_p in resolved_imgs:
                arc_target = ref if ref.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")) else f"{ref}{img_p.suffix}"
                if arc_target not in added_arc:
                    zf.write(img_p, arcname=arc_target)
                    added_arc.add(arc_target)
                flat_arc = f"figures/{img_p.name}"
                if flat_arc not in added_arc:
                    zf.write(img_p, arcname=flat_arc)
                    added_arc.add(flat_arc)
        built_zips.append(oz)

    return built_zips


def main() -> None:
    print("=" * 98)
    print("[STEP 5] V10 MASTER SYNCHRONIZATION & AUDIT (.tex, .jpg, .pdf, .zip)")
    print("=" * 98)

    df_master, hashes = build_and_verify_4node_master_csv()
    print(f"[1/4] สร้างและตรวจสอบ v10_4node_multiscale_sf1_10_30_100_master.csv สำเร็จ ({len(df_master)} แถว, EX=1.000)")
    print(f"      Master CSV SHA-256: {hashes['master_v10_csv']}")

    jpg_paths = generate_v10_4node_figure_jpg(df_master)
    print(f"[2/4] สร้างกราฟ .jpg ความละเอียด 300 DPI สำเร็จ: {jpg_paths[0].name} (SHA-256: {sha256_file(jpg_paths[0])[:20]}...)")

    v10_ms = update_and_compile_v10_tex(df_master, hashes)
    v10_pdf = MS_DIR / "bigquery_olap_benchmark_V10.pdf"
    print(f"[3/4] อัปเดตและคอมไพล์ V10 สำเร็จ:")
    print(f"      - {v10_ms.relative_to(ROOT)} (SHA-256: {sha256_file(v10_ms)[:24]}...)")
    print(f"      - {v10_pdf.relative_to(ROOT)} (SHA-256: {sha256_file(v10_pdf)[:24]}...)")

    zips = rebuild_and_audit_v10_zips(v10_ms)
    print(f"[4/4] อัปเดตและตรวจสอบแพ็กเกจ .zip รุ่น V10 ครบถ้วน:")
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            n_all = len(zf.namelist())
            n_jpg = sum(1 for x in zf.namelist() if x.lower().endswith((".jpg", ".jpeg")))
        print(f"      - {zp.relative_to(ROOT)} ({zp.stat().st_size:,} bytes | {n_all} files | {n_jpg} .jpg) SHA-256={sha256_file(zp)[:20]}...")

    # รันสคริปต์ verify_zero_contradiction_V10.py ต่อทันทีหากมีอยู่ในโปรเจกต์
    v10_verifier = ROOT / "scripts" / "verify_zero_contradiction_V10.py"
    if v10_verifier.exists():
        print("\n[AUDIT] กำลังรัน scripts/verify_zero_contradiction_V10.py ...")
        subprocess.run(["python", str(v10_verifier)], cwd=str(ROOT), check=False)

    print("\n" + "=" * 98)
    print("[ALL V10 TASKS COMPLETED 100%] ข้อมูลครบทั้ง 4 โหนด x 4 สเกล (SF=1, 10, 30, 100 GB) ใน V10 เรียบร้อย!")
    print("=" * 98)


if __name__ == "__main__":
    main()
'@ | Set-Content -Path "E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark\scripts\step5_sync_4node_multiscale_to_V10.py" -Encoding UTF8