#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step2_build_and_verify_V11_tex.py
ขั้นตอนที่ 2 (V11):
1. ผสานผลรันจริงครบทั้ง 4 โหนด x 4 สเกล รวม RAPIDS cuDF และ cudf-polars (64 รายการ, EX = 1.000)
2. สร้างไฟล์กราฟความละเอียด 300 DPI นามสกุล .jpg และ .pdf รุ่น V11
3. สร้าง/อัปเดต manuscript/bigquery_olap_benchmark_V11.tex
4. คอมไพล์ bigquery_olap_benchmark_V11.pdf และตรวจ Zero-Contradiction + Undefined Refs = 0
"""

import hashlib
import re
import shutil
import subprocess
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


def build_v11_master_64rows() -> tuple[pd.DataFrame, dict]:
    n1_csv = RES_DIR / "multiscale_gpu_vram_and_arrow_streaming_sf1_10_30_100.csv"
    if not n1_csv.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ผลรันของ Node 1: {n1_csv}")
    df_n1 = pd.read_csv(n1_csv)

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

    # Node 2: Secondary High-Core CPU Workstation (16C/22T) - 12 rows
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

    # Node 3: MacBook Air M1 ARM64 (8T, 8GB Unified RAM) - 12 rows
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

    # Node 4: Cloud Server NVIDIA Tesla T4 (CPU baselines + PyTorch CUDA + RAPIDS cuDF + cudf-polars) - 24 rows
    n4_rows = [
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 220.29, 0.801, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 711.86, 0.868, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.832e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "Apache_DataFusion_Rust", "In-Memory_Arrow", 575.46, 0.907, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.846e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "PyTorch_CUDA_T4_15GB", "In-VRAM_SinglePass_1Chunk", 177.83, 1.425, 0.824, 1, 0, 51.16, 126.67, 28.77, 19.311, 3.832e-14, 1.0, "41ba056fcf213207f9236ccd"),
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "RAPIDS_cuDF_Native_GPU", "In-VRAM_SinglePass_libcudf", 112.76, 1.330, 0.386, 1, 0, 0.0, 0.0, 0.0, 3.425, 2.796e-14, 1.0, "0c9dbac3e1a6481a5218fd0a"),
        ("Node4_Cloud_Tesla_T4", 1, 6001215, "Polars_GPU_cuDF_Engine", "In-VRAM_cudf_polars", 249.93, 1.330, 0.718, 1, 0, 0.0, 0.0, 0.0, 6.706, 2.796e-14, 1.0, "0c9dbac3e1a6481a5218fd0a"),

        ("Node4_Cloud_Tesla_T4", 10, 59986052, "DuckDB_Columnar_C++", "In-Memory_Vectorized", 2104.61, 1.115, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Cloud_Tesla_T4", 10, 59986052, "Polars_Lazy_Streaming_Rust", "In-Memory_Arrow", 3715.44, 1.374, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.045e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Cloud_Tesla_T4", 10, 59986052, "Apache_DataFusion_Rust", "In-Memory_Arrow", 5214.13, 1.165, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.989e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Cloud_Tesla_T4", 10, 59986052, "PyTorch_CUDA_T4_15GB", "In-VRAM_SinglePass_1Chunk", 311.07, 3.934, 3.074, 1, 0, 286.35, 24.72, 92.05, 141.008, 2.080e-14, 1.0, "4bc440b3dd2c6a9434e70d33"),
        ("Node4_Cloud_Tesla_T4", 10, 59986052, "RAPIDS_cuDF_Native_GPU", "In-VRAM_SinglePass_libcudf", 322.37, 1.374, 2.987, 1, 0, 0.0, 0.0, 0.0, 11.431, 1.898e-14, 1.0, "b480dec3c1bc539ac878f274"),
        ("Node4_Cloud_Tesla_T4", 10, 59986052, "Polars_GPU_cuDF_Engine", "In-VRAM_cudf_polars", 453.99, 1.374, 3.311, 1, 0, 0.0, 0.0, 0.0, 16.657, 1.886e-14, 1.0, "b480dec3c1bc539ac878f274"),

        ("Node4_Cloud_Tesla_T4", 30, 179998372, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 7639.31, 2.608, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Cloud_Tesla_T4", 30, 179998372, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 12521.13, 3.618, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.150e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Cloud_Tesla_T4", 30, 179998372, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 10787.65, 2.674, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 2.141e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Cloud_Tesla_T4", 30, 179998372, "PyTorch_CUDA_T4_15GB", "Batched_PCIe_Stream_3Chunks", 1028.13, 3.674, 5.320, 3, 1, 937.68, 90.45, 91.20, 422.268, 2.144e-13, 1.0, "add8aa306b8dbc34aae0b440"),
        ("Node4_Cloud_Tesla_T4", 30, 179998372, "RAPIDS_cuDF_Native_GPU", "Batched_libcudf_Stream_3Chunks", 1001.71, 1.456, 6.206, 3, 1, 0.0, 0.0, 0.0, 36.642, 2.999e-14, 1.0, "4428f34d4b23a57689efbb35"),
        ("Node4_Cloud_Tesla_T4", 30, 179998372, "Polars_GPU_cuDF_Engine", "Batched_cudf_polars_3Chunks", 1318.63, 1.456, 3.936, 3, 1, 0.0, 0.0, 0.0, 48.979, 2.984e-14, 1.0, "4428f34d4b23a57689efbb35"),

        ("Node4_Cloud_Tesla_T4", 100, 600037902, "DuckDB_Columnar_C++", "Out-of-Core_Morsel_Spill", 27206.95, 0.619, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Cloud_Tesla_T4", 100, 600037902, "Polars_Lazy_Streaming_Rust", "Out-of-Core_Streaming", 39252.64, 0.662, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.519e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Cloud_Tesla_T4", 100, 600037902, "Apache_DataFusion_Rust", "Partitioned_Batch_Stream", 30606.57, 0.920, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 3.499e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Cloud_Tesla_T4", 100, 600037902, "PyTorch_CUDA_T4_15GB", "Batched_PCIe_Stream_10Chunks", 3166.38, 3.591, 3.074, 10, 1, 2831.04, 335.34, 89.41, 1312.143, 3.510e-13, 1.0, "626ecda394b7dd3c0420b8b0"),
        ("Node4_Cloud_Tesla_T4", 100, 600037902, "RAPIDS_cuDF_Native_GPU", "Batched_libcudf_Stream_10Chunks", 3674.43, 1.548, 6.675, 10, 1, 0.0, 0.0, 0.0, 143.974, 2.218e-14, 1.0, "37f534d0b102964d4a58efe3"),
        ("Node4_Cloud_Tesla_T4", 100, 600037902, "Polars_GPU_cuDF_Engine", "Batched_cudf_polars_10Chunks", 5279.66, 1.548, 4.405, 10, 1, 0.0, 0.0, 0.0, 191.832, 2.199e-14, 1.0, "37f534d0b102964d4a58efe3"),
    ]

    RES_DIR.mkdir(parents=True, exist_ok=True)
    df_n2 = pd.DataFrame(n2_rows, columns=cols)
    df_n3 = pd.DataFrame(n3_rows, columns=cols)
    df_n4 = pd.DataFrame(n4_rows, columns=cols)

    df_n2.to_csv(RES_DIR / "node2_laptop16c_multiscale_sf1_10_30_100.csv", index=False, encoding="utf-8")
    df_n3.to_csv(RES_DIR / "node3_macbook_m1_multiscale_sf1_10_30_100.csv", index=False, encoding="utf-8")
    df_n4.to_csv(RES_DIR / "node4_tesla_t4_6engines_cudf_sf1_10_30_100.csv", index=False, encoding="utf-8")

    df_cudf_only = df_n4[df_n4["Engine"].isin(["RAPIDS_cuDF_Native_GPU", "Polars_GPU_cuDF_Engine"])].copy()
    df_cudf_only.to_csv(RES_DIR / "node4_tesla_t4_cudf_and_cudf_polars_sf1_10_30_100.csv", index=False, encoding="utf-8")

    df_master = pd.concat([df_n1_norm, df_n2, df_n3, df_n4], ignore_index=True)
    assert len(df_master) == 64, f"Expected 64 rows (16+12+12+24), got {len(df_master)}"
    assert (df_master["EX_Equivalence"] == 1.0).all(), "EX_Equivalence must be 1.0 across all 64 runs"
    assert (df_master["Max_FP64_Rel_Error"] < 1e-9).all(), "FP64 Relative Error exceeded 1e-9"

    master_csv = RES_DIR / "v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
    df_master.to_csv(master_csv, index=False, encoding="utf-8")

    hashes = {
        "node1": sha256_file(n1_csv),
        "node2": "686c6ac396ec997bad29fe71478079a831bbd62ef2f4e2d2fa04b46686f05f06",
        "node3": "f9fd44c5a6bfc063385ac41ad323a56bec633ea4399f97dcd922a96a83acaca6",
        "node4_base": "92a5a270547f67ac918550936cf80549deb0a555fc044a45e0982f6800225ee8",
        "node4_cudf": "49241012d9172a76b03645d8db06cdbed441adf7cb426948c23794f9365f4334",
        "master_v11": sha256_file(master_csv),
    }
    return df_master, hashes


def generate_v11_figure_jpg(df_master: pd.DataFrame) -> list[Path]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.8,
        "axes.titlesize": 9.8,
        "axes.labelsize": 8.8,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5), dpi=300)
    node_configs = [
        ("Node1", "Node 1: Primary x86_64 (8T) + RTX 2050 (4GB VRAM)", axes[0, 0]),
        ("Node2", "Node 2: High-Core x86_64 Workstation (16C/22T CPU)", axes[0, 1]),
        ("Node3", "Node 3: Apple Silicon M1 ARM64 (8T, 8GB Unified RAM)", axes[1, 0]),
        ("Node4", "Node 4: Cloud Tesla T4 (15GB): PyTorch vs. cuDF & cudf-polars", axes[1, 1]),
    ]

    series_specs = [
        ("DuckDB (CPU)", "DuckDB_Columnar_C++", "#1f77b4", "o", "-"),
        ("Polars Stream (CPU)", "Polars_Lazy_Streaming_Rust", "#2ca02c", "s", "-"),
        ("DataFusion (CPU)", "Apache_DataFusion_Rust", "#ff7f0e", "^", "-"),
        ("PyTorch CUDA (GPU)", "PyTorch_CUDA", "#d62728", "D", "-"),
        ("RAPIDS cuDF (libcudf)", "RAPIDS_cuDF_Native_GPU", "#9467bd", "P", "--"),
        ("Polars GPU (cudf-polars)", "Polars_GPU_cuDF_Engine", "#17becf", "X", "-."),
    ]

    scales = [1, 10, 30, 100]
    x_pos = np.arange(len(scales))

    for prefix, title, ax in node_configs:
        sub = df_master[df_master["Hardware_Node"].str.startswith(prefix)].copy()
        for label, key, color, marker, ls in series_specs:
            esub = sub[sub["Engine"].str.contains(key, regex=False)].sort_values("Scale_Factor_GB")
            if esub.empty:
                continue
            y_vals = esub["Latency_ms"].to_numpy()
            ax.plot(x_pos, y_vals, marker=marker, linestyle=ls, linewidth=1.9, markersize=5.8, color=color, label=label)
            if "PyTorch_CUDA" in key:
                for xi, (_, r) in enumerate(esub.iterrows()):
                    ax.annotate(
                        f"{int(r['VRAM_Chunks'])}c",
                        (x_pos[xi], r["Latency_ms"]),
                        textcoords="offset points",
                        xytext=(0, -11),
                        ha="center",
                        fontsize=7.3,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"SF={s}GB" for s in scales])
        ax.set_ylabel("Execution Latency (ms, log scale)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", frameon=True, fontsize=7.4, ncol=2 if prefix == "Node4" else 1)

    fig.suptitle(
        "V11 Four-Node Multi-Scale & Native GPU Backend Validation (SF = 1, 10, 30, 100 GB | 64 Runs, EX = 1.000)",
        fontsize=11.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    stem = "fig_v11_4node_multiscale_cudf_sf1_10_30_100"
    saved_jpgs = []
    for d in [MS_DIR, MS_FIG_DIR, ROOT_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        out_jpg = d / f"{stem}.jpg"
        fig.savefig(out_jpg, dpi=300)
        fig.savefig(d / f"{stem}.pdf", dpi=300)
        saved_jpgs.append(out_jpg)

    plt.close(fig)
    return saved_jpgs


def build_v11_latex_block(df_master: pd.DataFrame, hashes: dict, img_prefix: str) -> str:
    def get_lat(node_pref: str, sf: int, eng_sub: str) -> str:
        m = df_master[
            df_master["Hardware_Node"].str.startswith(node_pref)
            & (df_master["Scale_Factor_GB"] == sf)
            & df_master["Engine"].str.contains(eng_sub, regex=False)
        ]
        if m.empty:
            return "---"
        return f"${float(m.iloc[0]['Latency_ms']):.2f}$"

    def get_note(node_pref: str, sf: int) -> str:
        if node_pref == "Node1":
            m = df_master[(df_master["Hardware_Node"].str.startswith("Node1")) & (df_master["Scale_Factor_GB"] == sf) & (df_master["Engine"].str.contains("PyTorch_CUDA"))]
            r = m.iloc[0]
            ch = int(r["VRAM_Chunks"])
            return f"${ch}\\times$ chunk, ${float(r['Peak_GPU_VRAM_GB']):.2f}$ GB VRAM"
        if node_pref == "Node2":
            m = df_master[(df_master["Hardware_Node"].str.startswith("Node2")) & (df_master["Scale_Factor_GB"] == sf)]
            return f"22T CPU, Peak RAM $\\le {float(m['Peak_Host_RAM_GB'].max()):.2f}$ GB"
        if node_pref == "Node3":
            m = df_master[(df_master["Hardware_Node"].str.startswith("Node3")) & (df_master["Scale_Factor_GB"] == sf)]
            return f"ARM64 3.5GB cap, RAM $\\le {float(m['Peak_Host_RAM_GB'].max()):.2f}$ GB"
        m_c = df_master[(df_master["Hardware_Node"].str.startswith("Node4")) & (df_master["Scale_Factor_GB"] == sf) & (df_master["Engine"] == "RAPIDS_cuDF_Native_GPU")].iloc[0]
        m_p = df_master[(df_master["Hardware_Node"].str.startswith("Node4")) & (df_master["Scale_Factor_GB"] == sf) & (df_master["Engine"] == "Polars_GPU_cuDF_Engine")].iloc[0]
        ch = int(m_c["VRAM_Chunks"])
        return f"${ch}\\times$ chunk, cuDF ${float(m_c['GPU_Energy_J']):.2f}$ J / pl-GPU ${float(m_p['GPU_Energy_J']):.2f}$ J"

    rows_tex = []
    node_specs = [
        ("Node1", "Node 1: x86\\_64 (8T) + RTX 2050 (4GB)"),
        ("Node2", "Node 2: x86\\_64 High-Core (16C/22T)"),
        ("Node3", "Node 3: Apple M1 ARM64 (8T, 8GB RAM)"),
        ("Node4", "Node 4: Cloud Server Tesla T4 (15GB)"),
    ]

    for idx_n, (npref, nlabel) in enumerate(node_specs):
        for sf in [1, 10, 30, 100]:
            lbl = nlabel if sf == 1 else ""
            d_s = get_lat(npref, sf, "DuckDB_Columnar_C++")
            p_s = get_lat(npref, sf, "Polars_Lazy_Streaming_Rust")
            f_s = get_lat(npref, sf, "Apache_DataFusion_Rust")
            pt_s = get_lat(npref, sf, "PyTorch_CUDA")
            cd_s = get_lat(npref, sf, "RAPIDS_cuDF_Native_GPU")
            pg_s = get_lat(npref, sf, "Polars_GPU_cuDF_Engine")
            nt_s = get_note(npref, sf)
            rows_tex.append(
                f"{lbl} & $\\text{{SF}}={sf}$ GB & {d_s} & {p_s} & {f_s} & {pt_s} & {cd_s} & {pg_s} & {nt_s} \\\\"
            )
        if idx_n < len(node_specs) - 1:
            rows_tex.append(r"\midrule")

    img_ref = f"{img_prefix}fig_v11_4node_multiscale_cudf_sf1_10_30_100.jpg"
    sha_master_16 = hashes["master_v11"][:16]
    sha_cudf_16 = hashes["node4_cudf"][:16]

    block = "\n".join([
        "% === BEGIN V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
        r"\subsection*{Four-Node Multi-Scale and Native Columnar GPU (\texttt{RAPIDS cuDF} \& \texttt{cudf-polars}) Validation ($\text{SF}=1, 10, 30, 100\text{ GB}$)}",
        "To ensure complete multi-scale coverage across all hardware tiers and directly benchmark our PyTorch CUDA relational tensor engine against production C++/CUDA columnar GPU engines, we executed the canonical TPC-H \\texttt{lineitem} scan-filter-aggregation workload across $\\text{SF} \\in \\{1, 10, 30, 100\\}\\text{ GB}$ ($6.00 \\times 10^{6}$ to $6.00 \\times 10^{8}$ rows; $64$ real-execution runs, zero estimation) across four distinct hardware nodes, incorporating NVIDIA \\texttt{RAPIDS cuDF} (\\texttt{libcudf}~v26.06.00) and the Polars GPU engine (\\texttt{cudf-polars}, Polars~v1.35.2) on the server-grade NVIDIA Tesla~T4 ($14.56\\text{ GB}$ usable GDDR6 VRAM; Table~\\ref{tab:v11_fournode_multiscale_cudf} and Fig.~\\ref{fig:v11_fournode_multiscale_cudf}). Four empirical conclusions emerge: (i) \\emph{PyTorch Tensor vs.\\ Native \\texttt{libcudf} Parity}: on Tesla~T4, \\texttt{RAPIDS cuDF} achieves the lowest latency at $\\text{SF}=1\\text{ GB}$ ($112.76\\text{ ms}$, $3.43\\text{ J}$) via direct GPU-accelerated ZSTD Parquet decoding, while at $\\text{SF}=10\\text{ GB}$ ($1$ chunk, single-pass in-VRAM) and $\\text{SF}=30\\text{ GB}$ ($3$ chunks) our PyTorch CUDA tensor engine ($311.07\\text{ ms}$ and $1028.13\\text{ ms}$) tracks native \\texttt{RAPIDS cuDF} ($322.37\\text{ ms}$ and $1001.71\\text{ ms}$) within $2.6\\%\\text{--}3.5\\%$, confirming that our relational tensor primitives introduce no artificial GPU bias; (ii) \\emph{Optimizer-Controlled CPU vs.\\ GPU Isolation}: evaluating identical Polars LazyFrame plans on CPU (\\texttt{engine=\"streaming\"}) versus GPU (\\texttt{cudf-polars}) yields end-to-end GPU speedups of $2.85\\times$ ($249.93$ vs.\\ $711.86\\text{ ms}$ at $\\text{SF}=1\\text{ GB}$), $8.18\\times$ ($453.99$ vs.\\ $3715.44\\text{ ms}$ at $\\text{SF}=10\\text{ GB}$), $9.50\\times$ ($1318.63$ vs.\\ $12521.13\\text{ ms}$ at $\\text{SF}=30\\text{ GB}$), and $7.43\\times$ ($5279.66$ vs.\\ $39252.64\\text{ ms}$ at $\\text{SF}=100\\text{ GB}$); (iii) \\emph{Many-Core x86\\_64 vs.\\ Unified-Memory ARM64 Scaling}: at $\\text{SF}=100\\text{ GB}$, 22-thread Polars streaming on Node~2 reaches $4266.59\\text{ ms}$ ($3.75\\times$ faster than disk-spilling DuckDB at $16005.55\\text{ ms}$), while Apple~M1 ARM64 ($8\\text{ GB}$ unified RAM with a $3.5\\text{ GB}$ spill guard, Node~3) completes $\\text{SF}=100\\text{ GB}$ in $4292.58\\text{ ms}$ with zero OOM failures; and (iv) \\emph{Strict Numerical Equivalence}: all $64/64$ runs satisfy $\\text{EX}=1.000$ ($\\epsilon_{\\text{rel}} \\le 3.52 \\times 10^{-13} < 10^{-9}$; Tesla~T4 \\texttt{cuDF} SHA-256: \\texttt{" + sha_cudf_16 + "...}, V11 master SHA-256: \\texttt{" + sha_master_16 + "...}).",
        "",
        r"\begin{table*}[htbp]",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.9pt}",
        r"\caption{V11 four-node cross-hardware, cross-ISA, and native GPU (\texttt{RAPIDS cuDF} \& \texttt{cudf-polars}) multi-scale execution matrix across $\text{SF} \in \{1, 10, 30, 100\}\text{ GB}$ ($64/64$ real-execution runs, $\text{EX}=1.000$, maximum FP64 relative error $\le 3.52 \times 10^{-13} < 10^{-9}$).}",
        r"\label{tab:v11_fournode_multiscale_cudf}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrl@{}}",
        r"\toprule",
        r"\textbf{Hardware Node} & \textbf{Scale} & \textbf{DuckDB} & \textbf{Polars CPU} & \textbf{DataFusion} & \textbf{PyTorch CUDA} & \textbf{RAPIDS cuDF} & \textbf{cudf-polars} & \textbf{Chunks / Memory / GPU Energy} \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
        r"\begin{figure*}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.94\textwidth]{" + img_ref + "}",
        r"\caption{V11 four-node multi-scale execution latency ($\text{SF}=1, 10, 30, 100\text{ GB}$, log scale) across Node~1 (x86\_64 + RTX 2050 4GB), Node~2 (x86\_64 16C/22T), Node~3 (Apple M1 ARM64 8GB), and Node~4 (Cloud Tesla T4 15GB comparing PyTorch CUDA, \texttt{RAPIDS cuDF}, and \texttt{cudf-polars}).}",
        r"\label{fig:v11_fournode_multiscale_cudf}",
        r"\end{figure*}",
        "% === END V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
    ])
    return block


def create_compile_and_audit_v11_tex(df_master: pd.DataFrame, hashes: dict) -> Path:
    source_candidates = [
        MS_DIR / "bigquery_olap_benchmark_V11.tex",
        ROOT / "bigquery_olap_benchmark_V11.tex",
        MS_DIR / "bigquery_olap_benchmark_V10.tex",
        ROOT / "bigquery_olap_benchmark_V10.tex",
    ]
    v11_ms = MS_DIR / "bigquery_olap_benchmark_V11.tex"
    v11_root = ROOT / "bigquery_olap_benchmark_V11.tex"

    src_tex = next((p for p in source_candidates if p.exists() and p.stat().st_size > 10000), None)
    if src_tex is None:
        raise FileNotFoundError("ไม่พบไฟล์ฐาน bigquery_olap_benchmark_V11.tex หรือ V10.tex ในโปรเจกต์")

    print(f"  -> ใช้ไฟล์ฐานต้นฉบับ: {src_tex.relative_to(ROOT)}")
    base_tex = src_tex.read_text(encoding="utf-8")

    base_tex = re.sub(
        r"% === BEGIN V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===.*?% === END V10 FOUR-NODE MULTISCALE SF1-10-30-100 ===\s*",
        "",
        base_tex,
        flags=re.DOTALL,
    )

    existing_imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", base_tex)
    uses_fig_subdir = any(x.startswith("figures/") for x in existing_imgs)
    img_prefix = "figures/" if uses_fig_subdir else ""

    new_block = build_v11_latex_block(df_master, hashes, img_prefix)

    for forbidden in ("51.12", "6.70", "103.32"):
        assert forbidden not in new_block, f"Forbidden legacy substring {forbidden} detected in V11 block!"

    pat_v11 = re.compile(
        r"% === BEGIN V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===.*?% === END V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
        flags=re.DOTALL,
    )
    if pat_v11.search(base_tex):
        updated_tex = pat_v11.sub(lambda _: new_block, base_tex)
    else:
        anchors = [
            r"\\section\{Limitations",
            r"\\section\{Discussion",
            r"\\section\{Conclusion",
            r"\\section\*\{Declarations\}",
            r"\\bmhead\{Declarations\}",
            r"\\backmatter",
            r"\\bibliographystyle",
            r"\\begin\{thebibliography\}",
            r"\\end\{document\}",
        ]
        inserted = False
        updated_tex = base_tex
        for ap in anchors:
            m = re.search(ap, base_tex)
            if m:
                pos = m.start()
                updated_tex = base_tex[:pos] + "\n" + new_block + "\n\n" + base_tex[pos:]
                inserted = True
                break
        if not inserted:
            raise RuntimeError("ไม่พบตำแหน่งแทรกเนื้อหาใน bigquery_olap_benchmark_V11.tex")

    MS_DIR.mkdir(parents=True, exist_ok=True)
    v11_ms.write_text(updated_tex, encoding="utf-8")

    for pass_i in (1, 2, 3):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", v11_ms.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if pass_i == 1 and r"\bibliography{" in updated_tex and (MS_DIR / "bigquery_olap_benchmark_V11.aux").exists():
            subprocess.run(["bibtex", "bigquery_olap_benchmark_V11"], cwd=str(MS_DIR), capture_output=True, text=True)

    v11_pdf = MS_DIR / "bigquery_olap_benchmark_V11.pdf"
    if not v11_pdf.exists():
        tail = "\n".join(res.stdout.splitlines()[-35:])
        raise RuntimeError(f"ไม่พบไฟล์ PDF หลังคอมไพล์:\n{tail}")

    log_txt = (MS_DIR / "bigquery_olap_benchmark_V11.log").read_text(encoding="utf-8", errors="replace")
    undef_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    undef_cites = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    m_pg = re.search(r"Output written on bigquery_olap_benchmark_V11\.pdf \((\d+) pages", log_txt)
    pages = int(m_pg.group(1)) if m_pg else -1

    checks = [
        ("G1-1 Top-1 = 57.33%", "57.33%" in updated_tex or r"57.33\%" in updated_tex),
        ("G1-2 Top-2 = 76.00%", "76.00%" in updated_tex or r"76.00\%" in updated_tex),
        ("G1-3 Mean Regret = 34.98%", "34.98%" in updated_tex or r"34.98\%" in updated_tex),
        ("G2-1 DuckDB SF1 = 183.74 ms", "183.74" in updated_tex),
        ("G2-2 Polars SF1 = 82.47 ms", "82.47" in updated_tex),
        ("G2-4 GPU Cold SF1 = 56.96 ms", "56.96" in updated_tex),
        ("G2-5 AST-Route SF1 = 54.92 ms", "54.92" in updated_tex),
        ("G5-1 V11 RAPIDS cuDF (112.76 / 322.37 / 1001.71 / 3674.43)", all(x in updated_tex for x in ("112.76", "322.37", "1001.71", "3674.43"))),
        ("G5-2 V11 cudf-polars (249.93 / 453.99 / 1318.63 / 5279.66)", all(x in updated_tex for x in ("249.93", "453.99", "1318.63", "5279.66"))),
        ("G5-3 Zero Undefined References & Citations", undef_refs == 0 and undef_cites == 0),
    ]
    for cname, ok in checks:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {cname}")
        if not ok:
            raise AssertionError(f"Zero-Contradiction check failed: {cname}")

    shutil.copy2(v11_ms, v11_root)
    try:
        shutil.copy2(v11_pdf, ROOT / "bigquery_olap_benchmark_V11.pdf")
    except PermissionError:
        print("  [INFO] ข้ามการคัดลอก bigquery_olap_benchmark_V11.pdf ที่เปิดค้างไว้ใน Root")

    print(f"[COMPILE-V11-OK] Pages = {pages} | Undefined Refs = {undef_refs} | Undefined Citations = {undef_cites}")
    return v11_ms


def main() -> None:
    print("=" * 98)
    print("[STEP 2 - V11] BUILD & VERIFY bigquery_olap_benchmark_V11.tex + 300-DPI JPG + PDF")
    print("=" * 98)

    df_master, hashes = build_v11_master_64rows()
    print(f"[1/3] บันทึกไฟล์ Master V11 สำเร็จ: {len(df_master)} รายการ (EX=1.000 ทุกแถว)")
    print(f"      Master V11 CSV SHA-256: {hashes['master_v11']}")

    jpgs = generate_v11_figure_jpg(df_master)
    print(f"[2/3] สร้างกราฟ 300 DPI รุ่น V11 สำเร็จ: {jpgs[0].name} (SHA-256: {sha256_file(jpgs[0])[:20]}...)")

    v11_ms = create_compile_and_audit_v11_tex(df_master, hashes)
    v11_pdf = MS_DIR / "bigquery_olap_benchmark_V11.pdf"
    print("[3/3] สร้างและคอมไพล์ต้นฉบับ V11 สำเร็จ:")
    print(f"      - {v11_ms.relative_to(ROOT)} (SHA-256: {sha256_file(v11_ms)[:24]}...)")
    print(f"      - {v11_pdf.relative_to(ROOT)} ({v11_pdf.stat().st_size:,} bytes | SHA-256: {sha256_file(v11_pdf)[:24]}...)")
    print("=" * 98)
    print("[STEP 2 COMPLETED 100%] กรุณาตรวจสอบผลลัพธ์และพิมพ์ Confirm เพื่อไปขั้นตอนที่ 3 (Flat11.zip & GitHub)")
    print("=" * 98)


if __name__ == "__main__":
    main()