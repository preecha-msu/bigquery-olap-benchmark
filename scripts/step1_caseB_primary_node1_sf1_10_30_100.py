#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step1_caseB_primary_node1_sf1_10_30_100.py
ขั้นตอนที่ 1 (กรณี B - Primary Node 1: PREECHA-IT-MSU):
1. ตรวจสอบและสร้างไฟล์ Parquet ZSTD สำหรับ SF = 1, 10, 30, 100 GB (สร้างเฉพาะสเกลที่ยังไม่มี)
2. รันวัดผลจริงครบ 4 สเกล (1, 10, 30, 100 GB) บน DuckDB, Polars, Apache DataFusion, และ PyTorch CUDA GPU (RTX 2050)
3. ตรวจสอบความเท่าเทียมของผลลัพธ์ (EX = 1.000, FP64 Relative Error < 1e-9) และบันทึก SHA-256
"""

import gc
import hashlib
import os
import platform
import shutil
import socket
import time
from pathlib import Path

import datafusion
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SPILL_DIR = DATA_DIR / "_duckdb_spill_tmp"
TARGET_SCALES = [1, 10, 30, 100]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_lineitem_parquet(sf: int) -> Path:
    sf_dir = DATA_DIR / f"sf{sf}"
    pq_dir = sf_dir / "parquet_zstd"
    pq_dir.mkdir(parents=True, exist_ok=True)
    SPILL_DIR.mkdir(parents=True, exist_ok=True)

    li_pq = pq_dir / "lineitem.parquet"
    if li_pq.exists() and li_pq.stat().st_size > 1024 * 1024:
        print(f"[CACHE-OK] พบไฟล์ SF={sf} GB แล้วที่: {li_pq} ({li_pq.stat().st_size / (1024**3):.2f} GB)")
        return li_pq

    print(f"\n[PROVISION] ไม่พบ {li_pq} -> กำลังสร้างข้อมูลจริง TPC-H SF={sf} GB ด้วย DuckDB...")
    db_path = sf_dir / f"tpch_sf{sf}_gen.duckdb"
    threads = max(2, psutil.cpu_count(logical=True) or 8)
    mem_gb = max(4, int((psutil.virtual_memory().total / (1024 ** 3)) * 0.55))

    con = duckdb.connect(str(db_path))
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"SET memory_limit='{mem_gb}GB';")
    con.execute(f"SET temp_directory='{SPILL_DIR.as_posix()}';")
    con.execute("SET max_temp_directory_size='150GB';")
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    con.execute(
        f"COPY lineitem TO '{li_pq.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);"
    )
    con.close()
    if db_path.exists():
        db_path.unlink(missing_ok=True)
    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    print(f"[PROVISION-DONE] สร้าง {li_pq} สำเร็จ ({li_pq.stat().st_size / (1024**3):.2f} GB)")
    return li_pq


def benchmark_scale_on_primary_node(sf: int, li_pq: Path) -> list[dict]:
    proc = psutil.Process(os.getpid())
    hostname = socket.gethostname()
    pf = pq.ParquetFile(li_pq)
    n_rows = pf.metadata.num_rows
    n_rg = pf.num_row_groups
    pq_gb = li_pq.stat().st_size / (1024 ** 3)
    li_sha = sha256_file(li_pq)

    threads = max(2, psutil.cpu_count(logical=True) or 8)
    SPILL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 96)
    print(f"[NODE 1 ({hostname}) | SF={sf} GB] Rows={n_rows:,} | RowGroups={n_rg} | Parquet={pq_gb:.2f} GB")
    print("=" * 96)

    records = []

    # 1. DuckDB (Reference Checksum + Out-of-Core Spill Guard)
    gc.collect()
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={threads};")
    con.execute("SET memory_limit='9GB';")
    con.execute(f"SET temp_directory='{SPILL_DIR.as_posix()}';")
    ram_b = proc.memory_info().rss / (1024 ** 3)
    t0 = time.perf_counter()
    res_duck = con.execute(f"""
        SELECT
            COUNT(*) AS cnt,
            SUM(CAST(l_quantity AS DOUBLE)) AS sum_qty,
            SUM(CAST(l_extendedprice AS DOUBLE) * (1.0 - CAST(l_discount AS DOUBLE))) AS sum_disc_price,
            SUM(CAST(l_extendedprice AS DOUBLE) * (1.0 - CAST(l_discount AS DOUBLE)) * (1.0 + CAST(l_tax AS DOUBLE))) AS sum_charge
        FROM read_parquet('{li_pq.as_posix()}')
        WHERE l_shipdate <= DATE '1998-09-02'
    """).fetchone()
    duck_ms = (time.perf_counter() - t0) * 1000.0
    ram_a = proc.memory_info().rss / (1024 ** 3)
    con.close()
    ref_sum_disc = float(res_duck[2])
    ref_cnt = int(res_duck[0])

    records.append({
        "Hardware_Node": f"Node1_{hostname}",
        "Scale_Factor_GB": sf,
        "Lineitem_Rows": n_rows,
        "Engine": "DuckDB_Columnar_C++",
        "Execution_Mode": "In-Memory_Vectorized" if sf <= 10 else "Out-of-Core_Morsel_Spill",
        "Latency_ms": round(duck_ms, 2),
        "Peak_Host_RAM_GB": round(max(ram_b, ram_a), 3),
        "Peak_GPU_VRAM_GB": 0.0,
        "VRAM_Chunks": 0,
        "Naive_FullLoad_OOM": 0,
        "PCIe_H2D_ms": 0.0,
        "GPU_Warm_Kernel_ms": 0.0,
        "Phi_PCIe_Pct": 0.0,
        "Max_FP64_Rel_Error": 0.0,
        "EX_Equivalence": 1.0,
        "Parquet_SHA256": li_sha[:24],
    })

    # 2. Polars Lazy Streaming Engine (Rust Arrow)
    gc.collect()
    ram_b_pl = proc.memory_info().rss / (1024 ** 3)
    t0 = time.perf_counter()
    pl_q = (
        pl.scan_parquet(str(li_pq))
        .filter(pl.col("l_shipdate") <= pl.lit("1998-09-02").str.to_date())
        .select([
            pl.len().alias("cnt"),
            pl.col("l_quantity").cast(pl.Float64).sum().alias("sum_qty"),
            (pl.col("l_extendedprice").cast(pl.Float64) * (1.0 - pl.col("l_discount").cast(pl.Float64))).sum().alias("sum_disc_price"),
        ])
    )
    try:
        df_pl = pl_q.collect(engine="streaming")
    except Exception:
        df_pl = pl_q.collect(streaming=True)
    pl_ms = (time.perf_counter() - t0) * 1000.0
    ram_a_pl = proc.memory_info().rss / (1024 ** 3)
    pl_sum_disc = float(df_pl["sum_disc_price"][0])
    pl_rel_err = abs(pl_sum_disc - ref_sum_disc) / max(abs(ref_sum_disc), 1e-9)

    records.append({
        "Hardware_Node": f"Node1_{hostname}",
        "Scale_Factor_GB": sf,
        "Lineitem_Rows": n_rows,
        "Engine": "Polars_Lazy_Streaming_Rust",
        "Execution_Mode": "In-Memory_Arrow" if sf <= 10 else "Out-of-Core_Streaming",
        "Latency_ms": round(pl_ms, 2),
        "Peak_Host_RAM_GB": round(max(ram_b_pl, ram_a_pl), 3),
        "Peak_GPU_VRAM_GB": 0.0,
        "VRAM_Chunks": 0,
        "Naive_FullLoad_OOM": 0,
        "PCIe_H2D_ms": 0.0,
        "GPU_Warm_Kernel_ms": 0.0,
        "Phi_PCIe_Pct": 0.0,
        "Max_FP64_Rel_Error": float(f"{pl_rel_err:.3e}"),
        "EX_Equivalence": 1.0 if pl_rel_err < 1e-9 else 0.0,
        "Parquet_SHA256": li_sha[:24],
    })

    # 3. Apache DataFusion (Rust Arrow Partitioned Batch Stream)
    gc.collect()
    ctx = datafusion.SessionContext()
    ctx.register_parquet("lineitem", str(li_pq))
    ram_b_df = proc.memory_info().rss / (1024 ** 3)
    t0 = time.perf_counter()
    df_batches = ctx.sql("""
        SELECT
            COUNT(*) AS cnt,
            SUM(CAST(l_quantity AS DOUBLE)) AS sum_qty,
            SUM(CAST(l_extendedprice AS DOUBLE) * (1.0 - CAST(l_discount AS DOUBLE))) AS sum_disc_price
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-09-02'
    """).collect()
    df_ms = (time.perf_counter() - t0) * 1000.0
    ram_a_df = proc.memory_info().rss / (1024 ** 3)
    df_tbl = pa.Table.from_batches(df_batches)
    df_sum_disc = float(df_tbl["sum_disc_price"][0].as_py())
    df_rel_err = abs(df_sum_disc - ref_sum_disc) / max(abs(ref_sum_disc), 1e-9)
    del ctx
    gc.collect()

    records.append({
        "Hardware_Node": f"Node1_{hostname}",
        "Scale_Factor_GB": sf,
        "Lineitem_Rows": n_rows,
        "Engine": "Apache_DataFusion_Rust",
        "Execution_Mode": "In-Memory_Arrow" if sf <= 10 else "Partitioned_Batch_Stream",
        "Latency_ms": round(df_ms, 2),
        "Peak_Host_RAM_GB": round(max(ram_b_df, ram_a_df), 3),
        "Peak_GPU_VRAM_GB": 0.0,
        "VRAM_Chunks": 0,
        "Naive_FullLoad_OOM": 0,
        "PCIe_H2D_ms": 0.0,
        "GPU_Warm_Kernel_ms": 0.0,
        "Phi_PCIe_Pct": 0.0,
        "Max_FP64_Rel_Error": float(f"{df_rel_err:.3e}"),
        "EX_Equivalence": 1.0 if df_rel_err < 1e-9 else 0.0,
        "Parquet_SHA256": li_sha[:24],
    })

    # 4. PyTorch CUDA GPU (RTX 2050 4.0 GB VRAM: Single-Pass @ SF=1 vs. Batched PCIe Stream @ SF=10, 30, 100)
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        naive_oom = 0
        if sf >= 10:
            try:
                _test = torch.empty((n_rows, 12), dtype=torch.float64, device=dev)
                del _test
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    naive_oom = 1
                    torch.cuda.empty_cache()

        # กำหนดจำนวน Chunks ตามเพดาน 4.0 GB VRAM: SF=1->1, SF=10->4, SF=30->12, SF=100->40
        target_chunks = {1: 1, 10: 4, 30: 12, 100: 40}.get(sf, max(1, sf // 2))
        rg_step = max(1, int(np.ceil(n_rg / target_chunks)))
        actual_chunks = 0

        torch.cuda.reset_peak_memory_stats()
        t_h2d_ms = 0.0
        t_kern_ms = 0.0
        gpu_sum_disc = 0.0
        cols = ["l_quantity", "l_extendedprice", "l_discount", "l_shipdate"]
        cutoff_days = (np.datetime64("1998-09-02") - np.datetime64("1970-01-01")).astype(np.int32)

        for rg_start in range(0, n_rg, rg_step):
            actual_chunks += 1
            rg_idx = list(range(rg_start, min(rg_start + rg_step, n_rg)))
            tbl_chunk = pf.read_row_groups(rg_idx, columns=cols)

            p_np = pc.cast(tbl_chunk["l_extendedprice"], pa.float64()).to_numpy(zero_copy_only=False)
            d_np = pc.cast(tbl_chunk["l_discount"], pa.float64()).to_numpy(zero_copy_only=False)
            s_np = pc.cast(tbl_chunk["l_shipdate"], pa.date32()).to_numpy(zero_copy_only=False).astype(np.int32)

            ev_s = torch.cuda.Event(enable_timing=True)
            ev_h = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)

            ev_s.record()
            t_p = torch.from_numpy(p_np).to(dev, non_blocking=True)
            t_d = torch.from_numpy(d_np).to(dev, non_blocking=True)
            t_s = torch.from_numpy(s_np).to(dev, non_blocking=True)
            ev_h.record()

            mask = t_s <= int(cutoff_days)
            disc_p = torch.where(mask, t_p * (1.0 - t_d), torch.zeros_like(t_p))
            chunk_sum = torch.sum(disc_p)
            ev_e.record()
            torch.cuda.synchronize()

            gpu_sum_disc += float(chunk_sum.cpu().item())
            t_h2d_ms += float(ev_s.elapsed_time(ev_h))
            t_kern_ms += float(ev_h.elapsed_time(ev_e))
            del t_p, t_d, t_s, mask, disc_p, chunk_sum, tbl_chunk

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_gpu_ms = t_h2d_ms + t_kern_ms
        phi_pcie = (t_h2d_ms / max(total_gpu_ms, 1e-6)) * 100.0
        gpu_rel_err = abs(gpu_sum_disc - ref_sum_disc) / max(abs(ref_sum_disc), 1e-9)

        records.append({
            "Hardware_Node": f"Node1_{hostname}",
            "Scale_Factor_GB": sf,
            "Lineitem_Rows": n_rows,
            "Engine": f"PyTorch_CUDA_RTX2050_{vram_total_gb:.1f}GB",
            "Execution_Mode": "In-VRAM_SinglePass" if actual_chunks == 1 else f"Batched_PCIe_Stream_{actual_chunks}Chunks",
            "Latency_ms": round(total_gpu_ms, 2),
            "Peak_Host_RAM_GB": round(proc.memory_info().rss / (1024 ** 3), 3),
            "Peak_GPU_VRAM_GB": round(peak_vram, 3),
            "VRAM_Chunks": actual_chunks,
            "Naive_FullLoad_OOM": naive_oom,
            "PCIe_H2D_ms": round(t_h2d_ms, 2),
            "GPU_Warm_Kernel_ms": round(t_kern_ms, 2),
            "Phi_PCIe_Pct": round(phi_pcie, 2),
            "Max_FP64_Rel_Error": float(f"{gpu_rel_err:.3e}"),
            "EX_Equivalence": 1.0 if gpu_rel_err < 1e-9 else 0.0,
            "Parquet_SHA256": li_sha[:24],
        })
        torch.cuda.empty_cache()

    for r in records:
        print(
            f"  -> [{r['Engine']:<30}] Mode={r['Execution_Mode']:<28} | "
            f"Latency={r['Latency_ms']:>8.2f} ms | RAM={r['Peak_Host_RAM_GB']:>5.2f} GB | "
            f"VRAM={r['Peak_GPU_VRAM_GB']:>4.2f} GB | RelErr={r['Max_FP64_Rel_Error']:.2e} | EX={r['EX_Equivalence']}"
        )

    return records


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    for sf in TARGET_SCALES:
        li_pq = ensure_lineitem_parquet(sf)
        all_records.extend(benchmark_scale_on_primary_node(sf, li_pq))

    df_out = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / "multiscale_gpu_vram_and_arrow_streaming_sf1_10_30_100.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")

    # อัปเดตไฟล์เดิมที่เคยมีแค่ 1, 10, 100 ให้มีครบ 1, 10, 30, 100 ด้วย
    legacy_csv = RESULTS_DIR / "multiscale_gpu_vram_and_arrow_streaming.csv"
    df_out.to_csv(legacy_csv, index=False, encoding="utf-8")

    csv_sha = sha256_file(out_csv)
    print("\n" + "=" * 96)
    print(f"[STEP 1 (CASE B) COMPLETED] บันทึกผลครบทั้ง 4 สเกล (SF=1, 10, 30, 100 GB) เรียบร้อยแล้ว!")
    print(f"  -> Output CSV : {out_csv}")
    print(f"  -> CSV SHA-256: {csv_sha}")
    print("=" * 96)
    print(df_out[[
        "Scale_Factor_GB", "Engine", "Execution_Mode", "Latency_ms",
        "Peak_Host_RAM_GB", "Peak_GPU_VRAM_GB", "VRAM_Chunks", "Phi_PCIe_Pct", "EX_Equivalence"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
