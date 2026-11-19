#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/v13_b1_cloud_gpu_8to16vcpu_r5.py
================================================================================
GROUP B - ITEM 1 (PART B: CLOUD GPU VM 8-16 vCPU + T4/L4/A100, R=5, 6 ENGINES)
  - Isolates true GPU performance (PyTorch CUDA, RAPIDS cuDF, cudf-polars) from
    the 2-vCPU Colab host decode bottleneck.
  - Auto-provisions canonical TPC-H Parquet (SF = 1, 10, 30, 100 GB) in
    memory-safe partitions via DuckDB dbgen.
  - Runs 1 untimed warmup + R=5 timed repetitions per (SF, Engine) cell.
  - Enforces EX=1.000 (FP64 relative error < 1e-9) on every cell & repetition.
  - Reports Mean, SD, CV(%), exact Student-t 95% CI, 10k Bootstrap 95% CI,
    and SHA-256 checksums.
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import platform
import shutil
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
from scipy import stats

try:
    import cudf  # type: ignore
    HAS_CUDF = True
except Exception:
    HAS_CUDF = False

try:
    _ = pl.GPUEngine
    HAS_POLARS_GPU_API = True
except Exception:
    HAS_POLARS_GPU_API = False

CUT_DATE = datetime.date(1995, 1, 1)
CUT_EPOCH_DAYS = (CUT_DATE - datetime.date(1970, 1, 1)).days
BATCH_ROWS = 15_000_000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provision_sf_parts(data_root: Path, sf: int) -> tuple[Path, list[Path], int]:
    """Generates canonical TPC-H lineitem Parquet in memory-safe partitions."""
    parts_dir = data_root / f"sf{sf}" / "lineitem_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(parts_dir.glob("lineitem_part_*.parquet"))
    n_children = 1 if sf == 1 else (4 if sf == 10 else 10)

    if len(existing) != n_children:
        for old_p in existing:
            old_p.unlink(missing_ok=True)
        print(f"[PROVISION SF={sf} GB] Generating {n_children} ZSTD Parquet partition(s) via DuckDB dbgen...")
        for step in range(n_children):
            out_p = parts_dir / f"lineitem_part_{step:02d}.parquet"
            con = duckdb.connect(":memory:")
            con.execute("INSTALL tpch; LOAD tpch;")
            if n_children == 1:
                con.execute(f"CALL dbgen(sf={sf});")
            else:
                con.execute(f"CALL dbgen(sf={sf}, children={n_children}, step={step});")
            con.execute(
                f"COPY (SELECT CAST(l_extendedprice AS DOUBLE) AS l_extendedprice, "
                f"CAST(l_discount AS DOUBLE) AS l_discount, "
                f"CAST(l_quantity AS DOUBLE) AS l_quantity, "
                f"CAST(l_shipdate AS DATE) AS l_shipdate FROM lineitem) "
                f"TO '{out_p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);"
            )
            con.close()
            gc.collect()
        existing = sorted(parts_dir.glob("lineitem_part_*.parquet"))

    total_rows = sum(pq.read_metadata(p).num_rows for p in existing)
    return parts_dir, existing, total_rows


def run_duckdb_once(parts_dir: Path, tmp_spill: Path) -> float:
    tmp_spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"SET temp_directory='{tmp_spill.as_posix()}';")
    glob_p = (parts_dir / "lineitem_part_*.parquet").as_posix()
    sql = (
        f"SELECT SUM(l_extendedprice * (1.0 - l_discount)) "
        f"FROM read_parquet('{glob_p}') "
        f"WHERE l_shipdate >= DATE '1995-01-01' AND l_quantity <= 38.0"
    )
    val = float(con.execute(sql).fetchone()[0])
    con.close()
    return val


def run_polars_cpu_once(part_files: list[Path]) -> float:
    q = (
        pl.scan_parquet(part_files)
        .filter((pl.col("l_shipdate") >= pl.date(1995, 1, 1)) & (pl.col("l_quantity") <= 38.0))
        .select((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum())
    )
    try:
        df_out = q.collect(engine="streaming")
    except TypeError:
        df_out = q.collect(streaming=True)
    return float(df_out.item())


def run_datafusion_once(parts_dir: Path) -> float:
    ctx = datafusion.SessionContext()
    ctx.register_parquet("lineitem", parts_dir.as_posix())
    sql = (
        "SELECT SUM(l_extendedprice * (1.0 - l_discount)) "
        "FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_quantity <= 38.0"
    )
    batches = ctx.sql(sql).collect()
    return float(batches[0].column(0)[0].as_py())


def run_pytorch_cuda_once(part_files: list[Path], device: torch.device) -> tuple[float, float, float, float, int]:
    cols = ["l_extendedprice", "l_discount", "l_quantity", "l_shipdate"]
    decode_ns = 0
    gpu_ns = 0
    n_chunks = 0
    total_acc = torch.zeros(1, dtype=torch.float64, device=device)
    torch.cuda.synchronize()

    t_e2e0 = time.perf_counter_ns()
    for fp in part_files:
        pf = pq.ParquetFile(fp)
        it = pf.iter_batches(batch_size=BATCH_ROWS, columns=cols)
        while True:
            t_d0 = time.perf_counter_ns()
            try:
                batch = next(it)
            except StopIteration:
                break
            ext_np = np.array(pc.cast(batch.column(0), pa.float64()).to_numpy(zero_copy_only=False), copy=True)
            dsc_np = np.array(pc.cast(batch.column(1), pa.float64()).to_numpy(zero_copy_only=False), copy=True)
            qty_np = np.array(pc.cast(batch.column(2), pa.float64()).to_numpy(zero_copy_only=False), copy=True)
            shp_np = np.array(pc.cast(pc.cast(batch.column(3), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False), copy=True)
            decode_ns += (time.perf_counter_ns() - t_d0)

            t_g0 = time.perf_counter_ns()
            shp_g = torch.from_numpy(shp_np).to(device, non_blocking=False)
            qty_g = torch.from_numpy(qty_np).to(device, non_blocking=False)
            mask = (shp_g >= CUT_EPOCH_DAYS) & (qty_g <= 38.0)
            if bool(torch.any(mask)):
                ext_g = torch.from_numpy(ext_np).to(device, non_blocking=False)
                dsc_g = torch.from_numpy(dsc_np).to(device, non_blocking=False)
                total_acc += torch.sum(ext_g[mask] * (1.0 - dsc_g[mask]))
            torch.cuda.synchronize()
            gpu_ns += (time.perf_counter_ns() - t_g0)
            n_chunks += 1

    val = float(total_acc.item())
    t_e2e1 = time.perf_counter_ns()
    return val, gpu_ns / 1e6, decode_ns / 1e6, (t_e2e1 - t_e2e0) / 1e6, n_chunks


def run_rapids_cudf_once(part_files: list[Path]) -> tuple[float, int]:
    cols = ["l_extendedprice", "l_discount", "l_quantity", "l_shipdate"]
    cut_ts = np.datetime64("1995-01-01")
    total_sum = 0.0
    for fp in part_files:
        gdf = cudf.read_parquet(str(fp), columns=cols)
        sub = gdf[(gdf["l_shipdate"] >= cut_ts) & (gdf["l_quantity"] <= 38.0)]
        if len(sub) > 0:
            total_sum += float((sub["l_extendedprice"] * (1.0 - sub["l_discount"])).sum())
        del sub, gdf
    torch.cuda.synchronize()
    return total_sum, len(part_files)


def run_cudf_polars_once(part_files: list[Path]) -> tuple[float, str]:
    gpu_eng = pl.GPUEngine(raise_on_fail=True)
    try:
        q = (
            pl.scan_parquet(part_files)
            .filter((pl.col("l_shipdate") >= pl.date(1995, 1, 1)) & (pl.col("l_quantity") <= 38.0))
            .select((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum())
        )
        val = float(q.collect(engine=gpu_eng).item())
        torch.cuda.synchronize()
        return val, "GPUEngine_SinglePass"
    except Exception:
        # Part-by-part GPU streaming when dataset exceeds VRAM ceiling
        acc = 0.0
        for fp in part_files:
            q = (
                pl.scan_parquet(fp)
                .filter((pl.col("l_shipdate") >= pl.date(1995, 1, 1)) & (pl.col("l_quantity") <= 38.0))
                .select((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum())
            )
            acc += float(q.collect(engine=gpu_eng).item())
        torch.cuda.synchronize()
        return acc, "GPUEngine_PartStream"


def summarize_trials(vals: list[float], rng: np.random.Generator) -> dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    r_cnt = len(arr)
    mean_v = float(np.mean(arr))
    if r_cnt > 1:
        sd_v = float(np.std(arr, ddof=1))
        cv_pct = (sd_v / max(mean_v, 1e-9)) * 100.0
        t_crit = float(stats.t.ppf(0.975, df=r_cnt - 1))
        hw = t_crit * sd_v / np.sqrt(r_cnt)
        t_lo, t_hi = max(mean_v - hw, 0.0), mean_v + hw
        boots = np.mean(rng.choice(arr, size=(10000, r_cnt), replace=True), axis=1)
        b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
    else:
        sd_v, cv_pct, t_lo, t_hi, b_lo, b_hi = 0.0, 0.0, mean_v, mean_v, mean_v, mean_v
    return {
        "Mean_ms": round(mean_v, 2),
        "SD_ms": round(sd_v, 2),
        "CV_Pct": round(cv_pct, 2),
        "t95_CI_Low_ms": round(t_lo, 2),
        "t95_CI_High_ms": round(t_hi, 2),
        "Boot95_CI_Low_ms": round(float(b_lo), 2),
        "Boot95_CI_High_ms": round(float(b_hi), 2),
        "Min_ms": round(float(np.min(arr)), 2),
        "Max_ms": round(float(np.max(arr)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfs", type=int, nargs="+", default=[1, 10, 30, 100])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--data-dir", type=str, default="data_cloud_r5")
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument("--cleanup-after-sf", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required on Cloud GPU VM.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    vcpus = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    node_tag = f"CloudVM_{vcpus}vCPU_{int(round(ram_gb))}GBRAM_{gpu_name.replace(' ', '_')}_{vram_gb}GBVRAM"

    data_root = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_spill = data_root / "_duckdb_spill"

    # Probe cudf-polars availability on a tiny frame
    has_cudf_polars = False
    if HAS_POLARS_GPU_API:
        try:
            _ = pl.DataFrame({"a": [1.0, 2.0]}).lazy().select(pl.col("a").sum()).collect(engine=pl.GPUEngine(raise_on_fail=True))
            has_cudf_polars = True
        except Exception:
            has_cudf_polars = False

    print("=" * 98)
    print(f"CLOUD GPU VM R={args.reps} BENCHMARK | {node_tag}")
    print(f"Engines: DuckDB, Polars, DataFusion, PyTorch_CUDA | RAPIDS_cuDF={HAS_CUDF} | cudf_polars={has_cudf_polars}")
    print("=" * 98)

    raw_rows: list[dict] = []
    sum_rows: list[dict] = []
    rng = np.random.default_rng(20261120)

    for sf in args.sfs:
        parts_dir, part_files, n_rows = provision_sf_parts(data_root, sf)
        first_sha = sha256_file(part_files[0])[:20]
        print("-" * 98)
        print(f"[SF={sf} GB | Rows={n_rows:,} | Parts={len(part_files)} | Part0 SHA256={first_sha}...]")

        # Untimed warmup + golden FP64 reference
        ref_val = run_duckdb_once(parts_dir, tmp_spill)
        _ = run_polars_cpu_once(part_files)
        _ = run_datafusion_once(parts_dir)
        _ = run_pytorch_cuda_once(part_files, device)
        if HAS_CUDF:
            _ = run_rapids_cudf_once(part_files)
        if has_cudf_polars:
            _ = run_cudf_polars_once(part_files)
        gc.collect()

        eng_keys = [
            "DuckDB_Columnar_C++",
            "Polars_Lazy_Streaming_Rust",
            "Apache_DataFusion_Rust",
            "PyTorch_CUDA_PostArrow_H2D_Kernel",
            "PyTorch_CUDA_Full_Storage_ZSTD_E2E",
        ]
        if HAS_CUDF:
            eng_keys.append("RAPIDS_cuDF_Native_GPU")
        if has_cudf_polars:
            eng_keys.append("Polars_GPU_cudf_polars")
        eng_lats: dict[str, list[float]] = {k: [] for k in eng_keys}
        cp_mode = "N/A"
        pt_chunks = 1

        for r_idx in range(1, args.reps + 1):
            t0 = time.perf_counter_ns(); v_dk = run_duckdb_once(parts_dir, tmp_spill); ms_dk = (time.perf_counter_ns() - t0) / 1e6
            t0 = time.perf_counter_ns(); v_pl = run_polars_cpu_once(part_files); ms_pl = (time.perf_counter_ns() - t0) / 1e6
            t0 = time.perf_counter_ns(); v_df = run_datafusion_once(parts_dir); ms_df = (time.perf_counter_ns() - t0) / 1e6
            v_pt, ms_pt_post, ms_pt_dec, ms_pt_full, pt_chunks = run_pytorch_cuda_once(part_files, device)

            checks = [("DuckDB", v_dk), ("Polars", v_pl), ("DataFusion", v_df), ("PyTorch_CUDA", v_pt)]
            eng_lats["DuckDB_Columnar_C++"].append(ms_dk)
            eng_lats["Polars_Lazy_Streaming_Rust"].append(ms_pl)
            eng_lats["Apache_DataFusion_Rust"].append(ms_df)
            eng_lats["PyTorch_CUDA_PostArrow_H2D_Kernel"].append(ms_pt_post)
            eng_lats["PyTorch_CUDA_Full_Storage_ZSTD_E2E"].append(ms_pt_full)

            rep_map = {
                "DuckDB_Columnar_C++": (ms_dk, 0.0, 0, "CPU_Vectorized"),
                "Polars_Lazy_Streaming_Rust": (ms_pl, 0.0, 0, "CPU_Streaming"),
                "Apache_DataFusion_Rust": (ms_df, 0.0, 0, "CPU_ArrowSQL"),
                "PyTorch_CUDA_PostArrow_H2D_Kernel": (ms_pt_post, ms_pt_dec, pt_chunks, "GPU_PostArrow"),
                "PyTorch_CUDA_Full_Storage_ZSTD_E2E": (ms_pt_full, ms_pt_dec, pt_chunks, "GPU_FullZSTD"),
            }

            if HAS_CUDF:
                t0 = time.perf_counter_ns()
                v_cu, cu_chunks = run_rapids_cudf_once(part_files)
                ms_cu = (time.perf_counter_ns() - t0) / 1e6
                checks.append(("RAPIDS_cuDF", v_cu))
                eng_lats["RAPIDS_cuDF_Native_GPU"].append(ms_cu)
                rep_map["RAPIDS_cuDF_Native_GPU"] = (ms_cu, 0.0, cu_chunks, "GPU_NativeParquet")

            if has_cudf_polars:
                t0 = time.perf_counter_ns()
                v_cp, cp_mode = run_cudf_polars_once(part_files)
                ms_cp = (time.perf_counter_ns() - t0) / 1e6
                checks.append(("cudf_polars", v_cp))
                eng_lats["Polars_GPU_cudf_polars"].append(ms_cp)
                rep_map["Polars_GPU_cudf_polars"] = (ms_cp, 0.0, len(part_files), cp_mode)

            for nm, vv in checks:
                rel_e = abs(vv - ref_val) / max(abs(ref_val), 1e-12)
                if rel_e >= 1e-9:
                    raise AssertionError(f"EX failure SF={sf} rep={r_idx} {nm}: rel_err={rel_e:.2e}")

            for ek, (lat_v, dec_v, chk_v, md_v) in rep_map.items():
                raw_rows.append({
                    "Hardware_Node": node_tag, "vCPUs": vcpus, "RAM_GB": ram_gb, "GPU": gpu_name,
                    "VRAM_GB": vram_gb, "Scale_Factor_GB": sf, "Rows": n_rows, "Engine": ek,
                    "Mode": md_v, "Repetition": r_idx, "Latency_ms": round(lat_v, 3),
                    "Host_Decode_ms": round(dec_v, 3), "Chunks": chk_v, "EX_Exact": 1.0,
                })

            cu_str = f" | cuDF={eng_lats['RAPIDS_cuDF_Native_GPU'][-1]:7.1f}ms" if HAS_CUDF else ""
            cp_str = f" | cudf-polars={eng_lats['Polars_GPU_cudf_polars'][-1]:7.1f}ms" if has_cudf_polars else ""
            print(f"  [Rep {r_idx}/{args.reps}] DuckDB={ms_dk:7.1f}ms | Polars={ms_pl:7.1f}ms | DF={ms_df:7.1f}ms | PT_Post={ms_pt_post:7.1f}ms{cu_str}{cp_str}")

        for ek in eng_keys:
            st = summarize_trials(eng_lats[ek], rng)
            sum_rows.append({
                "Hardware_Node": node_tag, "vCPUs": vcpus, "GPU": gpu_name, "VRAM_GB": vram_gb,
                "Scale_Factor_GB": sf, "Rows": n_rows, "Engine": ek, "Repetitions_R": args.reps,
                **st,
            })

        if args.cleanup_after_sf:
            shutil.rmtree(data_root / f"sf{sf}", ignore_errors=True)

    out_raw = out_dir / "v13_groupB_b1_cloud_gpu_8to16vcpu_r5_raw_trials.csv"
    out_sum = out_dir / "v13_groupB_b1_cloud_gpu_8to16vcpu_r5_summary_ci.csv"
    pd.DataFrame(raw_rows).to_csv(out_raw, index=False, encoding="utf-8")
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False, encoding="utf-8")

    print("=" * 98)
    print(df_sum[["Scale_Factor_GB", "Engine", "Repetitions_R", "Mean_ms", "SD_ms", "CV_Pct", "t95_CI_Low_ms", "t95_CI_High_ms"]].to_string(index=False))
    print("-" * 98)
    print(f"[SAVED RAW    ] {out_raw} | SHA-256: {sha256_file(out_raw)}")
    print(f"[SAVED SUMMARY] {out_sum} | SHA-256: {sha256_file(out_sum)}")
    print("=" * 98)


if __name__ == "__main__":
    main()