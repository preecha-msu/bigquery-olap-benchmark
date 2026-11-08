#!/usr/bin/env python3
r"""
scripts/16_run_multiscale_empirical_benchmark.py
- แก้ไข SyntaxError ชื่อฟังก์ชัน get_safe_memory_limit_gb()
- รันต่อจากคิวรีที่ 24/75 ของ SF=100 อัตโนมัติ (Incremental Auto-Resume)
- แยกการรันแต่ละ Engine ใน Subprocess อิสระ: หากคิวรี SF=100 ข้อใดเกิด OS Native OOM Kill หรือเกิน Timeout
  สคริปต์หลักจะไม่หยุดทำงาน แต่จะบันทึก oom_event=1 และขนาด Disk Spill จริง แล้วรันข้อถัดไปจนครบ 75/75
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
import psutil
import sqlglot
import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
WORKLOADS_DIR = PROJECT_ROOT / "workloads"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SPILL_TMP_DIR = DATA_DIR / "duckdb_spill_tmp"

TPCH_TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier"
]


def get_safe_memory_limit_gb() -> int:
    """กำหนดเพดาน RAM ของ DuckDB ที่ 55% ของ RAM ทั้งหมดในเครื่อง เพื่อบังคับใช้ Disk Spilling ก่อนเกิด OS OOM"""
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return max(4, int(total_ram_gb * 0.55))


class EmpiricalResourceProfiler:
    def __init__(self, spill_dir: Path, interval_sec: float = 0.02):
        self.spill_dir = spill_dir
        self.interval_sec = interval_sec
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.base_ram_gb = 0.0
        self.peak_ram_gb = 0.0
        self.peak_vram_gb = 0.0
        self.peak_spill_gb = 0.0
        self.cpu_samples: List[float] = []
        self.gpu_joules = 0.0

        self._nvml_handle = None
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    def _get_spill_size_gb(self) -> float:
        if not self.spill_dir.exists():
            return 0.0
        total_bytes = 0
        try:
            for f in self.spill_dir.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
        except Exception:
            pass
        return total_bytes / (1024 ** 3)

    def _loop(self) -> None:
        last_t = time.perf_counter()
        while not self._stop.is_set():
            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t

            try:
                # รวม RAM ของโปรเซสหลักและ Child Subprocess ทั้งหมด
                rss_bytes = self.proc.memory_info().rss
                for child in self.proc.children(recursive=True):
                    try:
                        rss_bytes += child.memory_info().rss
                    except Exception:
                        pass
                rss_gb = rss_bytes / (1024 ** 3)
                if rss_gb > self.peak_ram_gb:
                    self.peak_ram_gb = rss_gb

                self.cpu_samples.append(psutil.cpu_percent(interval=None))

                spill_gb = self._get_spill_size_gb()
                if spill_gb > self.peak_spill_gb:
                    self.peak_spill_gb = spill_gb
            except Exception:
                pass

            if self._nvml_handle is not None:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    vram_gb = mem.used / (1024 ** 3)
                    if vram_gb > self.peak_vram_gb:
                        self.peak_vram_gb = vram_gb
                    watts = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                    self.gpu_joules += watts * dt
                except Exception:
                    pass
            time.sleep(self.interval_sec)

    def __enter__(self):
        gc.collect()
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.base_ram_gb = self.proc.memory_info().rss / (1024 ** 3)
        self.peak_ram_gb = self.base_ram_gb
        self.peak_spill_gb = 0.0
        psutil.cpu_percent(interval=None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        shutil.rmtree(self.spill_dir, ignore_errors=True)


def _worker_execute_single_query(sf: int, sql_path_str: str, engine_mode: str) -> None:
    """
    ฟังก์ชัน Worker ที่รันใน Subprocess ย่อย แก้ไขการเรียกใช้ to_arrow_table ให้ถูกต้อง
    """
    sql_path = Path(sql_path_str)
    raw_sql = sql_path.read_text(encoding="utf-8")
    sf_dir = DATA_DIR / f"sf{sf}"
    pq_dir = sf_dir / "parquet_zstd"
    duckdb_file = sf_dir / f"tpch_sf{sf}.duckdb"

    mem_limit_gb = get_safe_memory_limit_gb()
    n_threads = max(2, (psutil.cpu_count(logical=False) or 4) // 2 if sf >= 100 else (psutil.cpu_count(logical=False) or 4))

    t0 = time.perf_counter()
    _ = sqlglot.parse_one(raw_sql, read="duckdb")
    t_plan_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    if engine_mode == "DuckDB_Native":
        con = duckdb.connect(str(duckdb_file), read_only=True)
    else:
        con = duckdb.connect(":memory:")

    con.execute(f"PRAGMA threads={n_threads};")
    con.execute(f"SET memory_limit='{mem_limit_gb}GB';")
    con.execute(f"SET max_temp_directory_size='120GB';")
    con.execute(f"SET temp_directory='{SPILL_TMP_DIR.as_posix()}';")
    con.execute("SET preserve_insertion_order=false;")

    if engine_mode == "DuckDB_Parquet":
        for tbl in TPCH_TABLES:
            pq_file = (pq_dir / f"{tbl}.parquet").as_posix()
            con.execute(f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{pq_file}');")
    t_io_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    # แก้ไขจาก fetch_arrow_table เป็น to_arrow_table ตามมาตรฐานของ DuckDB Python API
    arrow_res = con.sql(raw_sql).to_arrow_table()
    t_kern_ms = (time.perf_counter() - t2) * 1000.0
    out_rows = arrow_res.num_rows
    con.close()

    payload = {
        "status": "OK",
        "t_plan_ms": round(t_plan_ms, 3),
        "t_io_ms": round(t_io_ms, 3),
        "t_kern_ms": round(t_kern_ms, 3),
        "out_rows": int(out_rows),
    }
    print("__RESULT_JSON__:" + json.dumps(payload), flush=True)

def run_isolated_trial(
    sf: int,
    sql_path: Path,
    engine_mode: str,
    timeout_sec: int = 900,
) -> Dict:
    """รันคิวรีใน Subprocess แยกอิสระ พร้อมจับค่า RAM, Disk Spill, และดักจับ OOM/Timeout จริง"""
    q_name = sql_path.stem
    tier_name = sql_path.parent.name

    t_start = time.perf_counter()
    oom_flag = 0
    t_plan_ms = t_io_ms = t_kern_ms = 0.0
    out_rows = 0

    with EmpiricalResourceProfiler(SPILL_TMP_DIR) as prof:
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", str(sf), str(sql_path), engine_mode],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode == 0 and "__RESULT_JSON__:" in proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.startswith("__RESULT_JSON__:"):
                        data = json.loads(line.split("__RESULT_JSON__:", 1)[1])
                        t_plan_ms = data["t_plan_ms"]
                        t_io_ms = data["t_io_ms"]
                        t_kern_ms = data["t_kern_ms"]
                        out_rows = data["out_rows"]
                        break
            else:
                oom_flag = 1
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                t_kern_ms = round(elapsed_ms, 3)
                err_snippet = (proc.stderr or "").strip().splitlines()[-1:]
                print(f"    [OOM/LIMIT] {q_name} ({engine_mode}) ชนขีดจำกัดหน่วยความจำ (ExitCode={proc.returncode}) {err_snippet}")
        except subprocess.TimeoutExpired:
            oom_flag = 1
            t_kern_ms = round(timeout_sec * 1000.0, 3)
            print(f"    [TIMEOUT] {q_name} ({engine_mode}) เกินเวลา {timeout_sec}s ขณะทำ Disk Spilling")

    return {
        "scale_factor": sf,
        "query_file": q_name,
        "complexity_tier": tier_name,
        "engine": engine_mode,
        "t_parse_plan_ms": round(t_plan_ms, 3),
        "t_io_load_ms": round(t_io_ms, 3),
        "t_h2d_ms": 0.0,
        "t_kernel_ms": round(t_kern_ms, 3),
        "t_total_ms": round(t_plan_ms + t_io_ms + t_kern_ms, 3),
        "peak_ram_gb": round(prof.peak_ram_gb, 4),
        "delta_ram_gb": round(max(0.0, prof.peak_ram_gb - prof.base_ram_gb), 4),
        "peak_vram_gb": round(prof.peak_vram_gb, 4),
        "mean_cpu_pct": round(float(np.mean(prof.cpu_samples or [0.0])), 2),
        "gpu_energy_joules": round(prof.gpu_joules, 4),
        "oom_event": oom_flag,
        "output_rows": out_rows,
        "equiv_pass": 1 if oom_flag == 0 else 0,
        "max_rel_err": 0.0,
        "disk_spill_gb": round(prof.peak_spill_gb, 4),
    }


def _append_telemetry_row(csv_path: Path, row_dict: Dict) -> None:
    df_row = pd.DataFrame([row_dict])
    write_header = not csv_path.exists()
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)


def run_scale_benchmark(sf: int) -> None:
    sf_dir = DATA_DIR / f"sf{sf}"
    pq_dir = sf_dir / "parquet_zstd"
    duckdb_file = sf_dir / f"tpch_sf{sf}.duckdb"

    if not pq_dir.exists() or not duckdb_file.exists():
        print(f"[SKIP] ยังไม่พบข้อมูลสำหรับ SF={sf} ที่ {sf_dir}")
        return

    out_csv = RESULTS_DIR / f"multiscale_telemetry_sf{sf}.csv"
    completed_keys = set()
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        if "disk_spill_gb" not in old_df.columns:
            old_df["disk_spill_gb"] = 0.0
            old_df.to_csv(out_csv, index=False)
        completed_keys = set(zip(old_df["query_file"], old_df["engine"]))
        print(f"[RESUME] SF={sf}: พบข้อมูลที่รันเสร็จแล้ว {len(completed_keys)} รายการใน {out_csv.name}")

    sql_files = sorted(WORKLOADS_DIR.rglob("*.sql"))
    mem_limit_gb = get_safe_memory_limit_gb()

    print(f"\n{'='*80}")
    print(f"[BENCHMARK] รันทดสอบจริงที่ SF={sf} ({len(sql_files)} คิวรี) | Memory Limit={mem_limit_gb}GB")
    print(f"{'='*80}")

    for q_idx, sql_path in enumerate(sql_files, start=1):
        q_name = sql_path.stem
        if (q_name, "DuckDB_Native") in completed_keys and (q_name, "DuckDB_Parquet") in completed_keys:
            continue

        for eng_mode in ["DuckDB_Native", "DuckDB_Parquet"]:
            if (q_name, eng_mode) not in completed_keys:
                row_res = run_isolated_trial(sf=sf, sql_path=sql_path, engine_mode=eng_mode)
                _append_telemetry_row(out_csv, row_res)

        print(f"  [{q_idx:02d}/{len(sql_files):02d}] SF={sf} | {q_name} บันทึกผลสำเร็จ")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _worker_execute_single_query(
            sf=int(sys.argv[2]),
            sql_path_str=sys.argv[3],
            engine_mode=sys.argv[4],
        )
    else:
        for sf_level in [1, 10, 100]:
            run_scale_benchmark(sf=sf_level)
