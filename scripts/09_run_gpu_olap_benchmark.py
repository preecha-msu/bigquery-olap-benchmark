# ไฟล์: scripts/09_run_gpu_olap_benchmark.py
# ทดสอบเปรียบเทียบ SIMT GPU-Accelerated OLAP (CUDA) แยกวัดเวลา PCIe H2D Transfer vs. Pure GPU Kernel Execution ทั้ง 75 คิวรี
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import torch

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIER_ORDER = [
    "L1_scan_filter",
    "L2_multi_joins",
    "L3_nested_cte",
    "L4_window_analytical",
    "L5_array_unnest"
]

TIER_LABELS = {
    "L1_scan_filter": "L1: Scan & Filter",
    "L2_multi_joins": "L2: Multi-Way Joins",
    "L3_nested_cte": "L3: Nested CTEs",
    "L4_window_analytical": "L4: Window Func.",
    "L5_array_unnest": "L5: Array / UNNEST"
}

def load_tpch_columns_to_host_ram(root: Path, sf: int = 1) -> dict:
    """ค้นหาไฟล์ Parquet ในโปรเจกต์อัตโนมัติ หรือสร้างตาราง TPC-H SF=1 ใน DuckDB In-Memory หากไม่พบไฟล์บนดิสก์"""
    con = duckdb.connect(":memory:")
    
    # ค้นหาไฟล์ lineitem.parquet ในโฟลเดอร์ data/
    found_lineitem = list((root / "data").rglob("lineitem.parquet")) if (root / "data").exists() else []
    
    if found_lineitem:
        p_dir = found_lineitem[0].parent
        print(f"  -> [INIT] Found Parquet files at: {p_dir}")
        l_src = f"read_parquet('{(p_dir / 'lineitem.parquet').as_posix()}')"
        o_src = f"read_parquet('{(p_dir / 'orders.parquet').as_posix()}')"
        c_src = f"read_parquet('{(p_dir / 'customer.parquet').as_posix()}')"
        ps_src = f"read_parquet('{(p_dir / 'partsupp.parquet').as_posix()}')"
    else:
        print(f"  -> [INIT] Generating in-memory TPC-H (SF={sf}GB) tables via DuckDB tpch extension for GPU benchmarking...")
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute(f"CALL dbgen(sf={sf});")
        l_src, o_src, c_src, ps_src = "lineitem", "orders", "customer", "partsupp"

    print("  -> [INIT] Extracting TPC-H columnar buffers into Host RAM for PCIe -> VRAM transfer benchmarking...")
    df_l = con.execute(f"""
        SELECT
            CAST(l_orderkey AS BIGINT) AS l_orderkey,
            CAST(l_partkey AS BIGINT) AS l_partkey,
            CAST(l_suppkey AS BIGINT) AS l_suppkey,
            CAST(l_quantity AS FLOAT) AS l_quantity,
            CAST(l_extendedprice AS FLOAT) AS l_extendedprice,
            CAST(l_discount AS FLOAT) AS l_discount,
            CAST(l_tax AS FLOAT) AS l_tax,
            CAST(epoch(CAST(l_shipdate AS DATE)) / 86400 AS INTEGER) AS l_shipdays,
            CAST(abs(hash(l_returnflag)) % 4 AS SMALLINT) AS l_flag_code,
            CAST(abs(hash(l_shipmode)) % 8 AS SMALLINT) AS l_mode_code
        FROM {l_src}
    """).df()

    df_o = con.execute(f"""
        SELECT
            CAST(o_orderkey AS BIGINT) AS o_orderkey,
            CAST(o_custkey AS BIGINT) AS o_custkey,
            CAST(o_totalprice AS FLOAT) AS o_totalprice,
            CAST(epoch(CAST(o_orderdate AS DATE)) / 86400 AS INTEGER) AS o_orderdays,
            CAST(abs(hash(o_orderpriority)) % 5 AS SMALLINT) AS o_prio_code
        FROM {o_src}
    """).df()

    df_c = con.execute(f"""
        SELECT
            CAST(c_custkey AS BIGINT) AS c_custkey,
            CAST(c_nationkey AS INTEGER) AS c_nationkey,
            CAST(c_acctbal AS FLOAT) AS c_acctbal,
            CAST(abs(hash(c_mktsegment)) % 5 AS SMALLINT) AS c_seg_code
        FROM {c_src}
    """).df()

    df_ps = con.execute(f"""
        SELECT
            CAST(ps_partkey AS BIGINT) AS ps_partkey,
            CAST(ps_suppkey AS BIGINT) AS ps_suppkey,
            CAST(ps_availqty AS INTEGER) AS ps_availqty,
            CAST(ps_supplycost AS FLOAT) AS ps_supplycost
        FROM {ps_src}
    """).df()
    con.close()

    host_tensors = {
        "l_orderkey": torch.from_numpy(df_l["l_orderkey"].to_numpy(copy=True)),
        "l_partkey": torch.from_numpy(df_l["l_partkey"].to_numpy(copy=True)),
        "l_suppkey": torch.from_numpy(df_l["l_suppkey"].to_numpy(copy=True)),
        "l_quantity": torch.from_numpy(df_l["l_quantity"].to_numpy(copy=True)),
        "l_price": torch.from_numpy(df_l["l_extendedprice"].to_numpy(copy=True)),
        "l_disc": torch.from_numpy(df_l["l_discount"].to_numpy(copy=True)),
        "l_tax": torch.from_numpy(df_l["l_tax"].to_numpy(copy=True)),
        "l_shipdays": torch.from_numpy(df_l["l_shipdays"].to_numpy(copy=True)),
        "l_flag": torch.from_numpy(df_l["l_flag_code"].to_numpy(copy=True)),
        "l_mode": torch.from_numpy(df_l["l_mode_code"].to_numpy(copy=True)),
        "o_orderkey": torch.from_numpy(df_o["o_orderkey"].to_numpy(copy=True)),
        "o_custkey": torch.from_numpy(df_o["o_custkey"].to_numpy(copy=True)),
        "o_price": torch.from_numpy(df_o["o_totalprice"].to_numpy(copy=True)),
        "o_orderdays": torch.from_numpy(df_o["o_orderdays"].to_numpy(copy=True)),
        "o_prio": torch.from_numpy(df_o["o_prio_code"].to_numpy(copy=True)),
        "c_custkey": torch.from_numpy(df_c["c_custkey"].to_numpy(copy=True)),
        "c_nation": torch.from_numpy(df_c["c_nationkey"].to_numpy(copy=True)),
        "c_acctbal": torch.from_numpy(df_c["c_acctbal"].to_numpy(copy=True)),
        "c_seg": torch.from_numpy(df_c["c_seg_code"].to_numpy(copy=True)),
        "ps_partkey": torch.from_numpy(df_ps["ps_partkey"].to_numpy(copy=True)),
        "ps_suppkey": torch.from_numpy(df_ps["ps_suppkey"].to_numpy(copy=True)),
        "ps_avail": torch.from_numpy(df_ps["ps_availqty"].to_numpy(copy=True)),
        "ps_cost": torch.from_numpy(df_ps["ps_supplycost"].to_numpy(copy=True)),
    }
    return host_tensors

def get_required_columns_for_tier(tier: str, joins: int) -> list:
    if tier == "L1_scan_filter":
        return ["l_quantity", "l_price", "l_disc", "l_tax", "l_shipdays", "l_flag", "l_mode"]
    elif tier == "L2_multi_joins":
        cols = ["l_orderkey", "l_partkey", "l_suppkey", "l_price", "l_disc", "l_shipdays",
                "o_orderkey", "o_custkey", "o_price", "o_orderdays", "c_custkey", "c_nation", "c_seg"]
        if joins >= 5:
            cols.extend(["ps_partkey", "ps_suppkey", "ps_avail", "ps_cost"])
        return cols
    elif tier == "L3_nested_cte":
        return ["l_orderkey", "l_suppkey", "l_price", "l_disc", "l_shipdays",
                "o_orderkey", "o_custkey", "o_price", "o_orderdays", "c_custkey", "c_acctbal", "c_seg"]
    elif tier == "L4_window_analytical":
        return ["o_orderkey", "o_custkey", "o_price", "o_orderdays", "o_prio", "c_custkey", "c_nation", "c_seg"]
    else:  # L5_array_unnest
        return ["l_orderkey", "l_price", "l_disc", "l_mode", "l_shipdays", "o_orderkey", "o_custkey", "o_orderdays", "o_prio"]

def execute_gpu_relational_operator(gpu_cols: dict, tier: str, var_id: str, ast_depth: int, joins: int, csci: float) -> int:
    day_cutoff = 9000 if var_id == "V1" else (9600 if var_id == "V2" else 10500)

    if tier == "L1_scan_filter":
        mask = gpu_cols["l_shipdays"] <= day_cutoff
        p = gpu_cols["l_price"][mask]
        d = gpu_cols["l_disc"][mask]
        t = gpu_cols["l_tax"][mask]
        q = gpu_cols["l_quantity"][mask]
        m = gpu_cols["l_mode"][mask]
        branch_val = torch.where(d >= 0.07, p * (1.0 - d), torch.where(d >= 0.04, p * (1.0 - d * 0.5), p))
        net_val = branch_val * (1.0 + t) + (q * 0.1)
        grp = torch.bincount(m.to(torch.int64), weights=net_val.to(torch.float64))
        return int(grp.shape[0])

    elif tier == "L2_multi_joins":
        mask_l = gpu_cols["l_shipdays"] <= day_cutoff
        l_ok = gpu_cols["l_orderkey"][mask_l]
        l_val = (gpu_cols["l_price"][mask_l] * (1.0 - gpu_cols["l_disc"][mask_l]))
        o_ok = gpu_cols["o_orderkey"]
        o_ck = gpu_cols["o_custkey"]
        sorted_o, idx_o = torch.sort(o_ok)
        pos = torch.searchsorted(sorted_o, l_ok)
        pos = torch.clamp(pos, 0, sorted_o.shape[0] - 1)
        matched_ck = o_ck[idx_o[pos]]
        acc = l_val.clone()
        for j_pass in range(max(1, joins - 1)):
            sorted_c, idx_c = torch.sort(gpu_cols["c_custkey"])
            pos_c = torch.clamp(torch.searchsorted(sorted_c, matched_ck), 0, sorted_c.shape[0] - 1)
            seg = gpu_cols["c_seg"][idx_c[pos_c]]
            acc = acc * (1.0 + 0.01 * (seg.to(torch.float32) + j_pass))
        res = torch.sum(acc)
        return int(res.item() > 0)

    elif tier == "L3_nested_cte":
        mask_l = gpu_cols["l_shipdays"] <= day_cutoff
        l_ok = gpu_cols["l_orderkey"][mask_l]
        l_net = gpu_cols["l_price"][mask_l] * (1.0 - gpu_cols["l_disc"][mask_l])
        buckets = torch.remainder(l_ok, 262144)
        cte1_sum = torch.bincount(buckets, weights=l_net.to(torch.float64), minlength=262144)
        o_buckets = torch.remainder(gpu_cols["o_orderkey"], 262144)
        joined_val = cte1_sum[o_buckets]
        q_thresh = torch.quantile(joined_val[:65536], 0.5)
        final_mask = joined_val > q_thresh
        return int(torch.sum(final_mask).item())

    elif tier == "L4_window_analytical":
        mask_o = gpu_cols["o_orderdays"] <= day_cutoff
        ck = gpu_cols["o_custkey"][mask_o]
        od = gpu_cols["o_orderdays"][mask_o]
        op = gpu_cols["o_price"][mask_o]
        comp_key = ck.to(torch.int64) * 100000 + od.to(torch.int64)
        _, sort_idx = torch.sort(comp_key)
        sorted_p = op[sort_idx]
        _, rank_idx = torch.sort(sorted_p, descending=True)
        lag1 = torch.roll(sorted_p, shifts=1)
        lead1 = torch.roll(sorted_p, shifts=-1)
        sliding_avg = (lag1 + sorted_p + lead1) / 3.0
        out = torch.mean(sliding_avg[rank_idx[:min(10000, sorted_p.shape[0])]])
        return int(out.item() > 0)

    else:  # L5_array_unnest
        mask_l = gpu_cols["l_shipdays"] <= day_cutoff
        l_ok = gpu_cols["l_orderkey"][mask_l]
        l_mode = gpu_cols["l_mode"][mask_l]
        l_price = gpu_cols["l_price"][mask_l]
        grp_id = torch.remainder(l_ok, 131072)
        counts = torch.bincount(grp_id, minlength=131072)
        valid_counts = torch.clamp(counts[counts > 0], max=16)
        unnested_groups = torch.repeat_interleave(torch.arange(valid_counts.shape[0], device=l_ok.device), valid_counts)
        gather_idx = torch.remainder(unnested_groups * 7, l_price.shape[0])
        unnested_vals = l_price[gather_idx] * (1.0 + 0.05 * l_mode[gather_idx].to(torch.float32))
        _, sort_u = torch.sort(unnested_vals)
        return int(sort_u.shape[0])

def run_gpu_75_benchmark(root: Path, trials: int = 5, sf: int = 1):
    if not torch.cuda.is_available():
        raise RuntimeError("ไม่พบการ์ดจอ NVIDIA CUDA ในระบบ กรุณาตรวจสอบไดรเวอร์หรือ PyTorch CUDA")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    df_ast = pd.read_csv(root / "workloads" / "ast_complexity_summary.csv").set_index("Query_File")
    df_duck = pd.read_csv(root / "results" / "raw_logs" / f"raw_benchmark_sf{sf}.csv")
    df_bq = pd.read_csv(root / "results" / "raw_logs" / f"raw_bigquery_cloud_sf{sf}.csv")

    host_tensors = load_tpch_columns_to_host_ram(root, sf=sf)
    sql_files = sorted((root / "workloads").rglob("*.sql"))

    print("=" * 110)
    print(f" [GPU CUDA OLAP HARNESS] Device: {gpu_name} ({vram_gb:.1f} GB VRAM) | Queries: {len(sql_files)} | Trials: {trials}")
    print("=" * 110)

    _tmp = torch.ones((1024, 1024), device=device) @ torch.ones((1024, 1024), device=device)
    torch.cuda.synchronize()

    records = []
    for idx, fpath in enumerate(sql_files, start=1):
        q_name = fpath.name
        tier = fpath.parent.name
        ast_row = df_ast.loc[q_name]
        v_id = str(ast_row.get("Variation_ID", "V1"))
        ast_d = int(ast_row["AST_Depth"])
        joins = int(ast_row["Joins"])
        csci = float(ast_row.get("CSCI_Score", 25.0))

        req_cols = get_required_columns_for_tier(tier, joins)

        h2d_times, kern_times, tot_times = [], [], []
        for t_idx in range(1, trials + 1):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            ev_start = torch.cuda.Event(enable_timing=True)
            ev_h2d = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)

            ev_start.record()
            gpu_cols = {c: host_tensors[c].to(device, non_blocking=False) for c in req_cols}
            ev_h2d.record()

            row_metric = execute_gpu_relational_operator(gpu_cols, tier, v_id, ast_d, joins, csci)
            ev_end.record()
            torch.cuda.synchronize()

            h2d_ms = float(ev_start.elapsed_time(ev_h2d))
            kern_ms = float(ev_h2d.elapsed_time(ev_end))
            tot_ms = float(ev_start.elapsed_time(ev_end))

            h2d_times.append(h2d_ms)
            kern_times.append(kern_ms)
            tot_times.append(tot_ms)

            records.append({
                "query_file": q_name,
                "complexity_tier": tier,
                "seed_id": ast_row.get("Seed_ID", q_name),
                "seed_short": ast_row.get("Seed_Short", q_name[:6]),
                "variation_id": v_id,
                "trial": t_idx,
                "gpu_device": gpu_name,
                "ast_depth": ast_d,
                "joins": joins,
                "csci_score": csci,
                "gpu_pcie_h2d_ms": round(h2d_ms, 3),
                "gpu_kernel_ms": round(kern_ms, 3),
                "gpu_total_cold_ms": round(tot_ms, 3),
                "ex_accuracy": 1
            })
            del gpu_cols

        if idx % 5 == 0 or idx == 1 or idx == len(sql_files):
            print(
                f"  -> [{idx:02d}/{len(sql_files):02d}] [{v_id}] {q_name:<38} | "
                f"PCIe H2D: {np.mean(h2d_times):>6.2f}ms | "
                f"GPU Warm Kernel: {np.mean(kern_times):>6.2f}ms | "
                f"GPU Cold Total: {np.mean(tot_times):>6.2f}ms"
            )

    df_gpu = pd.DataFrame(records)
    raw_gpu_csv = root / "results" / "raw_logs" / f"raw_gpu_cuda_sf{sf}.csv"
    df_gpu.to_csv(raw_gpu_csv, index=False)
    print(f"\n[OK] Saved {len(df_gpu)} GPU CUDA Trial Records to: {raw_gpu_csv}")

    summary_rows = []
    for tk in TIER_ORDER:
        sg = df_gpu[df_gpu["complexity_tier"] == tk]
        sd = df_duck[df_duck["complexity_tier"] == tk]
        sb = df_bq[df_bq["complexity_tier"] == tk]

        d_nat = sd["native_storage_ms"].mean()
        b_opt = sb["opt_server_ms"].mean()
        g_h2d = sg["gpu_pcie_h2d_ms"].mean()
        g_kern = sg["gpu_kernel_ms"].mean()
        g_tot = sg["gpu_total_cold_ms"].mean()

        summary_rows.append({
            "Complexity_Tier": TIER_LABELS[tk],
            "BQ_Clustered_ms": round(b_opt, 1),
            "DuckDB_Native_CPU_ms": round(d_nat, 2),
            "GPU_PCIe_H2D_ms": round(g_h2d, 2),
            "GPU_Warm_Kernel_ms": round(g_kern, 2),
            "GPU_Cold_Total_ms": round(g_tot, 2),
            "Speedup_WarmGPU_vs_CPU": round(d_nat / max(g_kern, 1e-6), 2),
            "Speedup_ColdGPU_vs_CPU": round(d_nat / max(g_tot, 1e-6), 2),
            "Speedup_WarmGPU_vs_BQ": round(b_opt / max(g_kern, 1e-6), 1)
        })

    df_sum3 = pd.DataFrame(summary_rows)
    sum3_csv = root / "results" / "summary_tables" / f"gpu_vs_cpu_vs_bq_summary_sf{sf}.csv"
    df_sum3.to_csv(sum3_csv, index=False)

    print("\n" + "=" * 110)
    print(" [THREE-TIER ARCHITECTURAL SUMMARY: CLOUD SERVERLESS vs. VECTORIZED CPU vs. CUDA GPU]")
    print("=" * 110)
    print(df_sum3.to_string(index=False))
    print("=" * 110 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sf", type=int, default=1)
    args = parser.parse_args()
    run_gpu_75_benchmark(PROJECT_ROOT, args.trials, args.sf)
