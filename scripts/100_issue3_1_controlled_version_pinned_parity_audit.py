#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/100_issue3_1_controlled_version_pinned_parity_audit.py
Controlled 2x2 Engine-Version vs. Platform Isolation on Notebook1 (8 Threads, 5 Trials)
"""
import argparse, hashlib, os, platform, time
from pathlib import Path

os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import datafusion

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RES_DIR = ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_8table_tpch_sf1() -> tuple[duckdb.DuckDBPyConnection, Path, str]:
    pq_dir = ROOT / "data" / "sf1" / "parquet_zstd"
    if not (pq_dir / "lineitem.parquet").exists():
        alt = ROOT / "_reexec_tpch_data"
        if (alt / "lineitem.parquet").exists():
            pq_dir = alt
    pq_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8;")
    con.execute("SET memory_limit='9GB';")
    tables = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

    if not all((pq_dir / f"{t}.parquet").exists() for t in tables):
        con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1.0);")
        for t in tables:
            con.execute(f"COPY {t} TO '{(pq_dir / f'{t}.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880);")
    else:
        for t in tables:
            con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet('{(pq_dir / f'{t}.parquet').as_posix()}');")

    return con, pq_dir, sha256_file(pq_dir / "lineitem.parquet")


QUERIES_15 = [
    ("L1_V1_NarrowScan", "L1", "V1", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l_extendedprice * (1.0 - l_discount)), 0.0) AS chk_sum FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate < DATE '1995-04-01' AND l_quantity < 15.0"),
    ("L1_V2_MedScan",    "L1", "V2", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(CASE WHEN l_returnflag = 'R' THEN l_extendedprice * (1.0 - l_discount) ELSE l_extendedprice END), 0.0) AS chk_sum FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1996-01-01'"),
    ("L1_V3_FullScan",   "L1", "V3", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)), 0.0) AS chk_sum FROM lineitem WHERE l_shipdate <= DATE '1998-09-01'"),
    ("L2_V1_NarrowJoin", "L2", "V1", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l.l_extendedprice * (1.0 - l.l_discount)), 0.0) AS chk_sum FROM lineitem l INNER JOIN orders o ON l.l_orderkey = o.o_orderkey INNER JOIN customer c ON o.o_custkey = c.c_custkey WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate >= DATE '1995-01-01' AND o.o_orderdate < DATE '1995-04-01'"),
    ("L2_V2_MedJoin",    "L2", "V2", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l.l_extendedprice * (1.0 - l.l_discount)), 0.0) AS chk_sum FROM lineitem l INNER JOIN orders o ON l.l_orderkey = o.o_orderkey INNER JOIN customer c ON o.o_custkey = c.c_custkey INNER JOIN supplier s ON l.l_suppkey = s.s_suppkey AND c.c_nationkey = s.s_nationkey WHERE o.o_orderdate >= DATE '1994-01-01' AND o.o_orderdate < DATE '1996-01-01'"),
    ("L2_V3_FullJoin",   "L2", "V3", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l.l_extendedprice), 0.0) AS chk_sum FROM lineitem l INNER JOIN orders o ON l.l_orderkey = o.o_orderkey INNER JOIN part p ON l.l_partkey = p.p_partkey WHERE p.p_size >= 5 AND o.o_orderdate < DATE '1998-06-01'"),
    ("L3_V1_NarrowCTE",  "L3", "V1", "WITH o_agg AS (SELECT l_orderkey, SUM(l_extendedprice * (1.0 - l_discount)) AS o_rev FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate < DATE '1995-04-01' GROUP BY l_orderkey) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(o_rev), 0.0) AS chk_sum FROM o_agg WHERE o_rev > 10000.0"),
    ("L3_V2_MedCTE",     "L3", "V2", "WITH s_agg AS (SELECT l_suppkey, SUM(l_extendedprice * (1.0 - l_discount)) AS s_rev FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1996-01-01' GROUP BY l_suppkey), s_avg AS (SELECT AVG(s_rev) AS mean_rev FROM s_agg) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(s_rev), 0.0) AS chk_sum FROM s_agg CROSS JOIN s_avg WHERE s_agg.s_rev > s_avg.mean_rev"),
    ("L3_V3_FullCTE",    "L3", "V3", "WITH p_agg AS (SELECT l_partkey, COUNT(*) AS cnt, SUM(l_quantity) AS q_sum, SUM(l_extendedprice) AS p_sum FROM lineitem WHERE l_shipdate < DATE '1998-06-01' GROUP BY l_partkey) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(p_sum), 0.0) AS chk_sum FROM p_agg WHERE q_sum > 100.0"),
    ("L4_V1_NarrowWin",  "L4", "V1", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l_extendedprice), 0.0) AS chk_sum FROM (SELECT l_extendedprice, ROW_NUMBER() OVER (PARTITION BY l_returnflag ORDER BY l_extendedprice DESC, l_orderkey DESC, l_linenumber DESC) AS rn FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate < DATE '1995-03-01') t WHERE rn <= 1000"),
    ("L4_V2_MedWin",     "L4", "V2", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(l_extendedprice), 0.0) AS chk_sum FROM (SELECT l_extendedprice, ROW_NUMBER() OVER (PARTITION BY l_shipmode ORDER BY l_extendedprice DESC, l_orderkey DESC, l_linenumber DESC) AS rn FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-07-01') t WHERE rn <= 2500"),
    ("L4_V3_FullWin",    "L4", "V3", "SELECT COUNT(*) AS row_cnt, COALESCE(SUM(o_totalprice), 0.0) AS chk_sum FROM (SELECT o_totalprice, ROW_NUMBER() OVER (PARTITION BY o_orderpriority ORDER BY o_totalprice DESC, o_orderkey DESC) AS rn FROM orders WHERE o_orderdate < DATE '1998-06-01') t WHERE rn <= 5000"),
    ("L5_V1_NarrowUnnest","L5","V1", "WITH base AS (SELECT l_partkey, SUM(l_quantity) AS q_tot, SUM(l_extendedprice * (1.0 - l_discount)) AS net_rev FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate < DATE '1995-04-01' AND l_shipmode IN ('AIR', 'MAIL') GROUP BY l_partkey) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(net_rev), 0.0) AS chk_sum FROM base WHERE q_tot > 10.0"),
    ("L5_V2_MedUnnest",  "L5", "V2", "WITH base AS (SELECT l_partkey, SUM(l_quantity) AS q_tot, SUM(l_extendedprice * (1.0 - l_discount)) AS net_rev FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1996-01-01' AND l_shipmode IN ('AIR', 'MAIL', 'SHIP') GROUP BY l_partkey) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(net_rev), 0.0) AS chk_sum FROM base WHERE q_tot > 25.0"),
    ("L5_V3_FullUnnest", "L5", "V3", "WITH base AS (SELECT l_suppkey, SUM(l_quantity) AS q_tot, SUM(l_extendedprice * (1.0 - l_discount)) AS net_rev FROM lineitem WHERE l_shipdate < DATE '1998-06-01' AND l_discount IN (0.01, 0.03, 0.05, 0.07) GROUP BY l_suppkey) SELECT COUNT(*) AS row_cnt, COALESCE(SUM(net_rev), 0.0) AS chk_sum FROM base WHERE q_tot > 50.0"),
]


def run_benchmark(tag: str) -> None:
    con, pq_dir, lineitem_sha = load_8table_tpch_sf1()
    tables = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
    pl_frames = {t: pl.scan_parquet(pq_dir / f"{t}.parquet") for t in tables}

    df_cfg = datafusion.SessionConfig().with_target_partitions(8)
    df_ctx = datafusion.SessionContext(df_cfg)
    for t in tables:
        df_ctx.register_parquet(t, str(pq_dir / f"{t}.parquet"))

    rows = []
    print(f"\n[{tag}] Running 15 Canonical Queries x 5 Trials (8 Threads) | DuckDB={duckdb.__version__}, Polars={pl.__version__}, DF={datafusion.__version__}")
    for qid, tier, var_id, sql in QUERIES_15:
        con.execute(sql).fetchall()
        _ = pl.SQLContext(frames=pl_frames).execute(sql).collect()
        _ = df_ctx.sql(sql).collect()

        t_dk, t_pl, t_df = [], [], []
        r_dk = r_pl = r_df = (0, 0.0)
        for _ in range(5):
            t0 = time.perf_counter()
            r_dk = con.execute(sql).fetchone()
            t_dk.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            res_p = pl.SQLContext(frames=pl_frames).execute(sql).collect()
            t_pl.append((time.perf_counter() - t0) * 1000.0)
            r_pl = (int(res_p.row(0)[0]), float(res_p.row(0)[1]))

            t0 = time.perf_counter()
            batches = df_ctx.sql(sql).collect()
            t_df.append((time.perf_counter() - t0) * 1000.0)
            d_dict = batches[0].to_pydict()
            ks = list(d_dict.keys())
            r_df = (int(d_dict[ks[0]][0]), float(d_dict[ks[1]][0]))

        denom = max(abs(float(r_dk[1])), 1e-9)
        rel_err = max(abs(float(r_dk[1]) - r_pl[1]) / denom, abs(float(r_dk[1]) - r_df[1]) / denom)
        eq_ok = (int(r_dk[0]) == r_pl[0] == r_df[0]) and (rel_err < 1e-9)

        rows.append({
            "Env_Tag": tag,
            "Hostname": platform.node(),
            "Threads": 8,
            "DuckDB_Ver": duckdb.__version__,
            "Polars_Ver": pl.__version__,
            "DataFusion_Ver": datafusion.__version__,
            "Dataset_SHA256_Prefix16": lineitem_sha[:16],
            "Query_ID": qid,
            "Complexity_Tier": tier,
            "DuckDB_Mean_ms": round(float(np.mean(t_dk)), 3),
            "Polars_Mean_ms": round(float(np.mean(t_pl)), 3),
            "DataFusion_Mean_ms": round(float(np.mean(t_df)), 3),
            "Verified_Row_Count": int(r_dk[0]),
            "FP64_Checksum": round(float(r_dk[1]), 2),
            "Max_FP64_Rel_Error": f"{rel_err:.2e}",
            "Result_Equivalence": "PASS (EX=1.000)" if eq_ok else "CHECK",
        })
        print(f"  {qid:<18} | DuckDB={np.mean(t_dk):7.2f} ms | Polars={np.mean(t_pl):7.2f} ms | DF={np.mean(t_df):7.2f} ms | {rows[-1]['Result_Equivalence']}")

    con.close()
    out_p = RES_DIR / f"issue3_1_controlled_parity_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_p, index=False)
    print(f"  -> Saved {out_p.name} (SHA-256: {sha256_file(out_p)[:20]}...)")


def summarize_comparison() -> None:
    p_pin = RES_DIR / "issue3_1_controlled_parity_nb1_pinned.csv"
    p_cur = RES_DIR / "issue3_1_controlled_parity_nb1_current.csv"
    p_mac = RES_DIR / "macbook_reexecution_audit.csv"
    if not (p_pin.exists() and p_cur.exists()):
        return

    g_pin = pd.read_csv(p_pin).groupby("Complexity_Tier")[["DuckDB_Mean_ms", "Polars_Mean_ms", "DataFusion_Mean_ms"]].mean().round(2)
    g_cur = pd.read_csv(p_cur).groupby("Complexity_Tier")[["DuckDB_Mean_ms", "Polars_Mean_ms", "DataFusion_Mean_ms"]].mean().round(2)
    g_mac = pd.read_csv(p_mac).set_index("Complexity_Tier") if p_mac.exists() else None

    rows = []
    for t in ["L1", "L2", "L3", "L4", "L5"]:
        r = {
            "Tier": t,
            "NB1_x86_Pinned(1.4.5/1.36.1/50.1)_Duck_ms": g_pin.loc[t, "DuckDB_Mean_ms"],
            "NB1_x86_Primary_Duck_ms": g_cur.loc[t, "DuckDB_Mean_ms"],
            "NB1_x86_Pinned_Polars_ms": g_pin.loc[t, "Polars_Mean_ms"],
            "NB1_x86_Primary_Polars_ms": g_cur.loc[t, "Polars_Mean_ms"],
            "NB1_x86_Pinned_DF_ms": g_pin.loc[t, "DataFusion_Mean_ms"],
            "NB1_x86_Primary_DF_ms": g_cur.loc[t, "DataFusion_Mean_ms"],
        }
        if g_mac is not None and t in g_mac.index:
            r["MacM1_ARM64_Pinned(1.4.5/1.36.1/50.1)_Duck_ms"] = round(float(g_mac.loc[t, "DuckDB_Mean_ms"]), 2)
            r["MacM1_ARM64_Pinned_Polars_ms"] = round(float(g_mac.loc[t, "Polars_Mean_ms"]), 2)
            r["MacM1_ARM64_Pinned_DF_ms"] = round(float(g_mac.loc[t, "DataFusion_Mean_ms"]), 2)
        rows.append(r)

    df_s = pd.DataFrame(rows)
    out_s = RES_DIR / "issue3_1_version_vs_platform_isolation_summary.csv"
    df_s.to_csv(out_s, index=False)
    print("\n" + "=" * 105)
    print("ISSUE 3.1 CONTROLLED ISOLATION SUMMARY (SOFTWARE VERSION EFFECT vs. PLATFORM/ISA EFFECT @ 8 THREADS):")
    print("=" * 105)
    print(df_s.to_string(index=False))
    print("-" * 105)
    print(f"[SUMMARY CSV] {out_s.name} | SHA-256: {sha256_file(out_s)}")
    print("=" * 105)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "compare"], required=True)
    ap.add_argument("--tag", default="nb1_pinned")
    args = ap.parse_args()
    if args.mode == "run":
        run_benchmark(args.tag)
    else:
        summarize_comparison()

