#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_independent_replication.py
Self-contained independent replication harness for AST-Route V13.
  1. Auto-provisions canonical TPC-H Parquet via DuckDB dbgen.
  2. Executes 15 stratified OLAP queries (L1..L5 x V1..V3) live across
     DuckDB, Polars, and Apache DataFusion (1 warmup + R timed runs).
  3. Verifies EX=1.000 (FP64 relative error < 1e-9) on every query.
  4. Emits independent_replication_results.csv and independent_replication_receipt.json
     with host metadata, engine versions, and SHA-256 verification hash.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import datafusion
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import psutil
import sqlglot


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def to_compat_arrow(df_pl: pl.DataFrame):
    try:
        return df_pl.to_arrow(compat_level=pl.CompatLevel.oldest())
    except TypeError:
        return df_pl.to_arrow()


def build_15_queries() -> list[dict]:
    horizons = [("V1", "1998-09-01"), ("V2", "1997-01-01"), ("V3", "1993-01-01")]
    qs = []
    for v_tag, cut_dt in horizons:
        qs.append({
            "qid": f"L1_{v_tag}", "tier": "L1", "cut_dt": cut_dt,
            "sql": f"SELECT SUM(l_extendedprice * (1.0 - l_discount)) AS val FROM lineitem WHERE l_shipdate >= DATE '{cut_dt}' AND l_quantity <= 38.0",
        })
        qs.append({
            "qid": f"L2_{v_tag}", "tier": "L2", "cut_dt": cut_dt,
            "sql": f"SELECT SUM(l.l_extendedprice + o.o_totalprice * 0.01) AS val FROM lineitem l INNER JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE l.l_shipdate >= DATE '{cut_dt}'",
        })
        qs.append({
            "qid": f"L3_{v_tag}", "tier": "L3", "cut_dt": cut_dt,
            "sql": f"WITH c AS (SELECT l_partkey, l_extendedprice * (1.0 - l_discount) AS net_p FROM lineitem WHERE l_shipdate >= DATE '{cut_dt}' AND l_discount >= 0.02) SELECT SUM(net_p) AS val FROM c WHERE l_partkey <= 150000",
        })
        qs.append({
            "qid": f"L4_{v_tag}", "tier": "L4", "cut_dt": cut_dt,
            "sql": f"SELECT SUM(w_avg) AS val FROM (SELECT AVG(l_extendedprice) OVER (PARTITION BY l_returnflag) AS w_avg FROM lineitem WHERE l_shipdate >= DATE '{cut_dt}') sub_w",
        })
        qs.append({
            "qid": f"L5_{v_tag}", "tier": "L5", "cut_dt": cut_dt,
            "sql": f"SELECT SUM(CASE WHEN l_quantity > 25.0 THEN l_extendedprice * 2.0 ELSE l_extendedprice END) AS val FROM lineitem WHERE l_shipdate >= DATE '{cut_dt}'",
        })
    return qs


def run_polars(q: dict, pl_li: pl.DataFrame, pl_ord: pl.DataFrame) -> float:
    y, m, d = (int(x) for x in q["cut_dt"].split("-"))
    lf = pl_li.lazy().filter(pl.col("l_shipdate") >= pl.date(y, m, d))
    t = q["tier"]
    if t == "L1":
        out = lf.filter(pl.col("l_quantity") <= 38.0).select((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum()).collect()
    elif t == "L2":
        out = lf.join(pl_ord.lazy(), left_on="l_orderkey", right_on="o_orderkey", how="inner").select((pl.col("l_extendedprice") + pl.col("o_totalprice") * 0.01).sum()).collect()
    elif t == "L3":
        out = lf.filter((pl.col("l_discount") >= 0.02) & (pl.col("l_partkey") <= 150000)).select((pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum()).collect()
    elif t == "L4":
        out = lf.with_columns(pl.col("l_extendedprice").mean().over("l_returnflag").alias("w_avg")).select(pl.col("w_avg").sum()).collect()
    else:
        out = lf.select(pl.when(pl.col("l_quantity") > 25.0).then(pl.col("l_extendedprice") * 2.0).otherwise(pl.col("l_extendedprice")).sum()).collect()
    return float(out.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="replication_output")
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print(f"AST-ROUTE V13 INDEPENDENT REPLICATION HARNESS (SF={args.sf} GB, R={args.reps})")
    print("=" * 84)

    with tempfile.TemporaryDirectory() as tmpd:
        li_p = Path(tmpd) / "lineitem.parquet"
        ord_p = Path(tmpd) / "orders.parquet"
        print("[1/4] Provisioning canonical TPC-H Parquet via DuckDB dbgen...")
        c_gen = duckdb.connect(":memory:")
        c_gen.execute(f"INSTALL tpch; LOAD tpch; CALL dbgen(sf={args.sf});")
        c_gen.execute(f"COPY lineitem TO '{li_p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        c_gen.execute(f"COPY orders TO '{ord_p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        c_gen.close()

        pl_li = pl.read_parquet(
            li_p, columns=["l_orderkey", "l_partkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipdate", "l_returnflag"]
        ).with_columns([
            pl.col("l_orderkey").cast(pl.Int64),
            pl.col("l_partkey").cast(pl.Int64),
            pl.col("l_quantity").cast(pl.Float64),
            pl.col("l_extendedprice").cast(pl.Float64),
            pl.col("l_discount").cast(pl.Float64),
            pl.col("l_shipdate").cast(pl.Date),
            pl.col("l_returnflag").cast(pl.String),
        ])
        pl_ord = pl.read_parquet(ord_p, columns=["o_orderkey", "o_totalprice"]).with_columns([
            pl.col("o_orderkey").cast(pl.Int64),
            pl.col("o_totalprice").cast(pl.Float64),
        ])

        ar_li = to_compat_arrow(pl_li)
        ar_ord = to_compat_arrow(pl_ord)

        con_dk = duckdb.connect(":memory:")
        con_dk.register("lineitem", ar_li)
        con_dk.register("orders", ar_ord)

        df_ctx = datafusion.SessionContext()
        df_ctx.register_record_batches("lineitem", [ar_li.to_batches()])
        df_ctx.register_record_batches("orders", [ar_ord.to_batches()])

        queries = build_15_queries()
        print("[2/4] Warmup & FP64 Multiset Equivalence Check (EX=1.000, eps < 1e-9)...")
        max_rel_err = 0.0
        for q in queries:
            v_dk = float(con_dk.execute(q["sql"]).fetchone()[0])
            v_pl = run_polars(q, pl_li, pl_ord)
            v_df = float(df_ctx.sql(q["sql"]).collect()[0].column(0)[0].as_py())
            for vv in (v_pl, v_df):
                rel = abs(vv - v_dk) / max(abs(v_dk), 1e-12)
                max_rel_err = max(max_rel_err, rel)
                if rel >= 1e-9:
                    raise AssertionError(f"Equivalence failure on {q['qid']}: rel={rel:.2e}")

        print(f"[3/4] Running {args.reps} timed repetitions across 15 queries & 3 CPU engines...")
        rows = []
        ast_us_list = []
        for q in queries:
            t_ast0 = time.perf_counter_ns()
            parsed = sqlglot.parse_one(q["sql"])
            _ = sum(1 for _ in parsed.walk())
            ast_us_list.append((time.perf_counter_ns() - t_ast0) / 1000.0)

            dk_l, pl_l, df_l = [], [], []
            for _ in range(args.reps):
                t0 = time.perf_counter_ns()
                _ = con_dk.execute(q["sql"]).fetchone()
                dk_l.append((time.perf_counter_ns() - t0) / 1e6)

                t0 = time.perf_counter_ns()
                _ = run_polars(q, pl_li, pl_ord)
                pl_l.append((time.perf_counter_ns() - t0) / 1e6)

                t0 = time.perf_counter_ns()
                _ = df_ctx.sql(q["sql"]).collect()
                df_l.append((time.perf_counter_ns() - t0) / 1e6)

            m_dk, m_pl, m_df = float(np.mean(dk_l)), float(np.mean(pl_l)), float(np.mean(df_l))
            cv_dk = (float(np.std(dk_l, ddof=1)) / max(m_dk, 1e-9) * 100.0) if args.reps > 1 else 0.0
            cv_pl = (float(np.std(pl_l, ddof=1)) / max(m_pl, 1e-9) * 100.0) if args.reps > 1 else 0.0
            cv_df = (float(np.std(df_l, ddof=1)) / max(m_df, 1e-9) * 100.0) if args.reps > 1 else 0.0
            best_eng = min([("DuckDB", m_dk), ("Polars", m_pl), ("DataFusion", m_df)], key=lambda x: x[1])[0]
            rows.append({
                "Query_ID": q["qid"],
                "Tier": q["tier"],
                "DuckDB_ms": round(m_dk, 3),
                "DuckDB_CV_Pct": round(cv_dk, 2),
                "Polars_ms": round(m_pl, 3),
                "Polars_CV_Pct": round(cv_pl, 2),
                "DataFusion_ms": round(m_df, 3),
                "DataFusion_CV_Pct": round(cv_df, 2),
                "Best_Engine": best_eng,
                "EX_Exact": 1.0,
            })
        con_dk.close()

    df_res = pd.DataFrame(rows)
    out_csv = out_dir / "independent_replication_results.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")

    l4_sub = df_res[df_res["Tier"] == "L4"]
    receipt = {
        "protocol": "AST-Route V13 Independent Docker Replication Receipt",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host_environment": {
            "os": platform.platform(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "in_docker": os.path.exists("/.dockerenv"),
        },
        "engine_versions": {
            "duckdb": duckdb.__version__,
            "polars": pl.__version__,
            "datafusion": getattr(datafusion, "__version__", "installed"),
            "sqlglot": sqlglot.__version__,
        },
        "verification": {
            "scale_factor_gb": args.sf,
            "repetitions_r": args.reps,
            "queries_verified": len(df_res),
            "ex_multiset_match_rate": 1.0,
            "max_fp64_rel_error": max_rel_err,
            "median_ast_parse_us": round(float(np.median(ast_us_list)), 2),
            "duckdb_mean_ms": round(float(df_res["DuckDB_ms"].mean()), 3),
            "polars_mean_ms": round(float(df_res["Polars_ms"].mean()), 3),
            "datafusion_mean_ms": round(float(df_res["DataFusion_ms"].mean()), 3),
            "l4_window_polars_mean_ms": round(float(l4_sub["Polars_ms"].mean()), 3),
            "l4_window_duckdb_mean_ms": round(float(l4_sub["DuckDB_ms"].mean()), 3),
            "l4_window_datafusion_mean_ms": round(float(l4_sub["DataFusion_ms"].mean()), 3),
            "l4_window_polars_beats_datafusion": bool(l4_sub["Polars_ms"].mean() < l4_sub["DataFusion_ms"].mean()),
        },
        "results_csv_sha256": sha256_file(out_csv),
    }
    rec_p = out_dir / "independent_replication_receipt.json"
    rec_p.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("[4/4] Replication Complete!")
    print(df_res[["Query_ID", "Tier", "DuckDB_ms", "Polars_ms", "DataFusion_ms", "Best_Engine", "EX_Exact"]].to_string(index=False))
    print("-" * 84)
    print(f"[RECEIPT SAVED] {rec_p} | Results CSV SHA-256: {receipt['results_csv_sha256'][:24]}...")
    print("=" * 84)


if __name__ == "__main__":
    main()