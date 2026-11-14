#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/102_issue3_3_three_component_cost_audit.py
================================================================================
Issue 3.3 Three-Component Cost Decomposition Audit (Per 10^6 Queries):
  1. Cloud_Query_Charge_USD_1M : Marginal BigQuery On-Demand Scan Charge ($6.25/TiB, 10 MiB floor)
  2. Local_Compute_Amort_USD_1M: Time-proportional CPU/GPU Hardware Amortization ($0.038/hr, 3-yr life)
  3. Energy_Electricity_USD_1M : Active Device Energy (Joules) x Assumed Tariff ($0.12/kWh)
  Explicitly distinguishes Marginal Cloud Scan Cost from Reference Local Footprint (Non-TCO claim).
================================================================================
"""

import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
RES_DIR = ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_HW_AMORT_USD = 0.038          # 3-year workstation + RTX 2050 lifecycle (~$1,000 / 26,280 hrs)
ELEC_TARIFF_USD_PER_KWH = 0.12       # Assumed electricity rate ($0.12 / kWh = $0.12 / 3.6e6 J)
BQ_PRICE_PER_TIB_USD = 6.25          # Google BigQuery On-Demand scan rate ($6.25 / TiB)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_local_amort_1m(latency_ms: float) -> float:
    return (latency_ms / 1000.0) * (HOURLY_HW_AMORT_USD / 3600.0) * 1_000_000.0


def compute_energy_cost_1m(joules: float) -> float:
    return (joules / 3_600_000.0) * ELEC_TARIFF_USD_PER_KWH * 1_000_000.0


def main() -> None:
    # Canonical empirical latencies (ms), active energy (J), and billed cloud scan charges ($/1M)
    records = [
        # --- SF = 1 GB (N=75 Factorial Workload) ---
        ("SF=1 GB", "SQLite 3 (Row-Store Baseline)",     8417.33, 126.260, 0.00,    "Local CPU B-Tree (Sunk HW + High Elec.)"),
        ("SF=1 GB", "Always DuckDB (C++ Vectorized)",     183.74,   6.397, 0.00,    "Local CPU Pkg Est. ($0 Marginal Cloud Fee)"),
        ("SF=1 GB", "Always DataFusion (Rust Arrow)",      96.30,   3.393, 0.00,    "Local CPU Pkg Est. ($0 Marginal Cloud Fee)"),
        ("SF=1 GB", "Always Polars (Rust Arrow)",          82.47,   3.170, 0.00,    "Local CPU Pkg Est. ($0 Marginal Cloud Fee)"),
        ("SF=1 GB", "Always GPU Cold (RTX 2050)",          56.96,   0.644, 0.00,    "Meas. NVML GPU Board ($1.58 J Host+GPU sys)"),
        ("SF=1 GB", "Always BigQuery (Unoptimized)",     1412.60,   0.000, 2072.45, "Remote Serverless Scan ($6.25/TiB, 0 J Local)"),
        ("SF=1 GB", "Always BigQuery (Clustered)",        745.39,   0.000, 1267.66, "Clustered Partition/Block Pruning (10 MiB floor)"),
        ("SF=1 GB", "Proposed AST-Route (B1 Pure Lat.)",   54.92,   1.381, 0.00,    "Hybrid Local CPU+GPU ($0 Marginal Cloud Charge)"),
        ("SF=1 GB", "Proposed AST-Route (B5 Energy)",      70.20,   0.748, 0.00,    "Energy-Prioritized Mode (-45.8% Energy)"),
        # --- SF = 100 GB (Extreme Spill & Cloud-Elastic Crossover, Table 9 B7 vs B8) ---
        ("SF=100 GB", "Always DuckDB (Out-of-Core Spill)", 8699.98, 385.400, 0.00,  "3,302 MB Disk Spill ($0 Cloud, High Local Time)"),
        ("SF=100 GB", "Always Polars (Streaming Arrow)",   6551.42, 256.200, 0.00,  "Host DRAM Cap Spill ($0 Cloud, High Local Time)"),
        ("SF=100 GB", "Always BigQuery (Clustered)",       2571.60,   0.000, 126766.00, "Full Cloud Offload at SF=100 GB ($6.25/TiB)"),
        ("SF=100 GB", "AST-Route B7 (Cloud-Elastic)",      1142.80,  12.840, 507.06, "Routes Spill-Bound L1/L4 to BQ ($507.06/1M Scan)"),
        ("SF=100 GB", "AST-Route B8 (Zero-Cloud Cap)",     1326.90,  16.350, 0.00,  "Hard Cloud Cap C_cloud=$0 (40x Chunked Local)"),
    ]

    rows = []
    for sf, policy, lat_ms, eng_j, cloud_usd, note in records:
        is_pure_cloud = "Always BigQuery" in policy
        hw_amort = 0.00 if is_pure_cloud else round(compute_local_amort_1m(lat_ms), 3)
        elec_usd = 0.00 if is_pure_cloud else round(compute_energy_cost_1m(eng_j), 3)
        ref_sum = round(cloud_usd + hw_amort + elec_usd, 3)
        rows.append({
            "Scale_Factor": sf,
            "Engine_or_Policy": policy,
            "Mean_Latency_ms": lat_ms,
            "Local_Active_Energy_J": eng_j,
            "1_Est_Cloud_Query_Charge_USD_1M": round(cloud_usd, 2),
            "2_Local_HW_Amortization_USD_1M": hw_amort,
            "3_Assumed_Electricity_Cost_USD_1M": elec_usd,
            "Reference_Combined_Footprint_USD_1M": ref_sum,
            "Accounting_Scope_Note": note,
        })

    df_out = pd.DataFrame(rows)
    out_csv = RES_DIR / "issue3_3_three_component_cost_decomposition.csv"
    df_out.to_csv(out_csv, index=False)

    print("=" * 110)
    print("ISSUE 3.3 THREE-COMPONENT COST DECOMPOSITION (USD PER 10^6 QUERIES — NON-TCO MARGINAL vs. LOCAL FOOTPRINT):")
    print("=" * 110)
    cols = ["Scale_Factor", "Engine_or_Policy", "1_Est_Cloud_Query_Charge_USD_1M", "2_Local_HW_Amortization_USD_1M", "3_Assumed_Electricity_Cost_USD_1M", "Reference_Combined_Footprint_USD_1M"]
    print(df_out[cols].to_string(index=False))
    print("-" * 110)
    print(f"[SAVED CSV] {out_csv.name} | SHA-256: {sha256_file(out_csv)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
