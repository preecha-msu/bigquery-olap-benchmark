# Pre-Specification & Blinded Third-Party Holdout Registration Manifest (AST-Route V9.8)

**Timestamp:** 2026-11-14T20:05:00+07:00  
**Repository Snapshot:** `https://github.com/preecha-msu/bigquery-olap-benchmark` (`v9.0-jbd-submission`)  
**Primary Analysis Specification (Sole Primary Endpoint):** Nested Inner-Fold Cross-Validation (`tau_guard = 8%` modal across 25/25 folds, Mondrian Conformal `alpha = 0.10`, pre-specified equivalence margin `Delta_eq = +-5.0 ms` bounded by cold router overhead `4.15-4.39 ms`). Fixed `tau_guard = 12%` is reported strictly as a secondary sensitivity reference.

## 1. Strict Separation of Development vs. Untouched Blinded Holdout Workloads

| Workload Category | Suite Name | Queries | Query Author / Provenance | Touched During Model Design? | Primary Role |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Development Set** | Core TPC-H Factorial (`L1-L5 x V1-V3`) | 75 | Author-stratified from TPC-H Clause 4 | Yes (Evaluated via 25-Seed LOSO-CV) | Cost-model & CSCI ablation |
| **Prospective Seed Holdout** | Unseen Seed `S6` (`L1-L5 x V1-V3`) | 15 | Prospective frozen-weight seed cluster | **No** (Weights frozen prior to generation) | Prospective template holdout |
| **Third-Party Standard Holdout 1** | **Official TPC-H (`Q1-Q22`)** | 22 | **TPC Council Standard Specification** | **No** (Unmodified official queries) | Choke-point & subquery holdout |
| **Third-Party Standard Holdout 2** | **Official IMDB JOB (`1a-33a`)** | 33 | **Leis et al. (VLDB 2015)** | **No** (Unmodified canonical templates) | Real-world skewed join holdout |
| **Third-Party Standard Holdout 3** | **Official Star Schema Benchmark (`SSB Q1.1-Q4.3`)** | 13 | **O'Neil et al. (TPCTC 2009)** | **No** (Unmodified standard flights) | Star-join warehouse holdout |
| **Third-Party Standard Holdout 4** | **Official ClickBench (`Q0-Q42`)** | 43 (20 Core + 23 Dialect Audit) | **ClickHouse Inc. (ClickBench)** | **No** (Unmodified web analytics queries) | Wide-table & SQL dialect holdout |
| **Third-Party Standard Holdout 5** | **Official TPC-DS (`Q1-Q99` via DuckDB `dsdgen`)** | **99** | **TPC Council / DuckDB Official `tpcds` Extension** | **No** (100% unmodified official TPC-DS SQL) | **Blinded 99-query enterprise holdout** |
| **Derived / Synthetic Holdouts** | `TPC-DS-Derived` (30), `BIRD-Inspired` (20), `+50` Extended (`TPC-DS-Complex-25`, `Redset-BI-25`) | 100 | Schema-faithful stress templates | **No** (Zero-shot & LOWSO-CV holdouts) | Multi-channel & BI fleet stress |

## 2. Summary of Unmodified Official 99-Query TPC-DS Blinded Holdout (`v98_partB_official_tpcds_99queries_blinded_holdout.csv`)
- **SQLGlot AST Parsing Coverage:** `99/99` (`100.0%`) parsed into 9D vectors (`mean CSCI = 42.49`).
- **Native Engine Execution Coverage:** DuckDB `99/99` (`100.0%`), Apache DataFusion `93/99` (`93.94%` native; `6/99` via DuckDB fallback), Polars `81/99` (`81.82%` via `SQLGlot` SQL-89 comma-join -> SQL-92 `INNER JOIN` transpilation; `18/99` correlated subqueries fallback to DuckDB), GPU `78/99` (`78.79%` kernel coverage).
- **Zero-Shot Routing Performance (`0` TPC-DS Training Labels):**
  - Static Baselines: DuckDB `44.55 ms`, Polars `46.85 ms`, DataFusion `45.51 ms`, Always GPU `35.56 ms`.
  - **AST-Route (Zero-Shot 9D):** **`30.49 ms` exec** (`93.94%` Top-1, `3.23%` mean regret, `15.01%` P95 regret).
  - **AST-Route (Zero-Shot Pure-SQL Pool):** **`36.15 ms` exec** (`84.85%` Top-1, `4.19%` mean regret).
