# AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Execution Protocol](https://img.shields.io/badge/Protocol-100%25%20Full--Scale%20Zero--Estimation-success.svg)](#)
[![Validation](https://img.shields.io/badge/Validation-25--Seed%20LOSO--CV%20%7C%2010k%20Bootstrap%2095%25%20CI-informational.svg)](#)
[![Hardware Telemetry](https://img.shields.io/badge/Telemetry-CPU%20%7C%20RAM%20%7C%20NVIDIA%20NVML%20GPU%20%7C%20Joules-orange.svg)](#)

Official open-source implementation, 75-query factorial SQL workload suite, hardware/energy sensor logs, and automated reproducibility verification pipeline for the **AST-Route Framework**.

---

## 1. Overview of the AST-Route Framework

Modern analytical data infrastructures span heterogeneous execution paradigms—serverless cloud data warehouses, in-process C++ and Rust Apache Arrow vectorized CPU engines, and SIMT GPU-accelerated relational engines. However, no single engine dominates across all query topologies: low-complexity scans suffer from PCIe host-to-device (H2D) transfer penalties on GPUs and stateless scheduling overheads on serverless clouds, whereas complex multi-axis window sorting and lateral array unnesting (`UNNEST`) overwhelm single-node CPU pipelines.

**AST-Route** is a four-layer pre-execution query routing and cost-prediction framework that dispatches incoming SQL queries to their optimal execution engine in **< 0.5 ms** without runtime profiling overhead:

1. **Layer 1 (Static AST Structural & Zone-Map Selectivity Profiler):** Parses input SQL queries via `SQLGlot` into a 9-dimensional structural and selectivity vector $\mathbf{x}_q = [D(q), N(q), J(q), S(q), W(q), U(q), C(q), \text{CSCI}(q), \sigma(q)]^\top \in \mathbb{R}^9$, featuring a construct-validated **Composite Structural Complexity Index (CSCI)** (Spearman's $\rho = 0.4935, p = 6.84 \times 10^{-6}$; Monte Carlo weight stability $\bar{\rho} = 0.8189$).
2. **Layer 2 (Cross-Paradigm Analytical Cost, PCIe Crossover, and Green Energy Models):** Formulates unified analytical models for the GPU PCIe bottleneck fraction ($\Phi_{\text{PCIe}}$), effective analytical scan throughput ($\Theta_{\text{eff}}$), cloud slot concurrency multiplier ($\mathcal{M}_{\text{slot}}$), Cloud FinOps savings ($\Delta C_{\text{1M}}$), and Green OLAP performance per watt ($\eta_{\text{watt}}$ in $\text{MB/J}$).
3. **Layer 3 (Multi-Output Cost & Resource Regression Engine, $\mathcal{R}_{\text{GBR-9D}}$):** Simultaneously predicts log-transformed execution latency, per-query energy footprint ($\text{Joules}$), and memory consumption ($\text{RAM/VRAM GB}$) across candidate engines ($R^2 = 0.815\text{--}0.875$ under 25-fold Leave-One-Seed-Out cross-validation).
4. **Layer 4 (Policy-Driven Optimal Engine Dispatcher):** Selects the optimal target engine across **DuckDB (C++ DBMS)**, **Polars / Apache DataFusion (Rust Arrow)**, **CUDA GPU (NVIDIA RTX 2050)**, and **Google BigQuery (Serverless Cloud)** according to user-defined latency, energy, or FinOps objectives.

---

## 2. Key Empirical & Routing Results (100% Full-Scale TPC-H SF = 1 GB)

### 2.1 Six-Engine Cross-Paradigm Telemetry Summary ($N = 75$ Queries, 25 Canonical Seeds $\times$ 3 Horizons)

| Execution Engine (Paradigm) | Mean Latency (95% Seed-Clustered CI) | Scan Throughput ($\Theta_{\text{eff}}$) | Active Util. (CPU / GPU) | Memory Footprint (RAM / VRAM) | Mean Energy ($E(q)$) | Perf. per Watt ($\eta_{\text{watt}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SQLite 3** (Full B-Tree Row-Store) | `8417.33 ms` (`[6412.19, 10819.70]`) | `0.049 GB/s` | `8.86%` CPU | `2.26 GB` RAM | `126.26 J` | `4.05 MB/J` |
| **Google BigQuery** (Serverless Clustered) | `745.39 ms` (`[515.64, 1026.49]`) | `0.611 GB/s` | $\mathcal{M}_{\text{slot}} = 6.37\times$ | Serverless Slots | `15.72 J` | V1 Prune: `78.4%` |
| **DuckDB Native** (C++ Vectorized DBMS) | `183.74 ms` (`[110.71, 274.89]`) | `3.03 GB/s` | `66.10%` CPU | `4.71 GB` RAM | `6.40 J` | `243.0 MB/J` |
| **Apache DataFusion** (Rust Arrow Planner) | `96.30 ms` (`[78.31, 114.92]`) | `4.83 GB/s` | `70.12%` CPU | `4.65 GB` RAM | `3.39 J` | `157.1 MB/J` |
| **Polars Lazy SQL** (Rust Arrow Streaming) | `82.47 ms` (`[67.93, 98.65]`) | `5.07 GB/s` | `77.44%` CPU | `4.65 GB` RAM | `3.17 J` | `159.7 MB/J` |
| **CUDA GPU (Cold)** (PCIe H2D + Kernel) | `56.96 ms` (`[46.23, 68.44]`) | $\Phi_{\text{PCIe}} = 60.9\%$ | `31.04%` GPU | `1.30 GB` VRAM | `0.65 J` | `672.3 MB/J` |
| **CUDA GPU (Warm)** (Pure In-VRAM SIMT) | **`22.26 ms` (`[16.78, 28.68]`)** | **`18.89 GB/s`** | `31.04%` GPU | `1.30 GB` VRAM | **`0.25 J`** | **`1942.8 MB/J`** |

* **Omnibus Non-Parametric Friedman Test:** $\chi^2_F(5) = 321.50, p = 2.382 \times 10^{-67}$, Kendall's Concordance $W = 0.8573$, Exact Equivalence $\text{EX} = 1.000$ ($75/75$ queries).

### 2.2 Five-Policy Automated Query Routing Comparison (Leak-Free 25-Seed LOSO-CV)

| Routing Model / Policy | Top-1 Acc. | Near-Opt. ($\le 10\%$) | Mean Latency (95% CI) | P95 Latency | Latency Regret | Energy Cost | Memory Cost | Cloud FinOps | Router Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Always DuckDB** | `28.00%` | `32.00%` | `183.74 ms` (`[110.7, 274.9]`) | `731.22 ms` | `271.10%` | `6.397 J` | `4.71 GB` | `$0.00` | `0.00 µs` |
| **2. Always GPU (Cold)** | `49.33%` | `50.67%` | `56.96 ms` (`[46.2, 68.4]`) | `113.70 ms` | `60.76%` | `0.644 J` | `1.30 GB` | `$0.00` | `0.00 µs` |
| **3. Always BigQuery** | `0.00%` | `0.00%` | `745.39 ms` (`[515.6, 1026.5]`) | `2794.05 ms` | `1709.86%` | `15.720 J` | `2.13 GB` | `$1267.66/1M` | `0.00 µs` |
| **4. Rule-Based Routing** | `49.33%` | `58.67%` | `71.48 ms` (`[52.8, 91.7]`) | `230.71 ms` | `65.63%` | `2.374 J` | `3.43 GB` | `$0.00` | `34.07 µs` |
| **5a. AST-Route RF (9D)** | `52.00%` | `56.00%` | `62.46 ms` (`[48.9, 77.4]`) | `138.17 ms` | `51.94%` | `1.467 J` | `2.65 GB` | `$0.00` | `425.53 µs` |
| **5b. AST-Route Cost-Reg (9D)** | **`57.33%`** | **`65.33%`** | **`54.92 ms` (`[42.8, 68.5]`)** | **`129.37 ms`** | **`34.98%`** | **`1.381 J`** | **`2.91 GB`** | **`$0.00`** | **`497.16 µs`** |
| *Oracle Ideal Lower Bound* | `100.00%` | `100.00%` | `42.64 ms` | `101.92 ms` | `0.00%` | `0.975 J` | `3.01 GB` | `$0.00` | `0.00 µs` |

---

## 3. Repository Structure & Cryptographic SHA-256 Telemetry Audit

- `workloads/`: 75 stratified SQL queries across `L1_scan_filter` through `L5_array_unnest` and `ast_complexity_summary.csv` (`SHA-256: 4cb5f7477fd24dc77a6fd2b5...`).
- `results/raw_logs/`:
  - `raw_benchmark_sf1.csv` (DuckDB Parquet vs. Native, `SHA-256: 7d69610e86863845ae7cbb5c...`)
  - `raw_bigquery_cloud_sf1.csv` (Google BigQuery Unoptimized vs. Clustered, `SHA-256: 7aee23711627a085062c7cf8...`)
  - `raw_gpu_cuda_sf1.csv` (NVIDIA RTX 2050 Hardware-Timed PCIe H2D & Warm Kernel, `SHA-256: 50201ab2ea90e96f4b49796c...`)
  - `raw_multi_software_sf1.csv` (Polars, Apache DataFusion, and Full-Scale SQLite 3, `SHA-256: 4d6ba9d612f4c309ec096d61...`)
  - `raw_hardware_energy_sf1.csv` (CPU/GPU Utilization %, RAM/VRAM GB, Energy Joules, `SHA-256: 366d77e7426a78b6089254eb...`)
  - `raw_query_routing_predictions_sf1.csv` (25-Seed LOSO-CV Routing Predictions, `SHA-256: 8d077e29f9b9f3fd40cc7f3e...`)
- `results/summary_tables/`: Aggregated CSV tables (`six_software_comparison_sf1.csv`, `hardware_energy_summary_sf1.csv`, `query_routing_evaluation_sf1.csv`, `bootstrap_ci_and_validation_sf1.csv`, `framework_ablation_and_robustness_sf1.csv`).
- `scripts/`: End-to-end execution, telemetry, validation, and figure generation scripts (`00_init_and_dryrun.py` through `18_run_framework_reviewer_defense_experiments.py`).
- `manuscript/`: Complete LaTeX source (`main.tex`) and unified 300 DPI vector figures (`figures/`).

---

## 4. Author & Funding Acknowledgement

- **Author:** Preecha Noiumkar (Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand; Correspondence: `preecha.n@msu.ac.th`)
- **Funding:** This research project was financially supported by Mahasarakham University.
