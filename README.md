# AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Execution Protocol](https://img.shields.io/badge/Protocol-100%25%20Full--Scale%20Zero--Estimation-success.svg)](#)
[![Big Data Scales](https://img.shields.io/badge/TPC--H%20Scales-SF%3D1%2C%2010%2C%2030%2C%20100%20GB%20(866M%20Rows)-purple.svg)](#)
[![Validation](https://img.shields.io/badge/Validation-25--Seed%20LOSO--CV%20%7C%20Unseen%20Holdout%20%7C%20TPC--H%20Q1--Q22-informational.svg)](#)
[![Hardware Telemetry](https://img.shields.io/badge/Telemetry-CPU%20%7C%20RAM%20Spill%20%7C%20NVML%20GPU%20%7C%20Joules-orange.svg)](#)

Official open-source implementation, 97-query SQL workload suite (75 AST-stratified factorial queries + 22 official TPC-H Q1–Q22 queries), multi-scale execution harnesses ($	ext{SF}=1, 10, 30, 	ext{and } 100	ext{ GB}$), hardware/energy sensor logs, and automated reproducibility verification pipeline for the **AST-Route Framework**.

---

## 1. Overview of the AST-Route Architecture

Modern analytical data infrastructures span four distinct execution paradigms: serverless cloud data warehouses (**Google BigQuery**), in-process C++ vectorized DBMS (**DuckDB**), Rust Apache Arrow streaming engines (**Polars** and **Apache DataFusion**), traditional B-Tree row-stores (**SQLite 3**), and SIMT GPU-accelerated relational tensor engines (**PyTorch CUDA on NVIDIA GeForce RTX 2050**). Committing an analytical workload to any single static engine incurs severe performance regret, excessive energy consumption, out-of-core disk spilling, or high cloud FinOps expenditure.

**AST-Route** is a four-layer pre-execution query routing and cost-prediction framework that dispatches incoming SQL queries in **497.2 µs** inference time (**4.145 ms** total end-to-end SQL parsing + feature walk + inference overhead):

1. **Layer 1 (Static AST Structural & Zone-Map Selectivity Profiler):** Parses SQL queries via `SQLGlot` into a 9-dimensional vector $\mathbf{x}_q = [D(q), N(q), J(q), S(q), W(q), U(q), C(q), 	ext{CSCI}(q), \sigma(q)]^	op \in \mathbb{R}^9$, featuring a construct-validated **Composite Structural Complexity Index (CSCI)** (Spearman's $ho = 0.4935, p = 6.84 	imes 10^{-6}$; 10,000-trial Monte Carlo rank stability $ar{ho} = 0.9880$ at $\pm 20\%$ weight perturbation and $ho = 0.9812$ against equal weights).
2. **Layer 2 (Cross-Paradigm Analytical Cost, PCIe Crossover, Memory Wall & Green Energy Models):** Formulates unified analytical models for the GPU PCIe bottleneck fraction ($\Phi_{	ext{PCIe}}$), effective scan throughput ($\Theta_{	ext{eff}}$), cloud slot-to-wall concurrency proxy ($\mathcal{M}_{	ext{slot}}$), multi-scale Cloud FinOps savings ($\Delta C_{	ext{1M}}$), and Green OLAP performance per watt ($\eta_{	ext{watt}}$ in $	ext{MB/J}$).
3. **Layer 3 (Multi-Output Cost & Resource Regression Engine, $\mathcal{R}_{	ext{GBR-9D}}$):** Simultaneously predicts log-latency, energy footprint ($	ext{Joules}$), and working-set memory ($	ext{GB}$), achieving median multiplicative **Q-Errors (P50)** of **1.053** on CUDA GPU ($R^2 = 0.875$), **1.095** on Polars ($R^2 = 0.822$), **1.105** on DataFusion, **1.172** on Energy ($R^2 = 0.815$), and **1.007** on Memory ($	ext{MAPE} = 2.21\%$).
4. **Layer 4 (Pareto-Configurable Multi-Objective Dispatcher):** Navigates the Latency–Energy Pareto frontier ($lpha \in [0, 1]$) across local CPU, discrete GPU, and serverless cloud backends.

---

## 2. Key Empirical Results (100% Full-Scale Real Execution)

### 2.1 Four-Scale Big Data Evaluation ($	ext{SF} = 1, 10, 30, 	ext{and } 100	ext{ GB}$) & Memory/VRAM Wall Profiling

| Scale Factor (TPC-H Spec) | Dataset Rows (`lineitem` / Total) | DuckDB Out-of-Core | Polars Arrow Stream | DataFusion Arrow | RTX 2050 GPU (PCIe Chunks) | BigQuery Cloud | **Proposed AST-Route** | Host RAM Cap & Disk Spill | GPU VRAM & Energy Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SF = 1 GB** | `6.0M / 8.7M` | `51.12 ms` | `52.69 ms` | `61.95 ms` | **`6.70 ms` (`1x`)** | `745.40 ms` | **`6.70 ms` (`7.63x`)** | `4.71 GB` / `0.0 MB` Spill | `1.30 GB` VRAM / `5934 MB/J` |
| **SF = 10 GB** | `60.0M / 86.6M` | `489.40 ms` | `482.56 ms` | `591.59 ms` | **`105.75 ms` (`4x`)** | `1252.27 ms` | **`105.75 ms` (`4.63x`)** | `7.85 GB` / `125.0 MB` Spill | `3.45 GB` VRAM / `11886 MB/J` |
| **SF = 30 GB** | `180.0M / 259.8M` | `1588.05 ms` | `1619.46 ms` | `1555.46 ms` | `256.82 ms` (`12x`) | `1788.96 ms` | **`250.70 ms` (`6.33x`)** | `9.00 GB` Cap / `864.0 MB` Spill | `3.68 GB` VRAM / `10987 MB/J` |
| **SF = 100 GB** | `600.1M / 866.1M` | `6414.31 ms` | `6906.39 ms` | `6972.37 ms` | `870.96 ms` (`40x`) | `2571.63 ms`* | **`846.41 ms` (`7.58x`)** | `9.00 GB` Cap / `3302.0 MB` Spill | `3.85 GB` VRAM / `11882 MB/J` |

* *Note:* At **$	ext{SF} = 100	ext{ GB}$**, Google BigQuery Serverless Cloud (`1493.51 ms` in L1, `1700.85 ms` in L3, `1286.16 ms` in L4) overtakes all local CPU engines (`4729.95–12860.49 ms`) by **3.17x–10.00x** via horizontal slot elasticity.

### 2.2 Evaluation Across 25-Seed LOSO-CV ($N=75$), Unseen Holdout Seeds ($N=15$), and Official TPC-H Q1–Q22 ($N=22$)

| Validation Protocol & Workload Suite | Top-1 Routing Acc. | Near-Optimal Hit Rate | Mean Latency (95% CI) | Median / Mean Regret | Mean Energy | Speedup vs. Baselines |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 25-Seed LOSO-CV** ($N=75$ Factorial Queries, L1–L5) | **`57.33%`** | **`65.33%`** | **`54.92 ms`** (`[42.78, 68.50]`) | **`0.00%`** / `34.98%` | `1.381 J` | `3.35x` vs. DuckDB ($p=1.83	imes 10^{-6}$), `1.50x` vs. Polars, `1.30x` vs. Rule-Based ($p=0.0187$) |
| **2. Strict Unseen Seed Holdout** (5 Locked Seeds `L1_S5`–`L5_S5`, $N=15$) | **`73.33%`** | **`73.33%`** | **`47.01 ms`** (P95: `79.64 ms`) | **`0.00%`** / `44.88%` | `1.112 J` | Matches exact Oracle P95 (`79.64 ms`); `4.48x` vs. DuckDB, `1.73x` vs. Polars, `1.37x` vs. Rule-Based |
| **3. Official Standard TPC-H Q1–Q22** (22-Fold LOO-CV, $N=22$) | **`72.73%`** | **`90.91%`** | **`15.84 ms`** (Oracle: `12.79 ms`) | **`0.00%`** / `18.47%` | `0.413 J` | `1.44x` vs. Always DuckDB (`22.82 ms`), `1.75x` vs. Always GPU (`27.71 ms`), `2.70x` vs. Polars (`42.79 ms`) |

---

## 3. Repository Structure & Reproducibility Scripts

- `workloads/`: 75 TPC-H-derived factorial SQL queries (`L1_scan_filter` through `L5_array_unnest`) and `ast_complexity_summary.csv`.
- `results/raw_logs/`: Per-trial ground-truth execution logs across all 6 engines (`raw_benchmark_sf1.csv`, `raw_bigquery_cloud_sf1.csv`, `raw_gpu_cuda_sf1.csv`, `raw_multi_software_sf1.csv`, `raw_hardware_energy_sf1.csv`, `raw_query_routing_predictions_sf1.csv`).
- `results/summary_tables/`: All 13 manuscript tables exported directly from raw logs (`six_software_comparison_sf1.csv` through `environment_manifest_sf1_sf100.csv`).
- `results/figures_300dpi/` & `manuscript/figures/`: All 11 vector PDF figures (`300 DPI`) rendered in unified academic serif typography.
- `manuscript/main.tex`: Complete LaTeX source code for *Journal of Big Data* (SpringerOpen) submission.
- `scripts/`: End-to-end execution, telemetry, validation, and figure generation scripts (`00_init_and_dryrun.py` through `27_sync_github_readme_and_sha256_certificate.py`).
- `REPRODUCTION_CERTIFICATE.md`: Full 64-character cryptographic SHA-256 verification table for every dataset and summary file.

---

## 4. Author & Funding Acknowledgement

- **Author:** Preecha Noiumkar (Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand; Correspondence: `preecha.n@msu.ac.th`)
- **Funding:** This research project was financially supported by Mahasarakham University.
