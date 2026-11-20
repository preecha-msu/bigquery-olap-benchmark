# AST-Route: Tail-Safe CPU-GPU-Cloud OLAP Dispatch via PCIe Margin Guarding and Zero-Compile SQL Metadata

[![Manuscript](https://img.shields.io/badge/Manuscript-V13%20(Journal%20of%20Big%20Data)-005A9C)](manuscript/bigquery_olap_benchmark_V13.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Replication](https://img.shields.io/badge/Replication-Docker%20%2B%20SHA256%20Verified-success)](docker_replication/)

**Author:** Preecha Noiumkar (`preecha.n@msu.ac.th`)  
**Affiliation:** Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand  
**Manuscript Version:** **V13** (*Journal of Big Data*, SpringerOpen)

---

## 1. Overview & Key Empirical Findings

**AST-Route** is an engine-agnostic, pre-execution analytical SQL dispatch framework combining zero-compile `SQLGlot` AST parsing and Parquet row-group footer sketching (`172 us` cached; `4.15 ms` cold parse+sketch; `1.42 MB` RAM) with a calibrated PCIe transfer-margin abstention guard across **DuckDB**, **Polars**, **Apache DataFusion**, **SQLite 3**, **CUDA GPUs** (PyTorch CUDA, CuPy, NVIDIA RAPIDS `cuDF` / `cudf-polars`), and **Google BigQuery**.

1. **PCIe Margin Guarding Drives Tail Safety (81.3%):** Stepwise ablation across the 75-query TPC-H development suite ($K=25$ seed clusters, 25-fold LOSO-CV at $\text{SF}=1\text{ GB}$) shows that the PCIe margin guard accounts for **81.3%** of P95 tail-regret reduction—cutting P95 regret from `257.31%` (Always GPU) and `358.40%` (Always Polars) to **`118.40%`** (`8%` Nested-CV Protocol P1, `55.37 ms` net) and **`114.11%`** (`12%` Fixed Protocol P2, `54.92 ms` net).
2. **Minimal Two-Engine Pool Sufficiency (98.7%):** A compact **`{Polars, GPU}`** portfolio (`55.28 ms`) retains **98.7%** of full six-engine utility at $\text{SF}=1\text{ GB}$, while **DuckDB** handles correlated subqueries (Q16–Q22 / TPC-DS) and **BigQuery** offloads $\text{SF}=100\text{ GB}$ NVMe spill regimes.
3. **Cross-Dialect Zero-Compile Role:** Static `SQLGlot` AST + footer sketching avoids target DBMS planner/catalog lock-in across SQL dialects (note: single-dialect native DuckDB `EXPLAIN-DAG` is faster at `1.41 ms` on DuckDB-only SQL). Under $K=25$ cluster bootstrap, aggregate P95 differences vs. `Ridge+Guard` ($p=0.38$) and `EXPLAIN-DAG` ($p=0.74$) are non-significant (`TOST` at $\pm 25\text{ pp}$ inconclusive), while GBR+CSCI lowers L3–L5 subgroup mean regret (`18.98%` -> `8.49%`).
4. **External Consortium Holdouts ($K=38$ Clusters):** Across **155 untouched consortium queries** (99 Official TPC-DS + 43 ClickBench [20 native SQL + 23 dialect-transpiled] + 13 SSB), tail-regret reduction (`97.05%` vs. `644.62%` CPU and `588.99%` GPU P95; `0/38` LOCO sign flips) is **directional but not statistically significant** ($p_{\text{perm}} = 0.125$ vs. CPU, $p_{\text{perm}} = 0.198$ vs. GPU; random-intercept LME retired).
5. **Prospective OOD & Harness-Shift Boundary (`3/25` Top-1):** On 25 locked single-shot prospective queries (`Seed_S7` $1/15$, `DSB_Subset` $2/5$, `JCC_H_Skew` $0/5$) over compact 1–2 table in-memory Arrow buffers, zero-shot Top-1 drops to **`12.0%`** (`83.98%` mean regret, `346.00%` P95 regret; `19/25` routed to Tier-L1 DataFusion basin; `0/25` static Mahalanobis OOD triggers), and verifying that **$m=5$ stratified canary calibration** immediately eliminates the DataFusion trap (`76.0%` -> `9.7%`), lifts out-of-sample Top-1 to **`38.1%`**, and cuts mean/P95 regret to **`26.06%` / `72.57%`** ($p_{\text{perm}}=0.00002$).

---

## 2. Clean Repository Layout

| Path | Contents |
| :--- | :--- |
| `manuscript/bigquery_olap_benchmark_V13.pdf` | Compiled V13 Research Article PDF (26 pages, Springer *Journal of Big Data*) |
| `manuscript/bigquery_olap_benchmark_V13.tex` | Canonical V13 LaTeX source (28/28 consistency gates verified) |
| `Journal_of_Big_Data_Snapp_Submission_Flat13.zip` | Clean-Room verified flat submission package |
| `reproducibility/frozen_ast_route_v11_gbdt_model.json` | Frozen Cat-I GBDT routing ensemble (`SHA-256: f0e5f17c832472e3...`) |
| `results/v13_master_query_registry_row_level.csv` | 495-row SQL registry (`400` Primary Cat. I–IV + `25` Prospective OOD + `70` Enterprise Stress) |
| `docker_replication/` | Single-command containerized replication kit (`EX = 1.000`) |
| `workloads/` | SQL benchmark queries across all 4 epistemic categories and auxiliary suites |
| `results/` | Verified per-query & multi-scale ($\text{SF}=1\text{--}100\text{ GB}$) hardware/energy CSV logs |
| `scripts/` | Benchmark harnesses, router evaluation, and verification scripts |
| `SHA256SUMS.txt` | Cryptographic SHA-256 checksums of primary artifacts and logs |

---

## 3. Reproducibility & Verification Commands

**1. Containerized Clean-Room Replication (Docker):**
```bash
cd docker_replication
docker compose up --build
```

**2. Verify Manuscript Tables & Zero-Contradiction Gates (Python):**
```bash
pip install -r reproducibility/requirements-lock.txt
python reproducibility/verify_and_regenerate_main_tables_R1_R6.py
python scripts/verify_zero_contradiction_V13.py
```

---

## 4. Protocol Transparency

This study was **not prospectively preregistered on OSF**. Repository `.zenodo.json` files and Git tags denote code-freeze checkpoints; Git commit chronology and Nested-CV guard harmonization (`12%` V1–V9 Fixed Protocol P2 -> `8%` V10–V13 Inner-Fold Nested-CV Protocol P1) are documented in **Tables 8, 10, and 18** of the V13 manuscript.
