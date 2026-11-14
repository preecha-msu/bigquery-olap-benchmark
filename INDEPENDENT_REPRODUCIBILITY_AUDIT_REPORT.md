# Independent 6-Stage Reproducibility Audit Report
**Repository:** `bigquery-olap-benchmark`  
**Target Manuscript:** `manuscript/bigquery_olap_benchmark_V9.tex` (31 pages Main Text + 15 pages `Additional_File_1_Supplementary_Appendix.pdf`)  
**One-Command Verification Script:** `python reproducibility/verify_and_regenerate_main_tables_R1_R6.py`  
**Audit Execution Timestamp:** `2026-11-14 09:55:00`  
**Discrepancy Report SHA-256:** `ae88fba80e1913afa978daddddd87a72eb9040d9b876482debe8fbc3dc4dbf81`  

---

## Stage 1: Hardware & Software Environment Verification
| Component                      | Detected_Specification                             | Status   |
|:-------------------------------|:---------------------------------------------------|:---------|
| Operating System               | Windows 11 (10.0.26200)                            | VERIFIED |
| Host CPU Architecture          | AMD64 | 8 Physical / 12 Logical Cores              | VERIFIED |
| Host System RAM                | 63.71 GB Total DDR4                                | VERIFIED |
| Discrete GPU Accelerator       | NVIDIA GeForce RTX 2050 (4.0 GB VRAM)              | VERIFIED |
| NVIDIA Driver & CUDA Runtime   | Driver: 528.20 | CUDA: 12.4 | PyTorch: 2.6.0+cu124 | VERIFIED |
| Python Runtime                 | 3.13.9                                             | VERIFIED |
| DuckDB (C++ Vectorized DBMS)   | v2.0.0                                             | VERIFIED |
| Polars (Rust Apache Arrow)     | v2.1.0                                             | VERIFIED |
| Apache DataFusion (Rust Arrow) | v55.0.0                                            | VERIFIED |
| SQLite 3 (Row-Store Baseline)  | v3.51.0                                            | VERIFIED |
| SQLGlot & Scikit-Learn         | SQLGlot v30.24.0 | Scikit-Learn v1.7.2             | VERIFIED |

## Stage 2: Cryptographic Data & Raw Log Integrity (SHA-256)
| Artifact_Category   | Artifact_Name                                   |   Row_Count |   Size_Bytes | SHA256_Prefix24          | Integrity_Status        |
|:--------------------|:------------------------------------------------|------------:|-------------:|:-------------------------|:------------------------|
| SQL Workload Suite  | workloads/*.sql (75 SQL files)                  |          75 |        84539 | 7fb0ae30762e31094944f918 | PASS (100% Match)       |
| Telemetry Log CSV   | ast_complexity_summary.csv                      |          75 |        12216 | 4cb5f7477fd24dc77a6fd2b5 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_benchmark_sf1.csv                           |         375 |        53453 | 7d69610e86863845ae7cbb5c | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_bigquery_cloud_sf1.csv                      |         150 |        26594 | 7aee23711627a085062c7cf8 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_gpu_cuda_sf1.csv                            |         375 |        60905 | 50201ab2ea90e96f4b49796c | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_multi_software_sf1.csv                      |         375 |        51996 | 4d6ba9d612f4c309ec096d61 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_hardware_energy_sf1.csv                     |         375 |       163502 | 1a607af26e1bfddaef3e4466 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | raw_query_routing_predictions_sf1.csv           |          75 |        11680 | 8d077e29f9b9f3fd40cc7f3e | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | merged_75queries_all_engines_sf1.csv            |          75 |        22216 | 5eb9a13b43a164f5f11bbee9 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | query_routing_evaluation_sf1.csv                |          10 |         1054 | 89b1ec64cca15b2dd41a4ad2 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | framework_ablation_and_robustness_sf1.csv       |          14 |         1781 | 24053808e1e0e4a910b148b7 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | priority3_unseen_holdout_and_cost_model_sf1.csv |          13 |         2036 | d3b6edd646b4b71c51842ae0 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | official_tpch_q1_q22_summary_sf1.csv            |          22 |         4136 | 5538c2065d686171f879b970 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | multiscale_sf1_sf10_sf30_sf100_summary.csv      |           4 |          527 | c47702b2e3ea590a3e230f35 | PASS (SHA-256 Verified) |
| Telemetry Log CSV   | concurrent_workload_benchmark_sf1.csv           |           5 |          787 | fb82beb361435f3b72f9ed44 | PASS (SHA-256 Verified) |

## Stage 3: Stratified Independent Query Re-Execution (L1–L5 Live Run)
| Query_ID            | Tier   |   DuckDB_Live_ms |   Polars_Live_ms |   SQLite3_Live_ms |   GPU_Tensor_Live_ms |   Output_Row_Count |   FP64_Aggregate_Checksum |
|:--------------------|:-------|-----------------:|-----------------:|------------------:|---------------------:|-------------------:|--------------------------:|
| L1_Scan_Filter_Rep  | L1     |            8.268 |           10.557 |            18.139 |              802.492 |             101767 |               2.4603e+08  |
| L2_Multi_Join_Rep   | L2     |            5.85  |            6.145 |           449.276 |                1.347 |             121952 |               3.11014e+08 |
| L3_Staged_CTE_Rep   | L3     |            8.534 |           10.557 |            94.946 |                0.399 |              40150 |               4.81675e+08 |
| L4_Window_Rank_Rep  | L4     |            7.316 |           17.508 |           274.47  |                0.354 |               3000 |               1.48876e+07 |
| L5_Array_Unnest_Rep | L5     |            5.442 |            3.37  |            28.234 |                0.381 |             119889 |               7.78296e+09 |

## Stage 4: Multiset Result Equivalence Verification
| Query_ID            | Tier   | Row_Count_Exact_Match   | NULL_Semantics_Match   | Duplicate_Preservation   |   Max_FP64_Relative_Error |   Tolerance_Threshold | Equivalence_Verdict   |
|:--------------------|:-------|:------------------------|:-----------------------|:-------------------------|--------------------------:|----------------------:|:----------------------|
| L1_Scan_Filter_Rep  | L1     | PASS (100%)             | PASS (100%)            | PASS (100%)              |                  6.66e-15 |                 1e-09 | PASS (EX=1.000)       |
| L2_Multi_Join_Rep   | L2     | PASS (100%)             | PASS (100%)            | PASS (100%)              |                  9.2e-15  |                 1e-09 | PASS (EX=1.000)       |
| L3_Staged_CTE_Rep   | L3     | PASS (100%)             | PASS (100%)            | PASS (100%)              |                  4.95e-16 |                 1e-09 | PASS (EX=1.000)       |
| L4_Window_Rank_Rep  | L4     | PASS (100%)             | PASS (100%)            | PASS (100%)              |                  5e-16    |                 1e-09 | PASS (EX=1.000)       |
| L5_Array_Unnest_Rep | L5     | PASS (100%)             | PASS (100%)            | PASS (100%)              |                  3.92e-15 |                 1e-09 | PASS (EX=1.000)       |

## Stage 5 & 6: Statistical Recalculation & Automated Discrepancy Report (Published vs. Raw Logs)
| Table_Reference   | Metric_Name                                         |   Published_Manuscript_Value |   Recalculated_RawLog_Value |   Absolute_Discrepancy | Unit   | Audit_Verdict           |
|:------------------|:----------------------------------------------------|-----------------------------:|----------------------------:|-----------------------:|:-------|:------------------------|
| Table 1 & 5       | DuckDB Native Mean Latency (SF=1, N=75)             |                      183.74  |                     183.74  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 1           | DuckDB External Parquet Mean Latency (SF=1, N=75)   |                      254.42  |                     254.42  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 2 & 5       | Google BigQuery Clustered Mean Latency (SF=1, N=75) |                      745.39  |                     745.39  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 4 & 5       | RTX 2050 GPU Warm Kernel Mean Latency (SF=1, N=75)  |                       22.26  |                      22.26  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 4 & 5       | RTX 2050 GPU Cold Mean Latency (SF=1, N=75)         |                       56.96  |                      56.96  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 5 & 8       | Polars Rust Arrow Mean Latency (SF=1, N=75)         |                       82.47  |                      82.47  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 5 & 8       | Apache DataFusion Mean Latency (SF=1, N=75)         |                       96.3   |                      96.3   |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 5 & 10      | SQLite 3 Row-Store Mean Latency (SF=1, N=75)        |                     8417.33  |                    8417.33  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 7           | DuckDB Mean Active Device Energy (SF=1)             |                        6.4   |                       6.4   |                      0 | J      | PASS (Zero Discrepancy) |
| Table 7           | Polars Mean Active Device Energy (SF=1)             |                        3.17  |                       3.17  |                      0 | J      | PASS (Zero Discrepancy) |
| Table 7           | RTX 2050 GPU Cold Active Energy (SF=1)              |                        0.65  |                       0.65  |                      0 | J      | PASS (Zero Discrepancy) |
| Table 8 & 9       | Proposed AST-Route (9D) LOSO-CV Mean Latency        |                       54.92  |                      54.92  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 8 & 9       | Proposed AST-Route (9D) Top-1 Routing Accuracy      |                       57.33  |                      57.33  |                      0 | %      | PASS (Zero Discrepancy) |
| Table 8           | Proposed AST-Route (9D) Top-2 Routing Accuracy      |                       76     |                      76     |                      0 | %      | PASS (Zero Discrepancy) |
| Table 8 & 9       | Proposed AST-Route (9D) Mean Performance Regret     |                       34.98  |                      34.98  |                      0 | %      | PASS (Zero Discrepancy) |
| Table 8 & 11      | Proposed AST-Route (9D) P95 Tail Regret             |                      114.11  |                     114.11  |                      0 | %      | PASS (Zero Discrepancy) |
| Table 8 & 9       | Proposed AST-Route (9D) Mean Energy Consumption     |                        1.381 |                       1.381 |                      0 | J      | PASS (Zero Discrepancy) |
| Table 10          | Cliff's Delta: Polars vs. SQLite 3                  |                        1     |                       1     |                      0 | delta  | PASS (Zero Discrepancy) |
| Table 10          | Cliff's Delta: DuckDB Native vs. Parquet            |                        0.296 |                       0.296 |                      0 | delta  | PASS (Zero Discrepancy) |
| Table 11B         | Unseen Holdout Seeds (N=15) Top-1 Accuracy          |                       73.33  |                      73.33  |                      0 | %      | PASS (Zero Discrepancy) |
| Table 11B         | Unseen Holdout Seeds (N=15) Mean Latency            |                       47.01  |                      47.01  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 12C         | Official TPC-H Q1-Q22 Nested LOO-CV Mean Latency    |                       16.98  |                      16.98  |                      0 | ms     | PASS (Zero Discrepancy) |
| Table 12C         | Official TPC-H Q1-Q22 Nested LOO-CV Top-2 Accuracy  |                       90.91  |                      90.91  |                      0 | %      | PASS (Zero Discrepancy) |
