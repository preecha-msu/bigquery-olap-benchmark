# Cryptographic SHA-256 Telemetry & Zero-Discrepancy Audit Certificate

- **Framework Title:** AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines
- **Author:** Preecha Noiumkar (Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand)
- **Execution Protocol:** 100% Full-Scale Real Execution (Zero Estimation / Zero Extrapolation)
- **Evaluated Workloads:** 97 Total SQL Queries (75 TPC-H-Derived AST-Stratified Queries across L1-L5 + 22 Official Standard TPC-H Queries Q1-Q22)
- **Evaluated Data Scales:** Standard TPC-H Scale Factors SF = 1 GB (8.7M rows), SF = 10 GB (86.6M rows), SF = 30 GB (259.8M rows), and SF = 100 GB (866.1M total rows / 600.1M lineitem rows)

## Verified SHA-256 Checksums of All Ground-Truth Telemetry & Summary Tables

| File Path | Description / Manuscript Table Mapping | SHA-256 Prefix (24-Char) | Full 64-Character SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| `workloads/ast_complexity_summary.csv` | Static 9D AST Structural & CSCI Summary (75 Queries) | `4cb5f7477fd24dc77a6fd2b5...` | `4cb5f7477fd24dc77a6fd2b58063d9868cf7126ea89e6d902a695ccebd58d9d9` |
| `results/raw_logs/raw_benchmark_sf1.csv` | DuckDB External Parquet vs. Native Columnar Raw Telemetry | `7d69610e86863845ae7cbb5c...` | `7d69610e86863845ae7cbb5ce25d81202a6d25317db25a35c4a14b7715dca8c5` |
| `results/raw_logs/raw_bigquery_cloud_sf1.csv` | Google BigQuery Unoptimized vs. Clustered Raw Telemetry | `7aee23711627a085062c7cf8...` | `7aee23711627a085062c7cf883f9404ce3aceaac2e5c080fa3a9060563458e5f` |
| `results/raw_logs/raw_gpu_cuda_sf1.csv` | NVIDIA RTX 2050 CUDA PCIe H2D & Warm Kernel Raw Telemetry | `50201ab2ea90e96f4b49796c...` | `50201ab2ea90e96f4b49796c5707641699be5d38e37505bb2d0b085e6803c7d2` |
| `results/raw_logs/raw_multi_software_sf1.csv` | Polars, Apache DataFusion, and Full-Scale SQLite 3 Raw Telemetry | `4d6ba9d612f4c309ec096d61...` | `4d6ba9d612f4c309ec096d61ee368394fc9a5d343155647bd094f5f55b750524` |
| `results/raw_logs/raw_hardware_energy_sf1.csv` | High-Frequency CPU/GPU Util %, RAM/VRAM, and Energy (Joules) Logs | `366d77e7426a78b6089254eb...` | `366d77e7426a78b6089254ebac3d89d84fe4a7ec069e18e5abffe008eb9bd612` |
| `results/raw_logs/raw_query_routing_predictions_sf1.csv` | 25-Seed LOSO-CV Out-of-Fold Routing Predictions | `8d077e29f9b9f3fd40cc7f3e...` | `8d077e29f9b9f3fd40cc7f3e1d267b14a47bf50f624a9268c4912698701aeca9` |
| `results/summary_tables/six_software_comparison_sf1.csv` | Table 5 & 6: Six-Software Latency & Throughput Matrix | `b180bbc2ef1884d3316cec3d...` | `b180bbc2ef1884d3316cec3d69d581bc4639f3e95796d75ba1e007eda99d9fd9` |
| `results/summary_tables/hardware_energy_summary_sf1.csv` | Table 7: Hardware Resource & Green OLAP Energy Summary | `7d68a9edf2041ddf83f51545...` | `7d68a9edf2041ddf83f51545ad0d0cc6aa7eb6fdd812b82f9f26f579cc3493aa` |
| `results/summary_tables/query_routing_evaluation_sf1.csv` | Table 8: Five-Policy Automated Query Routing Evaluation | `89b1ec64cca15b2dd41a4ad2...` | `89b1ec64cca15b2dd41a4ad2f6f547529a4d14dd64dab4a49ca66f4ae7ce0838` |
| `results/summary_tables/framework_ablation_and_robustness_sf1.csv` | Table 9: Feature Ablation, Pareto Frontier & Selectivity Noise | `24053808e1e0e4a910b148b7...` | `24053808e1e0e4a910b148b73339d773f3f40adf5e14bafb020aa157b1631e39` |
| `results/summary_tables/reviewer_trust_audit_report_sf1.csv` | Table 10: 10,000-Bootstrap 95% CIs & Cliff's Delta Audit Matrix | `35ac0344bbff073696ad74c7...` | `35ac0344bbff073696ad74c7cf89499323b840fc273841e5487c48dade413c74` |
| `results/summary_tables/priority3_unseen_holdout_and_cost_model_sf1.csv` | Table 11: Cost Predictor Q-Error & Unseen Seed Holdout (L1_S5-L5_S5) | `d3b6edd646b4b71c51842ae0...` | `d3b6edd646b4b71c51842ae08c5f9a17e8f781ad623ba44368515440614199e2` |
| `results/summary_tables/official_tpch_q1_q22_summary_sf1.csv` | Table 12C: Official Standard TPC-H Q1-Q22 Benchmark Evaluation | `5538c2065d686171f879b970...` | `5538c2065d686171f879b9705b30557f01e65867a135e698ec15642f509189cc` |
| `results/summary_tables/multiscale_sf1_sf10_sf30_sf100_summary.csv` | Table 12A: 4-Scale Big Data (SF=1, 10, 30, 100 GB) & Spill Summary | `02361d1654718a1f39c537cf...` | `02361d1654718a1f39c537cf3340633b84b6600e7ff053cee8ec8f3cfa52be0c` |
| `results/summary_tables/multiscale_tier_breakdown_sf1_10_30_100.csv` | Table 12A/B: Tier-by-Tier Multi-Scale Breakdown (L1-L5) | `d19ea859b42b812ee443bcc9...` | `d19ea859b42b812ee443bcc9bf472353e01641494916a967589e2a51151d36a4` |
| `results/summary_tables/cross_platform_validation_summary.csv` | Table 12B: Cross-Platform Hardware Scaling (2, 4, 8 Cores) | `a89abc0352307355e2199eda...` | `a89abc0352307355e2199edafe995f6b2ca08b4a65470f316b88c68691e62d86` |
| `results/summary_tables/environment_manifest_sf1_sf100.csv` | Table 13: Hardware and Software Environment Manifest | `3ae73b1f0984d5dcd1058697...` | `3ae73b1f0984d5dcd1058697275a4ddc3d612734dbc8b7d8b3c9fd5505b705cf` |
