# AST-Route v13.0.0 — Official Archival Release for Journal of Big Data (SpringerOpen)

**Title:** AST-Route: Tail-Safe CPU-GPU-Cloud OLAP Dispatch via PCIe Margin Guarding and Zero-Compile SQL Metadata
**Author:** Preecha Noiumkar (Mahasarakham University, Thailand)

## Key Artifacts
- `manuscript/bigquery_olap_benchmark_V13.pdf` & `.tex`: V13 manuscript (30/30 consistency gates passed).
- `Journal_of_Big_Data_Snapp_Submission_Flat13.zip`: Clean-room verified Springer Snapp flat submission archive.
- `reproducibility/frozen_ast_route_v11_gbdt_model.json`: Frozen Cat-I 9D GBDT ensemble (`SHA-256: f0e5f17c832472e3...`).
- `results/v13_master_query_registry_row_level.csv`: 495-row zero-overlap SQL registry (400 Primary Cat. I-IV + 25 Prospective + 70 Enterprise Stress).
- `results/v13_prospective_m5_canary_recovery_audit.csv`: Out-of-sample $m=5$ stratified canary calibration recovery (`12.0%` -> `38.1%` Top-1; `76.0%` -> `9.7%` DataFusion trap; `83.98%` -> `26.06%` mean regret; `346.21%` -> `72.57%` median P95 regret, $p=0.00002$).
- `docker_replication/`: Single-command containerized replication kit (`EX = 1.000`).
