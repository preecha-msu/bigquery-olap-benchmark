# Cryptographic Reproduction & Zero-Discrepancy Audit Certificate (V2)

All empirical telemetry files, multi-scale TPC-H benchmarks (`SF=1`, `SF=10`, `SF=100`),
official TPC-H external validation logs (`Q1-Q22`), and AST query routing evaluations
have been verified from real hardware executions with zero statistical estimation.

| Artifact File | Rows | Cols | Size (Bytes) | SHA-256 Digest |
| :--- | ---: | ---: | ---: | :--- |
| `ast_query_router_evaluation_sf1.csv` | 8 | 9 | 870 | `56b52566ce859f99fb61e66280663ffa0dbc2b424915d9e2ade6969676cc845d` |
| `ast_query_router_per_query_sf1.csv` | 75 | 15 | 15,551 | `e899ac9a886c24d8055f72a08d4573bf3702723e292ad065356e7169345e7501` |
| `csci_validation_sensitivity_sf1.csv` | 9 | 9 | 1,747 | `fa89ea8b7f8790be6fb3f9bcae3388cc08fa0702c3006faf816c11846ed3d79a` |
| `gpu_cpu_3mode_fairness_comparison_sf1.csv` | 5 | 15 | 803 | `13dbb975161f943d00960c8b7085e91de8fbffa13ed1d48bdec23e8831791ba1` |
| `hardware_software_environment_manifest.csv` | 12 | 2 | 473 | `1d906b543ca39b3d682de15ceac00a285932c4891bc2d69309f94c3490e9a9ae` |
| `hierarchical_seed_cluster_stats_sf1.csv` | 9 | 8 | 1,530 | `8288e7166b387ea5b3a321526d0fecf61bf6a19dca180c3b9727c1fdf748b1ff` |
| `merged_75queries_all_engines_sf1.csv` | 75 | 27 | 22,216 | `5eb9a13b43a164f5f11bbee91a774372446182487cab251cc1048f540574d0e3` |
| `multiscale_gpu_vram_and_arrow_streaming.csv` | 9 | 8 | 925 | `5c31a038864caedbdd059a3f23f36e5f3aa07d49ea290f01c037ce38b9124b26` |
| `multiscale_manifest_sf10_sf100.csv` | 2 | 6 | 291 | `c12a2ba9946996ea95c9f6c78f738d20d6529ca0f0175af5fb9351988d3cac58` |
| `multiscale_scalability_summary_table.csv` | 6 | 11 | 649 | `370a5f0e7a053febae10bc65cbb694746675d92efd3650a2e9d349e715a783f2` |
| `multiscale_telemetry_sf1.csv` | 150 | 19 | 23,497 | `9be2a6c63a2b779871a1a42c87f5ec3dd4823fda06260464e4188b66aa1ef6ca` |
| `multiscale_telemetry_sf10.csv` | 150 | 19 | 23,770 | `623ef56203c6cd3d33acc89540b6e8780e46121cb54a298c97756b6c5e3b2690` |
| `multiscale_telemetry_sf100.csv` | 150 | 19 | 24,572 | `31518d87f5af08a75c5a6a9f7f033e9a3d173cb9916e960eaf2d6dec57b65bc8` |
| `multiscale_tier_breakdown.csv` | 7 | 10 | 699 | `8c7546fd8597280584370e10c14dd4f1e3ce0e76b4d1508fe86a2aaf06c8187a` |
| `multiset_equivalence_audit_6engines.csv` | 6 | 8 | 623 | `1fad2496951a49dc3674c62a2fe6bb6e4db6d5a006a0ce64e6fda40df5642544` |
| `official_tpch_q1_q22_external_validation.csv` | 22 | 15 | 1,770 | `f7f4490389ae0db27608e673368cbb816a85869af95c2c4400c26de4a55d4f69` |
| `reconciled_sf1_master_table.csv` | 6 | 16 | 1,780 | `92cf88a5ba93480c3fedf7eaf924fbc36f63c3badd09aa69c3ab1c28a5fa56df` |
| `sha256_manifest_v2.csv` | 19 | 6 | 2,685 | `a3bc4d18cc7c4b85a68e234f2a4f76a51c5b1d89303d481de5206f94aaf3b7c3` |
| `two_way_interaction_complexity_selectivity_sf1.csv` | 9 | 8 | 1,066 | `d3693d770dc7743b0020cb920c8668bb32df90979e087f44dfd92c5f5398b6de` |