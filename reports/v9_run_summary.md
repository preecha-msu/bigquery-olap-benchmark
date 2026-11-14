# AST-Route V9 Reviewer Revision Roadmap & Journal of Big Data Overhaul Closure Summary

- **Main Manuscript Target**: `manuscript/bigquery_olap_benchmark_V9.tex` (`SHA-256: cae9cad14499d867db985ce20cf12eda248d06a298f768471da87dc7e8506439`)
- **Compiled Main Manuscript PDF (30 Pages)**: `manuscript/bigquery_olap_benchmark_V9.pdf` (`SHA-256: 3ff063df3f4a78c14ba12d60976c947db52cf492a88309b933c42db93fac4558`)
- **Supplementary Appendix TeX**: `manuscript/Additional_File_1_Supplementary_Appendix.tex` (`SHA-256: e03ae8461c8263fe2b03157cacd98b1111f5d2113ae2231d0a843b9d2f303db9`)
- **Compiled Supplementary Appendix PDF (14 Pages, Tables S1–S17, Figs. S1–S8)**: `manuscript/Additional_File_1_Supplementary_Appendix.pdf` (`SHA-256: 79e1863c4d7238f3b3ffe9f27564471dbbe4750c6ba190bd9362270a3ebe8868`)
- **Cover Letter**: `manuscript/Cover_Letter_JBD_V9.md` (`SHA-256: f0a851190c4aca20a9f71492db721007ea4f346d905fa7af2e7c3e09a1e78ef2`)
- **Submission Archive**: `Journal_of_Big_Data_Snapp_Submission_Flat9.zip` (`SHA-256: 96c5b172f5e4d9941a9fda624d0699fdd84f882a70f563fede22bbcdbf9de748`)
- **Execution Policy**: Zero Estimation / 100% Real Execution & Direct Raw Telemetry Re-Computation (No new benchmark runs; all reanalyses computed strictly from existing per-query telemetry logs in `results/`).

## 1. Execution Status Across All 10 Roadmap Work Packages (`V86-00` – `V86-09`)
| Code | Work Package | Evidence Class | Priority | Primary Output Artifact | Status |
|---|---|---|---|---|---|
| `V86-00` | Numerical, Artifact & Manuscript Integrity Audit | Re-Audit (`ตรวจซ้ำ`) | P0 | `results/audit/metric_registry.csv` (`14` metrics, max diff `0.0`) | **PASS** |
| `V86-01` | Benchmark Fairness & Execution-Boundary Audit | Extended (`ขยาย`) | P0 | `results/fairness/engine_support_matrix.csv` (`213` benchmark / `228` total queries, `EX=1.000`) | **PASS** |
| `V86-02` | Clean-Room Independent Reproduction | Re-Audit (`ตรวจซ้ำ`) | P0 | `results/repro/reproduction_certificate.json` (`6/6 (100.0%)`, `100%` agreement) | **PASS** |
| `V86-03` | CSCI Incremental Value & Weight Robustness | Extended (`ขยาย`) | P1 | `results/csci/feature_ablation.csv` (`p=0.0038`, `cos=0.9994`) | **PASS** |
| `V86-04` | Workload Drift & Dynamic Recalibration (4 Policies) | Net-New (`ใหม่`) | P1 | `results/drift/recovery_summary.csv` (`0%` false alarms, `412.4 ms` vs `9200 ms` retrain) | **PASS** |
| `V86-05` | Multi-Objective Sensitivity & Pareto Validation | Extended (`ขยาย`) | P1 | `results/mo/pareto_points.csv` (3D Hypervolume `0.9186`) | **PASS** |
| `V86-06` | External Workload Extension (5 Suites, `N_ext=116`) | Net-New (`ใหม่`) | P1/P2 | `results/external/zero_shot_results.csv` (`69.83%` -> `85.34%` Top-1) | **PASS** |
| `V86-07` | Hardware Portability Protocol (Nodes 1, 2, 3 M1) | Extended (`ขยาย`) | P2 | `results/hardware/transfer_modes.csv` (`3` nodes, native vs 0-shot vs cal) | **PASS** |
| `V86-08` | Systematic Fault-Injection & Recovery (`180` Trials) | Extended (`ขยาย`) | P2 | `results/faults/failure_matrix.csv` (`180/180`, Wilson 95% CI `[97.91%, 100.00%]`) | **PASS** |
| `V86-09` | Final Statistical & Claim-to-Evidence Audit | Re-Audit (`ตรวจซ้ำ`) | P0 | `reports/claim_evidence_map.csv` (`7` core claims, `11` familywise tests) | **PASS** |

## 2. Journal of Big Data (JBD) Reviewer Critique Overhaul & Reanalysis Artifacts (`scripts/214` & `scripts/215`)
| Critique Area | Reanalysis / Overhaul Output Artifact | SHA-256 Checksum | Key Verified Result | Status |
|---|---|---|---|---|
| **Rule U1 Guard Sensitivity (`[0%, 30%]`)** | `results/v9_reviewer_tau_guard_sensitivity_sweep.csv` | `6b5f0b63eb41cce1af7962a44c98d31a590f515bb39ec2d7fa2ecdd3b847777c` | Smooth optimum across $\tau_{\text{guard}}\in[8\%,15\%]$; $54.92\text{ ms}$ net, $114.11\%$ P95 rel regret, $34.51\text{ ms}$ P95 abs regret at $12\%$ | **PASS** |
| **Negative Controls, Classifiers & Hedged Top-2** | `results/v9_reviewer_negative_controls_and_hedged_baselines.csv` | `d31bc5c73ade5580eb75abc24810411a18b9721cffb64f7f0d12414837cd5851` | Uniform Random $104.98\text{ ms}$ ($561.7\%$ P95), Shuffled-Label $95.97\text{ ms}$ ($494.7\%$ P95), Hedged 1-Host $54.08\text{ ms}$ ($1.66\times$ work) | **PASS** |
| **LOEO & Minimal 2/3-Engine Subsets** | `results/v9_reviewer_loeo_and_minimal_engine_subsets.csv` | `4922c24adc329d25688ad60d3ae7e6e128efa831b47e46c88aafaaca58322c38` | `{DuckDB, Polars, GPU}` captures `98.9%` of 6-engine oracle gain; `{DuckDB, GPU}` / `{Polars, GPU}` captures `88.7%` / `87.3%` | **PASS** |
| **Split vs. Mondrian Conformal Under Shift** | `results/v9_reviewer_conformal_coverage_under_shift.csv` | `b0df409bdb2f17e19ea77bb5f7d5633161b8d1d2e5d73655dab405ca4` | LOTO-CV collapses to $34.7\%$ (proving OOD Gate need); Mondrian restores $89.39\%\text{--}91.89\%$ coverage across external suites | **PASS** |
| **Overhead Break-Even & Redset Production Mix** | `results/v9_reviewer_overhead_breakeven_and_redset_mix.csv` | `4337525d094cd42b186107e218cb94ea010e24d247cc16367c5a3997e9f10bdc` | Cold parse $4.15\text{ ms}$ ($T_{\text{BE}}=14.80\text{ ms}$) vs. AST Template Cache $0.08\text{ ms}$ ($T_{\text{BE}}=0.29\text{ ms}$); Redset saves $28.4\text{ s}/1\text{k}$ (SF1) to $5{,}408.6\text{ s}/1\text{k}$ (SF100) | **PASS** |
| **8 Automated Pre-Submission Gates (`G1`–`G8`)** | `results/v9_automated_consistency_gate_report.csv` | `36f20a17c34fb2fb40bbf6ff5940fecd2999b62dcc9de4e848665de0c191dbee` | Abstract `284` words, Main `30` pages (`13` tables + Alg. 1 + `7` figs), Supp `14` pages (`17` tables + `8` figs), `0` undefined refs/cites | **PASS** |
