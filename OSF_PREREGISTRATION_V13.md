# OSF Prospective Preregistration Protocol (AST-Route V13)

## 1. Study Title
Tail-Safe CPU-GPU-Cloud OLAP Dispatch via PCIe Margin Guarding and Zero-Compile SQL Metadata

## 2. Prior Post-Hoc Harmonization Disclosure
All experiments in Manuscript V13 (400 SQL workloads across Categories I-IV and 64 single-pass cross-hardware cells) were completed prior to this registration; specifically, the inner-fold 8% PCIe margin guard (Protocol P1) and +10% non-inferiority threshold were harmonized post-hoc during V10-V12 (replacing the initial V1-V9 fixed 12% guard and +-5 ms TOST criterion).

## 3. Prospective Locked-Model Evaluation Plan (For New Unseen Workloads)
- **Frozen Repository:** `https://github.com/preecha-msu/db_cluster_distributed`
- **Locked Primary Endpoint:** 95th-percentile (P95) relative latency regret vs. per-query oracle across strictly unseen prospective queries (TPC-DS SF=10/100 GB & JCC-H skewed joins), evaluated with fixed 8% inner-fold abstention guard (`tau_guard = 0.08`) and zero weight/threshold retuning.
- **Locked Statistical Test:** Paired within-query contrast analyzed via Random-Slope REML (`regret ~ policy + (1 + policy | template_cluster)`) and 50,000-iteration cluster-level sign-flip permutation at `alpha = 0.05` (Holm-adjusted).

## 4. How to Mint Official Zenodo DOI & Link to Manuscript in 2 Minutes
1. Push `clean_archival_repo_v13/` to `https://github.com/preecha-msu/db_cluster_distributed`.
2. Enable `db_cluster_distributed` at https://zenodo.org/account/settings/github/ and publish GitHub Release `v13.0.0`.
3. Copy the minted `10.5281/zenodo.XXXXXXX` string into `ZENODO_DOI.txt` in the project root and re-run `scripts/step8_v13_groupA_empirical_tests_and_complete_overhaul.py` to embed the live DOI across PDF and ZIPs.
