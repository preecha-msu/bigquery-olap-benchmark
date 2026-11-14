# AST-Route V10 Reviewer Revision & Research Protocol Roadmap

**Baseline Version:** AST-Route V9.8 / V9.11 (`manuscript/bigquery_olap_benchmark_V9.pdf`, 32 pages; `Additional_File_1_Supplementary_Appendix.pdf`, 16 pages)  
**Serialized Frozen Model Artifact (`reproducibility/frozen_ast_route_v9_gbdt_model.json`):** `f0e5f17c832472e30be67dbbee30020dbf23e94658458a2c59acd9b00ed315cf`  
**Master Workload Accounting (`400` Unique SQL Queries):** `75` Category I Core Dev + `55` Category II Diagnostic Standard (`22` Official TPC-H + `33` IMDB JOB) + `115` Category III Author-Derived Zero-Shot (`15` S6 + `30` TPC-DS-Derived + `20` BIRD-Inspired + `25` TPC-DS-Complex + `25` Redset-BI) + `155` Category IV Strictly Untouched Third-Party Holdouts (`99` Unmodified Official TPC-DS + `43` Official ClickBench + `13` Official SSB).

---

## Overview of Evidence Already Completed in V9 vs. Hardware Extensions Designated for V10

| Experiment ID | Priority | V9 Status (Completed from Existing Telemetry in `v911_reviewer_v98_complete_synthesis_audit.csv`) | V10 Physical Hardware Extension Scope |
| :--- | :---: | :--- | :--- |
| **E1 — Independent GPU Backend Validation** | **Priority 1** | **Completed on `CuPy v14.2.0` CUDA 12 & `Polars-Arrow-CuPy` (`60/60` native L1–L4 & `75/75` CSR: $	ext{EX}=1.000$, `100.0%` routing decision agreement, $ho=0.9999$)** | Add native C++ `RAPIDS cuDF` / `HeavyDB` execution on Linux/WSL2 and cloud Tesla T4/A100 |
| **E2 — Multi-Objective Pareto Validation** | **Priority 1** | **Completed across Policies `P1–P4` & 15-Weight Simplex ($	ext{HV}=0.8965	ext{--}0.9340$, `0.0%` Dominated Solution Ratio vs `100%` static, $\mathcal{R}_T/\mathcal{R}_E/\mathcal{R}_C$ reported)** | Extend Pareto frontier to dynamic per-query electricity spot pricing and multi-cloud (Snowflake/Athena) billing |
| **E3 — Whole-System Energy Validation** | **Priority 2** | **Completed across 4 scopes: Measured NVML GPU (`1.24%` MAPE), Estimated CPU Package (`3.09%–3.21%` MAPE), Composite (`1.96%`), and 45s ACPI DC Battery Shunt (`2,817` queries, $ho=0.9945$)** | Add external inline AC wall-socket power meter (Yokogawa WT310E / Hoboware USB wattmeter) + Linux ring-0 Intel RAPL |
| **E4 — Large-Scale TPC-DS Validation** | **Priority 2** | **Completed on Official 99Q (`SF=0.02`), Snowflake 55Q (`SF=1 GB`, `2.83M` rows, `83.64%` 9D / `89.09%` SQL-5 Top-1), and Snowflake 30Q (`SF=10 GB`, `28.3M` rows, `86.67%` 9D / `90.00%` 12D Top-1)** | Execute all `99/99` unmodified Official TPC-DS queries at `SF=10 GB` and `SF=100 GB` on 16–40 GB VRAM server GPUs |
| **E5 — Conformal Guard Robustness** | **Priority 2** | **Completed across $	au_{	ext{guard}} \in \{0\%, 4\%, 8\%, 10\%, 12\%, 15\%, 20\%, 30\%\}$ with Seed-Level 95% CIs, Nadeau–Bengio $p_{	ext{NB}}$, and Primary (`8%`) vs Sensitivity (`12%`) separation** | Evaluate adaptive online conformal step-size updates ($\eta_t$) under continuous 24-hour production query streams |
| **E6 — Independent Reproducibility Audit** | **Priority 2** | **Completed across 3 physical nodes (`x86_64`+RTX 2050, 22T `x86_64`, Apple M1 `ARM64`) and one-command `R1–R6` (`0.000` table discrepancy, `140/140` SHA-256 verified)** | External blind third-party reproduction in an independent laboratory container with time-stamped OSF registration |

---

## Detailed 9-Point Research Protocol for Each Experiment (E1–E6)

### Experiment E1 — Independent GPU Backend Validation (Priority 1)
1. **Research Hypothesis & Scientific Motivation:** $H_1$: AST-Route's latency gains and tail-regret compression do not depend on the author-developed PyTorch CUDA tensor prototype; substituting an independent columnar GPU execution engine preserves $\ge 95\%$ routing decision agreement and statistically significant speedups over static CPU/GPU assignment.
2. **Experimental Design & Control Variables:** Hold hardware (`Node 1: RTX 2050 4 GB VRAM, PCIe Gen4 x4 at 5.69 GB/s`), input data (zero-copy Apache Arrow / DLPack buffers and cold ZSTD Parquet), thread count ($T=8$), and frozen router weights (`f0e5f17c832472e3`) strictly constant while varying the GPU execution backend across: (a) `PyTorch 2.6.0+cu124`, (b) `CuPy v14.2.0` CUDA 12 Thrust/CUB columnar primitives, (c) `Polars-Arrow-CuPy Hybrid`, and (d) `RAPIDS cuDF` / `HeavyDB` (V10 Linux extension).
3. **Dataset & SQL Query Inventory:** Common supported subset $N=60$ queries (Tiers L1–L4) and full $N=75$ Core TPC-H suite (with CSR `cp.repeat` offset expansion for Tier L5 `UNNEST`).
4. **Python Scripts & Role:** `scripts/216_v97_reviewer_experiments_E1_to_E7.py` (Module E1), `scripts/217_v98_reviewer_action_plan_A_to_G.py` (Part A), and `reproducibility/verify_and_regenerate_main_tables_R1_R6.py`.
5. **Statistical Tests & Mathematical Formulas:** Spearman rank concordance $ho$, exact routing decision match rate $rac{1}{N}\sum_{i=1}^N \mathbb{I}(\hat{e}_{	ext{PyTorch}}(q_i) = \hat{e}_{	ext{CuPy}}(q_i))$, multiset accuracy $	ext{EX}=1.000$ ($\|\mathbf{r}_1 - \mathbf{r}_2\|_{\infty} \le 10^{-9}$), and Nadeau–Bengio CV-corrected paired $t$-test $t_{	ext{NB}} = ar{d} / \sqrt{(K^{-1} + n_{	ext{test}}/n_{	ext{train}})s_d^2}$.
6. **Result Tables:** Main Text Table 10, Supplementary Table S17 (Rows 3A–3B), and `results/v97_exp_e1_external_gpu_backend_validation.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Pass if $	ext{EX} = 100\%$ on supported queries and Routing Decision Agreement $\ge 95\%$ (Measured in V9: **`100.0%` [`60/60` and `75/75`], $ho=0.9999$, `54.10 ms` vs `53.76 ms` net**). If an external GPU engine lacks L5 `UNNEST` and falls back to Polars CPU, report the exact fallback latency (`75.06 ms`, still beating Polars `82.47 ms`).
8. **Manuscript Integration Locations:** Section 2.3 (Item 4), Section 4.6.4 (Regime 3), Section 5.1 (Internal Validity), and Table S17.
9. **Acceptance Probability Impact if Negative:** If decision agreement were $<85\%$, claims would be restricted to tensor-lowered operators; because measured agreement is `100.0%`, internal validity is strongly reinforced.

### Experiment E2 — Multi-Objective Pareto Validation (Priority 1)
1. **Research Hypothesis & Scientific Motivation:** $H_2$: Adjusting the multi-objective preference vector $(lpha, eta, \gamma) \in \Delta^2$ over latency, energy, and cloud FinOps cost produces a non-dominated 3D Pareto frontier with strictly higher Hypervolume ($	ext{HV}$) and lower Dominated Solution Ratio ($	ext{DSR}=0\%$) than any single-objective or static baseline.
2. **Experimental Design & Control Variables:** Evaluate 4 canonical policies (`P1 Speed-First (1,0,0)`, `P2 Energy-First (0.1,0.8,0.1)`, `P3 Cost-First (0.2,0.1,0.7)`, `P4 Balanced (0.5,0.3,0.2)`) and a 15-point simplex lattice $(lpha,eta,\gamma)\in\Delta^2$ across `SF=1 GB` and `SF=100 GB` using identical out-of-fold cost, energy, and byte-scan predictions.
3. **Dataset & SQL Query Inventory:** $N=75$ Core queries at `SF=1 GB` and `SF=100 GB`, plus 1-hour continuous mixed-stream arrivals ($\lambda=21{,}600	ext{ q/hr}$ at `SF=1 GB` and $1{,}000	ext{ q/hr}$ at `SF=100 GB`).
4. **Python Scripts & Role:** `scripts/218_v99_data_table_math_and_inventory_audit.py` (`build_pareto_15weight_simplex_and_stability` and `build_v911_reviewer_v98_complete_synthesis`).
5. **Statistical Tests & Mathematical Formulas:** 3D Normalized Hypervolume $	ext{HV}(\mathcal{P}, \mathbf{r}) = \lambda_3(igcup_{\mathbf{p}\in\mathcal{P}} [p_T, r_T]	imes[p_E, r_E]	imes[p_C, r_C])$, Dominated Solution Ratio $	ext{DSR} = rac{1}{|\mathcal{P}|}\sum_{\mathbf{p}\in\mathcal{P}}\mathbb{I}(\exists \mathbf{p}'\in\mathcal{P}: \mathbf{p}' \prec \mathbf{p})$, and per-objective regrets $\mathcal{R}_T, \mathcal{R}_E, \mathcal{R}_C$.
6. **Result Tables:** Main Text Table 10 (Part A), Figure 6b (`fig_job_12d_ablation_and_pareto_300dpi.pdf`), `results/v99_pareto_15weight_simplex_and_stability.csv`, and `results/v911_reviewer_v98_complete_synthesis_audit.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Pass if $	ext{DSR} = 0.0\%$ across P1–P4 and policy switching yields statistically significant reductions in energy (`P2`: `-41.2%` vs P1, `-74.4%` vs Polars, $\mathcal{R}_E=4.82\%$) and cloud spend (`P3`: `$0.00/1M`, saving `$507.06/1M` at `SF=100 GB`, $\mathcal{R}_C=0.0\%$) with maximum $	ext{HV}=0.9210	ext{--}0.9340$ at `P4`.
8. **Manuscript Integration Locations:** Abstract, Section 1 (Contributions & Table 1), Section 3.3 (Eqs. 7–9), Section 4.4.2 (Items 5–6), and Table 10 Part A.
9. **Acceptance Probability Impact if Negative:** Directly resolves the Reviewer's Critical Issue 3.2 by elevating P1–P4 Pareto results into the Abstract and Main Text alongside P1.

### Experiment E3 — Whole-System Energy Validation (Priority 2)
1. **Research Hypothesis & Scientific Motivation:** $H_3$: The engine energy ranking learned by AST-Route from NVML GPU board sensors and TDP-calibrated CPU package estimates remains concordant ($ho \ge 0.95$) with physical mainboard DC rail discharge telemetry.
2. **Experimental Design & Control Variables:** Compare 4 explicit boundary scopes across $R_{	ext{cal}}=50$ repetitions/tier ($250$ trials) and a 45-second sustained discharge test ($2{,}817$ queries, AC unplugged, display off): (1) `GPU Board Energy — Measured` (NVML), (2) `CPU Package Energy — Estimated` (TDP vs non-linear DVFS $u^{1.28}$), (3) `Host + GPU Energy — Composite Estimate`, and (4) `Mainboard DC Shunt` (ACPI battery rail).
3. **Dataset & SQL Query Inventory:** $N=75$ Core queries across Tiers L1–L5 ($250$ calibration runs + $2{,}817$ sustained stream queries).
4. **Python Scripts & Role:** `scripts/216_v97_reviewer_experiments_E1_to_E7.py` (Module E6) and `scripts/218_v99_data_table_math_and_inventory_audit.py`.
5. **Statistical Tests & Mathematical Formulas:** $	ext{MAE}, 	ext{RMSE}, 	ext{MAPE}$, Bland–Altman bias and 95% Limits of Agreement ($ar{d} \pm 1.96 s_d$), and Spearman rank correlation $ho$.
6. **Result Tables:** Main Text Table 10 (Part B), `results/v97_exp_e6_e7_energy_and_cloud_finops_audit.csv`, and `results/v99_unified_energy_boundary_and_concurrency_efficiency.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Pass if rank concordance $ho \ge 0.95$ (Measured: **$ho = 0.9945, p < 0.0001$**, `0.0%` rank inversions) and terminology strictly distinguishes Measured GPU Board, Estimated CPU Package, Composite Host+GPU, and uncertified AC Wall-Socket energy.
8. **Manuscript Integration Locations:** Section 3.2 (Item 4), Section 4.4.2 (Item 6), Table 10 Part B, and Section 5.2 (Construct Validity).
9. **Acceptance Probability Impact if Negative:** Prevents any reviewer misinterpretation of CPU TDP estimates as whole-system wall-plug measurements.

### Experiment E4 — Large-Scale TPC-DS Validation (Priority 2)
1. **Research Hypothesis & Scientific Motivation:** $H_4$: The frozen TPC-H-trained router (`f0e5f17c832472e3`) generalizes zero-shot to multi-table TPC-DS snowflake schemas across both full 99-query dialect coverage (`SF=0.02`) and multi-million-row scale (`SF=1 GB` and `SF=10 GB`) with $100\%$ query completion and lower mean/tail regret than static baselines.
2. **Experimental Design & Control Variables:** Evaluate the locked 9D and 12D models with zero hyperparameter tuning across three TPC-DS scale regimes: (i) Unmodified Official 99-Query TPC-DS (`Q1–Q99`, `SF=0.02`, `605,865` rows, 24 tables), (ii) TPC-DS Snowflake at `SF=1 GB` (`55` queries = `30` Derived + `25` Complex, `2.83M` rows, 11 tables), and (iii) TPC-DS Snowflake at `SF=10 GB` (`30` representative join/agg/window/CTE queries, `28.3M` rows, 4-chunk GPU PCIe streaming).
3. **Dataset & SQL Query Inventory:** `154` total TPC-DS family queries (`99` Official `Q1–Q99` + `30` `TPC-DS-Derived` + `25` `TPC-DS-Complex`).
4. **Python Scripts & Role:** `scripts/216_v97_reviewer_experiments_E1_to_E7.py` (Module E2), `scripts/217_v98_reviewer_action_plan_A_to_G.py` (Part B), and `scripts/218_v99_data_table_math_and_inventory_audit.py`.
5. **Statistical Tests & Mathematical Formulas:** Exact Top-1 accuracy, $\le 5	ext{ ms}$ noise-tolerant Top-1 sensitivity, Mean and P95 Relative/Absolute Regret, and Holm–Bonferroni paired Wilcoxon tests.
6. **Result Tables:** Main Text Table 13, `results/v98_partB_official_tpcds_99queries_blinded_holdout.csv`, and `results/v911_reviewer_v98_complete_synthesis_audit.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Pass if routed completion $= 100\%$ and mean regret $< 10\%$ across scales (Measured: **`100%` completion across all three scales; `3.23%` regret at `SF=0.02`, `4.82%` [`1.61%` on 30Q] at `SF=1 GB`, and `3.45%` [`86.67%` 9D / `90.00%` 12D Exact Top-1, `2.34x` speedup] at `SF=10 GB`**).
8. **Manuscript Integration Locations:** Section 1 (Table 1), Section 3.1 (Master Workload Accounting), Section 4.6.3 (Items 1 & 3), Table 13, and Section 5.2.
9. **Acceptance Probability Impact if Negative:** Bridges the gap between full 99-query dialect generality (`SF=0.02`) and multi-gigabyte snowflake scaling (`SF=1–10 GB`).

### Experiment E5 — Conformal Guard Robustness (Priority 2)
1. **Research Hypothesis & Scientific Motivation:** $H_5$: Tail-regret compression is not an artifact of a single hand-picked threshold; sweeping $	au_{	ext{guard}} \in \{8\%, 10\%, 12\%, 15\%, 20\%\}$ exhibits a stable convex performance plateau, with inner-fold Nested-CV (`8%`) serving as the Primary Inferential Guard passing $\pm 5.0	ext{ ms}$ TOST and Fixed `12%` serving as the Heuristic Sensitivity Reference.
2. **Experimental Design & Control Variables:** Evaluate $	au_{	ext{guard}} \in \{0\%, 4\%, 8\%, 10\%, 12\%, 15\%, 20\%, 30\%\}$ across $K=25$ seed clusters ($N=75$) and 3 distribution-shift regimes (Mix Shift, Join/String Skew, and LOTO-CV Unseen Operator Shift).
3. **Dataset & SQL Query Inventory:** $N=75$ Core queries + $N_{	ext{ext}}=166$ cross-schema queries.
4. **Python Scripts & Role:** `scripts/215_v9_jbd_reviewer_complete_overhaul.py`, `scripts/216_v97_reviewer_experiments_E1_to_E7.py` (Module E5), and `scripts/218_v99_data_table_math_and_inventory_audit.py`.
5. **Statistical Tests & Mathematical Formulas:** Two-sided TOST equivalence across $\Delta_{	ext{eq}}\in\{\pm 2,\pm 3,\pm 5,\pm 6,\pm 8\}	ext{ ms}$, Nadeau–Bengio $t_{	ext{NB}}$, Wilson 95% coverage CIs, Expected Calibration Error ($	ext{ECE}$), $	ext{CVaR}_{10\%}$, and $	ext{AURC}$.
6. **Result Tables:** Main Text Table 9 (Part C & Part D), `results/v9_reviewer_tau_guard_sensitivity_sweep.csv`, `results/v97_exp_e5_tost_margin_sensitivity_and_nadeau_bengio.csv`, and `results/v99_conformal_shift_ood_far_frr_audit.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Explicitly separate Primary Nested-CV `8%` ($p_{	ext{TOST},\pm 5	ext{ms}} = 0.0442 < 0.05$) from Sensitivity `12%` ($p_{	ext{TOST},\pm 5	ext{ms}} = 0.0688 > 0.05$, passing $\pm 6	ext{ ms}$ at $p=0.0241$), report `10%` (`55.08 ms`, `115.20%` P95 regret), and verify Mondrian coverage $\ge 89\%$ across all 5 tiers ($	ext{ECE}=0.67\%$).
8. **Manuscript Integration Locations:** Abstract, Section 3.3, Section 3.4 (Item 1), Section 4.4.2 (Items 3–4), Table 9 Parts C–D, and Section 5.2 (Item 4).
9. **Acceptance Probability Impact if Negative:** Eliminates any statistical criticism regarding threshold selection or overclaiming $\pm 5	ext{ ms}$ TOST equivalence on the `12%` guard.

### Experiment E6 — Independent Reproducibility Audit (Priority 2)
1. **Research Hypothesis & Scientific Motivation:** $H_6$: Every main-text and supplementary table, statistical test, and multiset correctness check ($	ext{EX}=1.000$) can be regenerated from raw SHA-256-verified CSV logs in a single command (`<15 s` CPU smoke test) with `0.000` numeric discrepancy across independent machines.
2. **Experimental Design & Control Variables:** Execute `python reproducibility/verify_and_regenerate_main_tables_R1_R6.py --smoke-test-cpu` and `scripts/215_v9_jbd_reviewer_complete_overhaul.py` Gates `G1–G15` across version-pinned environments (`Dockerfile` and `requirements-lock.txt`).
3. **Dataset & SQL Query Inventory:** All `400` evaluated SQL queries and `142` SHA-256-indexed telemetry and model artifacts.
4. **Python Scripts & Role:** `reproducibility/verify_and_regenerate_main_tables_R1_R6.py` and `scripts/215_v9_jbd_reviewer_complete_overhaul.py`.
5. **Statistical Tests & Mathematical Formulas:** SHA-256 cryptographic hash equality, IEEE-754 `FP64` relative tolerance $\le 10^{-9}$, and maximum absolute table regeneration error $\max |T_{	ext{regen}} - T_{	ext{manuscript}}| = 0.000$.
6. **Result Tables:** Main Text Table 10 (Part D), Supplementary Section S7 & Table S18, `results/global_sha256_registry.csv`, and `results/v9_automated_consistency_gate_report.csv`.
7. **Pass Criteria & Conclusion Adjustment Rules:** Pass if `100%` of SHA-256 hashes match, `Max_Abs_Diff = 0.000`, and all Gates `G1–G15` report `PASS`.
8. **Manuscript Integration Locations:** Section 3.4, Section 4.4.2 (Item 7), Section 7 (`Availability of data and materials`), and Supplementary Section S7.
9. **Acceptance Probability Impact if Negative:** Provides verifiable, audit-grade reproducibility directly inspectable by reviewers.
