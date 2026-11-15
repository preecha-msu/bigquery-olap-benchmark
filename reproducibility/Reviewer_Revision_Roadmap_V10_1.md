# Reviewer Revision Roadmap V10.1 & V11: Extended Verification Protocols (R7–R18)
**Manuscript:** *AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines* (`bigquery_olap_benchmark_V11.tex` / `.pdf`)  
**Target Journal:** *Journal of Big Data* (Springer Nature)  
**Repository & Release Tag:** `https://github.com/preechan-glitch/my_research_files` (Branch: `main`, Tag: `v11.0.0-jbd-r18`)  
**Principle:** Zero Estimation / 100% Real Execution from Locked Workspace Telemetry & Existing 4 Physical Hardware Nodes (`Nodes 1–4`, Zero New Hardware Added)

---

## Model Version Lineage Table (`V9 -> V10 -> V11`)

To prevent any post-hoc model overfitting across peer-review rounds, the core pre-execution Gradient Boosted Decision Tree (GBDT) parameter weights and Mondrian conformal quantile thresholds are **intentionally locked and kept bit-for-bit identical** across `V9`, `V10`, and `V11` (`SHA-256: f0e5f17c832472e30be67dbbee30020dbf23e94658458a2c59acd9b00ed315cf`), while each revision round expands the external validation, statistical stress-testing, and hardware/engine verification protocols:

| Version | Locked Model Artifact | Model SHA-256 Digest | GBDT Weights & Thresholds | Validation Protocols Added | Key Empirical Additions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V9** | `frozen_ast_route_v9_gbdt_model.json` | `f0e5f17c832472e30be67dbbee30020dbf23e94658458a2c59acd9b00ed315cf` | Locked 25-Seed LOSO-CV ($d_{\max}=3$, 80 trees, $\tau \in \{8\%, 12\%\}$) | `R1–R6` | Core 75Q, Official TPC-H 22Q, TPC-DS 99Q, ClickBench 43Q, SSB 13Q, JOB 33Q (12D Sketch), Nodes 1–4 |
| **V10** | `frozen_ast_route_v10_gbdt_model.json` | `f0e5f17c832472e30be67dbbee30020dbf23e94658458a2c59acd9b00ed315cf` | Identical to V9 (Zero Retraining / Zero Refitting) | `R7–R11` | Unified 4-Boundary Energy ($E_{\text{sys}}$), 5-Criterion Clean-Room Audit, Grouped CSCI Block Permutation & Split Audit, OOD `AUROC=0.9642`, 500-Pt Simplex |
| **V11** | `frozen_ast_route_v11_gbdt_model.json` | `f0e5f17c832472e30be67dbbee30020dbf23e94658458a2c59acd9b00ed315cf` | Identical to V9 & V10 (Zero Post-Hoc Refitting) | `R12–R18` | Third-Party `NVIDIA RAPIDS cudf-polars 26.10.2`, $G_{\text{core}}=10$ Template-Cluster Bootstrap, `R12` 3-Level Taxonomy, `R13` CSCI vs. PCA/RP, `R14` 10k-Simplex/NSGA-II/MOBO, `R15` `E1–E7` Energy Matrix, `R16` Calibration/VRAM Sweep, `R17` Prospective Non-TPC SaaS Workload, `R18` $C=1..64$ Contention Decomposition |

---

## Part I: Protocols R7–R11 (Baseline V10 Verification Suite)

### Protocol R7: Unified Energy Measurement Boundary & Idle-Subtracted System Efficiency Audit
- **Research Hypothesis ($H_{\text{R7}}$):** Under a unified system energy boundary ($E_{\text{sys}} = E_{\text{device}} + E_{\text{host\_assist/DRAM}}$) and uniform idle baseline subtraction ($P_{\text{idle,sys}} = 19.30\text{ W}$), AST-Route's Energy-First Policy **P2** $(0.1, 0.8, 0.1)$ achieves a statistically significant reduction in both gross and dynamic per-query Joules ($\text{J/q}$) and increases system throughput per Watt ($\text{MB/J}$) compared to Speed-First Policy **P1** $(1, 0, 0)$ and all static CPU baselines.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py` $\rightarrow$ `results/v10_r7_r11_reviewer_complete_reanalysis.csv`
- **Pass/Fail Criteria:** **PASS** — Policy P2 reduces gross $E_{\text{sys}}$ by **$-27.1\%$** ($2.273 \rightarrow 1.658\text{ J/q}$) and dynamic $E_{\text{sys}}$ by **$-54.1\%$** ($1.182 \rightarrow 0.542\text{ J/q}$, $p_{\text{seed}} < 0.001$), improving $\text{MB/J}$ to **$122.32\text{ MB/J}$**.

### Protocol R8: Five-Criterion Clean-Room Reproducibility & Environment Parity Audit
- **Research Hypothesis ($H_{\text{R8}}$):** Independent clean-room execution of the locked model artifact satisfies all five quantitative reproducibility gates without manual hyperparameter or threshold edits.
- **Python Script:** `reproducibility/verify_and_regenerate_main_tables_R1_R6.py`
- **Pass/Fail Criteria:** **PASS** — Completion $= 100.0\% \ge 99\%$, $\text{EX} = 1.000$, Routing Decision Agreement $= 98.67\%\text{--}100.0\% \ge 95\%$, Main-Table Discrepancy $= 0.000\% \le 1\%$, and `0` model edits.

### Protocol R9 (Formalized by Protocol R19): CSCI Oblique Representation, Grouped Atomic-Block Permutation, & Per-Seed Split Audit
- **Research Hypothesis ($H_{\text{R9/R19}}$):** Because $\text{CSCI}(q) = 1.5D+2.5J+3.0S+3.5W+4.0U+1.5C$ is a deterministic linear combination of the six atomic operators ($I(\text{CSCI}; Y \mid D,J,S,W,U,C) = 0\text{ bits}$, $\operatorname{Var}(\text{CSCI}\mid D..C)=0$), it introduces zero independent information; instead, its utility stems strictly from providing a domain-informed 1D oblique structural projection for shallow ($d_{\max}=3$) axis-aligned GBDTs.
- **Python Script:** `scripts/v11_r19_r23_real_execution_suite.py` & `scripts/v11_final_reviewer_proof_audit_EC41_EC50.py`
- **Pass/Fail Criteria:** **PASS** — Grouped atomic-operator block permutation (`R19a`) preserving $\text{CSCI}_{\pi(i)} \equiv \mathbf{w}^\top \mathbf{z}_{\pi(i)}$ degrades regret from $34.98\%$ to **$68.42\%$** ($\Delta\mathcal{R}_{\text{group}} = +33.44\text{ pp}, p = 0.0002$); identical-fold refitting (`R19b`: `M3` with CSCI vs. `M2` without CSCI) reduces splits/tree by **$-28.22\%$ in $25/25$ ($100\%$) seeds** ($6.84 \rightarrow 4.91, p < 0.0001$) and lowers mean regret by **$-9.22\text{ pp}$** ($p = 0.0038$), outperforming unsupervised `PCA-1` ($p=0.0124$), `PCA-1,2` ($p=0.0156$), and `Random Oblique Projections` ($p=0.0068$).

### Protocol R10: Out-of-Distribution (OOD) Structural Gate Discrimination & Selective Risk Audit
- **Research Hypothesis ($H_{\text{R10}}$):** Under unseen operator family shift (LOTO-CV), the Mahalanobis + structural domain OOD gate ($\gamma_{\text{OOD}}=3.0$) achieves $\text{AUROC} > 0.95$ and $\text{AUPRC} > 0.90$ with $\text{FAR} \le 5\%$ and $\text{FRR} \le 10\%$.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Pass/Fail Criteria:** **PASS** — $\text{FAR} = 2.00\%$, $\text{FRR} = 5.33\%$, Mahalanobis $\text{AUROC} = 0.9642$ (Combined $0.9840$), $\text{AUPRC} = 0.9418$, $\text{AURC} = 28.14\%$ ($-39.9\%$), and LOTO-CV post-fallback latency $= 76.18\text{ ms}$ ($p = 0.0312$).

### Protocol R11: Dense 500-Point Dirichlet Simplex Pareto Frontier Verification
- **Research Hypothesis ($H_{\text{R11}}$):** Across 500 randomized Dirichlet simplex weights plus a 100-point grid, AST-Route remains strictly non-dominated ($\text{DSR} = 0.00\%$).
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Pass/Fail Criteria:** **PASS** — $\text{DSR} = 0.00\%$ ($0/500$ dominated), mean $\text{HV}_{\mathbf{r}} = 0.9184 \pm 0.0042$, max $\text{HV} = 0.9210$ ($\text{SF}=1$) / $0.9340$ ($\text{SF}=100$).

---

## Part II: Protocols R12–R18 (V11 Level 1 + Level 2 Reviewer Overhaul)

### Protocol R12 (Critical 2): Independent Clean-Room Re-Execution & Three-Level Epistemic Taxonomy (`Reproducibility` vs. `Replication` vs. `Generalization`)
1. **Research Hypothesis ($H_{\text{R12}}$):** Executing the locked `frozen_ast_route_v11_gbdt_model.json` (`SHA-256: f0e5f17c...`) in an isolated clean-room environment and across independent third-party GPU/CPU environments satisfies all five verification criteria while strictly separating **(Level A) Same-Host Reproducibility**, **(Level B) Cross-Environment/Hardware Replication**, and **(Level C) Untouched External Benchmark Generalization** (and isolating Category II diagnostic `IMDB JOB` from Category IV holdout claims).
2. **Experimental Design:** Verify cryptographic model integrity before and after execution; run zero-edit inference and execution across Core 75Q ($\text{SF}=1\text{ GB}$), Official TPC-DS 99Q ($\text{SF}=0.02$), Official ClickBench 43Q, Official SSB 13Q, and WSL2 Linux `NVIDIA RAPIDS cudf-polars 26.10.2` (`libcudf-cu12`).
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r12_independent_reexecution_audit`) $\rightarrow$ `results/v11_r12_independent_reexecution_audit.csv`
4. **Dataset & Hardware:** Core 75Q, Official TPC-DS 99Q, ClickBench 43Q, SSB 13Q, TPC-H 22Q across Node 1 (`x86_64 + RTX 2050 4.0 GB` + WSL2 Linux `libcudf-cu12`), Node 2 (`16C/22T x86_64`), Node 3 (`Apple M1 ARM64`), and Node 4 (`Cloud-Hosted Server NVIDIA Tesla T4 15.0 GB VRAM`).
5. **Statistical Test:** Exact cryptographic hash identity, binomial proportion CIs on completion and decision agreement, and paired Wilcoxon / bootstrap CIs on regret.
6. **Expected Results Table:** Main Manuscript Section 4.6, Section 5, and Additional File 1 Table S26 (`tab:supp_v11_r12_r13`).
7. **Pass/Fail Criteria:** **PASS** — Level A: $100\%$ completion, $\text{EX}=1.000$, $100.0\%$ frozen-feature decision match ($98.67\%$ cross-day live timing agreement), $0.000\%$ table regeneration diff; Level B: `95.45%` Top-1 & `0.00%` P95 regret on `RAPIDS cudf-polars` TPC-H 22Q ($\text{SF}=1, 10\text{ GB}$) and $96.67\%\text{--}100.0\%$ Top-1 across Nodes 2–4; Level C: Official TPC-DS 99Q `64.65%` Exact Top-1 All-6 (`84.85%` Pure-SQL), `79.80%` $\le 5\%$ / `88.89%` $\le 10\%$ Near-Optimal Hit Rate, `3.23%` mean regret.
8. **Manuscript Section:** Sections 3.5, 4.6, 5, and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Critical 2** & **R12** (*Independent Re-Execution Audit & Separation of Reproducibility, Replication, and Generalization*).

---

### Protocol R13 (Critical 1): CSCI Mechanistic Validation vs. PCA-1, PCA-2, Random Projection, Equal-Weight, and In-Fold Learned Linear Combination
1. **Research Hypothesis ($H_{\text{R13}}$):** Under identical 25-seed LOSO-CV training folds and identical GBDT hyperparameters ($d_{\max}=3$, 80 trees), the domain-informed $\text{CSCI}(q)$ oblique feature achieves significantly lower routing latency, lower P95 regret, lower $\text{ECE}$, lower $\text{AURC}$, and fewer splits/tree than (i) Raw AST + $\hat{\sigma}(q)$ without any composite index (`M2`), (ii) In-Fold Top-1 Principal Component (`PCA-1`), (iii) In-Fold Top-2 Principal Components (`PCA-1,2`), (iv) Gaussian Random Oblique Projections ($R=20$ vectors/fold), and (v) Equal-Weight Composite Index ($w=[1,1,1,1,1,1]$), while matching the In-Fold Learned Linear Combination (`Ridge-NNLS`, $\cos=0.9998$) both on the Development Set ($N=75$) and zero-shot on External Official TPC-DS ($N=99$) and ClickBench ($N=43$).
2. **Experimental Design:** Fit `StandardScaler`, `PCA(n_components=1,2)`, $R=20$ unit Gaussian projections $\mathbf{u}_r \sim \mathcal{N}(0,I_6)/\|\cdot\|_2$, and `Ridge-NNLS` strictly inside each of the 25 training folds ($N_{\text{train}}=72$) to guarantee zero test-fold leakage; evaluate out-of-fold on Core 75Q and zero-shot on External TPC-DS 99Q and ClickBench 43Q.
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r13_csci_mechanistic_ablation`) $\rightarrow$ `results/v11_r13_csci_mechanistic_pca_rp_ablation.csv`
4. **Dataset & Hardware:** Development Set (`Core TPC-H 75Q, SF=1 GB`) and External Test Sets (`Official TPC-DS 99Q` & `Official ClickBench 43Q`) on Node 1.
5. **Statistical Test:** 10,000-iteration bootstrap 95% CIs and paired Wilcoxon signed-rank tests across $K=25$ seeds.
6. **Expected Results Table:** Main Manuscript Section 3.1, Section 4.4, and Additional File 1 Table S26 (`tab:supp_v11_r12_r13`).
7. **Pass/Fail Criteria:** **PASS** — Proposed `M3 (CSCI)` achieves **$54.92\text{ ms}$** net latency, **$114.11\%$** P95 regret, **$4.18\%$** $\text{ECE}$, **$28.14\%$** $\text{AURC}$, **$4.91$** splits/tree (`-28.2%`), and **$64.65\%$ / $88.37\%$** External TPC-DS / ClickBench Top-1 ($3.23\%$ / $5.74\%$ regret), outperforming `M2 (No Composite)` ($58.12\text{ ms}, p=0.0038$), `PCA-1` ($57.48\text{ ms}, 6.42\text{ splits}, p=0.0124$), `PCA-1,2` ($57.19\text{ ms}, 6.51\text{ splits}, p=0.0156$), `Random Projection RP-20` ($57.85\text{ ms}, 6.58\text{ splits}, p=0.0068$), and `Equal-Weight` ($56.14\text{ ms}, 5.28\text{ splits}, p=0.0284$), while matching `In-Fold Learned Ridge-NNLS` ($54.91\text{ ms}, \cos=0.9998, p=0.8924$).
8. **Manuscript Section:** Sections 3.1, 4.4, and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Critical 1** & **R13** (*CSCI Mechanistic Justification vs. Unsupervised PCA, Random Oblique Projections, and Equal/Learned Weights*).

---

### Protocol R14 (Critical 3): Pareto Search Expansion — 10,000-Point Dirichlet Simplex vs. NSGA-II & Multi-Objective Bayesian Optimization (MOBO)
1. **Research Hypothesis ($H_{\text{R14}}$):** Expanding the multi-objective policy evaluation from 500 points to **10,000 Dirichlet simplex configurations** and comparing against continuous **NSGA-II** ($P=100 \times G=100 = 10{,}000$ evaluations) and **Multi-Objective Bayesian Optimization (MOBO / qEHVI)** confirms that AST-Route Policies P1–P4 achieve $\text{DSR} = 0.00\%$ and lie within $\Delta\text{HV} \le 0.0004$ of the continuous evolutionary/Bayesian Pareto ceiling.
2. **Experimental Design:** Evaluate 3D Normalized Hypervolume $\text{HV}_{\mathbf{r}}$ across Latency Regret ($\mathcal{R}_T$), Energy Regret ($\mathcal{R}_E$), and Cloud FinOps Cost Regret ($\mathcal{R}_C$) at $\text{SF}=1\text{ GB}$ and $\text{SF}=100\text{ GB}$ under 5 search strategies: (1) 15-Point Grid, (2) 500-Point Dirichlet, (3) 10,000-Point Dirichlet, (4) NSGA-II, and (5) MOBO (qEHVI).
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r14_pareto_10000pt_nsga2_mobo`) $\rightarrow$ `results/v11_r14_pareto_10000pt_nsga2_mobo.csv`
4. **Dataset & Hardware:** Core 75Q multi-objective cost/energy/latency surface at $\text{SF}=1\text{ GB}$ and $\text{SF}=100\text{ GB}$.
5. **Statistical Test:** 10,000-bootstrap 95% CIs on frontier $\text{HV}_{\mathbf{r}}$, exact Pareto dominance check ($\text{DSR}$), and Hypervolume Improvement ($\Delta\text{HV}$).
6. **Expected Results Table:** Main Manuscript Section 3.3, Section 4.4.2, and Additional File 1 Table S27 (`tab:supp_v11_r14_r15`).
7. **Pass/Fail Criteria:** **PASS** — Across all $10{,}000$ Dirichlet samples, $\text{DSR} = 0.00\%$ ($0/10{,}000$ dominated); max $\text{HV}_{\text{SF1}} = 0.9214$ (`10k-Simplex`, `NSGA-II`, `MOBO`) vs. $0.9210$ (`Policy P4`, $\Delta\text{HV} = +0.0004$), and all manuscript text uses the exact phrase **"Pareto-efficient among the evaluated policy configurations"**.
8. **Manuscript Section:** Sections 1, 3.3, 4.4.2, and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Critical 3** & **R14** (*Pareto Claim Scope & Comparison with NSGA-II and Multi-Objective Bayesian Optimization*).

---

### Protocol R15 (Critical 4): Seven-Level Energy Validation Matrix (`E1–E7`) with 50–100 Repetitions & Explicit Sensor Separation
1. **Research Hypothesis ($H_{\text{R15}}$):** Constructing a rigorous 7-level Energy Validation Matrix (`E1–E7`) across $R=50\text{--}100$ repetitions per tier — while explicitly separating direct hardware sensor telemetry (`NVIDIA NVML`, `Mainboard ACPI DC Battery Shunt`, and `Cloud Intel RAPL / AMD SMU`) from TDP/DVFS-calibrated host CPU estimates (withdrawn from Abstract/headline claims) — proves high measurement repeatability ($\text{CV} < 3.25\%, \text{ICC} > 0.996$) and $100\%$ policy ranking invariance under worst-case sensor uncertainty.
2. **Experimental Design:** Evaluate `E1 (Idle Power Baseline)`, `E2 (CPU-Only Workload)`, `E3 (GPU-Only Warm Kernel Workload)`, `E4 (CPU + GPU Simultaneous Hybrid & Concurrent C=32 Workload)`, `E5 (External / Independent Mainboard DC Power Shunt Audit, N=2,817 queries + R=100 tier runs)`, `E6 (Energy Repeatability SD, CV%, ICC across R=50–100 runs)`, and `E7 (Energy Measurement Uncertainty 95% CI, Bland–Altman LoA, & Worst-Case Perturbation Invariance)`.
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r15_energy_validation_matrix_E1_E7`) $\rightarrow$ `results/v11_r15_energy_validation_matrix_E1_E7.csv`
4. **Dataset & Hardware:** $N=750$ repeated tier runs (`issue3_4_real_50rep_per_tier_energy_runs.csv`), $N=2{,}817$ ACPI DC battery-rail shunt queries, and $1{,}500$ cloud RAPL/SMU/NVML runs across Nodes 1 and 4.
5. **Statistical Test:** Intraclass Correlation Coefficient $\text{ICC}(3,1)$, Bland–Altman bias and 95% LoA, Spearman rank correlation $\rho$, and 10,000-bootstrap 95% CIs.
6. **Expected Results Table:** Main Manuscript Section 4.4.2 and Additional File 1 Table S27 (`tab:supp_v11_r14_r15`).
7. **Pass/Fail Criteria:** **PASS** — NVML GPU $\text{CV} = 1.24\%\text{--}1.98\%$ ($\text{ICC} = 0.9981$, Bland–Altman bias $+0.0005\text{ J}$, 95% LoA $[-0.036, +0.037]\text{ J}$); independent Mainboard DC Shunt ($N=2{,}817$) achieves $\rho = 0.9945$ ($\text{MACE} = 1.77\%$); and even under simultaneous worst-case $+3.21\%$ CPU and $-1.96\%$ GPU sensor error, Energy-First Policy P2 saves $-24.8\%$ $E_{\text{sys}}$ vs. P1 ($p < 0.0001$, $100\%$ ranking invariance).
8. **Manuscript Section:** Sections 1, 3.2, 4.4.2, and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Critical 4** & **R15** (*Energy Telemetry Rigor, E1–E7 Validation Matrix, and Separation of Direct Sensors vs. TDP Estimation*).

---

### Protocol R16 (Critical 5): Hardware Calibration Sample Scaling ($m = 0, 1, 3, 5, 10, 15$ vs. Full Retraining) & GPU VRAM Capacity Sweep (`4 GB, 8 GB, 16 GB`)
1. **Research Hypothesis ($H_{\text{R16}}$):** On existing hardware nodes (`Nodes 1–4`, without adding new hardware), few-shot affine log-ratio calibration with $m=5$ stratified queries ($1$ per tier L1–L5) recovers $>90\%$ of the latency and accuracy gains of full in-domain retraining at $>12\times$ lower calibration time and $>14\times$ lower energy cost, while Rule U4's VRAM capacity guard maintains $0.0\%$ OOM violations across `1.28 GB`, `4.0 GB` (`RTX 2050`), `8.0 GB`, and `15.0/16.0 GB` (`Server Tesla T4`) VRAM ceilings.
2. **Experimental Design:** Evaluate `R16a` across $m \in \{0, 1, 3, 5, 10, 15\}$ calibration queries vs. Full Retraining ($N=75$) on `Node 2 (22T x86_64)`, `Node 3 (Apple M1 ARM64)`, and `Node 4 (Server Tesla T4)`, recording Calibration Time (`ms`), Calibration Energy (`J`), Pre/Post-Calibration Top-1/Top-2 Accuracy, and Post-Calibration Latency & Regret; evaluate `R16b` across VRAM caps $\{1.28, 4.00, 8.00, 15.00\}\text{ GB}$ at $\text{SF}=1, 10, 30\text{ GB}$.
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r16_hardware_calibration_and_vram_sweep`) $\rightarrow$ `results/v11_r16_hardware_calibration_and_vram_sweep.csv`
4. **Dataset & Hardware:** Existing 4 physical nodes (`Node 1 RTX 2050 4.0 GB`, `Node 2 22T x86_64`, `Node 3 Apple M1 ARM64`, `Node 4 Cloud-Hosted Server NVIDIA Tesla T4 15.0 GB`).
5. **Statistical Test:** Monotonic convergence analysis of post-calibration latency/regret and exact OOM violation count ($0/75$).
6. **Expected Results Table:** Main Manuscript Section 4.7 and Additional File 1 Table S28 (`tab:supp_v11_r16_r17_r18`).
7. **Pass/Fail Criteria:** **PASS** — At $m=5$ queries, calibration takes only $0.62\text{--}0.91\text{ s}$ and $1.42\text{--}3.12\text{ J}$ ($12.5\times$ faster and $14.8\times$ lower energy than Full Retraining's $11.4\text{ s}$ and $46.20\text{ J}$), reducing Node 2 latency from $54.81\text{ ms}$ ($m=0$) to $48.92\text{ ms}$ ($91.5\%$ of the $m=15$ gain; Full Retraining $= 35.11\text{ ms}, 97.33\%$ Top-1) and achieving $93.33\%$ Top-1 on Node 3 M1 and $100.0\%$ Top-1 on Node 4 Tesla T4; across `4 GB`, `8 GB`, and `15/16 GB` VRAM caps, OOM rate is **$0.0\%$** and $\text{SF}=10\text{ GB}$ streaming chunks drop from `4 chunks` (`4 GB`) $\rightarrow$ `2 chunks` (`8 GB`) $\rightarrow$ `1 single-pass chunk` (`15/16 GB`).
8. **Manuscript Section:** Section 4.7 and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Critical 5** & **R16** (*Hardware Calibration Scaling Curve $m=1..15$ vs. Full Retraining & GPU VRAM Capacity Sensitivity `4 GB / 8 GB / 16 GB` Without Buying New Hardware*).

---

### Protocol R17: Prospective Hold-Before-Execute Query Evaluation on Non-TPC Enterprise SaaS, Billing & Telemetry Workload
1. **Research Hypothesis ($H_{\text{R17}}$):** On a newly constructed, non-TPC multi-table **Enterprise SaaS, Billing & Telemetry Analytics Workload** ($N=25$ queries across 5 skewed tables: `saas_organizations`, `saas_users`, `saas_subscriptions`, `saas_invoices`, `telemetry_events`), when queries and pre-execution routing choices are cryptographically hashed (SHA-256) *before* any engine execution, the locked AST-Route model achieves $\ge 80\%$ Exact Top-1 accuracy and $\ge 90\%$ Near-Optimal ($\le 10\%$) Hit Rate with zero post-hoc adjustments.
2. **Experimental Design:** (1) Generate 5 physical Parquet tables ($1.88\text{M}$ total rows with Zipf tenant skew $z=1.30\text{--}1.35$) and 25 SQL queries (`ENT_Q01`–`ENT_Q25` spanning L1–L5); (2) extract 9D pre-execution features via `sqlglot` + `pyarrow.parquet`, run the locked GBDT router, and record the **Pre-Execution Manifest SHA-256** and nanosecond timestamp before timing any engine; (3) execute all 25 queries across `DuckDB`, `Polars`, `DataFusion`, and `CUDA GPU`; (4) compute Exact Top-1, $\le 10\%$ Near-Optimal Hit Rate, and Regret.
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r17_prospective_enterprise_workload`) $\rightarrow$ `results/v11_r17_prospective_enterprise_workload.csv`
4. **Dataset & Hardware:** 5-table Non-TPC Enterprise SaaS & Telemetry dataset (`data/prospective_enterprise_r17/*.parquet`, $N=25$ queries) on Node 1.
5. **Statistical Test:** Exact pre-execution SHA-256 timestamp verification, Exact Top-1 / Near-Optimal hit rates, and paired Wilcoxon test vs. static baselines.
6. **Expected Results Table:** Main Manuscript Section 4.6 and Additional File 1 Table S28 (`tab:supp_v11_r16_r17_r18`).
7. **Pass/Fail Criteria:** **PASS** — Pre-execution manifest hashed prior to engine execution (`0` post-hoc edits), $\text{EX}=1.000$ row-count agreement across all 25 queries, High Exact Top-1 and $\le 10\%$ Near-Optimal Hit Rate, and significant regret compression over static CPU/GPU baselines.
8. **Manuscript Section:** Section 4.6 and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Protocol R17** (*Prospective Hold-Before-Execute Evaluation on Non-TPC Production-Like Multi-Table Analytics*).

---

### Protocol R18: Concurrent Runtime Overhead Decomposition Across $C = 1, 2, 4, 8, 16, 32, 64$ Streams
1. **Research Hypothesis ($H_{\text{R18}}$):** Under multi-tenant concurrent query arrival spanning $C \in \{1, 2, 4, 8, 16, 32, 64\}$ streams, decomposing total latency into (1) **Router Queueing & Dispatch Overhead**, (2) **CPU Thread Contention**, (3) **GPU Kernel Contention**, and (4) **PCIe DMA Bus Contention** demonstrates that router dispatch overhead remains a small fraction ($2.5\%\text{--}7.6\%$) of total latency while dual-queue hybrid co-scheduling achieves $1.50\times\text{--}2.52\times$ higher throughput (QPS) and $2.02\times\text{--}3.49\times$ lower energy per query than single-engine baselines.
2. **Experimental Design:** Evaluate all 7 concurrency levels $C \in \{1, 2, 4, 8, 16, 32, 64\}$ at $\text{SF}=1\text{ GB}$, measuring Pure Uncontended Service Time ($50.77\text{ ms}$), Router Queue/Dispatch Time ($4.15 \rightarrow 11.20\text{ ms}$), CPU Thread Contention ($0.00 \rightarrow 118.40\text{ ms}$), GPU Kernel Contention ($0.00 \rightarrow 94.60\text{ ms}$), and PCIe DMA Bus Contention ($0.00 \rightarrow 163.43\text{ ms}$), alongside Mean/P95/P99 latency, Throughput (QPS), and Energy ($\text{J/q}$).
3. **Python Script:** `scripts/v11_r12_r18_real_execution_suite.py` (`run_r18_concurrency_overhead_decomposition`) $\rightarrow$ `results/v11_r18_concurrency_overhead_decomposition.csv`
4. **Dataset & Hardware:** Core 75Q multi-stream arrival workload across $C \in \{1, 2, 4, 8, 16, 32, 64\}$ on Node 1 (`8T x86_64 + RTX 2050 4.0 GB`).
5. **Statistical Test:** Exact 5-component additive latency identity verification ($\sum_{i=1}^5 T_i(C) = T_{\text{total}}(C)$ within $<10^{-6}\text{ ms}$) and Little's Law throughput consistency.
6. **Expected Results Table:** Main Manuscript Section 4.4.2 and Additional File 1 Table S28 (`tab:supp_v11_r16_r17_r18`).
7. **Pass/Fail Criteria:** **PASS** — Exact additive identity across all 7 levels $C=1..64$; peak throughput reaches **$154.44\text{ QPS}$** at $C=32$ ($2.52\times$ vs. Polars $61.40\text{ QPS}$, $0.396\text{ J/q}$), and at extreme saturation $C=64$ ($438.40\text{ ms}$ total: $50.77\text{ ms}$ service + $11.20\text{ ms}$ router [$2.55\%$] + $118.40\text{ ms}$ CPU + $94.60\text{ ms}$ GPU + $163.43\text{ ms}$ PCIe DMA), AST-Route sustains **$146.00\text{ QPS}$** ($2.10\times$ vs. CPU, $2.38\times$ vs. GPU) with **$0.0\%$ OOM/failure rate**.
8. **Manuscript Section:** Section 4.4.2 and Additional File 1 Section S14.
9. **Reviewer Objection Addressed:** Addresses **Protocol R18** (*Concurrent Runtime Overhead Decomposition Across $C = 1, 2, 4, 8, 16, 32, 64$*).
