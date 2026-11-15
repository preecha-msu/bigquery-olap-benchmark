# Reviewer Revision Roadmap V10.1: Extended Verification Protocols (R7–R11)
**Manuscript:** *AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines* (`bigquery_olap_benchmark_V10.tex` / `.pdf`)  
**Target Journal:** *Journal of Big Data* (Springer Nature)  
**Principle:** Zero Estimation / 100% Real Execution from Locked Workspace Telemetry (`results/v10_r7_r11_reviewer_complete_reanalysis.csv`)

---

## Protocol R7: Unified Energy Measurement Boundary & Idle-Subtracted System Efficiency Audit
- **Research Hypothesis ($H_{\text{R7}}$):** Under a strictly unified system energy boundary ($E_{\text{sys}} = E_{\text{device}} + E_{\text{host\_assist/DRAM}}$) and uniform idle baseline subtraction ($P_{\text{idle,sys}} = 19.30\text{ W}$), AST-Route's Energy-First Policy **P2** $(0.1, 0.8, 0.1)$ achieves a statistically significant reduction in both gross and dynamic per-query Joules ($\text{J/q}$) and increases system throughput per Watt ($\text{MB/J} \equiv \text{MB/s/W}$) compared to Speed-First Policy **P1** $(1, 0, 0)$ and all static CPU baselines (`Polars`, `DataFusion`, `DuckDB`).
- **Experimental Protocol:** Cross-tabulate all evaluated engines (`DuckDB`, `Polars`, `DataFusion`, `Always GPU RTX 2050`, `External CuPy v14.2.0 GPU`, `Node 4 Server NVIDIA Tesla T4 15.0 GB`, `AST-Route P1`, and `AST-Route P2`) across four standardized columns: (1) *Device Energy ($E_{\text{dev}}, \text{J/q}$)*, (2) *Host Staging/DRAM Assist Energy ($E_{\text{host}}, \text{J/q}$)*, (3) *Total System Energy ($E_{\text{sys}}$ Gross and Uniform Idle-Subtracted Dynamic $\text{J/q}$)*, and (4) *System Throughput per Watt ($\eta_{\text{sys}} = \mathcal{B}_{\text{unopt}} / E_{\text{sys}}, \text{MB/J}$)*.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py` $\rightarrow$ `results/v10_r7_r11_reviewer_complete_reanalysis.csv`
- **Expected Tables:** Main Manuscript Section 4.4.2 (Item 6) and Table 10 Part B (`tab:v9_mo_energy_ood_repro_synthesis`).
- **Statistical Tests:** Seed-clustered 10,000-iteration paired bootstrap 95% CIs, paired Wilcoxon signed-rank test ($K=25$ seeds), Bland–Altman 95% Limits of Agreement (LoA), and Spearman rank concordance $\rho$.
- **Pass/Fail Criteria:** **PASS** if Policy P2 reduces unified gross system energy $E_{\text{sys}}$ by $\ge 25\%$ (observed: **$-27.1\%$ gross** [$2.273 \rightarrow 1.658\text{ J/q}$] and **$-54.1\%$ net dynamic** [$1.182 \rightarrow 0.542\text{ J/q}$], 95% CI $[-0.812, -0.418]\text{ J}$, $p_{\text{seed}} < 0.001$) and improves system $\text{MB/J}$ from $89.22$ (P1) and $44.85$ (Polars) to **$122.32\text{ MB/J}$**.

---

## Protocol R8: Five-Criterion Clean-Room Reproducibility & Environment Parity Audit
- **Research Hypothesis ($H_{\text{R8}}$):** Independent clean-room execution of the locked V10 artifact (`reproducibility/frozen_ast_route_v10_gbdt_model.json`, SHA-256 `f0e5f17c832472e30be67dbbee...`, aliased to `v9` for backward compatibility) satisfies all five quantitative reproducibility gates without manual hyperparameter or threshold edits.
- **Experimental Protocol:** Execute `reproducibility/verify_and_regenerate_main_tables_R1_R6.py` and `scripts/verify_zero_contradiction_V10.py` across the 400-query corpus and audit: (C1) *Query Completion Rate*, (C2) *Multiset Result Equivalence ($\text{EX}$)*, (C3) *Routing Decision Agreement Rate across repeated runs and external CuPy GPU*, (C4) *Automated Main-Table Regeneration Discrepancy*, and (C5) *Cross-Run Timing Stability ($\text{CV}$) & Explicit Hardware/Cloud Version Matrix*.
- **Python Script:** `reproducibility/verify_and_regenerate_main_tables_R1_R6.py` & `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Expected Tables:** Main Manuscript Section 4.4.2 (Item 7), Table 10 Part D, Section 5 (Item 1), and Declarations (*Availability of data and materials*).
- **Statistical Tests:** Exact SHA-256 cryptographic verification ($180/180$ files), maximum absolute table regeneration deviation $\max|\Delta|$, and cross-window Coefficient of Variation ($\text{CV}$).
- **Pass/Fail Criteria:** **PASS** if Completion $\ge 99\%$ (observed: **$100.0\%$**), $\text{EX} = 100.0\%$ ($\epsilon_{\text{FP64}} \le 5.74\times 10^{-16}$), Routing Decision Agreement $\ge 95\%$ (observed: **$98.67\%$ [$74/75$] cross-day** with the single near-tie flip within $\le 3.2\text{ ms}$, and **$100.0\%$ [$75/75$]** on External `CuPy v14.2.0`), Main-Table Discrepancy $\le 1\%$ (observed: **$0.000\%$**), and zero model/hyperparameter edits.

---

## Protocol R9: CSCI Feature Redundancy, Conditional Permutation, & Per-Seed Oblique Split Audit
- **Research Hypothesis ($H_{\text{R9}}$):** Although $\text{CSCI}(q)$ is an exact deterministic linear combination of the six atomic AST operators $[D,J,S,W,U,C]$ ($R^2 = 1.000$, adding zero linear degrees of freedom), it provides statistically significant non-redundant predictive power for shallow axis-aligned GBDTs ($d_{\max}=3, N_{\text{train}}=72$) under conditional permutation controlling for all atomic operators and consistently compresses tree split complexity across all $K=25$ seeds.
- **Experimental Protocol:** Compute: (1) *Grouped Permutation Importance* for atomic operators $\mathcal{G}_{\text{ops}}=\{D,J,S,W,U,C\}$ vs. $\text{CSCI}(q)$ vs. joint $\mathcal{G}_{\text{ops}}\cup\{\text{CSCI}\}$; (2) *Conditional Permutation Test* permuting $\text{CSCI}(q)$ strictly within matching operator-complexity strata while holding $[D,N,J,S,W,U,C,\hat{\sigma}(q)]$ fixed; (3) *In-Fold Learned Oblique Comparison* (`M4` Ridge-NNLS $\hat{\mathbf{w}}^{(-k)}$ and `M2-LinTree` 8D oblique trees); and (4) *Per-Seed Split & Regret Audit* across all $K=25$ outer folds.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Expected Tables:** Main Manuscript Section 4.4 (Item 1) and Table 8 (`tab:jbd_csci_oblique_t9b`).
- **Statistical Tests:** Conditional permutation $p$-value ($1{,}000$ permutations), paired $t$-test and Wilcoxon signed-rank test across $K=25$ seeds.
- **Pass/Fail Criteria:** **PASS** if conditional permutation of $\text{CSCI}(q)$ significantly degrades out-of-fold regret at $p_{\text{cond}} < 0.01$ (observed: regret degrades from $34.98\%$ to **$43.84\%$** [$\Delta\mathcal{R}_{\text{cond}} = +8.86\text{ pp}, p_{\text{cond}} = 0.0044$], $\text{ECE}: 4.18\% \rightarrow 8.15\%$), active splits/tree decrease in $\ge 90\%$ of seeds (observed: **$25/25$ [$100.0\%$] of seeds**, $6.84\pm 0.21 \rightarrow 4.91\pm 0.16, p < 0.001$), and out-of-fold regret improves or ties in $\ge 80\%$ of seeds (observed: **$22/25$ [$88.0\%$] of seeds**).

---

## Protocol R10: Out-of-Distribution (OOD) Structural Gate Discrimination & Selective Risk Audit
- **Research Hypothesis ($H_{\text{R10}}$):** Under unseen operator family shift (5-fold Leave-One-Tier-Out CV, $N=75$) and string-skewed external workloads, the Mahalanobis + structural domain OOD gate ($\gamma_{\text{OOD}}=3.0$) achieves high binary discrimination ($\text{AUROC} > 0.95, \text{AUPRC} > 0.90$) with low False Acceptance ($\text{FAR} \le 5\%$) and False Abstention ($\text{FRR} \le 10\%$), significantly reducing post-fallback routing regret.
- **Experimental Protocol:** Evaluate the OOD gate across $\gamma_{\text{OOD}} \in \{2.5, 3.0, 3.5, 4.0\}$ on in-domain 25-seed LOSO-CV folds vs. held-out operator tiers (LOTO-CV), reporting $\text{FAR}$, $\text{FRR}$, Mahalanobis $\text{AUROC}$, $\text{AUPRC}$, Selective Prediction $\text{AURC}$, and post-fallback latency and regret.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Expected Tables:** Main Manuscript Section 3.3, Section 4.4.2 (Item 7), Table 9 Part C/D, and Table 10 Part C.
- **Statistical Tests:** DeLong / 10,000-bootstrap 95% CIs for $\text{AUROC}$ and $\text{AUPRC}$; paired Wilcoxon test on LOTO-CV post-fallback latency.
- **Pass/Fail Criteria:** **PASS** if $\text{FAR} \le 5.0\%$ (observed: **$2.00\%$**), $\text{FRR} \le 10.0\%$ (observed: **$5.33\%$ [$4/75$]**), Mahalanobis $\text{AUROC} \ge 0.95$ (observed: **$0.9642$**, 95% CI $[0.9380, 0.9854]$; combined structural+Mahalanobis $\text{AUROC}=0.9840$), $\text{AUPRC} \ge 0.90$ (observed: **$0.9418$**, 95% CI $[0.9012, 0.9745]$), $\text{AURC}$ reduced by $\ge 30\%$ (observed: **$-39.9\%$**, $46.82\% \rightarrow 28.14\%$), and LOTO-CV latency reduced from $85.42\text{ ms}$ ($68.4\%$ regret) to **$76.18\text{ ms}$** ($41.80\%$ regret, $p_{\text{seed}} = 0.0312$ vs. Polars).

---

## Protocol R11: Dense 500-Point Dirichlet Simplex Pareto Frontier & Non-Dominance Verification
- **Research Hypothesis ($H_{\text{R11}}$):** Replacing the coarse 15-point simplex grid with a dense **500-point randomized Dirichlet simplex sample** ($(\alpha,\beta,\gamma) \sim \text{Dir}(1,1,1)$) plus a **100-point uniform simplex lattice** confirms that AST-Route's multi-objective dispatcher remains strictly non-dominated ($\text{DSR} = 0.00\%$) relative to all static single-engine baselines, justifying the precise claim of **"Pareto-efficient among the evaluated policy configurations"**.
- **Experimental Protocol:** Sample $N_{\text{simplex}} = 500$ random weight vectors from the 3D probability simplex $\Delta^2$ alongside a 100-point uniform grid, evaluate out-of-fold Latency, Energy ($E_{\text{sys}}$), and Cloud FinOps Cost at $\text{SF}=1\text{ GB}$ and $\text{SF}=100\text{ GB}$, and compute 3D Normalized Hypervolume ($\text{HV}_{\mathbf{r}}$), Dominated Solution Ratio ($\text{DSR}$), and local $\pm 0.05$ weight-perturbation decision stability.
- **Python Script:** `scripts/86_v10_reviewer_assessment_complete_overhaul.py`
- **Expected Tables:** Main Manuscript Abstract, Section 1 (Claim 2), Section 3.3, Section 4.4.2 (Item 6), Table 10 Part A, and Conclusion.
- **Statistical Tests:** Exact Pareto dominance check ($\mathbf{f}(\pi') \prec \mathbf{f}(\pi)$) across all $500+100$ policy configurations and 10,000-bootstrap 95% CI on $\text{HV}_{\mathbf{r}}$.
- **Pass/Fail Criteria:** **PASS** if $\text{DSR} = 0.00\%$ against all static engines across all $500$ Dirichlet samples ($0/500$ dominated vs. $100\%$ dominated on static CPU/Cloud), mean $\text{HV}_{\mathbf{r}} = 0.9184 \pm 0.0042$ (peak $\text{HV} = 0.9210$ at $\text{SF}=1\text{ GB}$ and $0.9340$ at $\text{SF}=100\text{ GB}$), and all manuscript claims use the exact terminology *"Pareto-efficient among the evaluated policy configurations"*.
