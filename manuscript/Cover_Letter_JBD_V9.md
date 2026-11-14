# Cover Letter — Journal of Big Data (SpringerOpen) Submission

**Date:** November 14, 2026  
**To:** The Editors-in-Chief, *Journal of Big Data* (SpringerOpen)  
**Manuscript Title:** *AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines*  
**Corresponding Author:** Preecha Noiumkar (Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand; `preecha.n@msu.ac.th`)  
**Article Type:** Research Article (Systems Architecture & Empirical Benchmark Evaluation)

---

Dear Editors-in-Chief,

Please find enclosed our manuscript, **"AST-Route: An AST-Aware Multi-Objective Query Routing and Cost-Prediction Framework Across Serverless Cloud, Vectorized CPU (C++/Rust), and GPU-Accelerated OLAP Engines,"** which we submit for consideration as a Research Article in the *Journal of Big Data*.

### 1. Core Contributions and Alignment with *Journal of Big Data*
Modern analytical data infrastructures span in-process C++ vectorized DBMSs (DuckDB), Rust Apache Arrow engines (Polars, Apache DataFusion), discrete GPU accelerators, and serverless cloud warehouses (Google BigQuery). Committing a heterogeneous SQL workload to any single static engine incurs severe performance cliffs (PCIe transfer stalls on small scans, host DRAM spilling on large joins, or cloud scheduling overhead), whereas naive point-estimate cost minimization ($\arg\min_e \hat{T}_e$) triggers severe tail-latency regret near hardware crossover boundaries.

Our study introduces **AST-Route**, a pre-execution SQL query routing framework evaluated across **213 benchmark SQL queries from seven suites** (97 core TPC-H queries [75 factorial + 22 official Q1–Q22] and 116 external cross-schema queries across 5 benchmarks: 33 IMDB JOB, 13 SSB, 20 ClickBench, 30 TPC-DS-derived snowflake, and 20 BIRD-inspired synthetic financial queries, plus 15 prospective Seed `S6` holdout queries = **228 total evaluated queries**) across $\text{SF} \in \{1, 10, 30, 100\}\text{ GB}$ and three hardware nodes (`x86_64`+GPU, 22-thread `x86_64`, and Apple M1 `ARM64`).

### 2. Explicit Separation of Designated Endpoints vs. Post-Hoc Exploratory Extensions
To ensure complete methodological transparency, the manuscript explicitly separates primary evaluation endpoints from post-hoc exploratory analyses:
1. **Primary Evaluation Endpoints ($K=25$ Independent Seed Clusters at $\text{SF}=1\text{ GB}$):**
   - **Primary Endpoint E1 (Tail-Risk Compression & Practical Latency Equivalence vs. Always GPU):** AST-Route maintains practical mean-latency equivalence with Always GPU ($54.92\text{ ms}$ vs. $56.96\text{ ms}$, two-sided $p_{\text{seed}} = 0.184$, Two One-Sided Tests [TOST] non-inferiority $p_{\text{non-inf}} = 0.00067$ and nested-CV $8\%$ guard TOST equivalence $p_{\text{TOST}} = 0.0442$ within $\pm 5.0\text{ ms}$) while cutting 95th-percentile relative regret by **$2.25\times$** ($114.11\%$ vs. $257.31\%$) and 95th-percentile absolute regret from **$51.42\text{ ms}$ to $34.51\text{ ms}$** (full-workload paired Wilcoxon $p_{\text{raw,seed}} = 0.0098, p_{\text{Holm,seed}} = 0.0196$ across all $N=25$ seeds [$75$ queries] and $p = 0.0042$ on the $n=24$ divergent seeds), halving $>100\%$ regret failures ($14.67\% \rightarrow 6.67\%$).
   - **Primary Endpoint E2 (Net Latency Reduction vs. Static CPU Baselines):** AST-Route outperforms Always Polars ($82.47\text{ ms}$, $p_{\text{Holm,seed}} = 0.0171$), Always DataFusion ($96.30\text{ ms}$, $p_{\text{Holm,seed}} = 0.0041$), and Always DuckDB ($183.74\text{ ms}$, $p_{\text{Holm,seed}} = 5.45\times 10^{-4}$).
2. **Backend Scope Transparency, CSCI Attribution, and Minimal Engine Subsets:**
   - We state upfront in the Abstract, Introduction (Table 1), Section 2.3, and Section 5.1 that our GPU backend is an *author-developed PyTorch CUDA relational tensor prototype* on an NVIDIA RTX 2050 ($4\text{ GB}$ VRAM) rather than a standalone SQL DBMS. At $\text{SF}=100\text{ GB}$, we explicitly reconcile the unfiltered full-table capacity ceiling ($14{,}280.0\text{ MB}$, $K_{\max}=40 \times 357.0\text{ MB}$) with the post-pushdown PCIe DMA transfer mean ($3{,}518.98\text{ MB}$ at $5.69\text{ GB/s} = 618.45\text{ ms}$ PCIe $+ 581.15\text{ ms}$ kernel/staging $= 1{,}199.60\text{ ms}$).
   - Pure native-SQL routing excluding the GPU prototype ($M_{\text{gpu\_max}}=0$) captures **$48.5\%$** of the total latency gain ($69.12\text{ ms}$, $74.67\%$ Top-1, $p_{\text{Holm}} = 0.0084$), and a minimal 2-engine subset (`{DuckDB, GPU}` or `{Polars, GPU}`) captures **$97.3\%\text{--}98.7\%$** of the realized 6-engine routed gain (**$87.3\%\text{--}88.7\%$** of the theoretical 6-engine oracle gain; `{DuckDB, Polars, GPU}` captures **$99.9\%$** routed / **$98.9\%$** oracle).
3. **Locked Zero-Shot vs. Post-Hoc Exploratory 12D Sketch Transfer Across 5 External Suites ($N_{\text{ext}}=116$):**
   - **Level 1 (Locked 9D Zero-Shot, 0 External Labels):** Trained strictly on the 75 TPC-H queries, AST-Route achieves $92.31\%$ Top-1 on SSB ($N=13$), $80.00\%$ on ClickBench ($N=20$), $93.33\%$ on TPC-DS-derived snowflake ($N=30$), $90.00\%$ on BIRD-inspired synthetic financial ($N=20$), and safely abstains to Polars on string-skewed IMDB JOB ($N=33$, $6.75\text{ ms}$, $21.21\%$ Top-1).
   - **Level 2 (Post-Hoc Exploratory 12D Catalog Sketch, 0 External Labels):** Designed after observing 9D min/max zone-map blindness to unanchored `LIKE '%...%'` filters on JOB, augmenting with a $10.18\text{ KB}$ static trigram/MCV sketch lifts JOB Top-1 accuracy to $69.70\%$ ($4.54\text{ ms}$) and pooled 5-suite Top-1 accuracy from $69.83\%$ ($81/116$) to $85.34\%$ ($99/116$).
   - **Level 3 (Nested 5-Fold Supervised In-Domain Calibration):** Reported strictly as an upper-bound reference when in-domain runtime labels are available.

### 3. Manuscript Organization (Main Text + Additional File 1 Supplementary Appendix)
To ensure high readability and focus in the Main Text (**32 pages**, Tables 1–14, Algorithm 1, and Figures 1–6), all granular per-tier tables, TDP/DVFS energy calibration and Bland–Altman analyses, $C=1\text{--}32$ concurrency manifests, 3-node cross-ISA tables, and verification registries are provided in **Additional File 1 (`Additional_File_1_Supplementary_Appendix.pdf`, Tables S1–S17 and Figures S1–S8)**. All code, SQL suites, and SHA-256-indexed telemetry logs are publicly available at `https://github.com/preecha-msu/bigquery-olap-benchmark` (with a reserved archival record on Zenodo at `https://doi.org/10.5281/zenodo.15024891`).

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests.

Sincerely,

**Preecha Noiumkar**  
Department of Information Technology, Faculty of Informatics  
Mahasarakham University, Maha Sarakham 44150, Thailand  
Email: `preecha.n@msu.ac.th`
