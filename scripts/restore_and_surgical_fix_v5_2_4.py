#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/restore_and_surgical_fix_v5_2_4.py
================================================================================
1. กู้คืนไฟล์ต้นฉบับเต็ม 30 หน้าจาก bigquery_olap_benchmark_V2_pre_step3_*.tex
2. ซ่อมทุกจุดแบบ Non-Crossing Surgical Regex (ห้ามข้ามขอบเขต \begin{table}):
   - Abstract (<= 350 คำ, 163 คิวรี, สมดุล \begin{quote}...\end{quote})
   - Section 1 (ลบประโยคค้าง This study makes five primary contributions + ใส่ 5 Contributions ใหม่)
   - Table 2B (เหลือตารางเดียวใน Section 3.1 โดยใช้ \renewcommand{\thetable}{2B} + \addtocounter{table}{-1}
     เพื่อไม่ให้เลข Table 1..14 เลื่อน และไม่เกิด ??)
   - Table 3 แถว V3 ($12.52, $125.17, $375.51, $1,251.70)
   - Table 10 & Sec 4.2 BigQuery Server 95% CI ([982.40, 1914.50] ms, 1.90x)
   - Table 12 Part B DataFusion (244.12 ms บน 2C, 168.45 ms บน 4C) & Table 13 Part C ($C=32$ Streams)
   - Section 4.7 บรรทัดแรก (Table 9) + ข้อ 4 (ลบ 54.92+4.15=59.07 ms) + ข้อ 6 & Table 14 ท้าย Sec 4.7
   - Section 5 ข้อ 2 (97 TPC-H + 66 external = 163 queries), Section 6 Conclusion, Availability, Ref [13]/[28]
3. ตรวจชื่อไฟล์จริงใน \includegraphics แล้วบันทึกภาพ Fig 1, Fig 2, Fig 9 ทับให้ตรงชื่อ 100%
4. Hard Assertions: หน้า PDF >= 28 หน้า, รูปครบ >= 11 รูป, ตารางครบ 16 ตาราง, Undefined Refs (??) == 0
================================================================================
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 1. FIND INTACT 30-PAGE BACKUP (> 95,000 CHARS WITH ALL TABLES & FIGURES)
# ==============================================================================
def load_intact_base_tex() -> str:
    candidates = sorted(MS_DIR.glob("bigquery_olap_benchmark_V2_pre_*.tex"), reverse=True)
    for c in candidates:
        txt = c.read_text(encoding="utf-8", errors="ignore")
        n_inc = len(re.findall(r"\\includegraphics", txt))
        if len(txt) > 95000 and n_inc >= 11 and "\\section{The Proposed AST-Route" in txt:
            print(f"  [RESTORE-OK] กู้คืนต้นฉบับเต็มจาก: {c.name} ({len(txt):,} ตัวอักษร, รูปภาพ {n_inc} จุด)")
            return txt
    raise RuntimeError("ไม่พบไฟล์สำรองที่มีความยาว > 95,000 ตัวอักษร กรุณาตรวจสอบโฟลเดอร์ manuscript/")


# ==============================================================================
# 2. REGENERATE FIGURES 1, 2, AND 9 TO EXACT FILENAMES REFERENCED IN .TEX
# ==============================================================================
def draw_card(ax, x, y, w, h, title, lines, header_color="#1f4e78", body_color="#f7f9fc", edge_color="#1f4e78", title_fs=9.0, body_fs=7.8):
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=body_color, edgecolor=edge_color, linewidth=1.3, zorder=2
    )
    ax.add_patch(body)
    hh = min(0.055, h * 0.26)
    hdr = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=header_color, edgecolor=edge_color, linewidth=1.0, zorder=3
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hh / 2, title, color="white", fontsize=title_fs, fontweight="bold", ha="center", va="center", zorder=4, family="serif")
    avail_h = h - hh - 0.016
    step = avail_h / (max(len(lines), 1) + 0.35)
    for i, line in enumerate(lines):
        ly = (y + h - hh - 0.012) - (i + 0.65) * step
        ax.text(x + 0.012, ly, line, color="#1a1a1a", fontsize=body_fs, ha="left", va="center", zorder=4, family="serif")


def regenerate_figures_matching_tex(tex: str):
    inc_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    print(f"  [FIG-SCAN] พบคำสั่ง \\includegraphics ในไฟล์ .tex ทั้งหมด {len(inc_paths)} รูป:")
    for p in inc_paths:
        print(f"             - {p}")

    plt.rcParams["font.family"] = "serif"
    c_w, c_gap = 0.232, 0.014
    c_xs = [0.015 + i * (c_w + c_gap) for i in range(4)]

    # --- Figure 1 ---
    fig1, ax1 = plt.subplots(figsize=(13.5, 7.4), dpi=300)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.text(0.5, 0.975, "Proposed AST-Route Four-Layer Multi-Objective Query Routing Framework",
             fontsize=13, fontweight="bold", ha="center", va="center", color="#0f2c59")
    draw_card(ax1, 0.015, 0.72, 0.47, 0.22, "LAYER 1A: Input SQL & Static AST Parsing (SQLGlot)", [
        "• Workload: 75 Factorial TPC-H + 22 Q1–Q22 + 66 External (N = 163 Queries)",
        "• Zero-Execution Static Parsing via SQLGlot (3.48 ms file parse / 0.17 ms cached)",
        "• Extracts 7 Structural Operators: Depth D(q), Nodes N(q), Joins J(q),",
        "  Subqueries/CTEs S(q), Windows W(q), UNNEST U(q), CASE Branches C(q)",
    ], "#2b579a", "#eef4fa", "#2b579a")
    draw_card(ax1, 0.515, 0.72, 0.47, 0.22, "LAYER 1B: 9D/10D Feature Vector, CSCI & Catalog Scale", [
        "• Validated Index: CSCI(q) = 1.5D + 2.5J + 3.0S + 3.5W + 4.0U + 1.5C",
        "• Operator-Tier Alignment: Partial rho = 0.5477 (p = 3.68e-7); 10,000-MC rho = 0.9880 (±20%)",
        "• Zone-Map Selectivity sigma_hat(q) in (0, 1] (0.172 ms cached / 0.413 ms cold footer)",
        "• 9D Vector x_q in R^9 (augmented with log10(1 + B_scan) to R^10 across scales)",
    ], "#1e6b52", "#eef8f3", "#1e6b52")
    draw_card(ax1, 0.015, 0.49, 0.97, 0.19, "LAYERS 2 & 3: Cross-Paradigm Analytical Cost Models & Multi-Output Cost-Regression Engine (R_GBR-9D)", [
        "• Analytical Models: PCIe Fraction Phi_PCIe(q), Effective Throughput Theta_eff (GB/s), Cloud Slot Proxy M_slot, Billed FinOps Savings Delta C_1M ($)",
        "• Pluggable Predictors: Speed-First Ridge (49.12 ms) / Shallow DT (51.35 ms) / k-NN (52.84 ms) vs. Tail-Safe Conformal Multi-Output GBR (R_GBR-9D)",
        "• Out-of-Fold Predictive Accuracy (25-Seed LOSO-CV): GPU R^2 = 0.875 (P50 Q-Err 1.053) | Polars R^2 = 0.822 | Energy R^2 = 0.815 (Inference: 497.2 us)",
    ], "#b85d18", "#fdf5ed", "#b85d18")
    draw_card(ax1, c_xs[0], 0.21, c_w, 0.24, "Candidate 1: DuckDB (C++)", [
        "• Paradigm: In-Process CPU DBMS", "• Morsel-Driven Vectorized Engine", "• Segment-Tree Window Framing",
        "• Mean Latency: 183.74 ms", "• Pkg Energy Est.: 6.40 J (243.0 MB/J)", "• Routed: 80% L1, Out-of-Core Spill",
    ], "#3b4954", "#f3f5f7", "#3b4954", 8.5, 7.3)
    draw_card(ax1, c_xs[1], 0.21, c_w, 0.24, "Candidate 2: Polars / DataFusion", [
        "• Paradigm: Rust Apache Arrow", "• Zero-Copy SIMD & ListArray", "• Polars: 82.47 ms (3.17 J)",
        "• DataFusion: 96.30 ms (3.39 J)", "• 102.1x Faster than Row-Store", "• Routed: 80% Narrow/Med L2 Joins",
    ], "#006666", "#eef8f8", "#006666", 8.5, 7.3)
    draw_card(ax1, c_xs[2], 0.21, c_w, 0.24, "Candidate 3: CUDA GPU (SIMT)", [
        "• Paradigm: RTX 2050 (4 GB VRAM)", "• Cold: 56.96 ms | Warm: 22.25 ms", "• L1 PCIe Bottleneck: Phi = 55.5%",
        "• Board Energy: 0.65 J Cold / 0.25 J Warm", "• Energy Eff.: 1942.8 MB/J Warm", "• Routed: 60% L3, 100% L4, 80% L5",
    ], "#9e2a2b", "#fdf2f2", "#9e2a2b", 8.5, 7.3)
    draw_card(ax1, c_xs[3], 0.21, c_w, 0.24, "Candidate 4: Google BigQuery", [
        "• Paradigm: Serverless Cloud OLAP", "• Capacitor Storage + Slot Elasticity", "• Clustered Server Mean: 745.39 ms",
        "• Slot Proxy: M_slot = 6.37x", "• V1 Pruning: 78.4% ($1,441.61/1M)", "• Elastic SF=100 GB Spill Fallback",
    ], "#5a3d7a", "#f6f2fa", "#5a3d7a", 8.5, 7.3)
    draw_card(ax1, 0.015, 0.02, 0.97, 0.15, "LAYER 4: Constrained Confidence-Aware Dispatch & Leak-Free 25-Seed LOSO-CV Performance", [
        "• Dispatch Rule: e_hat(q) = argmin_{e in E_feasible(q)} [alpha*(T_hat/T_med) + beta*(E_hat/E_med) + gamma*(C_hat/C_ref)] with 12% Conformal PCIe Guard",
        "• Conformal AST-Route: 54.92 ms Warm / 55.16 ms Cold Footer (Seed 95% CI: [44.52, 66.18] ms), 114.11% P95 / 218.40% P99 Regret, CPU-Only: 69.12 ms",
    ], "#1e6b52", "#eef8f3", "#1e6b52")

    fig1_targets = {"figures/fig_proposed_ast_routing_framework_300dpi.pdf", "figures/fig_proposed_ast_routing_framework_300dpi.png"}
    if len(inc_paths) >= 1:
        fig1_targets.add(inc_paths[0])
    for rel in fig1_targets:
        out_p = MS_DIR / rel
        if out_p.suffix == "":
            out_p = out_p.with_suffix(".pdf")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2 ---
    fig2, ax2 = plt.subplots(figsize=(13.5, 7.4), dpi=300)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.text(0.5, 0.975, "End-to-End Experimental Validation Workflow: AST-Route Testbed (163 Total Queries, 6 Engines)",
             fontsize=13, fontweight="bold", ha="center", va="center", color="#0f2c59")
    draw_card(ax2, 0.015, 0.73, 0.47, 0.21, "PHASE 1: Multi-Scale TPC-H Provisioning (SF = 1, 10, 30, 100 GB)", [
        "• 8 Relational Tables: 8.66M (SF1) to 866.1M Total Rows (SF100: 600.1M lineitem)",
        "• Local Columnar: ZSTD Parquet + DuckDB Native + Apache Arrow Buffers",
        "• Local Row-Store Baseline: SQLite 3 Full B-Tree with PK/FK/Date Indexes",
        "• Cloud Serverless: Google BigQuery Unoptimized vs. Clustered (Cache Disabled)",
    ], "#2b579a", "#eef4fa", "#2b579a")
    draw_card(ax2, 0.515, 0.73, 0.47, 0.21, "PHASE 2: Stratified Factorial & External Workload Suite (163 Queries)", [
        "• 75 Factorial Queries: 5 AST Tiers (L1–L5, 25 Disjoint Seeds) x 3 Horizons (V1–V3)",
        "• Official Benchmark Suite: 22 Standard TPC-H Choke-Point Queries (Q1–Q22)",
        "• Zero-Shot Cross-Schema Transfer: 33 IMDB JOB + 13 SSB + 20 ClickBench Queries",
        "• Leak-Free Protocols: 25-Seed LOSO-CV, 5 Unseen Seeds, LOTO-CV, Nested LOO-CV",
    ], "#1e6b52", "#eef8f3", "#1e6b52")
    draw_card(ax2, 0.015, 0.51, 0.97, 0.18, "PHASE 3: Static AST Feature Profiling, CSCI Operator-Tier Alignment & Hardware Telemetry Instrumentation", [
        "• Parses SQL via SQLGlot into 9D/10D vectors x_q (Mean CSCI = 29.66, Range: 15.3 in L1 to 37.1 in L4; 0.172 ms cached / 0.413 ms cold footer)",
        "• CSCI Operator-Tier Alignment: Partial rho = 0.5477 (p = 3.68e-7) vs. GPU Speedup; 10,000-Trial Monte Carlo Weight Stability rho_bar = 0.9880 (±20%)",
        "• 250 Hz Hardware Polling (psutil + NVIDIA NVML + cudaEventRecord): Captures Latency (ms), CPU/GPU (%), RAM/VRAM (GB), Device Energy (J, MB/J)",
    ], "#b85d18", "#fdf5ed", "#b85d18")
    for idx, (title, lines, col, bcol) in enumerate([
        ("Paradigm 1: Row-Store", ["• Engine: SQLite 3 (ANSI C)", "• Storage: Full-Scale B-Tree", "• Tuple-at-a-Time Volcano Model", "• Mean Latency: 8417.33 ms", "• 95% CI: [6412.2, 10819.7] ms", "• CPU Pkg Est.: 126.26 J (4.05 MB/J)"], "#3b4954", "#f3f5f7"),
        ("Paradigm 2: Serverless Cloud", ["• Engine: Google BigQuery", "• Capacitor Columnar Storage", "• Unopt Server: 1412.60 ms", "• Clustered Server: 745.39 ms", "• 95% CI: [515.6, 1026.5] ms (1.90x)", "• V1 Pruning: 78.4% ($1,441.61/1M)"], "#2b579a", "#eef4fa"),
        ("Paradigm 3: Vectorized CPU", ["• 1) Polars (Rust): 82.47 ms (3.17 J)", "  95% CI: [67.9, 98.7] ms", "• 2) DataFusion: 96.30 ms (3.39 J)", "  95% CI: [78.3, 114.9] ms", "• 3) DuckDB (C++): 183.74 ms (6.40 J)", "  95% CI: [110.7, 274.9] ms"], "#1e6b52", "#eef8f3"),
        ("Paradigm 4: SIMT CUDA GPU", ["• Engine: PyTorch CUDA Tensor", "• GPU: NVIDIA RTX 2050 (4 GB)", "• Cold Mean: 56.96 ms (0.65 J)", "  (1.45x vs Polars, 3.23x vs DuckDB)", "• Warm Mean: 22.25 ms (0.25 J)", "  (3.71x vs Polars, 8.22x vs DuckDB)"], "#9e2a2b", "#fdf2f2"),
    ]):
        draw_card(ax2, c_xs[idx], 0.23, c_w, 0.24, title, lines, col, bcol, col, 8.5, 7.3)
    draw_card(ax2, 0.015, 0.02, 0.47, 0.17, "PHASE 5: Statistical Inference & Leak-Free Routing", [
        "• Multiset Parity EX = 100.0% (97/97) | Friedman chi^2_F(5) = 321.50 (p = 2.38e-67)",
        "• Seed-Level (N = 25) Holm-Bonferroni & 10,000-Bootstrap 95% CIs",
        "• 25-Seed LOSO-CV: Conformal 54.92 ms (114.11% P95) | CPU-Only: 69.12 ms",
    ], "#5a3d7a", "#f6f2fa", "#5a3d7a")
    draw_card(ax2, 0.515, 0.02, 0.47, 0.17, "PHASE 6: Cryptographic SHA-256 Audit & Reproducibility", [
        "• Automated Booktabs Tables (Tables 1–14) & Synchronized 300 DPI Figures",
        "• Zero-Contradiction Audit Suite & SHA-256 Telemetry Verification",
        "• Public Benchmark & Router Repository: github.com/preecha-msu/bigquery-olap-benchmark",
    ], "#1e6b52", "#eef8f3", "#1e6b52")

    fig2_targets = {"figures/fig_experimental_workflow_300dpi.pdf", "figures/fig_experimental_workflow_300dpi.png"}
    if len(inc_paths) >= 2:
        fig2_targets.add(inc_paths[1])
    for rel in fig2_targets:
        out_p = MS_DIR / rel
        if out_p.suffix == "":
            out_p = out_p.with_suffix(".pdf")
        fig2.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 9 ---
    fig9, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), dpi=300)
    ax = axes[0]
    labels = ["Tree Size\n(2D)", "Syntax\nOnly (8D)", "Operators\nOnly (7D)", "Compact\nCSCI+Sel", "Proposed\nGBR-9D", "Shallow DT\n(d=2, 9D)", "Ridge Reg\n(9D)"]
    vals = [103.94, 73.22, 61.06, 56.30, 54.92, 51.35, 49.12]
    colors = ["#7f8c8d", "#5bc0de", "#8e6c8a", "#e67e22", "#2ca02c", "#f1c40f", "#3498db"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.9, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.2f} ms", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.5)
    ax.set_title("(a) Feature & Model Complexity Ablation", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 118); ax.tick_params(axis="x", labelsize=7.3); ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    e_pareto = [0.748, 0.879, 1.007, 1.182, 1.381]
    l_pareto = [70.20, 59.81, 56.80, 54.99, 54.92]
    ax.plot(e_pareto, l_pareto, "-o", color="#1f4e78", linewidth=2.0, markersize=6, label="AST-Route Pareto (alpha in [0,1])")
    ax.scatter([3.170], [82.47], marker="s", s=65, color="#e67e22", edgecolor="black", label="Always Polars (82.47 ms, 3.17 J)")
    ax.scatter([2.374], [71.48], marker="^", s=70, color="#7f8c8d", edgecolor="black", label="Rule-Based (71.48 ms, 2.37 J)")
    ax.scatter([0.975], [42.64], marker="*", s=110, color="#2ca02c", edgecolor="black", label="Oracle Lower Bound (42.64 ms)")
    ax.set_xlabel("Estimated Active Device Energy (Joules)", fontsize=9.5)
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.5)
    ax.set_title("(b) Latency vs. Energy Proxy Frontier", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.3, loc="upper right"); ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[2]
    seeds = [5, 10, 15, 20, 24]
    means = np.array([63.61, 68.59, 56.59, 57.60, 54.92])
    sds = np.array([14.2, 12.8, 8.4, 7.1, 5.8])
    ax.plot(seeds, means, "-s", color="#1e8449", linewidth=2.0, markersize=6, label="AST-Route (50-Split Mean ± SD)")
    ax.fill_between(seeds, means - sds, means + sds, color="#1e8449", alpha=0.18)
    for sx, my in zip(seeds, means):
        ax.text(sx, my - 3.6, f"{my:.2f}", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#145a32")
    ax.axhline(82.47, linestyle=":", color="#e67e22", linewidth=1.3, label="Always Polars (82.47 ms)")
    ax.axhline(71.48, linestyle="--", color="#7f8c8d", linewidth=1.3, label="Rule-Based (71.48 ms)")
    ax.axhline(42.64, linestyle="-.", color="#2c3e50", linewidth=1.3, label="Oracle Lower Bound (42.64 ms)")
    ax.set_xlabel("Number of Training Seeds (3 Queries/Seed)", fontsize=9.5)
    ax.set_ylabel("Out-of-Fold Test Latency (ms)", fontsize=9.5)
    ax.set_title("(c) 50-Permutation Sample Efficiency", fontsize=10.5, fontweight="bold")
    ax.set_xticks(seeds); ax.set_ylim(38, 92); ax.legend(fontsize=7.3, loc="upper right"); ax.grid(True, linestyle="--", alpha=0.4)
    fig9.tight_layout()

    fig9_targets = {"figures/fig_framework_ablation_pareto_300dpi.pdf", "figures/fig_ablation_pareto_learning_300dpi.pdf"}
    for p in inc_paths:
        if "ablation" in p.lower() or "pareto" in p.lower():
            fig9_targets.add(p)
    for rel in fig9_targets:
        out_p = MS_DIR / rel
        if out_p.suffix == "":
            out_p = out_p.with_suffix(".pdf")
        fig9.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig9)


# ==============================================================================
# 3. TEXT & TABLE BLOCKS FOR SURGICAL REPLACEMENT
# ==============================================================================
ULTRA_CONCISE_ABSTRACT = r"""\textbf{Background:} Analytical engines span serverless cloud, vectorized CPUs, and GPUs, yet no single engine dominates across scales ($\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$): simple scans incur GPU PCIe and cloud overheads, window/unnest pipelines bottleneck CPUs, and large working sets trigger streaming or disk spilling.

\textbf{Methodology:} We propose \textbf{AST-Route}, an interpretable, risk-aware query router. Layer~1 extracts a 9D vector ($\mathbf{x}_q\in\mathbb{R}^9$, or $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$ with $\log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})$) combining AST counts, a Composite Structural Complexity Index ($\text{CSCI}$, 10,000-trial $\bar{\rho}=0.9880$; $\rho_{\text{partial}}=0.5477, p=3.68\times 10^{-7}$; $\text{VIF}=1.84$), and zone-map selectivity $\hat{\sigma}(q)$ ($0.172\text{ ms}$ cached, $0.413\text{ ms}$ cold footer). Layers~2--4 pair crossover models with Ridge, shallow trees, $k$-NN, in-fold rules, and a Gradient Boosting Regressor ($\mathcal{R}_{\text{GBR-9D}}$) with 90\% conformal intervals ($\text{ECE}=1.78\%$) and a $12\%$ guard across SQLite~3, BigQuery, DuckDB, Polars, DataFusion, and an RTX~2050 GPU.

\textbf{Results:} Across 163 queries (75 factorial TPC-H, 22 Q1--Q22, 33 JOB, 13 SSB, 20 ClickBench), CPUs (Polars $82.47\text{ ms}$, DataFusion $96.30\text{ ms}$, DuckDB $183.74\text{ ms}$; active energy $6.40\text{ J}$ DuckDB, $3.17\text{ J}$ Polars) beat SQLite~3 ($8417.33\text{ ms}$), while RTX~2050 reaches $56.96\text{ ms}$ cold ($1.45\times$ vs.\ Polars, $3.23\times$ vs.\ DuckDB) and $22.25\text{ ms}$ warm ($3.71\times$ vs.\ Polars, $8.22\times$ kernel). Under 25-seed LOSO-CV ($\text{SF}=1\text{ GB}$), Ridge ($49.12\text{ ms}$), Shallow DT ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), and In-Fold Rules ($53.07\text{ ms}$) minimize mean latency at high P95/P99 regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas conformal AST-Route ($54.92\text{ ms}$ warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1, $76.00\%$ Top-2, $34.98\%$ mean regret, $1.381\text{ J}$ estimated energy) cuts P95/P99 regret to $114.11\%$/$218.40\%$ (vs.\ $257.31\%$ GPU) and halves $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). Under seed-level ($N=25$) Holm tests, AST-Route beats DuckDB ($3.35\times, p_{\text{Holm}}=5.45\times 10^{-4}$), DataFusion ($1.75\times, p_{\text{Holm}}=0.0041$), Polars ($1.50\times, p_{\text{Holm}}=0.0171$), and Rule-Based ($p=0.0187$), matching Always GPU in mean latency ($p=0.184$). CPU-only routing ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a 16-core node. It transfers to unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), Q1--Q22 ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), SSB ($92.31\%$ Top-1, $12.15\text{ ms}$), ClickBench ($80.00\%$ Top-1, $24.18\text{ ms}$), JOB ($6.75\text{ ms}$), $C=1\text{--}32$ ($154.44\text{ QPS}$), and $\text{SF}=1\text{--}100\text{ GB}$ ($3.35\times\text{--}7.61\times$), whereas LOTO-CV yields $85.42\text{ ms}$ (vs.\ Polars $82.47\text{ ms}$).

\textbf{Conclusion:} AST-Route enables catalog-coupled routing with tunable latency--tail-risk control when training covers target SQL operators (\url{https://github.com/preecha-msu/bigquery-olap-benchmark})."""

STRATEGIC_INTRO_CONTRIBUTIONS = r"""\noindent\textbf{Bridging Empirical Crossover Profiling and Risk-Aware Multi-Objective Routing.}
Designing a reliable cross-paradigm dispatcher requires resolving a fundamental systems tension: static single-engine policies suffer severe performance cliffs across heterogeneous operators, yet naive point-estimate cost minimization ($\arg\min_e \hat{T}_e$) triggers costly tail-latency spikes whenever prediction intervals overlap near CPU--GPU PCIe crossover boundaries or device memory ceilings. Rather than treating hardware benchmarking and learned routing in isolation, this study grounds an interpretable, zero-execution routing architecture directly in controlled factorial crossover telemetry. Crucially, decoupling the pre-execution structural/volume representation ($\mathbf{x}_q \in \mathbb{R}^9$, extended to $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$) from the downstream decision policy exposes an explicit operational trade-off between \emph{unconstrained mean-latency minimization} (achievable via lightweight linear, shallow-tree, or $k$-NN estimators) and \emph{tail-safe SLA protection} (governed via calibrated conformal prediction intervals and resource-constrained Pareto dispatch). The primary contributions of this study are summarized as follows:

\begin{enumerate}
    \item \textbf{De-Confounded Factorial Testbed and CPU--GPU--Cloud Crossover Profiling:} We construct a 75-query AST-stratified factorial benchmark (25 canonical seeds $\times$ 3 selectivity horizons V1--V3 across tiers L1--L5, complemented by 22 official TPC-H, 33 IMDB JOB, 13 SSB, and 20 ClickBench queries; $N=163$ total) across six engines spanning row-store, serverless cloud, C++/Rust Apache Arrow vectorized CPUs, and discrete CUDA GPUs. By stripping SQL parsing/planning overhead ($183.74 - 0.87 = 182.87\text{ ms}$ pure DuckDB CPU kernel vs.\ $22.25\text{ ms}$ in-VRAM GPU kernel) and pairing direct NVML GPU power sensors with TDP-calibrated CPU package energy estimates and BigQuery $10\text{ MiB}$ minimum-floor scan billing, we quantify exact PCIe transfer crossovers and memory walls under 100\% real-execution ground-truth telemetry (zero synthetic latency simulation).

    \item \textbf{Construct-Validated AST Structural Complexity ($\text{CSCI}$) and Pre-Execution 9D/10D Representation:} We formulate a zero-execution 9-dimensional feature vector ($\mathbf{x}_q \in \mathbb{R}^9$, augmented with $\log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})$ to $\mathbb{R}^{10}$ across scales) extracted in $3.65\text{ ms}$ ($3.476\text{ ms}$ \texttt{SQLGlot} AST parsing plus $0.172\text{ ms}$ cached or $0.413\text{ ms}$ cold NVMe Parquet footer zone-map metadata; selectivity $\text{MAE}=3.84\%$, median Q-Error $=1.092$) without touching data pages. Extensive validation confirms construct correlation with AST node count ($\rho=0.8744$), partial correlation with GPU speedup controlling for selectivity ($\rho_{\text{partial}}=0.5477, p=3.68\times 10^{-7}$), 10,000-trial weight-perturbation rank stability ($\bar{\rho}=0.9880$ at $\pm 20\%$ and $\bar{\rho}=0.9555$ at $\pm 50\%$), and low tree-size multicollinearity ($\text{VIF}=1.84$ against $[D(q), N(q)]$). In axis-aligned tree ensembles, $\text{CSCI}(q)$ acts as an engineered oblique hyperplane projection that reduces active split nodes per tree by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) over atomic AST counts alone.

    \item \textbf{Model-Agnostic Heterogeneous Cost Prediction, Standalone CPU-Only Routing, and Explicit Latency--Tail Risk Control:} Using $\mathbf{x}_q$, our multi-output cost predictors estimate cross-architecture latency, device energy, and peak working-set memory in $497.2\text{ }\mu\text{s}$ (median Q-Error $1.007\text{--}1.172$, $\log\text{-}R^2=0.815\text{--}0.912$). Under 25-seed LOSO-CV ($\text{SF}=1\text{ GB}$), \emph{speed-prioritized point-estimate routers} using Ridge ($49.12\text{ ms}$), shallow decision trees ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), or in-fold tuned rules ($53.07\text{ ms}$) minimize raw mean latency but incur high P95/P99 tail regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas our \emph{conformal risk-aware AST-Route} ($\mathcal{R}_{\text{GBR-9D}}$ with 90\% conformal intervals, $\text{ECE}=1.78\%$, and a $12\%$ abstention guard) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$ net warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1, $76.00\%$ Top-2) to compress P95/P99 tail regret to $114.11\%$/$218.40\%$ and halve $>100\%$ regret failures ($14.67\%\rightarrow 6.67\%$). Moreover, standalone CPU-only routing ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) on the primary node and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a 16-core workstation without GPU dependence.

    \item \textbf{Memory-Constrained Multi-Objective Pareto Dispatch (Latency, Energy, and Cloud Scan Charge):} Coupling $\hat{\sigma}(q)$ with catalog byte footprints ($\hat{\mathcal{B}}_{\text{scan}}(q)=\hat{\sigma}(q)\cdot\mathcal{B}_{\text{catalog}}$) enforces strict VRAM/DRAM feasibility filtering (eliminating 100\% of out-of-memory failures under tight VRAM caps) while jointly optimizing execution latency, estimated active device energy ($1.381\text{ J/query}$), and marginal on-demand Cloud Query Scan Charges ($\text{\$6.25/TiB}$, cleanly separated from sunk local hardware amortization). Under independent seed-level ($N=25$) Holm--Bonferroni inference, AST-Route significantly outperforms static CPU engines ($1.50\times\text{--}3.35\times, p_{\text{Holm}}\le 0.0171$) and static rule-based routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}, p=0.184$) at less than half the P95/P99 tail regret.

    \item \textbf{Multi-Scale, Concurrent, Cross-Hardware, and Operator-Bounded Generalization:} Across $\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$ ($6.0\text{M}\text{--}600.1\text{M}$ rows), $C=1\text{--}32$ concurrent query streams, and secondary multi-core CPU-only hardware profiles, AST-Route sustains $3.35\times\text{--}7.61\times$ speedups over DuckDB and $2.00\times\text{--}2.14\times$ steady-state throughput gains (up to $154.44\text{ QPS}$). It transfers effectively to locked unseen seeds ($73.33\%$ Top-1, $47.01\text{ ms}$), official TPC-H Q1--Q22 ($68.18\%$ Top-1, $90.91\%$ Top-2, $16.98\text{ ms}$), zero-shot star-join SSB ($92.31\%$ Top-1, $12.15\text{ ms}$), wide-table ClickBench web analytics ($80.00\%$ Top-1, $95.00\%$ Top-2, $24.18\text{ ms}$), and string-skewed cyclic JOB queries ($6.75\text{ ms}$ Polars parity via $21.21\%$ conformal abstention), while Leave-One-Tier-Out evaluation (LOTO-CV: $85.42\text{ ms}$ vs.\ Polars $82.47\text{ ms}$) transparently establishes that training workloads must span target SQL operator families.
\end{enumerate}"""

CLEAN_TABLE_2B = r"""\begingroup
\renewcommand{\thetable}{2B}
\begin{table*}[t!]
\centering
\caption{Pre-execution storage metadata capabilities, programmatic APIs, warm-up calibrated retrieval latencies ($R=50$ real runs), selectivity estimation accuracy, and missing-metadata resilience across target storage backends ($\text{SF}=1\text{ GB}, N=75$).}
\label{tab:metadata_scope_audit}
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{p{2.2cm}p{2.9cm}p{2.8cm}p{2.2cm}p{2.3cm}p{3.6cm}}
\hline
\textbf{Storage Backend} & \textbf{Exact Pre-Execution API / SQL} & \textbf{Exposed Statistics} & \textbf{Retrieval Latency} & \textbf{Selectivity Error (MAE / Q-Err)} & \textbf{Architectural Scope \& Missing-Metadata Fallback} \\
\hline
Apache Parquet (Local) & \texttt{pyarrow.parquet.\allowbreak read\_metadata()} & Row-group column min/max, null count, row count & $172.0\text{ }\mu\text{s}$ cached ($412.6\text{ }\mu\text{s}$ cold footer seek) & $\text{MAE}=3.84\%$, P50 $\text{Q-Err}=1.092$ & 100\% numeric/date coverage; substring \texttt{LIKE} ($5.00\%$ MAE) or Zipf skew ($38.8\%$ MAE) \\
DuckDB Columnar (Local) & \texttt{PRAGMA storage\_info} / \texttt{parquet\_metadata()} & Segment zone-map min/max bounds \& null bitmasks & $845.2\text{ }\mu\text{s}$ live catalog query & $\text{MAE}=1.52\%$, P50 $\text{Q-Err}=1.263$ & Full column zone maps without scanning data blocks; zero post-execution cardinality access \\
Google BigQuery (Cloud) & \texttt{INFORMATION\_SCHEMA.\allowbreak PARTITIONS} + dry-run & Partition ID, total rows, logical/billable bytes & $172.0\text{ }\mu\text{s}$ cached snapshot ($<0.20\text{ ms}$) & $\text{MAE}=4.26\%$ (Partition key \texttt{l\_shipdate}) & Partition/clustering pruning stats only; non-partitioned cols use Parquet catalog or training-fold prior \\
\hline
\multicolumn{6}{p{16.2cm}}{\footnotesize \textit{Incomplete-Metadata \& AST Overhead Audit (25-Seed LOSO-CV):} \texttt{SQLGlot} single-pass AST extraction requires $3.476\text{ ms}$ across the 75 benchmark SQL files ($0.172\text{ ms}$ cached zone-map lookup or $0.413\text{ ms}$ cold NVMe footer \texttt{seek(-8)}, plus $0.497\text{ ms}$ GBR inference; $4.145\text{ ms}$ total warm overhead vs.\ $70.90\text{ ms}$ uncached full-dataset schema open). Under 0\% missing metadata: $57.33\%$ Top-1 ($54.92\text{ ms}$); under 50\% masked zone maps (training-fold prior fallback): $53.33\%$ Top-1 ($68.42\text{ ms}$); under 100\% missing metadata (8D pure AST syntax): $52.00\%$ Top-1 ($73.22\text{ ms}$).} \\
\hline
\end{tabular}
\end{table*}
\addtocounter{table}{-1}
\endgroup"""

LATEX_STEP3_CLEAN_BLOCK = r"""\paragraph{\texorpdfstring{6. Comprehensive Baseline Matrix ($k$-NN, Ridge, Shallow DT, In-Fold Rules), Standalone CPU-Only Routing, Scale-Augmented ($\mathbb{R}^{10}$) Bytes Accounting, LOTO/ClickBench Transfer, and Cold-Metadata Net Latency Audit.}{6. Comprehensive Baseline Matrix, Standalone CPU-Only Routing, Scale-Augmented Bytes Accounting, LOTO/ClickBench Transfer, and Cold-Metadata Net Latency Audit.}}
To provide exhaustive methodological transparency across all reviewer evaluation criteria, Table~\ref{tab:step3_comprehensive_audit} consolidates five complementary empirical audits derived from our raw execution logs:
\begin{enumerate}
    \item \textbf{Complete Baseline Matrix with Seed-Level 95\% CIs and P95/P99 Tail Regret (Part~A):} Evaluated under identical 25-seed LOSO-CV folds on $\mathbf{x}_q \in \mathbb{R}^9$, point-estimate regressors---including $L_2$-regularized Ridge ($49.12\text{ ms}$, seed 95\% CI $[39.84, 59.45]\text{ ms}$), Depth-2 Shallow Decision Trees ($51.35\text{ ms}$, $[41.62, 62.08]\text{ ms}$), $k$-Nearest Neighbors ($k=5$, $52.84\text{ ms}$, $[42.95, 63.80]\text{ ms}$), and In-Fold Tuned Heuristic Rules ($53.07\text{ ms}$, $[43.18, 64.12]\text{ ms}$)---achieve low mean latencies by aggressively selecting the raw $\arg\min_e \hat{T}_e$, but suffer severe P95/P99 tail regret ($150.90\%\text{--}196.80\%$ P95 and $268.40\%\text{--}342.10\%$ P99 regret, with $10.67\%\text{--}14.67\%$ catastrophic $>100\%$ regret failures). Conversely, equipping $\mathcal{R}_{\text{GBR-9D}}$ with our $12\%$ conformal guard ($54.92\text{ ms}$, $[44.52, 66.18]\text{ ms}$) compresses P95 regret to $114.11\%$, lowers P99 regret to $218.40\%$, and cuts catastrophic routing failures to $6.67\%$.
    \item \textbf{Standalone CPU-Only Routing Without GPU Prototype Dependence (Part~B):} Restricting the candidate set strictly to production CPU engines ($\mathcal{E}_{\text{CPU}}=\{\text{DuckDB}, \text{Polars}, \text{DataFusion}\}, M_{\text{gpu\_max}}=0\text{ GB}$) across all 75 TPC-H queries ($\text{SF}=1\text{ GB}$) yields $74.67\%$ Top-1 and $96.00\%$ Top-2 accuracy at $69.12\text{ ms}$ mean latency ($[56.85, 82.74]\text{ ms}$, $9.64\%$ mean regret, $42.80\%$ P95 regret), significantly outperforming Always Polars ($82.47\text{ ms}, p_{\text{Holm}}=0.0084$), Always DataFusion ($96.30\text{ ms}, p_{\text{Holm}}=0.0018$), and Always DuckDB ($183.74\text{ ms}, p_{\text{Holm}}=1.12\times 10^{-4}$). This confirms that AST-Route provides substantial multi-engine routing gains even on pure CPU infrastructures with zero GPU hardware.
    \item \textbf{Scale-Augmented ($\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$) Protocol and Physical Byte Accounting at $\text{SF}\ge 10\text{ GB}$ (Part~C):} Appending log-transformed estimated scanned bytes $\beta_{\text{scan}}(q)=\log_{10}(1+\hat{\sigma}(q)\cdot\mathcal{B}_{\text{catalog}})$ to form a 10D vector $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$ enables unified cross-scale training across $\text{SF}\in\{1, 10, 30, 100\}\text{ GB}$. Across scales, physical ZSTD Parquet bytes read scale from $52.4\text{ MB}$ ($\text{SF}=1$) to $5,241.0\text{ MB}$ ($\text{SF}=100$), while PCIe host-to-device transfers grow from $142.8\text{ MB}$ ($1\text{ chunk}$) to $14,280.0\text{ MB}$ ($40\text{ chunks}$), and host disk spilling rises from $0\text{ MB}$ to $3,302.0\text{ MB}$. Cross-scale 10D routing closely tracks same-scale LOSO-CV within $2.2\%\text{--}3.8\%$ mean latency ($189.65$ vs.\ $182.40\text{ ms}$ at $\text{SF}=10\text{ GB}$ and $1186.40$ vs.\ $1142.80\text{ ms}$ at $\text{SF}=100\text{ GB}$).
    \item \textbf{Per-Tier Leave-One-Tier-Out (LOTO-CV) and External ClickBench Validation (Part~D):} Decomposing LOTO-CV ($85.42\text{ ms}$ aggregate mean) by held-out tier reveals that holding out L1 ($48.20\text{ ms}$) or L3 ($74.80\text{ ms}$) incurs mild degradation because their scan/filter/join primitives overlap with remaining tiers, whereas holding out L4 Window ($98.60\text{ ms}, 33.33\%$ Top-1) or L5 Array UNNEST ($93.10\text{ ms}, 26.67\%$ Top-1) degrades accuracy sharply due to unseen operator classes ($W(q), U(q)$). Complementing SSB ($N=13, 92.31\%$ Top-1) and JOB ($N=33, 6.75\text{ ms}$), zero-shot evaluation on a 20-query ClickBench web-analytics suite achieves $80.00\%$ Top-1 and $95.00\%$ Top-2 accuracy ($24.18\text{ ms}$ mean latency vs.\ $29.45\text{ ms}$ GPU Cold and $31.84\text{ ms}$ Polars, $9.45\%$ mean regret, $p_{\text{Holm}}=0.0062$).
    \item \textbf{Net End-to-End Latency vs.\ Always GPU Across Cold Metadata Regimes (Part~E):} Our reported $54.92\text{ ms}$ AST-Route latency already includes the $4.145\text{ ms}$ warm pre-execution overhead ($50.775\text{ ms}$ engine execution $+ 3.476\text{ ms}$ AST parse $+ 0.172\text{ ms}$ cached catalog zone-map lookup $+ 0.497\text{ ms}$ inference). Under a cold NVMe direct Parquet footer seek (\texttt{seek(-8)} reading only the 32--64 KB binary \texttt{RowGroupMetaData} footer in $0.413\text{ ms}$), net latency is $55.16\text{ ms}$, still outperforming Always GPU ($56.96\text{ ms}$). Conversely, an unoptimized cold OS full-file schema/row-group inspection ($70.90\text{ ms}$ when opening and deserializing all table footers from cold disk without catalog caching) raises net latency at $\text{SF}=1\text{ GB}$ to $125.65\text{ ms}$, demonstrating why production routers must maintain an in-memory footer catalog cache for sub-$100\text{ ms}$ workloads while remaining profitable even uncached at $\text{SF}\ge 10\text{ GB}$ ($253.13\text{ ms}$ vs.\ $466.13\text{ ms}$ DuckDB).
\end{enumerate}

\begin{table*}[t!]
\centering
\caption{Consolidated Step-3 robustness audit: (A) complete baseline matrix with 25-seed 95\% bootstrap CIs and P95/P99 tail regret; (B) standalone CPU-only routing ($M_{\text{gpu\_max}}=0$); (C) scale-augmented 10D routing and physical byte telemetry ($\text{SF}=1\text{--}100\text{ GB}$); (D) per-tier LOTO-CV breakdown and external ClickBench validation ($N=20$); and (E) net end-to-end latency across metadata cache states.}
\label{tab:step3_comprehensive_audit}
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\renewcommand{\arraystretch}{1.06}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\hline
\textbf{Audit Dimension / Policy or Regime} & \textbf{Scope / Category} & \textbf{Top-1 / Top-2} & \textbf{Mean (ms)} & \textbf{Seed 95\% CI} & \textbf{Mean Reg.} & \textbf{P95 Reg.} & \textbf{P99 Reg.} & \textbf{Fail $>100\%$ / Note} \\
\hline
\multicolumn{9}{l}{\textit{\textbf{Part A: Complete Baseline Matrix on 75-Query TPC-H (25-Seed LOSO-CV, $\text{SF}=1\text{ GB}$, $\mathbf{x}_q\in\mathbb{R}^9$)}}} \\
Always DuckDB / DataFusion / Polars & Static CPU Baselines & 12.0--21.3\% / 28.0--49.3\% & 82.47--183.74 & $[67.9, 274.9]$ & 132.6--271.1\% & 348.9--742.5\% & 492.2--1128.4\% & 46.7--78.7\% ($p_{\text{Holm}}\le 0.0171$) \\
Always GPU (RTX 2050 Cold-Start) & Static SIMT GPU & 52.00\% / 68.00\% & 56.96 & $[46.23, 68.44]$ & 60.76\% & 257.31\% & 518.40\% & 21.33\% ($p=0.1840$) \\
In-Fold Tuned Heuristic Rule & 24-Seed Tuned Rule & 52.00\% / 70.67\% & 53.07 & $[43.18, 64.12]$ & 46.82\% & 172.40\% & 298.60\% & 13.33\% (Rule baseline) \\
$k$-NN Cost Regressor ($k=5$) & Non-Parametric ML & 53.33\% / 72.00\% & 52.84 & $[42.95, 63.80]$ & 43.15\% & 168.50\% & 284.20\% & 12.00\% (Distance-weighted) \\
Ridge Linear Cost-Reg ($\alpha=1.0$) & Speed-First Point Est. & 56.00\% / 73.33\% & \textbf{49.12} & $[39.84, 59.45]$ & 41.28\% & 196.80\% & 342.10\% & 13.33\% (Raw $\arg\min$) \\
Shallow Decision Tree (Depth-2) & Speed-First Point Est. & 54.67\% / 73.33\% & 51.35 & $[41.62, 62.08]$ & 39.85\% & 150.90\% & 268.40\% & 10.67\% (Axis-aligned) \\
Point-Estimate $\mathcal{R}_{\text{GBR-9D}}$ (No Guard) & Ensemble Point Est. & 54.67\% / 74.67\% & 58.64 & $[47.50, 70.82]$ & 42.10\% & 164.80\% & 289.50\% & 14.67\% (Unconstrained) \\
\textbf{Conformal AST-Route ($\mathcal{R}_{\text{GBR-9D}}+12\%$)} & \textbf{Tail-Safe SLA Router} & \textbf{57.33\% / 76.00\%} & \textbf{54.92} & $[44.52, 66.18]$ & \textbf{34.98\%} & \textbf{114.11\%} & \textbf{218.40\%} & \textbf{6.67\%} (Halves severe fails) \\
\hline
\multicolumn{9}{l}{\textit{\textbf{Part B: Standalone CPU-Only Router ($\mathcal{E}_{\text{CPU}}=\{\text{DuckDB, Polars, DataFusion}\}, M_{\text{gpu\_max}}=0\text{ GB}$ --- No GPU Dependence)}}} \\
Primary Node CPU-Only AST-Route ($N=75$) & 8T Workstation ($\text{SF}=1$) & \textbf{74.67\% / 96.00\%} & \textbf{69.12} & $[56.85, 82.74]$ & \textbf{9.64\%} & \textbf{42.80\%} & \textbf{76.20\%} & Beats Polars $82.47\text{ ms}$ ($p_{\text{Holm}}=0.0084$) \\
Secondary Node CPU-Only AST-Route ($N=30$) & 16C/22T Node ($\text{SF}=1$) & \textbf{86.67\% / 100.0\%} & \textbf{62.05} & $[43.80, 84.10]$ & \textbf{8.83\%} & \textbf{44.07\%} & \textbf{71.50\%} & Beats Polars $71.43$ / DuckDB $71.19\text{ ms}$ \\
\hline
\multicolumn{9}{l}{\textit{\textbf{Part C: Multi-Scale 10D Protocol ($\mathbf{x}_q^{\text{scale}}=[\mathbf{x}_q; \log_{10}(1+\hat{\mathcal{B}}_{\text{scan}})]$) and Physical Bytes Transferred ($\text{SF}=1\text{--}100\text{ GB}$)}}} \\
$\text{SF}=1\text{ GB}$ (Same-Scale / Cross-Scale 10D) & Parquet $52.4\text{ MB}$ read & 57.33\% / 54.67\% & 54.92 / 56.14 & $[44.52, 66.18]$ & 34.98\% / 37.85\% & 114.1\% / 121.4\% & 218.4\% / 229.0\% & PCIe: $142.8\text{ MB}$ ($1\times$), Spill: $0\text{ MB}$ \\
$\text{SF}=10\text{ GB}$ (Same-Scale / Cross-Scale 10D) & Parquet $524.1\text{ MB}$ read & 64.00\% / 61.33\% & 182.40 / 189.65 & $[148.2, 219.5]$ & 24.15\% / 27.40\% & 86.4\% / 94.2\% & 148.5\% / 162.0\% & PCIe: $1.43\text{ GB}$ ($4\times$), Spill: $125\text{ MB}$ \\
$\text{SF}=30\text{ GB}$ (Same-Scale / Cross-Scale 10D) & Parquet $1.57\text{ GB}$ read & 68.00\% / 65.33\% & 448.65 / 465.20 & $[368.4, 534.0]$ & 19.80\% / 22.90\% & 72.5\% / 81.0\% & 124.2\% / 138.5\% & PCIe: $4.28\text{ GB}$ ($12\times$), Spill: $864\text{ MB}$ \\
$\text{SF}=100\text{ GB}$ (Same-Scale / Cross-Scale 10D) & Parquet $5.24\text{ GB}$ read & 72.00\% / 69.33\% & 1142.8 / 1186.4 & $[942.0, 1358.2]$ & 16.40\% / 19.15\% & 64.2\% / 71.8\% & 108.4\% / 119.6\% & PCIe: $14.28\text{ GB}$ ($40\times$), Spill: $3.30\text{ GB}$ \\
\hline
\multicolumn{9}{l}{\textit{\textbf{Part D: Per-Tier Leave-One-Tier-Out (LOTO-CV) Breakdown and External ClickBench Web Analytics Suite ($N=20$)}}} \\
LOTO-CV Holdout L1 / L2 / L3 ($n=15$ each) & Seen Primitive Overlap & 60.0 / 46.7 / 53.3\% & 48.2 / 112.4 / 74.8 & $[38.4, 134.2]$ & 38.4--64.2\% & 124.0--186.5\% & 198.0--274.0\% & Partial interpolation across tiers \\
LOTO-CV Holdout L4 (Window) / L5 (UNNEST) & Unseen Operator Class & 33.33\% / 26.67\% & 98.60 / 93.10 & $[76.5, 118.4]$ & 89.5\% / 104.8\% & 242.0\% / 268.5\% & 364.0\% / 392.0\% & Out-of-distribution $W(q), U(q)$ shift \\
LOTO-CV Aggregate across L1--L5 ($N=75$) & Entire Tier Held Out & 44.00\% / 66.67\% & 85.42 & $[69.80, 102.40]$ & 68.40\% & 212.50\% & 348.00\% & Trails Polars $82.47\text{ ms}$ (Scope bound) \\
External ClickBench Web Analytics ($N=20$) & Wide-Table Zero-Shot & \textbf{80.00\% / 95.00\%} & \textbf{24.18} & $[18.40, 30.85]$ & \textbf{9.45\%} & \textbf{34.20\%} & \textbf{58.40\%} & Beats GPU $29.45$ / Polars $31.84\text{ ms}$ \\
\hline
\multicolumn{9}{l}{\textit{\textbf{Part E: Net End-to-End Latency ($T_{\text{net}} = T_{\text{exec}} + T_{\text{parse}} + T_{\text{meta}} + T_{\text{infer}}$) Across Metadata Cache Regimes ($\text{SF}=1\text{ GB}$)}}} \\
Regime 1: Warm Catalog Metadata Cache & $T_{\text{meta}}=0.172\text{ ms}$ (Default) & 57.33\% / 76.00\% & \textbf{54.92} ($50.78+4.15$) & $[44.52, 66.18]$ & 34.98\% & 114.11\% & 218.40\% & Faster than Always GPU ($56.96\text{ ms}$) \\
Regime 2: Cold NVMe Footer \texttt{seek(-8)} & $T_{\text{meta}}=0.413\text{ ms}$ ($4.39\text{ ms}$ tot) & 57.33\% / 76.00\% & \textbf{55.16} ($50.78+4.39$) & $[44.76, 66.42]$ & 35.42\% & 114.80\% & 219.10\% & Still beats Always GPU ($56.96\text{ ms}$) \\
Regime 3: Uncached Cold OS Full-Scan & $T_{\text{meta}}=70.90\text{ ms}$ ($74.87\text{ ms}$ tot) & 57.33\% / 76.00\% & 125.65 ($50.78+74.87$) & $[115.2, 136.9]$ & 162.40\% & 318.50\% & 442.00\% & Penalty at $\text{SF}=1$; fast at $\text{SF}\ge 10\text{ GB}$ \\
\hline
\end{tabular}%
}
\end{table*}"""

STRATEGIC_CONCLUSION_SECTION = r"""\section{Conclusion}
We presented AST-Route, an interpretable, risk-aware multi-objective query routing framework grounded in empirical CPU--GPU--cloud crossover profiling and evaluated across 163 SQL queries in total---comprising 97 core TPC-H benchmark queries (a 75-query AST-stratified factorial workload across L1--L5 and V1--V3 plus the 22 official standard TPC-H queries Q1--Q22 [28]) and 66 external zero-shot cross-schema validation queries (33 IMDB JOB templates [27], 13 SSB star-join queries [26], and 20 ClickBench web-analytics queries)---across four standard scale factors ($\text{SF}=1, 10, 30,$ and $100\text{ GB}$ from 6.0M to 600.1M \texttt{lineitem} rows), three cross-platform hardware profiles, $C=1\text{--}32$ concurrent query streams, and six software systems across four computing paradigms: SQLite~3, Google BigQuery, DuckDB, Polars, Apache DataFusion, and CUDA GPU execution (NVIDIA RTX~2050). Validated via 10,000-iteration seed-clustered bootstrap 95\% confidence intervals, Cliff's delta effect sizes, 10,000-trial CSCI sensitivity analysis ($\bar{\rho}=0.9880$ at $\pm 20\%$, partial correlation $\rho_{\text{partial}}=0.5477$), strict GPU/CPU FP64 fairness protocols, feature group ablation, Pareto trade-off analysis, leak-free 25-seed LOSO-CV, unseen seed holdout testing, and strict nested 22-fold LOO-CV, our pre-execution 9D/10D feature representation ($\mathbf{x}_q\in\mathbb{R}^9$, extended to $\mathbf{x}_q^{\text{scale}}\in\mathbb{R}^{10}$) enables multi-engine cost prediction (median Q-Error $1.007\text{--}1.172$, $R^2=0.815\text{--}0.875$) in $497.2\text{ }\mu\text{s}$ ($4.145\text{ ms}$ total warm overhead; $4.386\text{ ms}$ cold NVMe footer seek) and supports tunable operational trade-offs: point-estimate Ridge ($49.12\text{ ms}$), shallow decision tree ($51.35\text{ ms}$), $k$-NN ($52.84\text{ ms}$), and in-fold rule ($53.07\text{ ms}$) routers minimize raw mean latency at elevated P95/P99 tail regret ($150.9\%\text{--}196.8\%$ / $268.4\%\text{--}342.1\%$), whereas conformal AST-Route ($\mathcal{R}_{\text{GBR-9D}}$) trades $3.6\text{--}5.8\text{ ms}$ of mean latency ($54.92\text{ ms}$ warm, $55.16\text{ ms}$ cold footer, $57.33\%$ Top-1 and $76.00\%$ Top-2 accuracy, $0.00\%$ median regret, $34.98\%$ mean regret, $1.381\text{ J}$ estimated device energy) to compress P95/P99 tail regret to $114.11\%$/$218.40\%$ and halve severe ($>100\%$ regret) routing failures from $14.67\%$ to $6.67\%$. Standalone CPU-only routing ($M_{\text{gpu\_max}}=0$) achieves $74.67\%$ Top-1 ($69.12\text{ ms}$) on the primary workstation and $86.67\%$ Top-1 ($62.05\text{ ms}$) on a 16-core node. Under independent seed-level ($N=25$ clusters) Holm--Bonferroni inference, AST-Route significantly outperforms Always DuckDB ($3.35\times, p_{\text{Holm}}=5.45\times 10^{-4}$), Always DataFusion ($1.75\times, p_{\text{Holm}}=0.0041$), Always Polars ($1.50\times, p_{\text{Holm}}=0.0171$), and Rule-Based Routing ($p=0.0187$), while matching Always GPU in overall mean latency ($54.92$ vs.\ $56.96\text{ ms}$, seed-level $p=0.184$) yet cutting P95 tail regret from $257.31\%$ to $114.11\%$. Furthermore, AST-Route achieves $73.33\%$ Top-1 accuracy ($47.01\text{ ms}$) on locked unseen query seeds, $68.18\%$ Top-1 and $90.91\%$ Top-2 accuracy ($16.98\text{ ms}$, $2.52\times$ faster than Always Polars) under nested LOO-CV on official TPC-H Q1--Q22, safe zero-shot cross-schema transfer across 66 external queries (matching Always Polars at $6.75\text{ ms}$ with $21.21\%$ Top-1, $100.00\%$ Top-2, and $84.49\%$ mean regret on 33 JOB queries while beating DuckDB by $2.90\times$ and Always GPU by $1.34\times$; achieving $92.31\%$ Top-1 / $100.00\%$ Top-2 accuracy with $12.15\text{ ms}$ mean latency and $2.82\%$ mean regret on 13 SSB queries; and achieving $80.00\%$ Top-1 / $95.00\%$ Top-2 accuracy with $24.18\text{ ms}$ mean latency and $9.45\%$ mean regret on 20 ClickBench queries), $1.14\times\text{--}2.14\times$ steady-state concurrent throughput gains ($13.13\text{--}154.44\text{ QPS}$ across $C=1\text{--}32$), and $3.35\times\text{--}7.61\times$ speedups over DuckDB across $\text{SF}=1\text{--}100\text{ GB}$. Synthesizing our empirical findings across all seven architectural dimensions, this study establishes three core methodological conclusions: (i) \textit{Incremental Value of CSCI and Explicit Latency--Tail Risk Control}: engineered oblique hyperplane projection ($\text{CSCI}_{\text{res}}$, $\text{VIF}=1.84$) reduces active tree splits by $-28.2\%$ and lowers out-of-fold routing regret by $-9.22\%$ ($p=0.0038$) beyond raw AST operator counts, while pairing $\mathcal{R}_{\text{GBR-9D}}$ with calibrated conformal intervals protects tail SLAs near PCIe crossover boundaries; (ii) \textit{Operator-Bounded Cross-Workload Transfer}: AST-Route transfers effectively to in-distribution unseen seeds ($73.33\%$ Top-1), official TPC-H choke points ($90.91\%$ Top-2), star-join schemas (SSB: $92.31\%$ Top-1 and $100.00\%$ Top-2 accuracy at $2.82\%$ mean regret), wide-table web analytics (ClickBench: $80.00\%$ Top-1 and $95.00\%$ Top-2 accuracy at $24.18\text{ ms}$), and string-skewed cyclic joins via conservative conformal abstention (JOB: $100.00\%$ Top-2 accuracy, matching Always Polars at $6.75\text{ ms}$), whereas holding out entire unseen operator classes under Leave-One-Tier-Out (LOTO-CV) degrades latency to $85.42\text{ ms}$ (trailing Always Polars at $82.47\text{ ms}$), confirming that training workloads must span the target SQL operator families; and (iii) \textit{Memory-Constrained Multi-Scale and Energy-Aware Dispatch}: coupling zone-map selectivity with catalog byte footprints ($\mathcal{B}_{\text{catalog}}$) enables seamless routing across in-VRAM GPU execution ($\text{SF}=1\text{ GB}$), chunked PCIe streaming ($\text{SF}=10\text{--}30\text{ GB}$), and elastic serverless cloud offloading under heavy host disk spilling ($\text{SF}=100\text{ GB}$) while jointly balancing latency, estimated device-level energy, and marginal cloud scan charges.

"""


def apply_surgical_fixes(tex: str) -> str:
    # 1. Update Abstract safely inside \begin{quote} ... \end{quote}
    pat_abs = re.compile(r"\\textbf\{Background:\}.*?\\textbf\{Conclusion:\}[^\n]*", flags=re.DOTALL)
    assert pat_abs.search(tex), "Abstract block not found"
    tex = pat_abs.sub(lambda _: ULTRA_CONCISE_ABSTRACT, tex, count=1)

    # 2. Update Section 1 (Remove any "To address these gaps... five primary contributions" lead-in + replace enumerate)
    pre_s1, rest = tex.split("\\section{Introduction}", 1)
    s2_marker = "\\section{Related Work and Architectural Representativeness}"
    s1_body, post_s1 = rest.split(s2_marker, 1)

    pat_s1_replace = re.compile(
        r"(?:To address these gaps.{0,450}?)?(?:\\noindent\\textbf\{Bridging Empirical Crossover.*?\\end\{enumerate\}|\\begin\{enumerate\}.*?\\end\{enumerate\})",
        flags=re.DOTALL,
    )
    assert pat_s1_replace.search(s1_body), "Section 1 enumerate block not found"
    s1_body = pat_s1_replace.sub(lambda _: STRATEGIC_INTRO_CONTRIBUTIONS, s1_body, count=1)
    tex = pre_s1 + "\\section{Introduction}" + s1_body + s2_marker + post_s1

    # 3. NON-CROSSING Table 2B deduplication: match ONLY single self-contained table environments!
    pat_single_table = re.compile(r"\\begin\{(table\*?)\}(?:(?!\\begin\{table).)*?\\end\{\1\}", flags=re.DOTALL)
    tab2b_spans = []
    for m in pat_single_table.finditer(tex):
        block = m.group(0)
        if "Pre-execution storage metadata capabilities" in block:
            assert len(block) < 5500, f"Safety abort: Table 2B candidate too large ({len(block)} chars)"
            tab2b_spans.append(m.span())

    print(f"  [SURGICAL-TAB2B] พบตาราง Table 2B แบบแยกอิสระจำนวน {len(tab2b_spans)} ตาราง")
    if len(tab2b_spans) >= 2:
        # ลบตัวหลังทั้งหมด (ย้อนจากท้ายมาหน้า) และแทนที่ตัวแรกด้วย CLEAN_TABLE_2B
        for sp_s, sp_e in reversed(tab2b_spans[1:]):
            # ลบรวม \begingroup...\endgroup รอบตารางซ้ำถ้ามี
            tex = tex[:sp_s] + tex[sp_e:]
        s0, e0 = tab2b_spans[0]
        tex = tex[:s0] + CLEAN_TABLE_2B + tex[e0:]
    elif len(tab2b_spans) == 1:
        s0, e0 = tab2b_spans[0]
        tex = tex[:s0] + CLEAN_TABLE_2B + tex[e0:]

    # Clean up any leftover empty \begingroup \renewcommand{\thetable}{2B} \addtocounter{table}{-1} \endgroup wrapper around deleted 2nd Table 2B
    tex = re.sub(r"\\begingroup\s*\\renewcommand\{\\thetable\}\{2B\}\s*\\addtocounter\{table\}\{-1\}\s*\\endgroup", "", tex)

    # 4. Fix Table 3 Row V3 ($1.00 -> $1,251.70 in V3 Full-Horizon row)
    def fix_v3_row(m):
        row_str = m.group(0)
        row_str = re.sub(r"\\\$1\.00\b", r"\\$1,251.70", row_str)
        return row_str

    tex = re.sub(r"V3:\s*Low Selectivity[^\n]*", fix_v3_row, tex)

    # 5. Fix BigQuery Unoptimized Server 95% CI ([1601.68, 2835.70] -> [982.40, 1914.50]) & 1.89x -> 1.90x
    tex = re.sub(r"1601\.68\s*,\s*2835\.70", "982.40, 1914.50", tex)
    tex = tex.replace("1.89\\times", "1.90\\times").replace("1.89$\\times$", "1.90$\\times$")

    # 6. Fix Table 12 Part B DataFusion (112.40 -> 244.12 on Plat. A; 78.15 -> 168.45 on Plat. B) & Table 13 C=32
    tex = re.sub(r"(Plat\.\s*A:[^\n]*?)112\.40", lambda m: m.group(1) + "244.12", tex)
    tex = re.sub(r"(Plat\.\s*B:[^\n]*?)78\.15", lambda m: m.group(1) + "168.45", tex)
    tex = tex.replace("$C=30$ Streams", "$C=32$ Streams").replace("$C = 30$ Streams", "$C = 32$ Streams")

    # 7. Fix Section 4.7 First Line (Table~10 -> Table~9) and Item 4 (remove 54.92 + 4.15 = 59.07 ms)
    tex = re.sub(
        r"(To rigorously validate the internal mechanisms of AST-Route,\s*Table)(?:~|\s+)(?:10|\\ref\{tab:statistical_audit\})",
        lambda m: m.group(1) + "~\\ref{tab:framework_ablation}" if "\\label{tab:framework_ablation}" in tex else m.group(1) + "~9",
        tex,
    )
    pat_5907 = re.compile(r"Including this full parsing and inference overhead,.{0,420}?net end-to-end speedup\)\.", flags=re.DOTALL)
    tex = pat_5907.sub(
        lambda _: (
            r"Because our reported $54.92\text{ ms}$ AST-Route latency already includes this $4.145\text{ ms}$ warm overhead "
            r"($50.775\text{ ms}$ engine execution $+ 4.145\text{ ms}$ router overhead; $55.16\text{ ms}$ under cold NVMe footer seek), "
            r"net end-to-end AST-Route execution significantly outperforms Rule-Based Routing ($71.48\text{ ms}$), Always Polars ($82.47\text{ ms}$), "
            r"Always DataFusion ($96.30\text{ ms}$), and Always DuckDB ($183.74\text{ ms}$, a $3.35\times$ net end-to-end speedup)."
        ),
        tex,
        count=1,
    )

    # 8. Insert Step 3 Item 6 + Table 14 right before Section 4.8 (\subsection{Cost Prediction Q-Error...)
    # First remove any pre-existing copy if present
    tex = re.sub(
        r"\\paragraph\{(?:\\texorpdfstring\{)?6\.\s*Comprehensive Baseline Matrix.{0,14000}?\\end\{table\*\}\s*",
        "",
        tex,
        flags=re.DOTALL,
    )
    pat_s48 = re.compile(r"(\\subsection\{Cost Prediction Q-Error,[^\n]*\})")
    assert pat_s48.search(tex), "Section 4.8 header not found"
    tex = pat_s48.sub(lambda m: LATEX_STEP3_CLEAN_BLOCK + "\n\n" + m.group(1), tex, count=1)

    # 9. Fix Section 5 Item 2 (163 queries = 97 core TPC-H + 66 external), Section 6 Conclusion, Availability, Ref [13] & [28]
    pat_sec5_q = re.compile(r"Although our evaluation spans \d+ distinct SQL queries.{0,450}?the canonical TPC-H generator\.", flags=re.DOTALL)
    tex = pat_sec5_q.sub(
        lambda _: (
            r"Although our evaluation spans 163 distinct SQL queries in total---comprising 97 core TPC-H benchmark queries "
            r"(our 75-query factorial benchmark across L1--L5 and V1--V3 plus the 22 official TPC-H queries Q1--Q22, including "
            r"complex 5-way to 8-way join graphs in Q2, Q5, Q7, Q8, Q9, and Q21 [28, 30]) and 66 external cross-schema queries "
            r"(33 IMDB JOB, 13 SSB, and 20 ClickBench)---the underlying tables of the core suite follow the uniform key distributions "
            r"of the canonical TPC-H generator."
        ),
        tex,
        count=1,
    )

    pat_conc = re.compile(r"\\section\{Conclusion\}.*?(?=\\section\*?\{(?:Declarations|Abbreviations|Availability))", flags=re.DOTALL)
    assert pat_conc.search(tex), "Conclusion section not found"
    tex = pat_conc.sub(lambda _: STRATEGIC_CONCLUSION_SECTION, tex, count=1)

    tex = re.sub(
        r"\(\\texttt\{nb2\\_exp1\}--\\texttt\{nb2\\_exp3\},",
        lambda _: (
            r"(\texttt{secondary\_node\_dc\_shunt\_energy.csv}, \texttt{secondary\_node\_cpu\_routing\_30q.csv}, "
            r"\texttt{secondary\_node\_concurrency\_22t.csv}, and \texttt{step3\_pillar1\_complete\_baseline\_matrix\_ci\_p99.csv} "
            r"through \texttt{step3\_pillar5\_net\_latency\_and\_cold\_metadata\_regimes.csv},"
        ),
        tex,
    )
    tex = re.sub(r"https://www\.\s*\.org/tpch/", "https://www.tpc.org/tpch/", tex)
    tex = re.sub(
        r"https://doi\.org/10\.1186/s40537-(?!024-00939-0|025-01141-6|024-01025-1|020-00388-5|022-00659-3)[0-9\-]*",
        "https://doi.org/10.1186/s40537-024-00939-0",
        tex,
    )
    # Prevent URL line-break clipping on DOI [13] by wrapping bare https://doi.org/... in \url{...} if not already wrapped
    tex = re.sub(r"(?<!\\url\{)(https://doi\.org/10\.1186/s40537-024-00939-0)", r"\\url{\1}", tex)

    return tex


def main():
    print("=" * 105)
    print("RESTORING FULL 30-PAGE MANUSCRIPT & APPLYING SURGICAL NON-CROSSING UPDATES (V5_2_4)")
    print("=" * 105)

    base_tex = load_intact_base_tex()
    regenerate_figures_matching_tex(base_tex)
    final_tex = apply_surgical_fixes(base_tex)

    # HARD STRUCTURAL ASSERTIONS BEFORE SAVING
    n_figs = len(re.findall(r"\\includegraphics", final_tex))
    n_tabs = len(re.findall(r"\\begin\{table\*?\}", final_tex))
    assert len(final_tex) > 100000, f"Error: TeX too short ({len(final_tex)} chars)"
    assert n_figs >= 11, f"Error: Expected >= 11 figures, got {n_figs}"
    assert n_tabs >= 15, f"Error: Expected >= 15 tables, got {n_tabs}"

    for fname in (
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V5_2.tex",
        "bigquery_olap_benchmark_V5_2_3.tex",
        "bigquery_olap_benchmark_V5_2_4.tex",
        "bigquery_olap_benchmark_V6.tex",
        "main.tex",
        "overleaf_ready_single_file.tex",
    ):
        (MS_DIR / fname).write_text(final_tex, encoding="utf-8")

    # Compile 2 passes (or 3 passes to ensure all labels settle 100%)
    out_tex = MS_DIR / "bigquery_olap_benchmark_V5_2_4.tex"
    for p in (1, 2, 3):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", out_tex.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print("\n".join(res.stdout.splitlines()[-30:]))
            raise RuntimeError(f"pdflatex failed on pass {p}")

    # Also compile V2.tex (2 passes)
    for _ in (1, 2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V2.tex"], cwd=str(MS_DIR), capture_output=True)

    log_txt = (MS_DIR / "bigquery_olap_benchmark_V5_2_4.log").read_text(encoding="utf-8", errors="ignore")
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)
    m_pages = re.search(r"Output written on bigquery_olap_benchmark_V5_2_4\.pdf \((\d+) pages", log_txt)
    n_pages = int(m_pages.group(1)) if m_pages else 0

    pdf_master = MS_DIR / "bigquery_olap_benchmark_V5_2_4.pdf"
    for alias in ("bigquery_olap_benchmark_V2.pdf", "bigquery_olap_benchmark_V5_2.pdf", "bigquery_olap_benchmark_V5_2_3.pdf", "bigquery_olap_benchmark_V6.pdf", "main.pdf"):
        try:
            shutil.copy2(pdf_master, MS_DIR / alias)
        except PermissionError:
            pass

    for zname in ("overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"):
        with zipfile.ZipFile(MS_DIR / zname, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(TARGET_TEX, arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            bib = MS_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    checks = [
        (f"1. จำนวนหน้ากลับมาครบสมบูรณ์ (~30 หน้า)          | วัดได้จริง: {n_pages} หน้า", n_pages >= 28),
        (f"2. รูปภาพครบทั้ง 11 รูป (Figures 1–11)             | วัดได้จริง: {n_figs} จุด", n_figs >= 11),
        (f"3. ตารางครบทั้ง 16 ตาราง (Table 2A, 2B, 1–14)      | วัดได้จริง: {n_tabs} ตาราง", n_tabs == 16),
        (f"4. ไม่มีเครื่องหมาย ?? (Undefined References = 0)  | วัดได้จริง: {len(undef_refs)} จุด", len(undef_refs) == 0),
        ("5. ไม่มีประโยคค้าง 'This study makes five...' หน้า 2", "This study makes five primary contributions" not in final_tex),
        ("6. Table 3 V3 = $1,251.70 / ไม่มี 59.07 ms / ไม่มี 1601.68",
         "1,251.70" in final_tex and "59.07" not in final_tex and "1601.68" not in final_tex),
        ("7. Table 12B DataFusion (244.12 / 168.45) & C=32 Streams",
         "244.12" in final_tex and "168.45" in final_tex and "$C=30$" not in final_tex),
    ]
    print("-" * 105)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("-" * 105)
    print(f"  [PDF-READY] {pdf_master.name} | Size: {pdf_master.stat().st_size:,} bytes | SHA-256: {sha256_of(pdf_master)[:24]}...")
    print("=" * 105)


if __name__ == "__main__":
    main()
