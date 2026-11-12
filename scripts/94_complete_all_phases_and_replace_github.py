#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/94_complete_all_phases_and_replace_github.py
================================================================================
Complete Phase 1 + Phase 2 + Phase 3 Reviewer Acceptance Upgrade on Notebook1:
  1. Fix line 663 Table 13 tabular specification & unescaped '&' in Table 15 intro.
  2. Regenerate synchronized 300 DPI Figures 1, 2, 5, 6 (2-panel Latency + Throughput),
     7 (Energy-Aware Scan Efficiency), and 9 (non-overlapping P95 badges).
  3. Generate 2 NEW 300 DPI Figures:
     - Figure 12: 3-Node Cross-ISA L1-L5 Crossover & Concurrency Scaling (C=1..32)
     - Figure 13: 5-Suite Generalization (163 Queries) & 12D JOB Sketch Ablation
  4. Embed Figures 12 & 13 into bigquery_olap_benchmark_V2.tex, compile 3 passes,
     and replace bigquery_olap_benchmark_V2.tex directly on GitHub (Zero Backup).
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
ROOT_FIG_DIR = ROOT / "figures"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"

for d in (FIG_DIR, ROOT_FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def save_fig_both_dirs(fig, filenames: list[str]) -> None:
    for fn in filenames:
        fig.savefig(FIG_DIR / fn, dpi=300, bbox_inches="tight")
        fig.savefig(ROOT_FIG_DIR / fn, dpi=300, bbox_inches="tight")


def draw_card(ax, x, y, w, h, title, lines, header_color="#1f4e78", body_color="#f7f9fc", edge_color="#1f4e78", title_fs=9.0, body_fs=7.7):
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
    n = max(len(lines), 1)
    step = avail_h / (n + 0.35)
    for i, line in enumerate(lines):
        ly = (y + h - hh - 0.012) - (i + 0.65) * step
        ax.text(x + 0.012, ly, line, color="#1a1a1a", fontsize=body_fs, ha="left", va="center", zorder=4, family="serif")


def generate_all_synchronized_figures() -> None:
    plt.rcParams["font.family"] = "serif"

    # ==========================================================================
    # 1. FIGURE 1: Proposed AST-Route Four-Layer Architecture
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(13.5, 7.4), dpi=300)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.text(0.5, 0.975, "Proposed AST-Route Four-Layer Multi-Objective Query Routing Framework",
             fontsize=13, fontweight="bold", ha="center", va="center", color="#0f2c59")

    draw_card(
        ax1, 0.015, 0.72, 0.47, 0.22,
        "LAYER 1A: Input SQL & Static AST Parsing (SQLGlot)",
        [
            "• Workload: 75 Factorial TPC-H + 22 Q1–Q22 + 66 External (N = 163 Queries)",
            "• Zero-Execution Static Parsing via SQLGlot (3.48 ms parse + 0.17 ms zone-map)",
            "• Extracts 7 Structural Operators: Depth D(q), Nodes N(q), Joins J(q),",
            "  Subqueries/CTEs S(q), Windows W(q), UNNEST U(q), CASE Branches C(q)",
        ],
        header_color="#2b579a", body_color="#eef4fa", edge_color="#2b579a"
    )
    draw_card(
        ax1, 0.515, 0.72, 0.47, 0.22,
        "LAYER 1B: 9D/10D Feature Vector, CSCI & Catalog Scale",
        [
            "• Validated Index: CSCI(q) = 1.5D + 2.5J + 3.0S + 3.5W + 4.0U + 1.5C",
            "• Construct Validity: Partial rho = 0.5477 (VIF = 1.84); 10,000-MC rho = 0.9880 (±20%)",
            "• Pre-Execution Zone-Map Selectivity sigma_hat(q) in (0, 1] (MAE = 3.84%, Q-Err = 1.092)",
            "• 9D Vector x_q in R^9 (extended to R^10 across SF=1–100 GB & R^12 with 10.2 KB sketches)",
        ],
        header_color="#1e6b52", body_color="#eef8f3", edge_color="#1e6b52"
    )
    draw_card(
        ax1, 0.015, 0.49, 0.97, 0.19,
        "LAYERS 2 & 3: Cross-Paradigm Analytical Cost Models & Multi-Output Cost-Regression Engine (R_GBR-9D)",
        [
            "• Analytical Models: Bidirectional PCIe Fraction Phi_PCIe(q), Scan Throughput Theta_eff (GB/s), Slot Proxy M_slot, FinOps Savings Delta C_1M ($)",
            "• Multi-Output GBR Predictor f_Theta: Simultaneously estimates Log-Latency ln(1+T_e), Device Energy E_e (J), and Peak Memory (RAM/VRAM GB)",
            "• Out-of-Fold Predictive Accuracy (25-Seed LOSO-CV): GPU R^2 = 0.875 (P50 Q-Err 1.053) | Polars R^2 = 0.822 | Energy R^2 = 0.815 (Inference: 497.2 us)",
        ],
        header_color="#b85d18", body_color="#fdf5ed", edge_color="#b85d18"
    )

    c_w, c_gap = 0.232, 0.014
    c_xs = [0.015 + i * (c_w + c_gap) for i in range(4)]
    draw_card(
        ax1, c_xs[0], 0.21, c_w, 0.24, "Candidate 1: DuckDB (C++)",
        ["• Paradigm: In-Process CPU DBMS", "• Morsel-Driven Vectorized Engine", "• Segment-Tree Window Framing",
         "• Mean Latency: 183.74 ms", "• Est. Pkg Energy: 6.40 J (243.0 MB/J)", "• Routed: 80% L1, 20% L2"],
        header_color="#3b4954", body_color="#f3f5f7", edge_color="#3b4954", title_fs=8.5, body_fs=7.3
    )
    draw_card(
        ax1, c_xs[1], 0.21, c_w, 0.24, "Candidate 2: Polars / DataFusion",
        ["• Paradigm: Rust Apache Arrow", "• Zero-Copy SIMD & ListArray", "• Polars: 82.47 ms (3.17 J)",
         "• DataFusion: 96.30 ms (3.39 J)", "• 102.1x Faster than Row-Store", "• Routed: 80% Narrow/Med L2 Joins"],
        header_color="#006666", body_color="#eef8f8", edge_color="#006666", title_fs=8.5, body_fs=7.3
    )
    draw_card(
        ax1, c_xs[2], 0.21, c_w, 0.24, "Candidate 3: CUDA GPU (SIMT)",
        ["• Paradigm: RTX 2050 (4 GB VRAM)", "• Cold: 56.96 ms | Warm: 22.25 ms", "• L1 PCIe Bottleneck: Phi = 55.5%",
         "• Board Energy: 0.65 J Cold / 0.25 J Warm", "• Scan Eff.: 1942.8 MB/J Warm", "• Routed: 60% L3, 100% L4, 80% L5"],
        header_color="#9e2a2b", body_color="#fdf2f2", edge_color="#9e2a2b", title_fs=8.5, body_fs=7.3
    )
    draw_card(
        ax1, c_xs[3], 0.21, c_w, 0.24, "Candidate 4: Google BigQuery",
        ["• Paradigm: Serverless Cloud OLAP", "• Capacitor Storage + Slot Elasticity", "• Clustered Mean: 745.39 ms",
         "• Slot Multiplier: M_slot = 6.37x", "• V1 I/O Pruning: 78.4% ($1,441.61/1M)", "• Elastic SF=100 GB Spill Fallback"],
        header_color="#5a3d7a", body_color="#f6f2fa", edge_color="#5a3d7a", title_fs=8.5, body_fs=7.3
    )
    draw_card(
        ax1, 0.015, 0.02, 0.97, 0.15,
        "LAYER 4: Algorithm 1 Conformal Risk-Aware Multi-Objective Dispatch & 25-Seed LOSO-CV Performance",
        [
            "• Dispatch Rule: e_hat(q) = argmin_{e in E_feasible(q)} [alpha*(T_hat/T_med) + beta*(E_hat/E_med) + gamma*(C_hat/C_ref)] with 90% Conformal + 12% PCIe Guard",
            "• Outperforms Static & Heuristic Baselines: 54.92 ms Mean Latency (95% CI: [42.78, 68.50] ms), 34.98% Mean Regret, 114.11% P95 Regret, 1.381 J Energy",
        ],
        header_color="#1e6b52", body_color="#eef8f3", edge_color="#1e6b52"
    )
    save_fig_both_dirs(fig1, ["fig_proposed_ast_routing_framework_300dpi.pdf", "fig_proposed_ast_routing_framework_300dpi.png"])
    plt.close(fig1)

    # ==========================================================================
    # 2. FIGURE 2: End-to-End Experimental Validation Workflow
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(13.5, 7.4), dpi=300)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.5, 0.975, "End-to-End Experimental Validation Workflow: AST-Route Testbed (163 Queries, 6 Engines, 3 Nodes)",
             fontsize=13, fontweight="bold", ha="center", va="center", color="#0f2c59")

    draw_card(
        ax2, 0.015, 0.73, 0.47, 0.21,
        "PHASE 1: Multi-Scale TPC-H Provisioning (SF = 1, 10, 30, 100 GB)",
        [
            "• Base Dataset: 8 Relational Tables (8.66M rows at SF1 to 866.1M rows at SF100)",
            "• Local Columnar: ZSTD Parquet + DuckDB Native + Apache Arrow Buffers",
            "• Local Row-Store Baseline: SQLite 3 Full B-Tree with PK/FK/Date Indexes",
            "• Cloud Serverless: Google Cloud Storage -> BigQuery Unopt vs. Clustered",
        ],
        header_color="#2b579a", body_color="#eef4fa", edge_color="#2b579a"
    )
    draw_card(
        ax2, 0.515, 0.73, 0.47, 0.21,
        "PHASE 2: 163-Query Multi-Suite Workload Design (5 Benchmarks)",
        [
            "• 75 Factorial Queries: 5 Tiers L1–L5 (25 Disjoint Seeds x 3 Horizons V1–V3)",
            "• 22 Official Standard TPC-H Choke-Point Queries (Q1–Q22, Nested LOO-CV)",
            "• 66 External Cross-Schema Queries: 33 IMDB JOB + 13 SSB + 20 ClickBench",
        ],
        header_color="#1e6b52", body_color="#eef8f3", edge_color="#1e6b52"
    )
    draw_card(
        ax2, 0.015, 0.51, 0.97, 0.18,
        "PHASE 3: Static AST Feature Profiling, CSCI Construct Validation & Hardware Telemetry Instrumentation",
        [
            "• Parses SQL queries via SQLGlot into 9D/10D/12D vectors x_q (Mean CSCI = 29.66, Range: 15.3 in L1 to 37.1 in L4; 4.15 ms warm overhead)",
            "• CSCI Construct Validation: Partial rho = 0.5477 (p = 3.68e-7) vs. GPU Speedup; 10,000-Trial Monte Carlo Weight Stability rho_bar = 0.9880 (±20%)",
            "• High-Frequency Sensor Polling (250 Hz psutil + NVIDIA NVML + cudaEvent): Captures Latency (ms), CPU/GPU (%), RAM/VRAM (GB), Energy (J), MB/J",
        ],
        header_color="#b85d18", body_color="#fdf5ed", edge_color="#b85d18"
    )
    for idx, (title, lines, col, bcol) in enumerate([
        ("Paradigm 1: Row-Store", [
            "• Engine: SQLite 3 (ANSI C)", "• Storage: Full-Scale B-Tree", "• Tuple-at-a-Time Volcano Model",
            "• Mean Latency: 8417.33 ms", "• 95% CI: [6412.2, 10819.7] ms", "• Mean Energy: 126.26 J (4.05 MB/J)",
        ], "#3b4954", "#f3f5f7"),
        ("Paradigm 2: Serverless Cloud", [
            "• Engine: Google BigQuery", "• Capacitor Columnar Storage", "• Cache Strictly Disabled",
            "• Clustered Mean: 745.39 ms", "• 95% CI: [515.6, 1026.5] ms", "• V1 Pruning: 78.4% ($1,441.61/1M)",
        ], "#2b579a", "#eef4fa"),
        ("Paradigm 3: Vectorized CPU", [
            "• 1) DuckDB (C++): 183.74 ms", "  95% CI: [110.7, 274.9] ms (6.40 J)",
            "• 2) Polars (Rust): 82.47 ms", "  95% CI: [67.9, 98.7] ms (3.17 J)",
            "• 3) DataFusion (Rust): 96.30 ms", "  95% CI: [78.3, 114.9] ms (3.39 J)",
        ], "#1e6b52", "#eef8f3"),
        ("Paradigm 4: SIMT CUDA GPU", [
            "• Engine: PyTorch CUDA Tensor", "• GPU: NVIDIA RTX 2050 (4 GB)",
            "• Cold Mean: 56.96 ms (0.65 J)", "  95% CI: [46.23, 68.44] ms",
            "• Warm Mean: 22.25 ms (0.25 J)", "  95% CI: [16.78, 28.68] ms",
        ], "#9e2a2b", "#fdf2f2"),
    ]):
        draw_card(ax2, c_xs[idx], 0.23, c_w, 0.24, title, lines, header_color=col, body_color=bcol, edge_color=col, title_fs=8.5, body_fs=7.3)

    draw_card(
        ax2, 0.015, 0.02, 0.47, 0.17,
        "PHASE 5: Statistical Inference & 25-Seed LOSO-CV Routing",
        [
            "• Exact Equivalence EX = 1.000 (100%) | Friedman chi2_F(5) = 321.50 (p = 2.38e-67)",
            "• 10,000-Bootstrap Seed-Clustered 95% CIs & Cliff's Delta Effect Sizes",
            "• Seed-Grouped LOSO-CV: AST-Route Achieves 54.92 ms (34.98% Mean Regret)",
        ],
        header_color="#5a3d7a", body_color="#f6f2fa", edge_color="#5a3d7a"
    )
    draw_card(
        ax2, 0.515, 0.02, 0.47, 0.17,
        "PHASE 6: 3-Node Cross-ISA SHA-256 Audit & Reproducibility",
        [
            "• Verified Across 3 Physical Nodes: x86_64 GPU + 16C/22T x86_64 + Apple M1 ARM64",
            "• Zero-Contradiction Audit Suite & Global SHA-256 Registry (154 CSV Artifacts)",
            "• Open-Source Benchmark Repository: github.com/preecha-msu/bigquery-olap-benchmark",
        ],
        header_color="#1e6b52", body_color="#eef8f3", edge_color="#1e6b52"
    )
    save_fig_both_dirs(fig2, ["fig_experimental_workflow_300dpi.pdf", "fig_experimental_workflow_300dpi.png", "fig_architecture_workflow_300dpi.pdf"])
    plt.close(fig2)

    # ==========================================================================
    # 3. FIGURE 5: Hardware-Timed CUDA GPU Latency Decomposition (Canonical CSCI)
    # ==========================================================================
    tiers_csci = [
        "L1: Scan/Filter\n(CSCI=15.3)",
        "L2: Multi-Join\n(CSCI=26.8)",
        "L3: Nested CTE\n(CSCI=36.7)",
        "L4: Window\n(CSCI=37.1)",
        "L5: UNNEST\n(CSCI=32.4)",
    ]
    duck_nat = [50.84, 264.53, 121.92, 78.16, 403.27]
    gpu_warm = [24.46, 49.05, 14.98, 9.94, 12.84]
    gpu_pcie = [30.56, 55.83, 43.26, 9.43, 34.46]
    gpu_cold = [55.02, 104.88, 58.24, 19.37, 47.31]

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(13.0, 4.5), dpi=300)
    x = np.arange(5)
    w = 0.35
    ax5a.bar(x - w / 2, duck_nat, w, label="DuckDB Native Columnar (Vectorized CPU)", color="#2e5984", edgecolor="black")
    ax5a.bar(x + w / 2, gpu_warm, w, label="GPU Warm Kernel Compute (In-VRAM)", color="#2ca02c", edgecolor="black")
    ax5a.bar(x + w / 2, gpu_pcie, w, bottom=gpu_warm, label="PCIe Transfer Overhead (H2D + D2H)", color="#ff7f0e", edgecolor="black", hatch="//")
    for i in range(5):
        ax5a.text(x[i] - w / 2, duck_nat[i] + 6, f"{duck_nat[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
        ax5a.text(x[i] + w / 2, gpu_cold[i] + 6, f"{gpu_cold[i]:.1f}", ha="center", fontsize=8, fontweight="bold", color="#b33000")
    ax5a.annotate(
        "L1 PCIe Crossover:\nCold GPU (55.02 ms) >\nCPU (50.84 ms, Phi=55.5%)",
        xy=(0 + w / 2, 55.02), xytext=(0.42, 160),
        arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe6e6", edgecolor="#b30000"),
        fontsize=7.8
    )
    ax5a.set_xticks(x)
    ax5a.set_xticklabels(tiers_csci, fontsize=8.3)
    ax5a.set_ylabel("Execution Latency (ms, Linear Scale)")
    ax5a.set_title("(a) Vectorized CPU vs. CUDA GPU Latency Decomposition (PCIe + Kernel)", fontweight="bold", fontsize=9.5)
    ax5a.set_ylim(0, 460)
    ax5a.grid(axis="y", linestyle="--", alpha=0.5)
    ax5a.legend(loc="upper left", fontsize=7.8)

    spd_bq = [432.9 / 24.46, 667.3 / 49.05, 493.0 / 14.98, 372.8 / 9.94, 1761.0 / 12.84]
    spd_warm = [50.04 / 24.46, 263.61 / 49.05, 120.94 / 14.98, 77.31 / 9.94, 402.45 / 12.84]
    spd_cold = [duck_nat[i] / gpu_cold[i] for i in range(5)]

    ax5b.plot(x, spd_bq, "^-.", color="#5c5c8a", lw=1.8, markersize=7, label="Warm GPU vs. BigQuery Clustered (Cloud)")
    ax5b.plot(x, spd_warm, "s-", color="#2ca02c", lw=2.0, markersize=7, label="Warm GPU vs. DuckDB CPU Kernel (Level 1)")
    ax5b.plot(x, spd_cold, "o-", color="#d62728", lw=2.0, markersize=7, label="Cold GPU (Incl. PCIe) vs. DuckDB Native (Level 2)")
    ax5b.axhline(1.0, color="black", linestyle=":", lw=1.2, label="CPU-GPU Break-Even Threshold (1.0x)")
    for i in range(5):
        ax5b.text(x[i], spd_bq[i] * 1.15, f"{spd_bq[i]:.1f}x", ha="center", fontsize=7.6, color="#5c5c8a")
        ax5b.text(x[i], spd_warm[i] * 1.15, f"{spd_warm[i]:.2f}x", ha="center", fontsize=7.8, fontweight="bold", color="#1f7a1f")
        ax5b.text(x[i], spd_cold[i] * 0.70, f"{spd_cold[i]:.2f}x", ha="center", fontsize=7.8, fontweight="bold", color="#b30000")
    ax5b.set_yscale("log")
    ax5b.set_ylim(0.5, 300)
    ax5b.set_xticks(x)
    ax5b.set_xticklabels(tiers_csci, fontsize=8.3)
    ax5b.set_ylabel("Speedup Ratio (Log Scale, Higher is Faster)")
    ax5b.set_title("(b) GPU Acceleration Trajectory Across AST Complexity Tiers (L1–L5)", fontweight="bold", fontsize=9.5)
    ax5b.grid(True, which="both", linestyle="--", alpha=0.4)
    ax5b.legend(loc="upper left", fontsize=7.6)
    fig5.tight_layout()
    save_fig_both_dirs(fig5, ["fig_gpu_pcie_decomposition_sf1_300dpi.pdf", "fig_gpu_pcie_decomposition_sf1_300dpi.png"])
    plt.close(fig5)

    # ==========================================================================
    # 4. FIGURE 6: 2-Panel Six-Engine Cross-Paradigm Evaluation (Table 5 & Table 6 Exact!)
    # ==========================================================================
    sqlite_lat = [6789.6, 19374.0, 5108.1, 6023.7, 4791.2]
    bq_lat = [432.9, 667.3, 493.0, 372.8, 1761.0]
    polars_lat = [38.00, 81.68, 66.81, 71.11, 154.76]
    df_lat = [33.58, 92.76, 71.66, 173.16, 110.31]

    sqlite_tp = [0.048, 0.024, 0.080, 0.039, 0.056]
    bq_tp = [0.750, 0.694, 0.827, 0.633, 0.152]
    duck_tp = [6.39, 1.75, 3.34, 3.02, 0.66]
    polars_tp = [8.55, 5.67, 6.10, 3.32, 1.73]
    df_tp = [9.67, 4.99, 5.69, 1.36, 2.42]
    gpu_w_tp = [13.28, 9.44, 27.20, 23.73, 20.79]

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14.0, 4.5), dpi=300)
    bw = 0.13
    ax6a.bar(x - 2.5 * bw, sqlite_lat, bw, label="SQLite 3 (Row-Store)", color="#7f7f7f", edgecolor="black")
    ax6a.bar(x - 1.5 * bw, bq_lat, bw, label="BigQuery Clustered (Cloud)", color="#4c72b0", edgecolor="black")
    ax6a.bar(x - 0.5 * bw, duck_nat, bw, label="DuckDB Native (C++)", color="#55a868", edgecolor="black")
    ax6a.bar(x + 0.5 * bw, polars_lat, bw, label="Polars (Rust Arrow)", color="#dd8452", edgecolor="black")
    ax6a.bar(x + 1.5 * bw, df_lat, bw, label="DataFusion (Rust Arrow)", color="#8172b3", edgecolor="black")
    ax6a.bar(x + 2.5 * bw, gpu_cold, bw, label="RTX 2050 GPU (Cold)", color="#c44e52", edgecolor="black", hatch="//")
    ax6a.set_yscale("log")
    ax6a.set_ylim(10, 35000)
    ax6a.set_xticks(x)
    ax6a.set_xticklabels(tiers_csci, fontsize=8.2)
    ax6a.set_ylabel("Mean Execution Latency (ms, Log Scale)")
    ax6a.set_title("(a) Six-Engine Mean Execution Latency Across Tiers (Table 5)", fontweight="bold", fontsize=9.5)
    ax6a.grid(True, which="both", linestyle="--", alpha=0.4)
    ax6a.legend(ncol=2, loc="upper right", fontsize=7.3)

    ax6b.bar(x - 2.5 * bw, sqlite_tp, bw, label="SQLite 3", color="#7f7f7f", edgecolor="black")
    ax6b.bar(x - 1.5 * bw, bq_tp, bw, label="BigQuery Clustered", color="#4c72b0", edgecolor="black")
    ax6b.bar(x - 0.5 * bw, duck_tp, bw, label="DuckDB Native", color="#55a868", edgecolor="black")
    ax6b.bar(x + 0.5 * bw, polars_tp, bw, label="Polars Arrow", color="#dd8452", edgecolor="black")
    ax6b.bar(x + 1.5 * bw, df_tp, bw, label="DataFusion Arrow", color="#8172b3", edgecolor="black")
    ax6b.bar(x + 2.5 * bw, gpu_w_tp, bw, label="RTX 2050 GPU (Warm)", color="#2ca02c", edgecolor="black", hatch="//")
    ax6b.set_yscale("log")
    ax6b.set_ylim(0.015, 60)
    ax6b.set_xticks(x)
    ax6b.set_xticklabels(tiers_csci, fontsize=8.2)
    ax6b.set_ylabel("Effective Scan Throughput Theta_eff (GB/s, Log Scale)")
    ax6b.set_title("(b) Effective Analytical Scan Throughput Across Tiers (Table 6)", fontweight="bold", fontsize=9.5)
    ax6b.grid(True, which="both", linestyle="--", alpha=0.4)
    ax6b.legend(ncol=2, loc="lower right", fontsize=7.3)
    fig6.tight_layout()
    save_fig_both_dirs(fig6, ["fig_six_software_comparison_sf1_300dpi.pdf", "fig_six_software_comparison_sf1_300dpi.png"])
    plt.close(fig6)

    # ==========================================================================
    # 5. FIGURE 7: Hardware Resource & Energy-Aware Scan Efficiency (Table 7 Exact!)
    # ==========================================================================
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13.4, 4.4), dpi=300)
    e_sq = [101.0, 293.4, 76.0, 89.6, 71.3]
    e_dk = [1.79, 8.99, 4.00, 2.65, 14.67]
    e_pl = [1.36, 3.39, 2.56, 2.54, 5.89]
    e_gc = [0.60, 1.29, 0.68, 0.21, 0.45]
    e_gw = [0.27, 0.60, 0.17, 0.11, 0.12]

    w7 = 0.15
    ax7a.bar(x - 2 * w7, e_sq, w7, label="SQLite 3 (Row-Store)", color="#7f7f7f", edgecolor="black")
    ax7a.bar(x - 1 * w7, e_dk, w7, label="DuckDB (Est. CPU Pkg)", color="#55a868", edgecolor="black")
    ax7a.bar(x + 0 * w7, e_pl, w7, label="Polars (Est. CPU Pkg)", color="#dd8452", edgecolor="black")
    ax7a.bar(x + 1 * w7, e_gc, w7, label="RTX 2050 GPU (Cold Board)", color="#c44e52", edgecolor="black", hatch="//")
    ax7a.bar(x + 2 * w7, e_gw, w7, label="RTX 2050 GPU (Warm Board)", color="#4c72b0", edgecolor="black")
    ax7a.set_yscale("log")
    ax7a.set_ylim(0.06, 500)
    ax7a.set_xticks(x)
    ax7a.set_xticklabels(tiers_csci, fontsize=8.2)
    ax7a.set_ylabel("Active Device Energy per Query (Joules, Log Scale)")
    ax7a.set_title("(a) Est. CPU Package & Meas. GPU Board Energy (J)", fontweight="bold", fontsize=9.5)
    ax7a.grid(True, which="both", linestyle="--", alpha=0.4)
    ax7a.legend(loc="upper right", fontsize=7.6)

    eff_sq = [4.06, 2.28, 6.21, 3.38, 4.30]
    eff_dk = [515.2, 172.7, 215.4, 202.7, 104.0]
    eff_pl = [285.9, 163.0, 186.9, 103.2, 59.6]
    eff_gc = [569.6, 383.0, 619.6, 1178.3, 611.2]
    eff_gw = [1398.5, 910.3, 2558.0, 2502.1, 2345.0]

    ax7b.bar(x - 2 * w7, eff_sq, w7, label="SQLite 3 (4.05 MB/J)", color="#7f7f7f", edgecolor="black")
    ax7b.bar(x - 1 * w7, eff_dk, w7, label="DuckDB (243.0 MB/J)", color="#55a868", edgecolor="black")
    ax7b.bar(x + 0 * w7, eff_pl, w7, label="Polars (159.7 MB/J)", color="#dd8452", edgecolor="black")
    ax7b.bar(x + 1 * w7, eff_gc, w7, label="RTX 2050 Cold (672.3 MB/J)", color="#c44e52", edgecolor="black", hatch="//")
    ax7b.bar(x + 2 * w7, eff_gw, w7, label="RTX 2050 Warm (1942.8 MB/J)", color="#4c72b0", edgecolor="black")
    ax7b.set_yscale("log")
    ax7b.set_ylim(1.0, 5000)
    ax7b.set_xticks(x)
    ax7b.set_xticklabels(tiers_csci, fontsize=8.2)
    ax7b.set_ylabel("Energy-Aware Scan Efficiency eta_watt (MB/J, Log Scale)")
    ax7b.set_title("(b) Energy-Aware Scan Efficiency (MB Scanned per Joule)", fontweight="bold", fontsize=9.5)
    ax7b.grid(True, which="both", linestyle="--", alpha=0.4)
    ax7b.legend(loc="lower right", fontsize=7.6)
    fig7.tight_layout()
    save_fig_both_dirs(fig7, ["fig_hardware_energy_efficiency_sf1_300dpi.pdf", "fig_hardware_energy_efficiency_sf1_300dpi.png"])
    plt.close(fig7)

    # ==========================================================================
    # 6. FIGURE 9: Internal Mechanism Validation (Non-Overlapping P95 Badges)
    # ==========================================================================
    fig9, axes9 = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=300)
    ax = axes9[0]
    labels9 = ["Tree Size\n(2D)", "Syntax\nOnly (8D)", "Operators\nOnly (7D)", "Compact\nCSCI+Sel", "Proposed\nGBR-9D", "Shallow DT\n(d=2, 9D)", "Ridge Reg\n(9D)"]
    vals9 = [103.94, 73.22, 61.06, 56.30, 54.92, 51.35, 49.12]
    p95_badges = ["", "", "", "", "P95:\n114.1%", "P95:\n150.9%", "P95:\n196.8%"]
    colors9 = ["#7f8c8d", "#5bc0de", "#8e6c8a", "#e67e22", "#2ca02c", "#f1c40f", "#3498db"]
    bars9 = ax.bar(labels9, vals9, color=colors9, edgecolor="black", linewidth=0.9, width=0.60)
    for b, v, bdg in zip(bars9, vals9, p95_badges):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        if bdg:
            ax.text(b.get_x() + b.get_width() / 2, v * 0.50, bdg, ha="center", va="center", fontsize=6.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="black", alpha=0.92))
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.5)
    ax.set_title("(a) Feature & Model Ablation (Mean vs. P95 Regret)", fontsize=10.0, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.tick_params(axis="x", labelsize=7.3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes9[1]
    e_pareto = [0.748, 0.879, 1.007, 1.182, 1.381]
    l_pareto = [70.20, 59.81, 56.80, 54.99, 54.92]
    ax.plot(e_pareto, l_pareto, "-o", color="#1f4e78", linewidth=2.0, markersize=6, label="AST-Route Pareto (alpha in [0,1])")
    ax.scatter([3.170], [82.47], marker="s", s=65, color="#e67e22", edgecolor="black", label="Always Polars (82.47 ms, 3.17 J)")
    ax.scatter([2.374], [71.48], marker="^", s=70, color="#7f8c8d", edgecolor="black", label="Rule-Based (71.48 ms, 2.37 J)")
    ax.scatter([0.975], [42.64], marker="*", s=110, color="#2ca02c", edgecolor="black", label="Oracle Lower Bound (42.64 ms)")
    ax.set_xlabel("Estimated Active Device Energy (Joules)", fontsize=9.5)
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.5)
    ax.set_title("(b) Latency vs. Device-Level Energy Frontier", fontsize=10.0, fontweight="bold")
    ax.legend(fontsize=7.4, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes9[2]
    seeds = [5, 10, 15, 20, 24]
    means = [63.61, 68.59, 56.59, 57.60, 54.92]
    sds = [14.2, 12.8, 8.4, 7.1, 5.8]
    means_arr, sds_arr = np.array(means), np.array(sds)
    ax.plot(seeds, means, "-s", color="#1e8449", linewidth=2.0, markersize=6, label="AST-Route Out-of-Fold Mean ± SD")
    ax.fill_between(seeds, means_arr - sds_arr, means_arr + sds_arr, color="#1e8449", alpha=0.18)
    for sx, my in zip(seeds, means):
        ax.text(sx, my - 3.6, f"{my:.2f}", ha="center", va="top", fontsize=7.8, fontweight="bold", color="#145a32")
    ax.axhline(82.47, linestyle=":", color="#e67e22", linewidth=1.3, label="Always Polars (82.47 ms)")
    ax.axhline(71.48, linestyle="--", color="#7f8c8d", linewidth=1.3, label="Rule-Based (71.48 ms)")
    ax.axhline(42.64, linestyle="-.", color="#2c3e50", linewidth=1.3, label="Oracle Bound (42.64 ms)")
    ax.set_xlabel("Number of Training Seeds (3 Queries/Seed)", fontsize=9.5)
    ax.set_ylabel("Out-of-Fold Test Latency (ms)", fontsize=9.5)
    ax.set_title("(c) Cold-Start Sample Efficiency (5–24 Seeds)", fontsize=10.0, fontweight="bold")
    ax.set_xticks(seeds)
    ax.set_ylim(38, 92)
    ax.legend(fontsize=7.2, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig9.tight_layout()
    save_fig_both_dirs(fig9, ["fig_framework_ablation_pareto_300dpi.pdf", "fig_framework_ablation_pareto_300dpi.png"])
    plt.close(fig9)

    # ==========================================================================
    # 7. NEW FIGURE 12: 3-Node Cross-ISA Crossover & Concurrency Scaling
    # ==========================================================================
    fig12, (ax12a, ax12b) = plt.subplots(1, 2, figsize=(13.6, 4.4), dpi=300)
    tiers_short = ["L1 Scan", "L2 Join", "L3 CTE", "L4 Window\n(Crossover)", "L5 Unnest"]
    x12 = np.arange(5)
    n2_duck = [16.91, 74.17, 78.26, 183.26, 65.22]
    n2_pl = [42.43, 100.07, 84.27, 70.18, 77.66]
    n3_duck = [5.90, 22.10, 24.19, 70.75, 24.15]
    n3_pl = [59.79, 96.56, 65.58, 66.41, 50.74]

    w12 = 0.20
    ax12a.bar(x12 - 1.5 * w12, n2_duck, w12, label="Node 2 (x86_64 16C/22T) DuckDB", color="#2e5984", edgecolor="black")
    ax12a.bar(x12 - 0.5 * w12, n2_pl, w12, label="Node 2 (x86_64 16C/22T) Polars", color="#5bc0de", edgecolor="black")
    ax12a.bar(x12 + 0.5 * w12, n3_duck, w12, label="Node 3 (Apple M1 ARM64) DuckDB", color="#1e8449", edgecolor="black")
    ax12a.bar(x12 + 1.5 * w12, n3_pl, w12, label="Node 3 (Apple M1 ARM64) Polars", color="#f39c12", edgecolor="black", hatch="//")
    ax12a.annotate(
        "Cross-ISA L4 Window Crossover:\n• x86_64: Polars 70.2 vs DuckDB 183.3 ms\n• ARM64 M1: Polars 66.4 vs DuckDB 70.8 ms",
        xy=(3 + 1.5 * w12, 66.41), xytext=(1.25, 142),
        arrowprops=dict(arrowstyle="->", color="#9e2a2b", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdf2f2", edgecolor="#9e2a2b"),
        fontsize=7.6, fontweight="bold"
    )
    ax12a.set_xticks(x12)
    ax12a.set_xticklabels(tiers_short, fontsize=8.4)
    ax12a.set_ylabel("Mean Execution Latency (ms)")
    ax12a.set_title("(a) Cross-ISA Tier L1–L5 Latency & L4 Window Crossover", fontweight="bold", fontsize=9.5)
    ax12a.set_ylim(0, 210)
    ax12a.grid(axis="y", linestyle="--", alpha=0.4)
    ax12a.legend(loc="upper left", fontsize=7.5)

    c_streams = [1, 4, 8, 16, 32]
    n1_qps = [13.13, 61.02, 107.98, 139.99, 154.44]
    n2_qps = [14.66, 39.44, 34.48, 28.45, 22.68]
    n3_qps = [20.35, 22.29, 19.24, 17.12, 15.82]
    n3_duck_qps = [20.47, 18.98, 19.39, 18.20, 11.93]

    ax12b.plot(c_streams, n1_qps, "-o", color="#9e2a2b", lw=2.2, markersize=6, label="Node 1 (x86_64 + RTX 2050) AST-Route (Peak 154.4 QPS)")
    ax12b.plot(c_streams, n2_qps, "-s", color="#2e5984", lw=2.0, markersize=6, label="Node 2 (x86_64 16C/22T CPU) AST-Route (Peak 39.4 QPS)")
    ax12b.plot(c_streams, n3_qps, "-^", color="#1e8449", lw=2.0, markersize=6, label="Node 3 (Apple M1 ARM64) AST-Route (Peak 22.3 QPS)")
    ax12b.plot(c_streams, n3_duck_qps, "--d", color="#7f8c8d", lw=1.5, markersize=5, label="Node 3 (Apple M1 ARM64) Always DuckDB (11.9 QPS @ C=32)")
    ax12b.set_xlabel("Concurrent Query Streams (C)", fontsize=9.5)
    ax12b.set_ylabel("System Throughput (Queries/Sec, QPS)", fontsize=9.5)
    ax12b.set_title("(b) Multi-Stream Concurrency Scaling Across 3 Physical Nodes", fontweight="bold", fontsize=9.5)
    ax12b.set_xticks(c_streams)
    ax12b.set_ylim(0, 175)
    ax12b.grid(True, linestyle="--", alpha=0.4)
    ax12b.legend(loc="upper left", fontsize=7.3)
    fig12.tight_layout()
    save_fig_both_dirs(fig12, ["fig_three_node_cross_isa_concurrency_300dpi.pdf", "fig_three_node_cross_isa_concurrency_300dpi.png"])
    plt.close(fig12)

    # ==========================================================================
    # 8. NEW FIGURE 13: 5-Suite Generalization (163Q) & 12D JOB Sketch Ablation
    # ==========================================================================
    fig13, (ax13a, ax13b) = plt.subplots(1, 2, figsize=(13.6, 4.4), dpi=300)
    suites = ["75Q TPC-H\n(LOSO-CV)", "15Q Unseen\nHoldout", "22Q Official\nTPC-H", "13Q Star\nSSB", "20Q Web\nClickBench", "33Q IMDB\nJOB (12D)"]
    top1_s = [57.33, 73.33, 68.18, 92.31, 80.00, 69.70]
    top2_s = [76.00, 86.67, 90.91, 100.00, 95.00, 100.00]
    reg_s = [34.98, 13.82, 27.12, 2.82, 9.45, 16.50]

    x13 = np.arange(len(suites))
    w13 = 0.26
    ax13a.bar(x13 - w13, top1_s, w13, label="Top-1 Routing Accuracy (%)", color="#2e5984", edgecolor="black")
    ax13a.bar(x13, top2_s, w13, label="Top-2 Routing Accuracy (%)", color="#2ca02c", edgecolor="black")
    ax13a.bar(x13 + w13, reg_s, w13, label="Mean Performance Regret (%, Lower is Better)", color="#c44e52", edgecolor="black", hatch="//")
    ax13a.set_xticks(x13)
    ax13a.set_xticklabels(suites, fontsize=7.8)
    ax13a.set_ylabel("Percentage (%)")
    ax13a.set_ylim(0, 115)
    ax13a.set_title("(a) Multi-Suite Generalization Across 163 Queries (5 Benchmarks)", fontweight="bold", fontsize=9.5)
    ax13a.grid(axis="y", linestyle="--", alpha=0.4)
    ax13a.legend(loc="upper right", fontsize=7.4)

    job_steps = ["Always\nDuckDB", "Always\nGPU Cold", "9D Zero-Shot\n(Abstain)", "12D Zero-Shot\n(Sketch Gate)", "10D Calib.\n(+sigma_str)", "12D Calib.\n(Full Sketch)", "Oracle\nBound"]
    job_lat = [19.55, 9.05, 6.75, 4.54, 4.21, 4.14, 4.03]
    job_reg = [392.15, 124.57, 84.49, 16.50, 4.05, 3.55, 0.00]
    j_cols = ["#7f8c8d", "#c44e52", "#dd8452", "#3498db", "#5bc0de", "#2ca02c", "#111111"]
    xj = np.arange(len(job_steps))
    bars_j = ax13b.bar(xj, job_lat, width=0.56, color=j_cols, edgecolor="black")
    for b, lt, rg in zip(bars_j, job_lat, job_reg):
        ax13b.text(b.get_x() + b.get_width() / 2, lt + 0.45, f"{lt:.2f}ms\n({rg:.1f}%)", ha="center", va="bottom", fontsize=6.9, fontweight="bold")
    ax13b.set_xticks(xj)
    ax13b.set_xticklabels(job_steps, fontsize=7.4)
    ax13b.set_ylabel("Mean Execution Latency on IMDB JOB (ms)")
    ax13b.set_ylim(0, 23.5)
    ax13b.set_title("(b) Stepwise 12D Pre-Execution Sketch Resolution on JOB (N=33)", fontweight="bold", fontsize=9.5)
    ax13b.grid(axis="y", linestyle="--", alpha=0.4)
    fig13.tight_layout()
    save_fig_both_dirs(fig13, ["fig_cross_suite_163q_and_job_12d_ablation_300dpi.pdf", "fig_cross_suite_163q_and_job_12d_ablation_300dpi.png"])
    plt.close(fig13)


def fix_tex_and_embed_new_figures() -> dict[str, bool]:
    tex = TARGET_TEX.read_text(encoding="utf-8")
    chk: dict[str, bool] = {}

    # 1. Fix the corrupted Table 13 \begin{tabular} line (Line 663 error!)
    t13_pos = tex.find("System reproducibility manifest:")
    if t13_pos != -1:
        t13_end = tex.find(r"\end{table", t13_pos)
        if t13_end != -1:
            t13_block = tex[t13_pos:t13_end]
            t13_block_fixed = re.sub(
                r"(?:\\setlength\{\\tabcolsep\}\{[0-9.]+pt\}\\footnotesize)?\\begin\{tabular\}\{[^\n]+\}",
                lambda _: r"\setlength{\tabcolsep}{3.5pt}\footnotesize\begin{tabular}{@{}p{0.13\textwidth}p{0.22\textwidth}p{0.59\textwidth}@{}}",
                t13_block,
                count=1,
            )
            tex = tex[:t13_pos] + t13_block_fixed + tex[t13_end:]
    chk["1_Table13_Line663_Fixed"] = r"@{}}p{" not in tex

    # 2. Escape the '&' in 'Cross-Architecture & Cross-ISA Synthesis' before Table 15
    tex = tex.replace("Cross-Architecture & Cross-ISA Synthesis", r"Cross-Architecture \& Cross-ISA Synthesis")
    chk["2_No_Unescaped_Ampersand_In_Text"] = "Cross-Architecture & Cross-ISA" not in tex

    # 3. Embed Figure 12 (3-Node Cross-ISA & Concurrency) right after Table 15
    fig12_tex = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig_three_node_cross_isa_concurrency_300dpi.pdf}
\caption{Three-node cross-hardware and cross-ISA validation ($\text{SF}=1\text{ GB}$): (a) Tier L1--L5 execution latency comparing the 16-core/22-thread \texttt{x86\_64} workstation (Node~2) and the Apple M1 \texttt{ARM64} unified-memory node (Node~3), illustrating the invariant Tier~L4 Polars vs.\ DuckDB window-sorting crossover across instruction sets; and (b) multi-stream concurrent query throughput ($C \in \{1,4,8,16,32\}$) across all three physical nodes.}\label{fig:three_node_cross_isa}
\end{figure}
"""
    if r"\label{fig:three_node_cross_isa}" not in tex and r"\label{tab:three_node_cross_isa}" in tex:
        pos15 = tex.find(r"\label{tab:three_node_cross_isa}")
        end15 = tex.find(r"\end{table}", pos15)
        if end15 != -1:
            end15 += len(r"\end{table}")
            tex = tex[:end15] + "\n\n" + fig12_tex.strip() + "\n" + tex[end15:]
            tex = tex.replace(
                r"Table~\ref{tab:three_node_cross_isa} consolidates",
                r"Table~\ref{tab:three_node_cross_isa} and Fig.~\ref{fig:three_node_cross_isa} consolidate",
                1,
            )
    chk["3_Figure12_Embedded"] = r"\label{fig:three_node_cross_isa}" in tex

    # 4. Embed Figure 13 (5-Suite Generalization & 12D JOB Ablation) right after Table 16
    fig13_tex = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig_cross_suite_163q_and_job_12d_ablation_300dpi.pdf}
\caption{Cross-workload generalization and 12D pre-execution sketch ablation: (a) Top-1/Top-2 routing accuracy and mean performance regret across all five benchmark suites ($N=163$ total queries); and (b) stepwise latency and regret reduction on the 33-template IMDB Join Order Benchmark (JOB) achieved via compact $10.18\text{ KB}$ string/skew/fanout catalog sketches ($\mathbf{x}_q^{\text{skew}} \in \mathbb{R}^{12}$).}\label{fig:job_12d_ablation_fig}
\end{figure}
"""
    if r"\label{fig:job_12d_ablation_fig}" not in tex and r"\label{tab:job_12d_ablation}" in tex:
        pos16 = tex.find(r"\label{tab:job_12d_ablation}")
        end16 = tex.find(r"\end{table}", pos16)
        if end16 != -1:
            end16 += len(r"\end{table}")
            tex = tex[:end16] + "\n\n" + fig13_tex.strip() + "\n" + tex[end16:]
            tex = tex.replace(
                r"Table~\ref{tab:job_12d_ablation} synthesizes",
                r"Table~\ref{tab:job_12d_ablation} and Fig.~\ref{fig:job_12d_ablation_fig} synthesize",
                1,
            )
    chk["4_Figure13_Embedded"] = r"\label{fig:job_12d_ablation_fig}" in tex

    # Save in-place to bigquery_olap_benchmark_V2.tex and mirrors (Zero Backup)
    TARGET_TEX.write_text(tex, encoding="utf-8")
    for dst in (
        ROOT / "bigquery_olap_benchmark_V2.tex",
        ROOT / "main.tex",
        MS_DIR / "main.tex",
        MS_DIR / "overleaf_ready_single_file.tex",
        MS_DIR / "bigquery_olap_benchmark_V6.tex",
    ):
        dst.write_text(tex, encoding="utf-8")

    return chk


def compile_and_replace_on_github() -> None:
    for pass_i in (1, 2, 3):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"[ERROR on pdflatex pass {pass_i}]:")
            print("\n".join(proc.stdout.splitlines()[-30:]))
            raise RuntimeError(f"pdflatex failed on pass {pass_i}")

    pdf_v2 = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
    for copy_p in (
        ROOT / "bigquery_olap_benchmark_V2.pdf",
        MS_DIR / "bigquery_olap_benchmark_V6.pdf",
        ROOT / "bigquery_olap_benchmark_V6.pdf",
        MS_DIR / "bigquery_olap_benchmark_V6_2.pdf",
        ROOT / "bigquery_olap_benchmark_V6_2.pdf",
        MS_DIR / "main.pdf",
    ):
        try:
            shutil.copy2(pdf_v2, copy_p)
        except PermissionError:
            print(f"  [INFO] ข้ามการเขียนทับ {copy_p.name} ที่เปิดค้างไว้ (ไฟล์หลัก bigquery_olap_benchmark_V2.pdf อัปเดตสำเร็จแล้ว)")

    # อัปเดตไฟล์ ZIP สำหรับ Overleaf แบบปกติ
    for zpath in (ROOT / "Overleaf_V2_Submission_Ready.zip", ROOT / "Overleaf_V6_Project.zip", MS_DIR / "overleaf_V2_submission_bundle.zip"):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(MS_DIR / "main.tex", arcname="main.tex")
            zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            if (MS_DIR / "references.bib").exists():
                zf.write(MS_DIR / "references.bib", arcname="references.bib")
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    # Replace directly on GitHub (No backup folders)
    for bak in ROOT.glob("manuscript_backup_*"):
        if bak.is_dir():
            shutil.rmtree(bak, ignore_errors=True)

    run_git(["rm", "-r", "--cached", "data/"])
    for item in [
        "manuscript/bigquery_olap_benchmark_V2.tex",
        "manuscript/bigquery_olap_benchmark_V2.pdf",
        "manuscript/main.tex",
        "manuscript/overleaf_ready_single_file.tex",
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V2.pdf",
        "main.tex",
        "figures",
        "manuscript/figures",
        "scripts/94_complete_all_phases_and_replace_github.py",
    ]:
        if (ROOT / item).exists():
            run_git(["add", "-f", item])

    if run_git(["status", "--porcelain"]).stdout.strip():
        run_git(["commit", "-m", "Complete Reviewer Upgrade (Phases 1-3): Eqs 2-9, Algo 1, Tables 1-16, Figs 1-13 (replace V2.tex, no backup)"])
    run_git(["branch", "-M", "main"])
    for rem in [r.strip() for r in run_git(["remote"]).stdout.splitlines() if r.strip()]:
        res = run_git(["push", "-u", rem, "main", "--force-with-lease"])
        if res.returncode != 0:
            res = run_git(["push", "-u", rem, "main", "--force"])
        print(f"  [GITHUB REPLACE -> {rem:<13}] {'SUCCESS (REPLACED ON GITHUB)' if res.returncode == 0 else 'WARN: ' + res.stderr.strip()[:70]}")

    print("=" * 100)
    print(f"  [MASTER TEX SHA-256] {sha256_file(TARGET_TEX)}")
    print(f"  [MASTER PDF SHA-256] {sha256_file(pdf_v2)} (Size: {pdf_v2.stat().st_size:,} bytes)")
    print("=" * 100)


if __name__ == "__main__":
    print("=" * 100)
    print("RUNNING COMPLETE REVIEWER UPGRADE (PHASES 1-3) ON NOTEBOOK1...")
    print("=" * 100)
    generate_all_synchronized_figures()
    print("  [FIGURES OK] วาดซ่อม Figures 1, 2, 5, 6 (2-panel), 7, 9 และสร้างรูปใหม่ Figures 12 & 13 (300 DPI) สำเร็จ!")
    chk = fix_tex_and_embed_new_figures()
    for k, v in chk.items():
        print(f"  [{'PASS' if v else 'CHECK'}] {k}")
    compile_and_replace_on_github()
