#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/v12_regenerate_fig1_fig2_300dpi.py
================================================================================
Generates publication-grade 300 DPI vector PDF and raster JPG versions of
Figure 1 (`fig_v12_proposed_ast_routing_framework_300dpi.pdf/.jpg`) and
Figure 2 (`fig_v12_experimental_workflow_300dpi.pdf/.jpg`) exclusively for V12
(leaving V11 figures 100% untouched).

Fixes Reviewer issues:
  1. Removes legacy "163 queries" text; displays exact V12 corpus:
     400 Benchmark Queries (Cat I-IV) + 80 Public SQL/Trace Queries
     (25 SDSS DR16 + 25 NYC TLC + 30 Amazon Redset at SF=10 GB) + 70 Stress Queries.
  2. Unifies GitHub repository URL to `https://github.com/preecha-msu/bigquery-olap-benchmark`
     (removes legacy `preecha-msu/bigquery-olap-benchmark`).
  3. Shows the 3-Way Tail-Safety Decomposition (Threshold Guard -> +Mondrian Conformal
     -> +Oblique CSCI), Stronger GPU Baselines (Resident-VRAM, Pinned+Overlap FP64/FP32,
     RAPIDS cuDF & cudf-polars), and 4-Node Multi-Scale Router-in-the-Loop (SF=1-100 GB).
================================================================================
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, body, facecolor="#f4f8fb", edgecolor="#005a8d", title_color="#003d66", fontsize=8.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0, y + h - 0.032,
        title,
        ha="center", va="top",
        fontsize=fontsize + 0.7, fontweight="bold", color=title_color,
        family="sans-serif",
    )
    ax.text(
        x + 0.014, y + h - 0.072,
        body,
        ha="left", va="top",
        fontsize=fontsize, color="#1a1a1a",
        family="sans-serif", linespacing=1.24,
    )


def generate_fig1_v12() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "AST-Route (V13): Pre-Execution PCIe/VRAM Crossover & Tail-Safe Multi-Engine SQL Routing Architecture",
        ha="center", va="center", fontsize=11.0, fontweight="bold", color="#003366",
    )

    # Layer 1
    draw_box(
        ax, 0.015, 0.50, 0.225, 0.41,
        "Layer 1: Zero-Plan AST & Zone-Map",
        "• Input: Raw SQL Query q (Cold O1: 4.15 ms;\n"
        "  Warm AST-Cache O2: 0.67 ms; 1D Gate O3: 0.08 ms)\n"
        "• SQLGlot Single-Pass AST Extraction:\n"
        "  7 Operators [D, N, J, S, W, U, C]\n"
        "• Oblique Structural Projection CSCI(q):\n"
        "  Prunes -28.2% GBDT splits (6.84 -> 4.91)\n"
        "• Pre-Execution Parquet/DuckDB Zone-Map:\n"
        "  Selectivity sigma_hat(q) in 172 us\n"
        "• Scale-Aware 10D Vector x_q^scale (SF=1..100 GB)\n"
        "  + Optional 10.18 KB 12D Join/String Sketch",
        facecolor="#eef6fc", edgecolor="#006699",
    )

    # Layer 2
    draw_box(
        ax, 0.265, 0.50, 0.225, 0.41,
        "Layer 2: Crossover & Cost Models",
        "• Physical PCIe/VRAM Crossover Law:\n"
        "  Phi_PCIe(q) = T_PCIe / (T_PCIe + T_Kernel)\n"
        "  Break-Even Speedup > (1 - Phi_PCIe)^(-1)\n"
        "• 5-Dim Hardware Vector h_hw:\n"
        "  [M_host, M_gpu, B_PCIe, N_CPU, B_MEM]\n"
        "  Nodes 1-4 (RTX 2050 4GB, 22T CPU, M1, T4 15GB)\n"
        "• Scope 1 Direct NVML GPU Energy Sensor\n"
        "  (rho_{N=75} = 0.9684 vs ACPI DC Shunt)\n"
        "• Cloud FinOps Scan Fee ($6.25/TiB, $1,267.66/1M)",
        facecolor="#f2f9f5", edgecolor="#1b7837", title_color="#115224",
    )

    # Layer 3
    draw_box(
        ax, 0.515, 0.50, 0.225, 0.41,
        "Layer 3: Multi-Output GBDT & Conformal",
        "• Locked Model (SHA-256: f0e5f17c832472e3...)\n"
        "  Predicts log(1+T_e), Energy E_e, RAM/VRAM\n"
        "• Out-of-Fold Q-Error P50 = 1.053 - 1.095\n"
        "• Tier-Stratified 90% Mondrian Conformal\n"
        "  Intervals [L_0.90, U_0.90] (90.93% coverage)\n"
        "• 3-Way Tail-Safety Decomposition (R29):\n"
        "  [1] 8% Margin Guard: -78.2 pp In-Domain P95\n"
        "  [2] +Mondrian Bound: -48.6 pp Drift/Spill P95\n"
        "  [3] +CSCI: -5.4 pp Mean Regret & +6.1% 0-Shot",
        facecolor="#fff8ee", edgecolor="#b35806", title_color="#7f3b08",
    )

    # Layer 4
    draw_box(
        ax, 0.765, 0.50, 0.220, 0.41,
        "Layer 4: Guarded Dispatch",
        "• Locked Primary Guard: Inner-Fold Nested-CV\n"
        "  tau_guard = 8% (Secondary sensitivity: 12%)\n"
        "• Rule U1 (PCIe Margin Guard): Abstains to\n"
        "  CPU if GPU gain < 8% (2.17x-2.77x P95 cut)\n"
        "• Rule U2 (Conformal Cloud Spill Gate)\n"
        "• Rule U3 (Unanchored String Guard)\n"
        "• Rule U4 (VRAM/DRAM Feasibility Filter +\n"
        "  Mahalanobis OOD Gate, 0.0% OOM)\n"
        "• Minimalist 2-Engine Pool {Polars, GPU}\n"
        "  captures 98.7% of 6-engine gain at SF=1 GB",
        facecolor="#f9f1fb", edgecolor="#762a83", title_color="#40004b",
    )

    # Bottom Execution Backends & Corpus Banner
    draw_box(
        ax, 0.015, 0.05, 0.970, 0.38,
        "Target Execution Paradigms, Stronger GPU Baselines (R30), & Multi-Scale Evaluation Corpus (SF = 1, 10, 30, 100 GB)",
        "1. Vectorized CPU Engines:  DuckDB v2.0 (C++ Morsel-Driven DBMS)  |  Polars v2.1 / v1.35.2 (Rust Arrow Lazy SIMD)  |  Apache DataFusion v55.0\n"
        "2. Stronger GPU Baselines (Node 1 RTX 2050 4GB & Node 4 Server Tesla T4 15GB):  (B1) Cold H2D+Kernel FP64 (56.96 ms, 257.3% P95)  |  "
        "(B2) Pinned Host RAM + 2-Stream Async Overlap FP64 (47.82 ms -> 44.18 ms routed, 86.2% P95)  |  (B3) Pinned + Async Overlap FP32 (39.64 ms -> 37.95 ms routed)  |  "
        "(B4) Resident-VRAM Warm Cache (22.25 ms at SF=1 GB; spills at SF>=10 GB)  |  (B5) Native GPU Engines: RAPIDS cuDF (libcudf 26.06) & RMM-Pooled cudf-polars 26.10.2 (C3: 63.82 ms, 2.77x P95 cut)\n"
        "3. Serverless Cloud Spill Target:  Google BigQuery (0% share at SF<=30 GB; 20% elastic spill offload at SF=100 GB where local CPUs spill 3.3 GB to NVMe)\n"
        "4. Evaluated SQL Corpus (G_total = 42 Independent Template Clusters, EX = 1.000):  400 Core/Standard Benchmark Queries (75 Cat I Core + 55 Cat II TPC-H/JOB + "
        "115 Cat III Derived + 155 Cat IV Untouched Official TPC-DS [SF=1 & 10 GB primary], ClickBench 43Q, SSB 13Q)  +  80 Public Real-World SQL & Fleet Trace Queries "
        "(25 SDSS SkyServer DR16 + 25 Public NYC TLC + 30 Amazon Redshift Redset Trace at SF=10 GB)  +  70 Author-Constructed Schema-Blinded Stress Queries",
        facecolor="#f7f7f7", edgecolor="#333333", title_color="#111111", fontsize=7.8,
    )

    # Arrows between layers
    for x0, x1 in [(0.240, 0.265), (0.490, 0.515), (0.740, 0.765)]:
        ax.annotate("", xy=(x1, 0.705), xytext=(x0, 0.705), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))
    ax.annotate("", xy=(0.50, 0.43), xytext=(0.50, 0.50), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))

    plt.tight_layout(pad=0.2)
    for out_p in [
        FIG_DIR / "fig_v12_proposed_ast_routing_framework_300dpi.pdf",
        FIG_DIR / "fig_v12_proposed_ast_routing_framework_300dpi.jpg",
        ROOT / "manuscript" / "fig_v12_proposed_ast_routing_framework_300dpi.pdf",
        ROOT / "manuscript" / "fig_v12_proposed_ast_routing_framework_300dpi.jpg",
    ]:
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Generated V12 Figure 1 (fig_v12_proposed_ast_routing_framework_300dpi.pdf/.jpg)")


def generate_fig2_v12() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "End-to-End Experimental Validation & Reproducibility Workflow (V13: 4 Nodes, SF = 1-100 GB, G = 42 Template Clusters)",
        ha="center", va="center", fontsize=10.8, fontweight="bold", color="#003366",
    )

    phases = [
        (
            0.015, 0.52, 0.31, 0.39,
            "Phase 1: Multi-Scale Data & Corpus (G=42)",
            "• 4 TPC-H Scale Tiers: SF = 1, 10, 30, 100 GB\n"
            "  (6.0M to 600.1M lineitem rows; ZSTD Parquet)\n"
            "• 400 Benchmark Queries (Cat I-IV): 75 Core,\n"
            "  22 TPC-H, 33 JOB, 99 Official TPC-DS (SF=1 & 10G),\n"
            "  43 ClickBench, 13 SSB, 115 Zero-Shot Derived\n"
            "• 80 Public Real-World & Trace Queries (SF=10 GB):\n"
            "  25 SDSS DR16 + 25 NYC TLC + 30 Amazon Redset\n"
            "• 70 Author-Constructed Schema-Blinded Queries",
            "#eef6fc", "#006699",
        ),
        (
            0.345, 0.52, 0.31, 0.39,
            "Phase 2: 4-Node Hardware & GPU Baselines",
            "• Node 1: x86_64 (8T, 9GB Cap) + RTX 2050 (4GB)\n"
            "• Node 2: x86_64 Many-Core (16C/22T, 15.5GB)\n"
            "• Node 3: Apple M1 ARM64 (8C, 8GB Unified RAM)\n"
            "• Node 4: Cloud 2-vCPU + NVIDIA Tesla T4 (15GB)\n"
            "• 5 GPU Modes (R30): Cold FP64, Pinned+Overlap\n"
            "  FP64, Pinned+Overlap FP32, Resident-VRAM,\n"
            "  RAPIDS cuDF (libcudf 26.06) & cudf-polars 26.10",
            "#f2f9f5", "#1b7837",
        ),
        (
            0.675, 0.52, 0.31, 0.39,
            "Phase 3: Pre-Execution AST & Zone-Map",
            "• Single-Pass SQLGlot AST Extraction (3.48 ms)\n"
            "  + Parquet Footer Zone-Map Lookup (0.17 ms)\n"
            "• 9D Fixed-Scale Vector x_q & 10D Multi-Scale\n"
            "  Vector x_q^scale + 5D Hardware Vector h_hw\n"
            "• Exact TreeSHAP Attribution (R33):\n"
            "  sigma_hat (34.2%), CSCI (24.8%), J (14.6%),\n"
            "  W (10.4%), U (7.8%), D+N+S+C (8.2%)",
            "#fff8ee", "#b35806",
        ),
        (
            0.015, 0.06, 0.31, 0.39,
            "Phase 4: Leak-Free Training & 3-Way Ablation",
            "• 25-Seed LOSO-CV & Strict LOTFO Zero-Leakage\n"
            "• Locked Frozen Model: f0e5f17c832472e3...\n"
            "• 3-Way Stepwise Ablation (R29):\n"
            "  [0] Unguarded 8D (214.6% P95)\n"
            "  [1] +8% Margin Guard (-78.2 pp In-Dom P95)\n"
            "  [2] +Mondrian Conformal (-48.6 pp Drift P95)\n"
            "  [3] +CSCI (-28.2% splits, +6.1% 0-Shot Top-1)",
            "#f9f1fb", "#762a83",
        ),
        (
            0.345, 0.06, 0.31, 0.39,
            "Phase 5: Multi-Scale Router-in-the-Loop",
            "• Router-in-the-Loop at SF = 10, 30, 100 GB (R31):\n"
            "  Node 1: 182.4 ms (SF10), 448.7 ms (SF30),\n"
            "  1,142.8 ms (SF100, 7.61x vs DuckDB)\n"
            "  Node 4 Tesla T4: 622.4 ms (SF10, 0.0% P95),\n"
            "  1,905.6 ms (SF30, 0.0% P95)\n"
            "• Break-Even (R28): Use O2/O3 or CPU for <10 ms\n"
            "  queries (R1); use AST-Route for R3/R4 (>=25 ms)",
            "#eef6fc", "#006699",
        ),
        (
            0.675, 0.06, 0.31, 0.39,
            "Phase 6: Holdout Permutation & Reproducibility",
            "• External Holdout Cluster Permutation (K = 38):\n"
            "  p_perm = 0.125 vs CPU, 0.198 vs GPU (LME retired)\n"
            "• Global Holm Control (M = 4 Primary Tests):\n"
            "  H1 (P95 vs GPU p_Holm=0.0204), H2 (vs CPU\n"
            "  p_Holm=0.0112), H3 (NI p_Holm=0.0048)\n"
            "• Canonical Repository & SHA-256 Deposit:\n"
            "  https://github.com/preecha-msu/bigquery-olap-benchmark",
            "#f2f9f5", "#1b7837",
        ),
    ]

    for x, y, w, h, title, body, fc, ec in phases:
        draw_box(ax, x, y, w, h, title, body, facecolor=fc, edgecolor=ec, fontsize=7.6)

    # Arrows
    ax.annotate("", xy=(0.345, 0.715), xytext=(0.325, 0.715), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))
    ax.annotate("", xy=(0.675, 0.715), xytext=(0.655, 0.715), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))
    ax.annotate("", xy=(0.170, 0.45), xytext=(0.830, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5, color="#003366", ls="--"))
    ax.annotate("", xy=(0.345, 0.255), xytext=(0.325, 0.255), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))
    ax.annotate("", xy=(0.675, 0.255), xytext=(0.655, 0.255), arrowprops=dict(arrowstyle="->", lw=1.8, color="#003366"))

    plt.tight_layout(pad=0.2)
    for out_p in [
        FIG_DIR / "fig_v12_experimental_workflow_300dpi.pdf",
        FIG_DIR / "fig_v12_experimental_workflow_300dpi.jpg",
        ROOT / "manuscript" / "fig_v12_experimental_workflow_300dpi.pdf",
        ROOT / "manuscript" / "fig_v12_experimental_workflow_300dpi.jpg",
    ]:
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Generated V12 Figure 2 (fig_v12_experimental_workflow_300dpi.pdf/.jpg)")


if __name__ == "__main__":
    generate_fig1_v12()
    generate_fig2_v12()
