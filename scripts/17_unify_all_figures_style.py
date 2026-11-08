# ไฟล์: scripts/17_unify_all_figures_style.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures_300dpi"
MS_FIG_DIR = PROJECT_ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ตั้งค่า Global Academic Style ตามต้นแบบ Figure 3 (Serif Font, Clean Ticks, Muted Deep Palette)
def set_academic_serif_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.85,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

# ชุดสีมาตรฐานเดียวกับ Figure 3 (Seaborn Deep / Academic Muted Palette)
PALETTE = {
    "slate_grey": "#7F8C8D",
    "deep_blue":  "#4C72B0",
    "sage_green": "#55A868",
    "amber_gold": "#DD8452",
    "royal_purp": "#8172B2",
    "crimson":    "#C44E52",
    "teal_cyan":  "#64B5CD",
    "dark_navy":  "#1B263B"
}

def draw_card_with_header(ax, x, y, w, h, header_text, bullets, hdr_color, bg_color, border_color, hdr_ratio=0.25, title_fs=8.8, body_fs=7.8):
    """วาดกล่องสไตล์ Figure 2: กรอบโค้งมนพร้อมแถบหัวสีเข้ม (Solid Header Banner) และรายการ Bullet ด้านใน"""
    # กล่องพื้นหลัง
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.35, zorder=2
    )
    ax.add_patch(body)

    # แถบหัวด้านบน
    hdr_h = h * hdr_ratio
    hdr = FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor=hdr_color, edgecolor=border_color, linewidth=1.1, zorder=3
    )
    ax.add_patch(hdr)

    ax.text(
        x + w / 2.0, y + h - hdr_h / 2.0, header_text,
        ha="center", va="center", fontsize=title_fs, fontweight="bold", color="white", zorder=4
    )

    # ข้อความด้านใน
    body_top = y + h - hdr_h - 0.014
    line_step = (h - hdr_h - 0.022) / max(len(bullets), 1)
    for idx, line in enumerate(bullets):
        yy = body_top - (idx + 0.45) * line_step
        ax.text(
            x + 0.014, yy, line,
            ha="left", va="center", fontsize=body_fs, color="#1E293B", zorder=4
        )

def draw_arrow(ax, x1, y1, x2, y2, color="#334155"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        linewidth=1.35, color=color, zorder=5
    )
    ax.add_patch(arr)

def save_both(fig, filename: str):
    fig.savefig(FIG_DIR / filename, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(MS_FIG_DIR / filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> [OK] Saved unified figure: {filename}")

# =================================================================================================
# FIGURE 1: AST-Route Four-Layer Framework Architecture (Redesigned to Match Figure 2 Banner Style!)
# =================================================================================================
def generate_figure1_framework():
    set_academic_serif_style()
    fig, ax = plt.subplots(figsize=(12.6, 7.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.50, 0.972,
        "Proposed AST-Route Four-Layer Multi-Objective Query Routing Framework",
        ha="center", va="center", fontsize=12.0, fontweight="bold", color="#0F172A"
    )

    # LAYER 1 (Top Left & Top Right like Phase 1 & Phase 2 in Fig 2)
    draw_card_with_header(
        ax, 0.02, 0.76, 0.46, 0.175,
        "LAYER 1A: Input SQL & Static AST Parsing (SQLGlot)",
        [
            "• Input Workload: 75 Factorial SQL Queries (25 Canonical Seeds x V1-V3)",
            "• Zero-Execution Static Parsing via SQLGlot (< 0.8 ms overhead)",
            "• Extracts Operator Topology: Depth D(q), Nodes N(q), Joins J(q),",
            "  Subqueries/CTEs S(q), Windows W(q), UNNEST U(q), CASE C(q)"
        ],
        hdr_color="#1D4ED8", bg_color="#EFF6FF", border_color="#1E40AF", hdr_ratio=0.25
    )

    draw_card_with_header(
        ax, 0.52, 0.76, 0.46, 0.175,
        "LAYER 1B: 9-Dimensional Feature Vector & CSCI",
        [
            "• Validated Composite Index: CSCI(q) = 1.5D + 2.5J + 3.0S + 3.5W + 4.0U + 1.5C",
            "• Construct Validity: Spearman rho = 0.4935, Weight Stability = 0.8189",
            "• Zone-Map Predicate Selectivity Factor: sigma(q) in (0, 1] across V1-V3",
            "• 9D Routing Vector: x_q = [D, N, J, S, W, U, C, CSCI, sigma(q)]^T in R^9"
        ],
        hdr_color="#047857", bg_color="#ECFDF5", border_color="#065F46", hdr_ratio=0.25
    )
    draw_arrow(ax, 0.48, 0.845, 0.52, 0.845)
    draw_arrow(ax, 0.75, 0.76, 0.50, 0.715)
    draw_arrow(ax, 0.25, 0.76, 0.50, 0.715)

    # LAYER 2 & 3 (Middle Full-Width Banner Card like Phase 3 in Fig 2)
    draw_card_with_header(
        ax, 0.02, 0.515, 0.96, 0.195,
        "LAYERS 2 & 3: Cross-Paradigm Analytical Cost Models & Multi-Output Cost-Regression Engine (R_GBR-9D)",
        [
            "• Analytical Models: PCIe Fraction Phi_PCIe(q), Effective Throughput Theta_eff (GB/s), Cloud Slot Multiplier M_slot, FinOps Savings Delta C_1M ($)",
            "• Multi-Output GBR Predictor f_Theta: R^9 -> R^M simultaneously estimates Log-Latency ln(1+T_e), Energy E_e (Joules), and Memory (RAM/VRAM GB)",
            "• Out-of-Fold Predictive Accuracy (25-Seed LOSO-CV): GPU Latency R^2 = 0.875 | Polars Latency R^2 = 0.822 | Energy R^2 = 0.815 (Inference: 497.2 us)"
        ],
        hdr_color="#B45309", bg_color="#FFFBEB", border_color="#92400E", hdr_ratio=0.24
    )

    # LAYER 4A: 4 Target Candidate Execution Paradigms (4 Columns exactly like Fig 2!)
    cards_4 = [
        (0.02, "Candidate 1: DuckDB (C++)", "#334155", "#F8FAFC", "#475569", [
            "• Paradigm: In-Process CPU DBMS",
            "• Morsel-Driven Vectorized Engine",
            "• Segment-Tree Window Framing",
            "• Mean Latency: 183.74 ms",
            "• Energy: 6.40 J (243.0 MB/J)",
            "• Routed: 80% L1, 20% L2"
        ]),
        (0.265, "Candidate 2: Polars / DataFusion", "#0F766E", "#F0FDFA", "#115E59", [
            "• Paradigm: Rust Apache Arrow",
            "• Zero-Copy SIMD & ListArray",
            "• Polars: 82.47 ms (3.17 J)",
            "• DataFusion: 96.30 ms (3.39 J)",
            "• 102.1x Faster than Row-Store",
            "• Routed: 60% L2 (Narrow/Med)"
        ]),
        (0.51, "Candidate 3: CUDA GPU (SIMT)", "#991B1B", "#FEF2F2", "#7F1D1D", [
            "• Paradigm: RTX 2050 (4GB VRAM)",
            "• Cold: 56.96 ms | Warm: 22.26 ms",
            "• L1 PCIe Bottleneck: Phi = 55.5%",
            "• Energy: 0.65 J Cold / 0.25 J Warm",
            "• Green Perf: 1942.8 MB/J Warm",
            "• Routed: 60% L3, 100% L4, 80% L5"
        ]),
        (0.755, "Candidate 4: Google BigQuery", "#1E40AF", "#EFF6FF", "#1E3A8A", [
            "• Paradigm: Serverless Cloud OLAP",
            "• Capacitor Storage + Slot Elasticity",
            "• Clustered Mean: 745.39 ms",
            "• Slot Multiplier: M_slot = 6.37x",
            "• V1 I/O Pruning: 78.4% ($1.47/1M)",
            "• Elastic Out-of-Core Fallback"
        ])
    ]
    for cx, ctitle, chdr, cbg, cbdr, cbul in cards_4:
        draw_card_with_header(ax, cx, 0.235, 0.225, 0.235, ctitle, cbul, chdr, cbg, cbdr, hdr_ratio=0.21, title_fs=8.2, body_fs=7.3)
        draw_arrow(ax, cx + 0.1125, 0.515, cx + 0.1125, 0.472)
        draw_arrow(ax, cx + 0.1125, 0.235, cx + 0.1125, 0.192)

    # LAYER 4B: Bottom Dispatcher & Verification Banner
    draw_card_with_header(
        ax, 0.02, 0.025, 0.96, 0.165,
        "LAYER 4: Policy-Driven Optimal Engine Recommendation & Leak-Free 25-Seed LOSO-CV Performance",
        [
            "• Optimal Dispatch Rule: e_hat(q) = argmin_e [ E(T_e | x_q) ] -> Avoids L1/L2 PCIe Bottlenecks (0% L1 GPU) while Offloading L3-L5 to CUDA GPU",
            "• Outperforms All Static & Heuristic Baselines: 54.92 ms Mean Latency (95% CI: [42.78, 68.50] ms), 34.98% Regret, 65.33% Near-Optimal Hit, 1.381 J Energy"
        ],
        hdr_color="#15803D", bg_color="#F0FDF4", border_color="#166534", hdr_ratio=0.27
    )

    save_both(fig, "fig_proposed_ast_routing_framework_300dpi.pdf")

# =================================================================================================
# FIGURE 2: End-to-End Experimental Workflow (Ensure identical serif font & matching dimensions)
# =================================================================================================
def generate_figure2_workflow():
    set_academic_serif_style()
    fig, ax = plt.subplots(figsize=(12.6, 7.4), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.50, 0.972,
        "End-to-End Experimental Validation Workflow: AST-Route Testbed (75 Queries, 6 Engines)",
        ha="center", va="center", fontsize=12.0, fontweight="bold", color="#0F172A"
    )

    draw_card_with_header(
        ax, 0.02, 0.76, 0.46, 0.175,
        "PHASE 1: Full-Scale TPC-H Data Provisioning (SF = 1 GB)",
        [
            "• Base Dataset: 8 Relational Tables (8.66M rows; lineitem = 6.00M rows)",
            "• Local Columnar: ZSTD Parquet + DuckDB Native + Apache Arrow Buffers",
            "• Local Row-Store Baseline: SQLite 3 Full B-Tree with PK/FK/Date Indexes",
            "• Cloud Serverless: Google Cloud Storage -> BigQuery Unopt vs. Clustered"
        ],
        hdr_color="#1D4ED8", bg_color="#EFF6FF", border_color="#1E40AF", hdr_ratio=0.25
    )

    draw_card_with_header(
        ax, 0.52, 0.76, 0.46, 0.175,
        "PHASE 2: 75-Query Factorial Workload Design (25 Seeds x 3 Horizons)",
        [
            "• 5 Structural Tiers (5 Canonical Seeds/Tier = 25 Disjoint Seed Groups):",
            "  L1 (Scan/Filter), L2 (Multi-Join), L3 (Staged CTE), L4 (Window), L5 (UNNEST)",
            "• 3 Selectivity Perturbations per Seed (Total N = 75 Queries):",
            "  V1: High Selectivity (1-5%) | V2: Medium (15-30%) | V3: Full-Horizon (65-95%)"
        ],
        hdr_color="#047857", bg_color="#ECFDF5", border_color="#065F46", hdr_ratio=0.25
    )
    draw_arrow(ax, 0.48, 0.845, 0.52, 0.845)
    draw_arrow(ax, 0.75, 0.76, 0.50, 0.715)
    draw_arrow(ax, 0.25, 0.76, 0.50, 0.715)

    draw_card_with_header(
        ax, 0.02, 0.515, 0.96, 0.195,
        "PHASE 3: Static AST Feature Profiling, CSCI Construct Validation & Hardware Telemetry Instrumentation",
        [
            "• Parses all 75 SQL queries via SQLGlot into 9D vectors x_q = [D, N, J, S, W, U, C, CSCI, sigma(q)]^T (Mean CSCI = 29.66; Range: 15.3 in L1 to 37.1 in L4)",
            "• CSCI Construct Validation: Spearman rho = 0.4935 (p = 6.84e-6) vs. GPU Speedup; 1,000-Trial Monte Carlo Weight Perturbation Stability rho = 0.8189",
            "• High-Frequency Sensor Polling (250 Hz psutil + NVIDIA NVML + cudaEvent): Captures Latency (ms), CPU/GPU (%), RAM/VRAM (GB), Energy (J), Perf/Watt"
        ],
        hdr_color="#B45309", bg_color="#FFFBEB", border_color="#92400E", hdr_ratio=0.24
    )

    p_cards = [
        (0.02, "Paradigm 1: Row-Store", "#334155", "#F8FAFC", "#475569", [
            "• Engine: SQLite 3 (ANSI C)",
            "• Storage: Full-Scale B-Tree",
            "• Tuple-at-a-Time Volcano Model",
            "• Mean Latency: 8417.33 ms",
            "• 95% CI: [6412.2, 10819.7] ms",
            "• Mean Energy: 126.26 J (4.05 MB/J)"
        ]),
        (0.265, "Paradigm 2: Serverless Cloud", "#1E40AF", "#EFF6FF", "#1E3A8A", [
            "• Engine: Google BigQuery",
            "• Capacitor Columnar Storage",
            "• Cache Strictly Disabled",
            "• Clustered Mean: 745.39 ms",
            "• 95% CI: [515.6, 1026.5] ms",
            "• V1 I/O Pruning: 78.4% ($1.47/1M)"
        ]),
        (0.51, "Paradigm 3: Vectorized CPU", "#0F766E", "#F0FDFA", "#115E59", [
            "• 1) DuckDB (C++): 183.74 ms",
            "  95% CI: [110.7, 274.9] ms (6.40 J)",
            "• 2) Polars (Rust): 82.47 ms",
            "  95% CI: [67.9, 98.7] ms (3.17 J)",
            "• 3) DataFusion (Rust): 96.30 ms",
            "  95% CI: [78.3, 114.9] ms (3.39 J)"
        ]),
        (0.755, "Paradigm 4: SIMT CUDA GPU", "#991B1B", "#FEF2F2", "#7F1D1D", [
            "• Engine: PyTorch CUDA Tensor",
            "• GPU: NVIDIA RTX 2050 (4GB)",
            "• Cold Mean: 56.96 ms (0.65 J)",
            "  95% CI: [46.23, 68.44] ms",
            "• Warm Mean: 22.26 ms (0.25 J)",
            "  95% CI: [16.78, 28.68] ms"
        ])
    ]
    for cx, ctitle, chdr, cbg, cbdr, cbul in p_cards:
        draw_card_with_header(ax, cx, 0.235, 0.225, 0.235, ctitle, cbul, chdr, cbg, cbdr, hdr_ratio=0.21, title_fs=8.2, body_fs=7.3)
        draw_arrow(ax, cx + 0.1125, 0.515, cx + 0.1125, 0.472)
        draw_arrow(ax, cx + 0.1125, 0.235, cx + 0.1125, 0.192)

    draw_card_with_header(
        ax, 0.02, 0.025, 0.46, 0.165,
        "PHASE 5: Statistical Inference & 25-Seed LOSO-CV Routing",
        [
            "• Exact Equivalence EX = 1 (100%) | Friedman chi2_F(5) = 321.50 (p = 2.38e-67)",
            "• 10,000-Bootstrap Seed-Clustered 95% CIs & Cliff's Delta Effect Sizes",
            "• 5-Policy Seed-Grouped LOSO-CV: AST-Route Achieves 54.92 ms (34.98% Regret)"
        ],
        hdr_color="#6D28D9", bg_color="#F5F3FF", border_color="#5B21B6", hdr_ratio=0.27
    )

    draw_card_with_header(
        ax, 0.52, 0.025, 0.46, 0.165,
        "PHASE 6: Cryptographic SHA-256 Audit & Reproducibility",
        [
            "• Automated Booktabs Tables (Tables 1-8) & Unified 300 DPI Vector Figures",
            "• Zero-Discrepancy Verification Suite & SHA-256 Telemetry Audit Certificate",
            "• Open-Source Benchmark & Router Repository: github.com/preecha-msu"
        ],
        hdr_color="#15803D", bg_color="#F0FDF4", border_color="#166534", hdr_ratio=0.27
    )
    draw_arrow(ax, 0.48, 0.105, 0.52, 0.105)

    save_both(fig, "fig_experimental_workflow_300dpi.pdf")

# =================================================================================================
# FIGURE 3: Cross-Engine BigQuery vs DuckDB (3-Panel, Synced with Exact Table 2 Scanned MB Values)
# =================================================================================================
def generate_figure3_cross_engine():
    set_academic_serif_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.8, 4.2), dpi=300)
    x = np.arange(5)
    xt = ["L1\nScan/Filter", "L2\nMulti-Join", "L3\nNested CTE", "L4\nWindow", "L5\nUNNEST"]

    bq_unopt = [684.0, 1842.0, 1265.0, 1380.0, 1892.0]
    bq_clust = [432.9, 667.3, 493.0, 372.8, 1761.0]
    dk_parq  = [81.60, 403.94, 232.34, 111.52, 442.70]
    dk_nat   = [50.84, 264.53, 121.92, 78.16, 403.27]

    slot_ms  = [1980.8, 5193.6, 4637.3, 1907.5, 8737.2]
    mb_unopt = [332.5, 474.0, 417.3, 241.5, 273.4]
    mb_clust = [166.9, 308.4, 246.2, 132.6, 184.4]

    w1 = 0.19
    ax1.bar(x - 1.5*w1, bq_unopt, w1, label="BigQuery (Unoptimized)", color=PALETTE["crimson"], edgecolor="black", lw=0.75)
    ax1.bar(x - 0.5*w1, bq_clust, w1, label="BigQuery (Clustered)", color=PALETTE["amber_gold"], edgecolor="black", lw=0.75)
    ax1.bar(x + 0.5*w1, dk_parq,  w1, label="DuckDB (External Parquet)", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    ax1.bar(x + 1.5*w1, dk_nat,   w1, label="DuckDB (Native Columnar)", color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Latency (ms, Log Scale)")
    ax1.set_title("(a) Cross-Engine Execution Latency (SF=1GB)")
    ax1.set_xticks(x); ax1.set_xticklabels(xt)
    ax1.set_ylim(30, 3500)
    ax1.grid(axis="y", which="both", linestyle="--")
    ax1.legend(loc="upper left", frameon=True)

    w2 = 0.34
    ax2.bar(x - w2/2, bq_clust, w2, label="BQ Server Wall Time (ms)", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    ax2.bar(x + w2/2, slot_ms,  w2, label="BQ Total Slot Time (ms)", color=PALETTE["royal_purp"], edgecolor="black", lw=0.75)
    ax2.set_yscale("log")
    ax2.set_ylabel("Time Duration (ms, Log Scale)")
    ax2.set_title("(b) BigQuery Distributed Slot Time vs. Wall Time")
    ax2.set_xticks(x); ax2.set_xticklabels(xt)
    ax2.set_ylim(200, 14000)
    ax2.grid(axis="y", which="both", linestyle="--")
    ax2.legend(loc="upper left", frameon=True)

    ax3.bar(x - w2/2, mb_unopt, w2, label="Scanned MB (Unoptimized)", color=PALETTE["crimson"], edgecolor="black", lw=0.75)
    ax3.bar(x + w2/2, mb_clust, w2, label="Scanned MB (Clustered)", color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax3.set_ylabel("Data Scanned per Query (MB)")
    ax3.set_title("(c) Storage I/O Pruning via Clustering Keys")
    ax3.set_xticks(x); ax3.set_xticklabels(xt)
    ax3.set_ylim(0, 560)
    ax3.grid(axis="y", linestyle="--")
    ax3.legend(loc="upper right", frameon=True)

    plt.tight_layout(w_pad=1.8)
    save_both(fig, "fig_cross_engine_bq_duckdb_sf1_300dpi.pdf")

# =================================================================================================
# FIGURE 5: GPU PCIe Decomposition (Styled Identically to Figure 3: Academic Serif, Normal Title)
# =================================================================================================
def generate_figure5_gpu_pcie():
    set_academic_serif_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.4), dpi=300)
    x = np.arange(5)
    xt_csci = [
        "L1\nScan/Filter\n(CSCI=15.3)",
        "L2\nMulti-Join\n(CSCI=26.8)",
        "L3\nNested CTE\n(CSCI=36.7)",
        "L4\nWindow\n(CSCI=37.1)",
        "L5\nUNNEST\n(CSCI=32.4)"
    ]

    cpu_ms  = [50.84, 264.53, 121.92, 78.16, 403.27]
    warm_ms = [24.46, 49.05, 14.98, 9.94, 12.84]
    h2d_ms  = [30.56, 55.83, 43.26, 9.43, 34.46]
    cold_ms = [55.02, 104.88, 58.24, 19.37, 47.31]

    sp_warm_bq  = [17.7, 13.6, 32.9, 37.5, 137.1]
    sp_warm_cpu = [2.08, 5.39, 8.14, 7.86, 31.40]
    sp_cold_cpu = [0.92, 2.52, 2.09, 4.03, 8.52]

    w = 0.34
    ax1.bar(x - w/2, cpu_ms,  w, label="DuckDB Native Columnar (Vectorized CPU)", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    ax1.bar(x + w/2, warm_ms, w, label="GPU Warm Kernel Compute (In-VRAM)", color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax1.bar(x + w/2, h2d_ms,  w, bottom=warm_ms, label="PCIe Host-to-Device (H2D) Transfer Overhead", color=PALETTE["amber_gold"], edgecolor="black", lw=0.75, hatch="//")
    ax1.set_ylabel("Execution Latency (ms, Linear Scale)")
    ax1.set_title("(a) Vectorized CPU vs. CUDA GPU Latency Decomposition")
    ax1.set_xticks(x); ax1.set_xticklabels(xt_csci)
    ax1.set_ylim(0, 465)
    ax1.grid(axis="y", linestyle="--")
    ax1.legend(loc="upper left", frameon=True)

    for i in range(5):
        ax1.text(x[i] - w/2, cpu_ms[i] + 6, f"{cpu_ms[i]:.1f}", ha="center", va="bottom", fontsize=8.0)
        ax1.text(x[i] + w/2, cold_ms[i] + 6, f"{cold_ms[i]:.1f}", ha="center", va="bottom", fontsize=8.0, color="#7F1D1D")

    ax1.annotate(
        "L1 PCIe Crossover:\nCold GPU (55.0 ms) >\nCPU (50.8 ms)",
        xy=(0 + w/2, 55.0), xytext=(0.45, 165),
        arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#7F1D1D", lw=0.8),
        fontsize=8.0, color="#7F1D1D"
    )

    ax2.plot(x, sp_warm_bq,  "^--", color=PALETTE["royal_purp"], lw=1.7, markersize=6.5, label="Warm GPU vs. BigQuery Clustered (Cloud)")
    ax2.plot(x, sp_warm_cpu, "s-",  color=PALETTE["sage_green"], lw=1.7, markersize=6.5, label="Warm GPU (In-VRAM) vs. DuckDB Native CPU")
    ax2.plot(x, sp_cold_cpu, "o-",  color=PALETTE["crimson"],    lw=1.7, markersize=6.5, label="Cold GPU (Incl. PCIe H2D) vs. DuckDB Native CPU")
    ax2.axhline(1.0, color="black", linestyle=":", lw=1.1, label="CPU-GPU Break-Even Threshold (1.0x)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Speedup Ratio (Log Scale, Higher is Faster)")
    ax2.set_title("(b) GPU Acceleration Trajectory Across Complexity Tiers (L1-L5)")
    ax2.set_xticks(x); ax2.set_xticklabels(xt_csci)
    ax2.set_ylim(0.5, 250)
    ax2.grid(axis="y", which="both", linestyle="--")
    ax2.legend(loc="upper left", frameon=True)

    for i in range(5):
        ax2.text(x[i], sp_warm_bq[i]*1.15,  f"{sp_warm_bq[i]:.1f}x",  ha="center", fontsize=7.8, color="#4C1D95")
        ax2.text(x[i], sp_warm_cpu[i]*1.15, f"{sp_warm_cpu[i]:.2f}x", ha="center", fontsize=7.8, color="#065F46")
        ax2.text(x[i], sp_cold_cpu[i]*0.70, f"{sp_cold_cpu[i]:.2f}x", ha="center", fontsize=7.8, color="#7F1D1D")

    plt.tight_layout(w_pad=2.0)
    save_both(fig, "fig_gpu_pcie_decomposition_sf1_300dpi.pdf")

# =================================================================================================
# FIGURE 6: Six-Software Cross-Paradigm Comparison (2-Panel Like Fig 3: (a) Latency & (b) Throughput!)
# =================================================================================================
def generate_figure6_six_software():
    set_academic_serif_style()
    df_six = pd.read_csv(PROJECT_ROOT / "results" / "summary_tables" / "six_software_comparison_sf1.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.4), dpi=300)
    x = np.arange(len(df_six))
    w = 0.135
    xt = ["L1\nScan/Filter", "L2\nMulti-Join", "L3\nNested CTE", "L4\nWindow", "L5\nUNNEST"]

    # Panel (a): Execution Latency (ms, Log Scale)
    ax1.bar(x - 2.5*w, df_six["SQLite3_Full_ms"],      w, label="SQLite 3 (Row-Store)", color=PALETTE["slate_grey"], edgecolor="black", lw=0.7)
    ax1.bar(x - 1.5*w, df_six["BigQuery_Cloud_ms"],    w, label="BigQuery (Clustered)", color=PALETTE["deep_blue"],  edgecolor="black", lw=0.7)
    ax1.bar(x - 0.5*w, df_six["DuckDB_Native_ms"],     w, label="DuckDB (C++ DBMS)",    color=PALETTE["sage_green"], edgecolor="black", lw=0.7)
    ax1.bar(x + 0.5*w, df_six["Polars_Rust_ms"],       w, label="Polars (Rust Arrow)",  color=PALETTE["amber_gold"], edgecolor="black", lw=0.7)
    ax1.bar(x + 1.5*w, df_six["DataFusion_Rust_ms"],   w, label="DataFusion (Rust)",    color=PALETTE["royal_purp"], edgecolor="black", lw=0.7)
    ax1.bar(x + 2.5*w, df_six["GPU_RTX2050_Cold_ms"],  w, label="RTX 2050 GPU (Cold)",  color=PALETTE["crimson"],    edgecolor="black", lw=0.7, hatch="//")
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Latency (ms, Log Scale)")
    ax1.set_title("(a) Six-Engine Execution Latency Across Tiers (SF=1GB)")
    ax1.set_xticks(x); ax1.set_xticklabels(xt)
    ax1.set_ylim(8, 85000)
    ax1.grid(axis="y", which="both", linestyle="--")
    ax1.legend(ncol=2, loc="upper left", frameon=True, fontsize=7.8)

    # Panel (b): Effective Scan Throughput (GB/s, Log Scale)
    ax2.bar(x - 2.5*w, df_six["Throughput_SQLite_GBs"],     w, label="SQLite 3 (Row-Store)", color=PALETTE["slate_grey"], edgecolor="black", lw=0.7)
    ax2.bar(x - 1.5*w, df_six["Throughput_BQ_GBs"],         w, label="BigQuery (Clustered)", color=PALETTE["deep_blue"],  edgecolor="black", lw=0.7)
    ax2.bar(x - 0.5*w, df_six["Throughput_DuckDB_GBs"],     w, label="DuckDB (C++ DBMS)",    color=PALETTE["sage_green"], edgecolor="black", lw=0.7)
    ax2.bar(x + 0.5*w, df_six["Throughput_Polars_GBs"],     w, label="Polars (Rust Arrow)",  color=PALETTE["amber_gold"], edgecolor="black", lw=0.7)
    ax2.bar(x + 1.5*w, df_six["Throughput_DataFusion_GBs"], w, label="DataFusion (Rust)",    color=PALETTE["royal_purp"], edgecolor="black", lw=0.7)
    ax2.bar(x + 2.5*w, df_six["Throughput_GPU_Warm_GBs"],   w, label="RTX 2050 GPU (Warm)",  color=PALETTE["crimson"],    edgecolor="black", lw=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("Effective Scan Throughput (GB/s, Log Scale)")
    ax2.set_title("(b) Effective Analytical Scan Throughput Across Tiers")
    ax2.set_xticks(x); ax2.set_xticklabels(xt)
    ax2.set_ylim(0.015, 120)
    ax2.grid(axis="y", which="both", linestyle="--")
    ax2.legend(ncol=2, loc="upper left", frameon=True, fontsize=7.8)

    plt.tight_layout(w_pad=2.0)
    save_both(fig, "fig_six_software_comparison_sf1_300dpi.pdf")

# =================================================================================================
# FIGURE 7: Hardware Resource & Green OLAP Energy Efficiency (Styled Identically to Figure 3!)
# =================================================================================================
def generate_figure7_hardware_energy():
    set_academic_serif_style()
    df_hw = pd.read_csv(PROJECT_ROOT / "results" / "summary_tables" / "hardware_energy_summary_sf1.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.4), dpi=300)
    x = np.arange(len(df_hw))
    w = 0.15
    xt = ["L1\nScan/Filter", "L2\nMulti-Join", "L3\nNested CTE", "L4\nWindow", "L5\nUNNEST"]

    ax1.bar(x - 2*w, df_hw["Energy_SQLite_J"],   w, label="SQLite 3 (Row-Store)",         color=PALETTE["slate_grey"], edgecolor="black", lw=0.75)
    ax1.bar(x - 1*w, df_hw["Energy_DuckDB_J"],   w, label="DuckDB (C++ Vectorized)",      color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax1.bar(x + 0*w, df_hw["Energy_Polars_J"],   w, label="Polars (Rust Arrow)",          color=PALETTE["amber_gold"], edgecolor="black", lw=0.75)
    ax1.bar(x + 1*w, df_hw["Energy_GPU_Cold_J"], w, label="RTX 2050 GPU (Cold Incl. PCIe)", color=PALETTE["crimson"],  edgecolor="black", lw=0.75, hatch="//")
    ax1.bar(x + 2*w, df_hw["Energy_GPU_Warm_J"], w, label="RTX 2050 GPU (Warm In-VRAM)",  color=PALETTE["deep_blue"],  edgecolor="black", lw=0.75)
    ax1.set_yscale("log")
    ax1.set_ylabel("Energy Consumption per Query (Joules, Log Scale)")
    ax1.set_title("(a) Per-Query Energy Footprint Across Paradigms")
    ax1.set_xticks(x); ax1.set_xticklabels(xt)
    ax1.set_ylim(0.06, 900)
    ax1.grid(axis="y", which="both", linestyle="--")
    ax1.legend(loc="upper right", frameon=True, fontsize=7.9)

    ax2.bar(x - 2*w, df_hw["PerfWatt_SQLite_MB_per_J"],     w, label="SQLite 3 (Row-Store)",        color=PALETTE["slate_grey"], edgecolor="black", lw=0.75)
    ax2.bar(x - 1*w, df_hw["PerfWatt_DuckDB_MB_per_J"],     w, label="DuckDB (C++ Vectorized)",     color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    ax2.bar(x + 0*w, df_hw["PerfWatt_Polars_MB_per_J"],     w, label="Polars (Rust Arrow)",         color=PALETTE["amber_gold"], edgecolor="black", lw=0.75)
    ax2.bar(x + 1*w, df_hw["PerfWatt_DataFusion_MB_per_J"], w, label="DataFusion (Rust Arrow)",     color=PALETTE["royal_purp"], edgecolor="black", lw=0.75)
    ax2.bar(x + 2*w, df_hw["PerfWatt_GPU_Warm_MB_per_J"],   w, label="RTX 2050 GPU (Warm In-VRAM)", color=PALETTE["deep_blue"],  edgecolor="black", lw=0.75)
    ax2.set_yscale("log")
    ax2.set_ylabel("Performance per Watt (MB/Joule, Log Scale)")
    ax2.set_title("(b) Green OLAP Energy Efficiency (Perf/Watt)")
    ax2.set_xticks(x); ax2.set_xticklabels(xt)
    ax2.set_ylim(1.5, 9000)
    ax2.grid(axis="y", which="both", linestyle="--")
    ax2.legend(loc="upper left", frameon=True, fontsize=7.9)

    plt.tight_layout(w_pad=2.0)
    save_both(fig, "fig_hardware_energy_efficiency_sf1_300dpi.pdf")

# =================================================================================================
# FIGURE 8: 5-Policy AST-Route Evaluation (Styled Identically to Figure 3!)
# =================================================================================================
def generate_figure8_ast_router():
    set_academic_serif_style()
    df_eval = pd.read_csv(PROJECT_ROOT / "results" / "summary_tables" / "query_routing_evaluation_sf1.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.4), dpi=300)

    core_policies = [
        "3. Always BigQuery (Cloud OLAP)",
        "1. Always DuckDB (CPU DBMS)",
        "   Always Polars (Rust Arrow)",
        "4. Rule-Based Routing (Heuristic)",
        "   Syntax-Only AST Router (8-Dim)",
        "2. Always GPU (CUDA Cold-Start)",
        "5b. Proposed AST-Aware Cost-Reg (9-Dim)",
        "   Oracle Ideal Router (Lower Bound)"
    ]
    sub_p = df_eval[df_eval["Routing_Model_Policy"].isin(core_policies)].copy()
    sub_p["Order"] = sub_p["Routing_Model_Policy"].apply(lambda s: core_policies.index(s))
    sub_p = sub_p.sort_values("Order")

    short_labels = [
        "Always\nBigQuery",
        "Always\nDuckDB",
        "Always\nPolars",
        "Rule-Based\nRouting",
        "Syntax-AST\n(8-Dim)",
        "Always\nGPU (Cold)",
        "Proposed\nAST-Route",
        "Oracle\nIdeal"
    ]
    colors_1 = [
        PALETTE["deep_blue"], PALETTE["slate_grey"], PALETTE["amber_gold"],
        PALETTE["royal_purp"], PALETTE["teal_cyan"], PALETTE["crimson"],
        PALETTE["sage_green"], PALETTE["dark_navy"]
    ]

    x1 = np.arange(len(sub_p))
    bars1 = ax1.bar(x1, sub_p["Exec_Time_Mean_ms"], width=0.56, color=colors_1, edgecolor="black", lw=0.75)
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean Execution Time (ms, Log Scale)")
    ax1.set_title("(a) Mean Execution Time & Regret Across Routing Models")
    ax1.set_xticks(x1); ax1.set_xticklabels(short_labels, fontsize=8.1)
    ax1.set_ylim(20, 2200)
    ax1.grid(axis="y", which="both", linestyle="--")

    for b, reg_v in zip(bars1, sub_p["Performance_Regret_Pct"]):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h * 1.12, f"{h:.1f}ms\n(+{reg_v:.1f}%)", ha="center", va="bottom", fontsize=7.3)

    local_policies = [
        "1. Always DuckDB (CPU DBMS)",
        "   Always Polars (Rust Arrow)",
        "2. Always GPU (CUDA Cold-Start)",
        "4. Rule-Based Routing (Heuristic)",
        "   Syntax-Only AST Router (8-Dim)",
        "5b. Proposed AST-Aware Cost-Reg (9-Dim)"
    ]
    sub_loc = df_eval[df_eval["Routing_Model_Policy"].isin(local_policies)].copy()
    sub_loc["Order"] = sub_loc["Routing_Model_Policy"].apply(lambda s: local_policies.index(s))
    sub_loc = sub_loc.sort_values("Order")
    loc_labels = [
        "Always\nDuckDB",
        "Always\nPolars",
        "Always\nGPU",
        "Rule-Based\nRouting",
        "Syntax-AST\n(8-Dim)",
        "Proposed\nAST-Route"
    ]
    x2 = np.arange(len(sub_loc))
    w2 = 0.27
    b1 = ax2.bar(x2 - w2, sub_loc["Routing_Accuracy_Top1_Pct"], w2, label="Top-1 Routing Accuracy (%)", color=PALETTE["deep_blue"], edgecolor="black", lw=0.75)
    b2 = ax2.bar(x2,      sub_loc["Near_Optimal_Hit_Pct"],      w2, label="Near-Optimal Hit Rate (<=10% Oracle)", color=PALETTE["sage_green"], edgecolor="black", lw=0.75)
    b3 = ax2.bar(x2 + w2, sub_loc["Performance_Regret_Pct"],    w2, label="Performance Regret (%, Lower is Better)", color=PALETTE["crimson"], edgecolor="black", lw=0.75, hatch="//")

    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(b) Routing Accuracy vs. Performance Regret (25-Seed LOSO-CV)")
    ax2.set_xticks(x2); ax2.set_xticklabels(loc_labels, fontsize=8.3)
    ax2.set_ylim(0, 315)
    ax2.grid(axis="y", linestyle="--")
    ax2.legend(loc="upper right", frameon=True, fontsize=7.9)

    for b_grp in [b1, b2, b3]:
        for rect in b_grp:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width()/2, h + 3.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.0)

    plt.tight_layout(w_pad=2.0)
    save_both(fig, "fig_ast_query_router_sf1_300dpi.pdf")

if __name__ == "__main__":
    print("=" * 110)
    print(" [UNIFYING ALL FIGURES TO FIGURE 2-3 VISUAL STYLE (ACADEMIC SERIF + BANNER-HEADER CARDS)]")
    print("=" * 110)
    generate_figure1_framework()
    generate_figure2_workflow()
    generate_figure3_cross_engine()
    generate_figure5_gpu_pcie()
    generate_figure6_six_software()
    generate_figure7_hardware_energy()
    generate_figure8_ast_router()
    print("=" * 110)
    print(" [OK] All figures in manuscript/figures/ are now 100% visually unified!")
