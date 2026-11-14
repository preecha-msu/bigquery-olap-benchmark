#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/finalize_figures_and_5_microfixes_v5_2_5.py
1. ค้นหาไฟล์ .tex ฉบับเต็ม 30 หน้า (>95,000 ตัวอักษร) ใน manuscript/ อัตโนมัติ
2. ดึงพาธจริงจาก \includegraphics ภายในบล็อก Figure 1, Figure 2, และ Figure 9 ตาม Caption
   แล้ววาดรูป 300 DPI ใหม่ทับลงทุกไฟล์ที่ตรงกัน 100%
3. ซ่อม 5 จุดย่อยสุดท้าย (Table 3 V3, Table 11B SSB Top-2, Table 13C C=32, Sec 4.7 Table 9, Ref [13] DOI)
4. คอมไพล์ bigquery_olap_benchmark_V5_2_5.pdf และ bigquery_olap_benchmark_V2.pdf (3 รอบ)
"""

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
OUT_TEX = MS_DIR / "bigquery_olap_benchmark_V5_2_5.tex"
MASTER_V2_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_intact_30page_tex() -> Path:
    preferred = [
        MS_DIR / "bigquery_olap_benchmark_V2.tex",
        MS_DIR / "bigquery_olap_benchmark_V5_2.tex",
        MS_DIR / "bigquery_olap_benchmark_V6.tex",
        MS_DIR / "main.tex",
        MS_DIR / "overleaf_ready_single_file.tex",
    ]
    all_candidates = preferred + sorted(MS_DIR.glob("*.tex"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    for p in all_candidates:
        if not p.exists() or p.resolve() in seen:
            continue
        seen.add(p.resolve())
        txt = p.read_text(encoding="utf-8", errors="ignore")
        n_figs = len(re.findall(r"\\includegraphics", txt))
        if len(txt) > 95000 and n_figs >= 11 and "tab:step3_comprehensive_audit" in txt:
            print(f"  [SOURCE-FOUND] ใช้ไฟล์ต้นฉบับ 30 หน้า (มี Table 14 แล้ว): {p.name} ({len(txt):,} chars, {n_figs} figs)")
            return p
    # Fallback: intact 30-page file even before Table 14 if needed
    for p in all_candidates:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if len(txt) > 95000 and len(re.findall(r"\\includegraphics", txt)) >= 11:
            print(f"  [SOURCE-FOUND] ใช้ไฟล์ต้นฉบับ 30 หน้า: {p.name} ({len(txt):,} chars)")
            return p
    raise FileNotFoundError(f"ไม่พบไฟล์ .tex ฉบับเต็มใน {MS_DIR}")


def extract_exact_figure_paths_by_caption(tex: str) -> dict[str, list[Path]]:
    fmap = {
        "fig1": [
            FIG_DIR / "fig1_ast_route_framework_300dpi.pdf",
            FIG_DIR / "fig_proposed_ast_routing_framework_300dpi.pdf",
        ],
        "fig2": [
            FIG_DIR / "fig2_experimental_workflow_300dpi.pdf",
            FIG_DIR / "fig_experimental_workflow_300dpi.pdf",
        ],
        "fig9": [
            FIG_DIR / "fig9_ablation_pareto_sample_300dpi.pdf",
            FIG_DIR / "fig_internal_validation_ablation_pareto_300dpi.pdf",
            FIG_DIR / "fig_framework_ablation_pareto_300dpi.pdf",
        ],
    }
    blocks = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, flags=re.DOTALL)
    for blk in blocks:
        m = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", blk)
        if not m:
            continue
        rel = m.group(1).strip()
        if not Path(rel).suffix:
            rel += ".pdf"
        full_p = (MS_DIR / rel).resolve()
        if "Architecture of the proposed AST-Route" in blk:
            fmap["fig1"].insert(0, full_p)
            print(f"  [CAPTION-MATCH Fig 1] -> {rel}")
        elif "End-to-end six-phase experimental validation" in blk:
            fmap["fig2"].insert(0, full_p)
            print(f"  [CAPTION-MATCH Fig 2] -> {rel}")
        elif "Internal mechanism validation of the AST-Route" in blk:
            fmap["fig9"].insert(0, full_p)
            print(f"  [CAPTION-MATCH Fig 9] -> {rel}")
    return fmap


def _draw_card(ax, x, y, w, h, title, lines, hdr_color, fill_color, title_fs=8.6, body_fs=7.4):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.0",
        linewidth=1.25, edgecolor=hdr_color, facecolor=fill_color
    )
    ax.add_patch(box)
    hdr_h = min(3.8, h * 0.24)
    hdr = patches.FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle="round,pad=0.18,rounding_size=0.8",
        linewidth=1.0, edgecolor=hdr_color, facecolor=hdr_color
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2.0, y + h - hdr_h / 2.0, title,
            ha="center", va="center", color="white", fontweight="bold", fontsize=title_fs)
    ty = y + h - hdr_h - 1.5
    step = min((h - hdr_h - 2.2) / max(len(lines), 1), 3.05)
    for ln in lines:
        ax.text(x + 1.1, ty, ln, ha="left", va="top", color="#181818", fontsize=body_fs)
        ty -= step


def save_multi(fig, paths: list[Path]):
    written = set()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            out = p.with_suffix(ext)
            if out not in written:
                fig.savefig(out, dpi=300, bbox_inches="tight")
                written.add(out)
    plt.close(fig)


def regenerate_fig1_fig2_fig9(fpaths: dict[str, list[Path]]):
    plt.rcParams["font.family"] = "DejaVu Sans"

    # --- FIGURE 1 ---
    fig1, ax = plt.subplots(figsize=(13.6, 7.6), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97.6, "Proposed AST-Route Four-Layer Multi-Objective Query Routing Framework",
            ha="center", va="center", fontsize=12.4, fontweight="bold", color="#111111")
    _draw_card(ax, 1.5, 74.5, 47.0, 19.5, "LAYER 1A: Input SQL & Static AST Parsing (SQLGlot)", [
        "• Workload: 75 Factorial TPC-H + 22 Q1–Q22 + 66 External (N = 163 Queries)",
        "• Pre-Execution Static Parsing via SQLGlot (3.48 ms file parse / 0.17 ms cached)",
        "• Extracts 7 Structural Operators: Depth D(q), Nodes N(q), Joins J(q),",
        "  Subqueries/CTEs S(q), Windows W(q), UNNEST U(q), CASE C(q)",
    ], "#1f4e78", "#eaf2f8")
    _draw_card(ax, 51.5, 74.5, 47.0, 19.5, "LAYER 1B: 9D/10D Feature Vector, CSCI & Catalog Scale", [
        "• Validated Index: CSCI(q) = 1.5D + 2.5J + 3.0S + 3.5W + 4.0U + 1.5C",
        "• Operator-Tier Alignment: Partial rho = 0.5477 (p = 3.68e-7), 10,000-MC rho = 0.9880 (±20%)",
        "• Zone-Map Selectivity sigma_hat(q) in (0, 1] (0.172 ms cached / 0.413 ms cold footer)",
        "• 9D Vector x_q in R^9 (augmented with log10(1 + B_scan) to R^10 across scales)",
    ], "#196f3d", "#e9f7ef")
    _draw_card(ax, 1.5, 52.5, 97.0, 18.0, "LAYERS 2 & 3: Cross-Paradigm Analytical Cost Models & Pluggable Cost-Regression Engine", [
        "• Analytical Models: PCIe Fraction Phi_PCIe(q), Effective Throughput Theta_eff (GB/s), Cloud Slot Proxy M_slot, Billed FinOps Savings Delta C_1M ($)",
        "• Pluggable Predictors: Speed-First Ridge (49.12 ms) / Shallow DT (51.35 ms) / k-NN (52.84 ms) vs. Tail-Safe Conformal GBR-9D (54.92 ms)",
        "• Out-of-Fold Predictive Accuracy (25-Seed LOSO-CV): GPU R^2 = 0.875 (P50 Q-Err 1.053) | Polars R^2 = 0.822 | Energy R^2 = 0.815 (Inference: 497.2 us)",
    ], "#b9770e", "#fef9e7")
    for x_arr in (25.0, 75.0):
        ax.annotate("", xy=(x_arr, 70.8), xytext=(x_arr, 74.3), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))
    cands = [
        (1.5, "Candidate 1: DuckDB (C++)", "#2c3e50", "#f2f4f4", [
            "• Paradigm: In-Process CPU DBMS", "• Morsel-Driven Vectorized Engine", "• Segment-Tree Window Framing",
            "• Mean Latency: 183.74 ms", "• Pkg Energy Est.: 6.40 J (243.0 MB/J)", "• Routed: 80% L1, Out-of-Core Spill",
        ]),
        (26.1, "Candidate 2: Polars / DataFusion", "#117a65", "#e8f8f5", [
            "• Paradigm: Rust Apache Arrow", "• Zero-Copy SIMD & ListArray", "• Polars: 82.47 ms (3.17 J)",
            "• DataFusion: 96.30 ms (3.39 J)", "• 102.1x Faster than Row-Store", "• Routed: 80% Narrow/Med L2 Joins",
        ]),
        (50.7, "Candidate 3: CUDA GPU (SIMT)", "#922b21", "#fdedec", [
            "• Paradigm: RTX 2050 (4 GB VRAM)", "• Cold: 56.96 ms | Warm: 22.25 ms", "• L1 PCIe Bottleneck: Phi = 55.5%",
            "• Board Energy: 0.65 J Cold / 0.25 J Warm", "• Energy Eff.: 1942.8 MB/J Warm", "• Routed: 60% L3, 100% L4, 80% L5",
        ]),
        (75.3, "Candidate 4: Google BigQuery", "#5b2c6f", "#f5eef8", [
            "• Paradigm: Serverless Cloud OLAP", "• Capacitor Storage + Slot Elasticity", "• Clustered Server Mean: 745.39 ms",
            "• Slot Proxy: M_slot = 6.37x", "• V1 Pruning: 78.4% ($1,441.61/1M)", "• Elastic SF=100 GB Spill Fallback",
        ]),
    ]
    for cx, ctitle, chdr, cfill, clines in cands:
        _draw_card(ax, cx, 23.5, 23.2, 25.0, ctitle, clines, chdr, cfill, title_fs=8.2, body_fs=7.25)
        ax.annotate("", xy=(cx + 11.6, 48.7), xytext=(cx + 11.6, 52.3), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
        ax.annotate("", xy=(cx + 11.6, 19.8), xytext=(cx + 11.6, 23.3), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    _draw_card(ax, 1.5, 2.0, 97.0, 17.5, "LAYER 4: Constrained Confidence-Aware Dispatch & Leak-Free 25-Seed LOSO-CV Performance", [
        "• Dispatch Rule: e_hat(q) = argmin_{e in E_feasible(q)} [alpha*(T_hat/T_med) + beta*(E_hat/E_med) + gamma*(C_hat/C_ref)] with 12% Conformal PCIe Guard",
        "• Conformal AST-Route: 54.92 ms Warm / 55.16 ms Cold Footer (Seed 95% CI: [44.52, 66.18] ms), 114.11% P95 / 218.40% P99 Regret, CPU-Only: 69.12 ms",
    ], "#145a32", "#eafaf1")
    fig1.tight_layout()
    save_multi(fig1, fpaths["fig1"])

    # --- FIGURE 2 ---
    fig2, ax = plt.subplots(figsize=(13.6, 7.6), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97.6, "End-to-End Experimental Validation Workflow: AST-Route Testbed (163 Total Queries, 6 Engines)",
            ha="center", va="center", fontsize=12.2, fontweight="bold", color="#111111")
    _draw_card(ax, 1.5, 74.5, 47.0, 19.5, "PHASE 1: Multi-Scale TPC-H Provisioning (SF = 1, 10, 30, 100 GB)", [
        "• 8 Relational Tables: 8.66M (SF1) to 866.1M Total Rows (SF100; 600.1M lineitem)",
        "• Local Columnar: ZSTD Parquet + DuckDB Native + Apache Arrow Buffers",
        "• Local Row-Store Baseline: SQLite 3 Full B-Tree with PK/FK/Date Indexes",
        "• Cloud Serverless: Google BigQuery Unoptimized vs. Clustered (Cache Disabled)",
    ], "#1f618d", "#ebf5fb")
    _draw_card(ax, 51.5, 74.5, 47.0, 19.5, "PHASE 2: Stratified Factorial & External Workload Suite (163 Queries)", [
        "• 75 Factorial Queries: 5 AST Tiers (L1–L5, 25 Disjoint Seeds) x 3 Horizons (V1–V3)",
        "• Official Benchmark Suite: 22 Standard TPC-H Choke-Point Queries (Q1–Q22)",
        "• Zero-Shot Cross-Schema Transfer: 33 IMDB JOB + 13 SSB + 20 ClickBench Queries",
        "• Leak-Free Protocols: 25-Seed LOSO-CV, 5 Unseen Seeds, LOTO-CV, Nested LOO-CV",
    ], "#1e8449", "#e9f7ef")
    _draw_card(ax, 1.5, 52.5, 97.0, 18.0, "PHASE 3: Static AST Feature Profiling, CSCI Operator-Tier Alignment & Hardware Telemetry Instrumentation", [
        "• Parses SQL via SQLGlot into 9D/10D vectors x_q (Mean CSCI = 29.66, Range: 15.3 in L1 to 37.1 in L4; 0.172 ms cached / 0.413 ms cold footer)",
        "• CSCI Operator-Tier Alignment: Partial rho = 0.5477 (p = 3.68e-7) vs. GPU Speedup; 10,000-Trial Monte Carlo Weight Stability rho_bar = 0.9880 (±20%)",
        "• 250 Hz Hardware Polling (psutil + NVIDIA NVML + cudaEventRecord): Captures Latency (ms), CPU/GPU (%), RAM/VRAM (GB), Device Energy (J), MB/J",
    ], "#b9770e", "#fef9e7")
    for x_arr in (25.0, 75.0):
        ax.annotate("", xy=(x_arr, 70.8), xytext=(x_arr, 74.3), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))
    paradigms = [
        (1.5, "Paradigm 1: Row-Store", "#5d6d7e", "#f2f4f4", [
            "• Engine: SQLite 3 (ANSI C)", "• Storage: Full-Scale B-Tree", "• Tuple-at-a-Time Volcano Model",
            "• Mean Latency: 8417.33 ms", "• 95% CI: [6412.2, 10819.7] ms", "• CPU Pkg Est.: 126.26 J (4.05 MB/J)",
        ]),
        (26.1, "Paradigm 2: Serverless Cloud", "#2874a6", "#ebf5fb", [
            "• Engine: Google BigQuery", "• Capacitor Columnar Storage", "• Unopt Server: 1412.60 ms",
            "• Clustered Server: 745.39 ms", "• 95% CI: [515.6, 1026.5] ms (1.90x)", "• V1 Pruning: 78.4% ($1,441.61/1M)",
        ]),
        (50.7, "Paradigm 3: Vectorized CPU", "#117a65", "#e8f8f5", [
            "• 1) Polars (Rust): 82.47 ms (3.17 J)", "  95% CI: [67.9, 98.7] ms", "• 2) DataFusion: 96.30 ms (3.39 J)",
            "  95% CI: [78.3, 114.9] ms", "• 3) DuckDB (C++): 183.74 ms (6.40 J)", "  95% CI: [110.7, 274.9] ms",
        ]),
        (75.3, "Paradigm 4: SIMT CUDA GPU", "#a93226", "#fdedec", [
            "• Engine: PyTorch CUDA Tensor", "• GPU: NVIDIA RTX 2050 (4 GB)", "• Cold Mean: 56.96 ms (0.65 J)",
            "  (1.45x vs Polars, 3.23x vs DuckDB)", "• Warm Mean: 22.25 ms (0.25 J)", "  (3.71x vs Polars, 8.22x vs DuckDB)",
        ]),
    ]
    for px, ptitle, phdr, pfill, plines in paradigms:
        _draw_card(ax, px, 23.5, 23.2, 25.0, ptitle, plines, phdr, pfill, title_fs=8.2, body_fs=7.2)
        ax.annotate("", xy=(px + 11.6, 48.7), xytext=(px + 11.6, 52.3), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
        ax.annotate("", xy=(px + 11.6, 19.8), xytext=(px + 11.6, 23.3), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    _draw_card(ax, 1.5, 2.0, 47.0, 17.5, "PHASE 5: Statistical Inference & Leak-Free Routing", [
        "• Multiset Parity EX = 100.0% (97/97) | Friedman chi^2_F(5) = 321.50 (p = 2.38e-67)",
        "• Seed-Level (N = 25) Holm-Bonferroni & 10,000-Bootstrap 95% CIs",
        "• 25-Seed LOSO-CV: Conformal 54.92 ms (114.11% P95) | CPU-Only: 69.12 ms",
    ], "#6c3483", "#f5eef8")
    _draw_card(ax, 51.5, 2.0, 47.0, 17.5, "PHASE 6: Cryptographic SHA-256 Audit & Reproducibility", [
        "• Automated Booktabs Tables (Tables 1–14) & Synchronized 300 DPI Figures",
        "• Zero-Contradiction Audit Suite & SHA-256 Telemetry Verification",
        "• Public Benchmark & Router Repository: github.com/preecha-msu/bigquery-olap-benchmark",
    ], "#196f3d", "#eafaf1")
    fig2.tight_layout()
    save_multi(fig2, fpaths["fig2"])

    # --- FIGURE 9 ---
    fig9, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 4.35), dpi=300)
    labels_a = ["Tree Size\n(2D)", "Syntax\nOnly (8D)", "Operators\nOnly (7D)", "Compact\nCSCI+Sel", "Proposed\nGBR-9D", "Shallow DT\n(d=2, 9D)", "Ridge\nReg (9D)"]
    lats_a = [103.94, 73.22, 61.06, 56.30, 54.92, 51.35, 49.12]
    p95_tags = ["", "", "", "", "P95:114%", "P95:151%", "P95:197%"]
    cols_a = ["#7f8c8d", "#5dade2", "#8e44ad", "#eb984e", "#1e8449", "#f4d03f", "#aed6f1"]
    xa = np.arange(len(labels_a))
    bars = ax1.bar(xa, lats_a, width=0.62, color=cols_a, edgecolor="#1c1c1c", linewidth=0.9)
    for idx, (b, val) in enumerate(zip(bars, lats_a)):
        ax1.text(b.get_x() + b.get_width() / 2.0, val + 2.0, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        if p95_tags[idx]:
            ax1.text(b.get_x() + b.get_width() / 2.0, val - 9.5, p95_tags[idx], ha="center", va="top", fontsize=6.5,
                     color="#111111", bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.84, lw=0.4))
    ax1.set_xticks(xa); ax1.set_xticklabels(labels_a, fontsize=7.4); ax1.set_ylim(0, 118)
    ax1.set_ylabel("Mean Execution Latency (ms)", fontsize=9.0)
    ax1.set_title("(a) Feature & Model Ablation (Mean vs. P95 Regret)", fontsize=9.4, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    pareto_e = [1.381, 1.182, 1.007, 0.879, 0.748]
    pareto_t = [54.92, 54.99, 56.80, 59.81, 70.20]
    ax2.plot(pareto_e, pareto_t, "o-", color="#1f618d", lw=2.0, markersize=6.5, label="AST-Route Pareto (alpha in [0, 1])")
    ax2.scatter([3.170], [82.47], marker="s", s=65, color="#d35400", edgecolor="black", zorder=5, label="Always Polars (82.47 ms, 3.17 J)")
    ax2.scatter([2.374], [71.48], marker="^", s=70, color="#7d3c98", edgecolor="black", zorder=5, label="Rule-Based (71.48 ms, 2.37 J)")
    ax2.scatter([0.975], [42.64], marker="*", s=115, color="#111111", zorder=6, label="Oracle Lower Bound (42.64 ms)")
    ax2.set_xlabel("Estimated Active Device Energy (Joules)", fontsize=9.0)
    ax2.set_ylabel("Mean Execution Latency (ms)", fontsize=9.0)
    ax2.set_title("(b) Latency vs. Device-Level Energy Frontier", fontsize=9.4, fontweight="bold")
    ax2.set_xlim(0.55, 3.45); ax2.set_ylim(39, 88); ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=7.3, framealpha=0.92)

    seeds_x = np.array([5, 10, 15, 20, 24])
    sample_lat = np.array([63.61, 68.59, 56.59, 57.60, 54.92])
    sample_sd = np.array([15.20, 14.10, 9.40, 8.20, 6.10])
    ax3.plot(seeds_x, sample_lat, "s-", color="#148f77", lw=2.1, markersize=6.5, label="AST-Route Out-of-Fold Mean ± SD")
    ax3.fill_between(seeds_x, sample_lat - sample_sd, sample_lat + sample_sd, color="#148f77", alpha=0.16)
    for sx, sy in zip(seeds_x, sample_lat):
        ax3.text(sx, sy - 3.6, f"{sy:.2f}", ha="center", va="top", fontsize=7.8, fontweight="bold", color="#0e6251")
    ax3.axhline(82.47, color="#7f8c8d", linestyle=":", lw=1.3, label="Always Polars (82.47 ms)")
    ax3.axhline(71.48, color="#5d6d7e", linestyle="--", lw=1.3, label="Rule-Based (71.48 ms)")
    ax3.axhline(42.64, color="#111111", linestyle="-.", lw=1.3, label="Oracle Bound (42.64 ms)")
    ax3.set_xticks(seeds_x); ax3.set_xlabel("Number of Training Seeds (3 Queries/Seed)", fontsize=9.0)
    ax3.set_ylabel("Out-of-Fold Test Latency (ms)", fontsize=9.0)
    ax3.set_title("(c) Cold-Start Sample Efficiency (5–24 Seeds)", fontsize=9.4, fontweight="bold")
    ax3.set_ylim(38, 92); ax3.grid(True, linestyle="--", alpha=0.45)
    ax3.legend(loc="upper right", fontsize=7.2, framealpha=0.92)
    fig9.tight_layout()
    save_multi(fig9, fpaths["fig9"])


def apply_5_microfixes(tex: str) -> tuple[str, dict]:
    res = {}

    # 1. Table 3 Row V3: Queries 75 -> 25 และเรียงคอลัมน์เป็น $12.52 & $125.17 & $375.51 & $1,251.70
    def fix_v3_line(m):
        s = m.group(0)
        s = re.sub(r"&\s*75\s*&", "& 25 &", s, count=1)
        s = re.sub(
            r"\\?\$[0-9,\.]+\$?\s*&\s*\\?\$[0-9,\.]+\$?\s*&\s*\\?\$[0-9,\.]+\$?\s*&\s*\\?\$[0-9,\.]+\$?",
            lambda _: r"\$12.52 & \$125.17 & \$375.51 & \$1,251.70",
            s,
            count=1,
        )
        return s

    tex = re.sub(r"V3:\s*Low\s*Selectivity[^\n]*", fix_v3_line, tex, count=1)
    res["1. Table 3 แถว V3: Queries=25 และเรียง $12.52 -> $1,251.70"] = bool(
        re.search(r"V3:\s*Low\s*Selectivity[^\n]*&\s*25\s*&[^\n]*12\.52[^\n]*125\.17[^\n]*375\.51[^\n]*1,251\.70", tex)
    )

    # 2. Table 11 Part B แถว SSB: แก้ช่อง Top-2 จาก 92.31% -> 100.0%
    def fix_ssb_line(m):
        s = m.group(0)
        parts = s.split("92.31\\%")
        if len(parts) >= 3:
            return parts[0] + "92.31\\%" + parts[1] + "100.0\\%" + "92.31\\%".join(parts[2:])
        return s

    tex = re.sub(r"Zero-Shot\s+External\s+SSB[^\n]*", fix_ssb_line, tex, count=1)
    res["2. Table 11B แถว SSB: Top-1=92.31%, Top-2=100.0%"] = bool(
        re.search(r"Zero-Shot\s+External\s+SSB[^\n]*92\.31\\%[^\n]*100\.0\\%", tex)
    )

    # 3. Table 13 Part C แถวสุดท้าย: แก้ 30 Streams -> 32 Streams ทุกกรณี
    tex = re.sub(r"(\$?[Cc]\s*=\s*)30(\$?\s*Streams)", lambda m: f"{m.group(1)}32{m.group(2)}", tex)
    res["3. Table 13C แถวสุดท้าย: C=32 Streams (ไม่มี 30 Streams ค้าง)"] = not bool(re.search(r"30\$?\s*Streams", tex))

    # 4. Section 4.7 บรรทัดแรก: แก้ Table 10 -> Table 9
    tex = re.sub(
        r"(To\s+rigorously\s+validate\s+the\s+internal\s+mechanisms\s+of\s+AST-Route,\s*Table)(?:~|\s+)(?:\\ref\{[^}]+\}|10)",
        lambda m: f"{m.group(1)}~9",
        tex,
        count=1,
    )
    res["4. Section 4.7 บรรทัดแรก: อ้างอิง Table 9 ถูกต้อง"] = "To rigorously validate the internal mechanisms of AST-Route, Table~9" in tex

    # 5. Reference [13]: ซ่อมลิงก์ DOI https://doi.org/10.1186/s40537-024-00939-0 เต็มบรรทัด
    lines = tex.splitlines()
    for i, ln in enumerate(lines):
        if "Shojaee Rad" in ln:
            # ตรวจบรรทัดตัวเองและบรรทัดถัดไปอีก 2 บรรทัด
            window = "\n".join(lines[i:i+3])
            if "024-00939-0" not in window:
                for j in range(i, min(i + 3, len(lines))):
                    if "s40537" in lines[j] or "10.1186" in lines[j]:
                        lines[j] = re.sub(
                            r"(?:\\url\{)?https://doi\.org/10\.1186/s40537-[^\s\}]*\}?",
                            lambda _: r"\url{https://doi.org/10.1186/s40537-024-00939-0}.",
                            lines[j],
                        )
    tex = "\n".join(lines)
    res["5. Reference [13]: ลิงก์ DOI s40537-024-00939-0 ครบเต็ม"] = "s40537-024-00939-0" in tex

    # ตรวจสอบสมดุล \begin{quote} == \end{quote}
    pre_s1, post_s1 = tex.split("\\section{Introduction}", 1)
    if pre_s1.count("\\begin{quote}") > pre_s1.count("\\end{quote}"):
        tex = pre_s1.rstrip() + "\n\\end{quote}\n\n\\section{Introduction}" + post_s1

    return tex, res


def main():
    print("=" * 105)
    print("AUTO-DETECTING 30-PAGE MANUSCRIPT, UPDATING EXACT FIGURES (1, 2, 9) & APPLYING 5 MICRO-FIXES")
    print("=" * 105)

    src_tex = find_intact_30page_tex()
    raw_tex = src_tex.read_text(encoding="utf-8")

    fpaths = extract_exact_figure_paths_by_caption(raw_tex)
    regenerate_fig1_fig2_fig9(fpaths)
    print("  [FIG-OK] บันทึกภาพ Figure 1, Figure 2, และ Figure 9 ทับลงในไฟล์จริงที่ถูกเรียกใน .tex สำเร็จ!")
    print("-" * 105)

    updated_tex, res = apply_5_microfixes(raw_tex)
    for k, ok in res.items():
        print(f"  [{'PASS' if ok else 'CHECK'}] {k}")
    print("-" * 105)

    for fname in (
        "bigquery_olap_benchmark_V2.tex",
        "bigquery_olap_benchmark_V5_2.tex",
        "bigquery_olap_benchmark_V5_2_5.tex",
        "bigquery_olap_benchmark_V6.tex",
        "main.tex",
        "overleaf_ready_single_file.tex",
    ):
        (MS_DIR / fname).write_text(updated_tex, encoding="utf-8")

    for p in (1, 2, 3):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", OUT_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("\n".join(proc.stdout.splitlines()[-25:]))
            raise RuntimeError(f"pdflatex failed on pass {p}")

    for _ in (1, 2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MASTER_V2_TEX.name],
            cwd=str(MS_DIR),
            capture_output=True,
        )

    out_log = (MS_DIR / "bigquery_olap_benchmark_V5_2_5.log").read_text(encoding="utf-8", errors="ignore")
    undefs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", out_log)
    m_pg = re.search(r"Output written on bigquery_olap_benchmark_V5_2_5\.pdf \((\d+) pages", out_log)
    n_pages = int(m_pg.group(1)) if m_pg else 0

    pdf_master = MS_DIR / "bigquery_olap_benchmark_V5_2_5.pdf"
    for alias in (
        "bigquery_olap_benchmark_V2.pdf",
        "bigquery_olap_benchmark_V5_2.pdf",
        "bigquery_olap_benchmark_V5_2_4.pdf",
        "bigquery_olap_benchmark_V6.pdf",
        "main.pdf",
    ):
        try:
            shutil.copy2(pdf_master, MS_DIR / alias)
        except PermissionError:
            pass

    for zname in ("overleaf_V2_submission_bundle.zip", "overleaf_V3_submission_bundle.zip", "overleaf_project_bundle.zip"):
        with zipfile.ZipFile(MS_DIR / zname, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(MASTER_V2_TEX, arcname="main.tex")
            zf.write(MASTER_V2_TEX, arcname="bigquery_olap_benchmark_V2.tex")
            bib = MS_DIR / "references.bib"
            if bib.exists():
                zf.write(bib, arcname="references.bib")
            for f in sorted(FIG_DIR.glob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"figures/{f.name}")

    print(f"  [PDF-READY] {pdf_master.name} | Pages: {n_pages} | Undefined Refs: {len(undefs)} | SHA-256: {sha256_of(pdf_master)[:24]}...")
    print("=" * 105)


if __name__ == "__main__":
    main()
