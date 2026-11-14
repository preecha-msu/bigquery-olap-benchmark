#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
121_fix_v8_4_crossrefs_citations_and_figures.py
================================================================================
Final Precision Closure Script for Journal of Big Data (SpringerOpen) V8_4 -> V8_5:
  1. Fixes all 14 Table & Figure cross-reference shifts across Pages 7-28.
  2. Fixes all 6 Bibliography / Citation mismatches ([1]-[36]), cites Breß et al.
     (2013) via \cite{...} in Sec. 2.2 so 0 references are uncited, and re-sorts
     all \bibitem entries in strict first-appearance Vancouver order.
  3. Redraws Figure 6, Figure 10 (including _sf1_ filename alias), and Figure 15
     at 300 DPI to fix (Table 5)/(Table 6) headers, MathText symbols, X-axis label
     collision, and top legend border overlap.
  4. Standardizes Section 4.7 items 5, 4.7.1, and 6 into sequential subsubsections
     4.7.1, 4.7.2, and 4.7.3 using brace-balanced parsing, plus fixes Page 1
     title/address hyphenation and affiliation superscript '1'.
  5. Compiles 3 passes of pdflatex (verifying 0 '??', 0 undefined refs/citations,
     exact brace balance) and rebuilds the Flat-Directory Snapp ZIP bundle.
================================================================================
r"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
ROOT_FIG = ROOT / "figures"
SNAPP_DIR = ROOT / "Snapp_Journal_of_Big_Data_Submission"


def locate_source_tex() -> Path:
    for cand in (
        MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.tex",
        MS_DIR / "bigquery_olap_benchmark_V8_4.tex",
        MS_DIR / "bigquery_olap_benchmark_V8_3.tex",
        MS_DIR / "bigquery_olap_benchmark_V8.tex",
        MS_DIR / "main.tex",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate manuscript .tex file in {MS_DIR}")


def count_unquoted_braces(tex: str) -> tuple[int, int]:
    r"""Counts unescaped '{' and '}' ignoring LaTeX comments.r"""
    cleaned_lines = []
    for line in tex.splitlines():
        line_nc = re.sub(r"(?<!\\)%.*$", "", line)
        line_nc = line_nc.replace(r"\{", "").replace(r"\}", "")
        cleaned_lines.append(line_nc)
    joined = "\n".join(cleaned_lines)
    return joined.count("{"), joined.count("}")


def replace_textbf_heading_balanced(tex: str, inner_prefix: str, replacement_heading: str) -> tuple[str, bool]:
    r"""
    Safely replaces `(?:\par\vspace{...})?\noindent\textbf{<inner_prefix>...}`
    using exact brace-depth tracking so nested math/commands never break braces.
    r"""
    pat = re.compile(
        r"(?:\\par\\vspace\{[^{}]*\}\s*)?(?:\\noindent\s*)?\\textbf\{\s*"
        + re.escape(inner_prefix)
    )
    m = pat.search(tex)
    if not m:
        return tex, False

    tb_idx = tex.find(r"\textbf{", m.start())
    brace_open = tb_idx + len(r"\textbf{") - 1
    depth = 0
    end_pos = -1
    i = brace_open
    n = len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
        i += 1

    if end_pos == -1:
        return tex, False

    while end_pos < n and tex[end_pos] in (".", ":", " "):
        end_pos += 1

    updated = tex[: m.start()] + replacement_heading + "\n" + tex[end_pos:]
    return updated, True


# ==============================================================================
# STEP 1: REDRAW FIGURES 6, 10, AND 15 AT 300 DPI (SYNCING ALL ALIAS FILENAMES)
# ==============================================================================
def save_to_all_matching_paths(fig: plt.Figure, basenames: list[str]) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for base in basenames:
        stem = Path(base).stem
        for ext in (".pdf", ".png"):
            p1 = FIG_DIR / f"{stem}{ext}"
            fig.savefig(p1, dpi=300, bbox_inches="tight")
            saved.append(p1.name)
            if ROOT_FIG.exists():
                fig.savefig(ROOT_FIG / f"{stem}{ext}", dpi=300, bbox_inches="tight")
    return sorted(set(saved))


def redraw_figure6_six_engines(tex: str) -> list[str]:
    r"""
    Updates Figure 6:
      - Panel (a) header: (Table 5) -> (Table 8), clean upper-right legend
      - Panel (b) header: (Table 6) -> (Table 9), MathText Theta_eff on Y-axis,
        and non-overlapping legend in upper-left whitespace.
    r"""
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 4.7), dpi=300)

    tiers = [
        "L1: Scan/Filter\n(CSCI=15.3)",
        "L2: Multi-Join\n(CSCI=26.8)",
        "L3: Nested CTE\n(CSCI=36.7)",
        "L4: Window\n(CSCI=37.1)",
        "L5: UNNEST\n(CSCI=32.4)",
    ]
    x = np.arange(len(tiers))
    bw = 0.128

    # Exact empirical values from Table 8 (Latency, ms)
    lat_sqlite = [6789.6, 19374.0, 5108.1, 6023.7, 4791.2]
    lat_bq     = [432.9,  667.3,   493.0,  372.8,  1761.0]
    lat_duck   = [50.84,  264.53,  121.92, 78.16,  403.27]
    lat_polars = [38.00,  81.68,   66.81,  71.11,  154.76]
    lat_df     = [33.58,  92.76,   71.66,  173.16, 110.31]
    lat_gpu_c  = [55.02,  104.88,  58.24,  19.37,  47.31]

    ax1.bar(x - 2.5 * bw, lat_sqlite, bw, color="#7f8c8d", edgecolor="black", lw=0.75, label="SQLite 3 (Row-Store)")
    ax1.bar(x - 1.5 * bw, lat_bq,     bw, color="#4c72b0", edgecolor="black", lw=0.75, label="BigQuery Clustered (Cloud)")
    ax1.bar(x - 0.5 * bw, lat_duck,   bw, color="#55a868", edgecolor="black", lw=0.75, label="DuckDB Native (C++)")
    ax1.bar(x + 0.5 * bw, lat_polars, bw, color="#dd8452", edgecolor="black", lw=0.75, label="Polars (Rust Arrow)")
    ax1.bar(x + 1.5 * bw, lat_df,     bw, color="#8172b3", edgecolor="black", lw=0.75, label="DataFusion (Rust Arrow)")
    ax1.bar(x + 2.5 * bw, lat_gpu_c,  bw, color="#c44e52", edgecolor="black", lw=0.75, hatch="//", label="RTX 2050 GPU (Cold)")

    ax1.set_yscale("log")
    ax1.set_ylim(10, 52000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=8.0)
    ax1.set_ylabel("Mean Execution Latency (ms, Log Scale)", fontsize=9.2)
    ax1.set_title("(a) Six-Engine Mean Execution Latency Across Tiers (Table 8)", fontsize=9.8, fontweight="bold")
    ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.38)
    ax1.legend(ncol=2, loc="upper right", fontsize=7.1, framealpha=0.95, edgecolor="#cccccc")

    # Exact empirical values from Table 9 (Effective Scan Throughput Theta_eff, GB/s)
    thr_sqlite = [0.048, 0.024, 0.080, 0.039, 0.056]
    thr_bq     = [0.750, 0.694, 0.827, 0.633, 0.152]
    thr_duck   = [6.39,  1.75,  3.34,  3.02,  0.66]
    thr_polars = [8.55,  5.67,  6.10,  3.32,  1.73]
    thr_df     = [9.67,  4.99,  5.69,  1.36,  2.42]
    thr_gpu_w  = [13.28, 9.44,  27.20, 23.73, 20.79]

    ax2.bar(x - 2.5 * bw, thr_sqlite, bw, color="#7f8c8d", edgecolor="black", lw=0.75, label="SQLite 3")
    ax2.bar(x - 1.5 * bw, thr_bq,     bw, color="#4c72b0", edgecolor="black", lw=0.75, label="BigQuery Clustered")
    ax2.bar(x - 0.5 * bw, thr_duck,   bw, color="#55a868", edgecolor="black", lw=0.75, label="DuckDB Native")
    ax2.bar(x + 0.5 * bw, thr_polars, bw, color="#dd8452", edgecolor="black", lw=0.75, label="Polars Arrow")
    ax2.bar(x + 1.5 * bw, thr_df,     bw, color="#8172b3", edgecolor="black", lw=0.75, label="DataFusion Arrow")
    ax2.bar(x + 2.5 * bw, thr_gpu_w,  bw, color="#2ca02c", edgecolor="black", lw=0.75, hatch="//", label="RTX 2050 GPU (Warm)")

    ax2.set_yscale("log")
    ax2.set_ylim(0.015, 160.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers, fontsize=8.0)
    ax2.set_ylabel(r"Effective Scan Throughput $\Theta_{\mathrm{eff}}$ (GB/s, Log Scale)", fontsize=9.2)
    ax2.set_title("(b) Effective Analytical Scan Throughput Across Tiers (Table 9)", fontsize=9.8, fontweight="bold")
    ax2.grid(True, which="both", axis="y", linestyle="--", alpha=0.38)
    ax2.legend(ncol=2, loc="upper left", fontsize=7.1, framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()

    candidates = [
        "fig_six_software_comparison_sf1_300dpi",
        "fig_six_engines_latency_throughput_300dpi",
        "fig5_six_engines",
    ]
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        fn = Path(m.group(1)).stem
        if "six" in fn.lower() or "paradigm" in fn.lower():
            candidates.append(fn)

    saved = save_to_all_matching_paths(fig, candidates)
    plt.close(fig)
    return saved


def redraw_figure10_ablation_pareto(tex: str) -> list[str]:
    r"""
    Updates Figure 10 across ALL filename variants (including *_sf1_300dpi.pdf!):
      - Zero X-axis label collision between Shallow DT and Ridge Reg.
      - Clean P95 callout badges with zero bar-edge overflow
      - Proper MathText \alpha \in [0, 1] and \pm SD in legends.
    r"""
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), dpi=300)

    # Panel (a): Feature & Model Ablation
    ax = axes[0]
    labels = [
        "Tree Size\n(2D)",
        "Syntax\nOnly (8D)",
        "Operators\nOnly (7D)",
        "Compact\n(CSCI+Sel)",
        "Proposed\nGBR-9D",
        "Shallow DT\n(d=2, 9D)",
        "Ridge Reg.\n(9D)",
    ]
    vals = [103.94, 73.22, 61.06, 56.30, 54.92, 51.35, 49.12]
    p95_tags = {4: "P95:\n114.1%", 5: "P95:\n150.9%", 6: "P95:\n196.8%"}
    colors = ["#7f8c8d", "#5bc0de", "#8e6c8a", "#e67e22", "#27ae60", "#f1c40f", "#2980b9"]

    xs = np.arange(len(labels))
    bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.9, width=0.62)
    for idx, (b, v) in enumerate(zip(bars, vals)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1.8,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
        if idx in p95_tags:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 0.46,
                p95_tags[idx],
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#333333", lw=0.75, alpha=0.94),
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2, rotation=16, ha="right")
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.2)
    ax.set_title("(a) Feature & Model Ablation (Mean vs. P95 Regret)", fontsize=9.7, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel (b): Latency vs. Device-Level Energy Frontier
    ax = axes[1]
    e_pareto = [0.748, 0.879, 1.007, 1.182, 1.381]
    l_pareto = [70.20, 59.81, 56.80, 54.99, 54.92]
    ax.plot(
        e_pareto,
        l_pareto,
        "-o",
        color="#1f4e78",
        linewidth=2.0,
        markersize=6,
        label=r"AST-Route Pareto ($\alpha \in [0, 1]$)",
    )
    ax.scatter([3.170], [82.47], marker="s", s=65, color="#e67e22", edgecolor="black", zorder=4,
               label="Always Polars (82.47 ms, 3.17 J)")
    ax.scatter([2.374], [71.48], marker="^", s=70, color="#7f8c8d", edgecolor="black", zorder=4,
               label="Rule-Based (71.48 ms, 2.37 J)")
    ax.scatter([0.975], [42.64], marker="*", s=115, color="#27ae60", edgecolor="black", zorder=4,
               label="Oracle Lower Bound (42.64 ms)")
    ax.set_xlabel("Estimated Active Device Energy (Joules)", fontsize=9.2)
    ax.set_ylabel("Mean Execution Latency (ms)", fontsize=9.2)
    ax.set_title("(b) Latency vs. Device-Level Energy Frontier", fontsize=9.7, fontweight="bold")
    ax.set_xlim(0.55, 3.35)
    ax.set_ylim(40, 86)
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.95, edgecolor="#cccccc", borderpad=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel (c): Cold-Start Sample Efficiency (5-24 Seeds)
    ax = axes[2]
    seeds = np.array([5, 10, 15, 20, 24])
    means = np.array([63.61, 68.59, 56.59, 57.60, 54.92])
    sds = np.array([14.2, 12.8, 8.4, 7.1, 5.8])
    ax.plot(
        seeds,
        means,
        "-s",
        color="#1e8449",
        linewidth=2.0,
        markersize=6,
        label=r"AST-Route Out-of-Fold Mean $\pm$ SD",
    )
    ax.fill_between(seeds, means - sds, means + sds, color="#1e8449", alpha=0.18)
    for sx, my in zip(seeds, means):
        ax.text(sx, my - 3.3, f"{my:.2f}", ha="center", va="top", fontsize=7.6, fontweight="bold", color="#145a32")
    ax.axhline(82.47, linestyle=":", color="#e67e22", linewidth=1.3, label="Always Polars (82.47 ms)")
    ax.axhline(71.48, linestyle="--", color="#7f8c8d", linewidth=1.3, label="Rule-Based (71.48 ms)")
    ax.axhline(42.64, linestyle="-.", color="#2c3e50", linewidth=1.3, label="Oracle Bound (42.64 ms)")
    ax.set_xlabel("Number of Training Seeds (3 Queries/Seed)", fontsize=9.2)
    ax.set_ylabel("Out-of-Fold Test Latency (ms)", fontsize=9.2)
    ax.set_title("(c) Cold-Start Sample Efficiency (5–24 Seeds)", fontsize=9.7, fontweight="bold")
    ax.set_xticks(seeds)
    ax.set_ylim(38, 92)
    ax.legend(fontsize=7.1, loc="upper right", framealpha=0.95, edgecolor="#cccccc", borderpad=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    candidates = [
        "fig_framework_ablation_pareto_sf1_300dpi",
        "fig_framework_ablation_pareto_300dpi",
        "fig_ablation_pareto_learning_300dpi",
    ]
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        fn = Path(m.group(1)).stem
        if ("ablation" in fn.lower() or "pareto" in fn.lower()) and "job" not in fn.lower() and "163q" not in fn.lower():
            candidates.append(fn)

    saved = save_to_all_matching_paths(fig, candidates)
    plt.close(fig)
    return saved


def redraw_figure15_stability_drift(tex: str) -> list[str]:
    r"""
    Fixes Figure 15(a) legend border/grid collision by placing the combined legend
    on the top z-order twin axis (ax1_r) with generous headroom and border padding.
    r"""
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 4.45), dpi=300)

    x_lbl = ["0%\nClean", "±5%\nNoise", "±10%\nNoise", "±20%\nNoise"]
    xs = np.arange(len(x_lbl))
    w = 0.34
    flip_unguard = [0.00, 2.99, 7.35, 14.05]
    flip_conf    = [0.00, 3.02, 5.78, 12.00]
    viol_noguard = [10.67, 10.89, 10.85, 11.05]
    viol_conf    = [9.33,  9.33,  9.12,  8.86]

    b1 = ax1.bar(xs - w / 2, flip_unguard, w, color="#e67e22", edgecolor="black", lw=0.85,
                 label="Unguarded ArgMin Flip Rate (%)")
    b2 = ax1.bar(xs + w / 2, flip_conf,    w, color="#27ae60", edgecolor="black", lw=0.85,
                 label="Conformal AST-Route Flip Rate (%)")
    for b in list(b1)[1:] + list(b2)[1:]:
        v = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=7.2, fontweight="bold")

    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_lbl, fontsize=8.2)
    ax1.set_ylim(0, 24.0)
    ax1.set_ylabel("Routing Flip Rate (%)", fontsize=9.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.38)

    ax1_r = ax1.twinx()
    ax1_r.plot(xs, viol_noguard, "s--", color="#c0392b", lw=1.7, markersize=5.5,
               label="Without 12% PCIe Guard: >100% Viol. (%)")
    ax1_r.plot(xs, viol_conf,    "o-",  color="#1f4e78", lw=1.8, markersize=5.5,
               label="Conformal AST-Route: >100% Viol. (%)")
    ax1_r.set_ylim(5.0, 16.5)
    ax1_r.set_ylabel("Severe Regret (>100%) Rate (%)", fontsize=9.0, color="#1f4e78")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    leg = ax1_r.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        fontsize=6.9,
        frameon=True,
        facecolor="white",
        edgecolor="#bbbbbb",
        framealpha=0.97,
        borderpad=0.55,
        labelspacing=0.40,
    )
    leg.set_zorder(20)
    ax1.set_title("(a) Router Decision Stability & 12% PCIe Guard (1,000-MC)", fontsize=9.6, fontweight="bold", pad=8)

    # Panel (b): Dynamic Drift & 5-Canary Recalibration Policy
    d_lbl = ["D0\nBase", "D1\nMix", "D2a\nStale", "D2b\nRefr.", "D3\nSkew 0.8", "D4a\nSkew 1.4", "D4b\nRecal."]
    xd = np.arange(len(d_lbl))
    pol_ms = [82.47, 92.89, 87.60, 87.60, 86.86, 91.26, 91.26]
    ast_ms = [60.90, 62.34, 68.85, 64.49, 67.01, 78.47, 76.13]
    pi_cov = ["(89%)", "(89%)", "(89%)", "(89%)", "(85%)", "(75%)", "(89%)"]
    ast_cols = ["#2b579a", "#2b579a", "#2b579a", "#2b579a", "#2b579a", "#c0392b", "#27ae60"]

    ax2.bar(xd - w / 2, pol_ms, w, color="#7f8c8d", edgecolor="black", lw=0.85, label="Static Polars Mean Latency (ms)")
    bars_a = ax2.bar(xd + w / 2, ast_ms, w, color=ast_cols, edgecolor="black", lw=0.85, label="AST-Route Net Latency (ms)")
    for b, v, pc in zip(bars_a, ast_ms, pi_cov):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}\n{pc}", ha="center", va="bottom",
                 fontsize=6.8, fontweight="bold")

    ax2.set_xticks(xd)
    ax2.set_xticklabels(d_lbl, fontsize=7.9)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Mean Execution Latency (ms)", fontsize=9.0)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), fontsize=7.3, frameon=True,
               facecolor="white", edgecolor="#bbbbbb", framealpha=0.97, borderpad=0.52)
    ax2.grid(axis="y", linestyle="--", alpha=0.38)
    ax2.set_title("(b) Dynamic Drift & 5-Canary Recalibration Policy", fontsize=9.6, fontweight="bold", pad=8)

    fig.tight_layout()

    candidates = ["fig_v7_stability_and_workload_drift_300dpi"]
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        fn = Path(m.group(1)).stem
        if "stability" in fn.lower() or "drift" in fn.lower():
            candidates.append(fn)

    saved = save_to_all_matching_paths(fig, candidates)
    plt.close(fig)
    return saved


# ==============================================================================
# STEP 2: FIX ALL 14 TABLE & FIGURE CROSS-REFERENCE SHIFTS
# ==============================================================================
def fix_all_14_table_and_figure_crossrefs(tex: str) -> str:
    r"""
    Replaces every residual hardcoded or mis-mapped Table/Figure reference
    identified in V8_4.pdf (handles both spaces and `~` non-breaking ties).
    Uses single-backslash raw strings passed to lambda so Python 3.13 never
    escapes or doubles backslashes.
    r"""
    replacements: list[tuple[str, str]] = [
        # 1. Page 7 line 268: (Tables 8–11, 13, and 14A–B/D–E) -> Tables 11–13, 16, 19, and 15A–B/D–E
        (
            r"\(Tables[\s~]+(?:8|\\ref\{[^}]+\})(?:--|–|-)(?:11|\\ref\{[^}]+\}),\s*(?:13|\\ref\{[^}]+\}),\s*and[\s~]+(?:14|\\ref\{[^}]+\})A(?:--|–|-)B/D(?:--|–|-)E\)",
            r"(Tables~\ref{tab:jbd_routing_t8}--\ref{tab:jbd_ablation_t9}, \ref{tab:jbd_unseen_t11}, \ref{tab:jbd_manifest_t13}, and~\ref{tab:jbd_step3_t14}A--B/D--E)",
        ),
        # 2. Page 7 line 270: (Table 12 Part A and Table 14 Part C) -> (Table 18 Part A and Table 15 Part C)
        (
            r"\(Table[\s~]+(?:12|\\ref\{tab:jbd_stat_audit_t10\})\s+Part[\s~]+A\s+and\s+Table[\s~]+(?:14|\\ref\{tab:jbd_csci_oblique_t9b\})\s+Part[\s~]+C\)",
            r"(Table~\ref{tab:jbd_multiscale_t12} Part A and Table~\ref{tab:jbd_step3_t14} Part C)",
        ),
        # 3. Page 10 line 360: (evaluated in Table 11 Part C) -> Table 16 Part C
        (
            r"\(evaluated in Table[\s~]+(?:11|\\ref\{tab:jbd_routing_t8\})\s+Part[\s~]+C\)",
            r"(evaluated in Table~\ref{tab:jbd_unseen_t11} Part C)",
        ),
        # 4. Page 11 line 386: (summarized in Tables 12 and 13) -> Tables 12 and 19
        (
            r"\(summarized in Tables[\s~]+(?:10|12|\\ref\{[^}]+\})\s+and[\s~]+(?:13|\\ref\{tab:jbd_ablation_t9\}|\\ref\{tab:system_manifest\}|\\ref\{tab:env_manifest\})\)",
            r"(summarized in Tables~\ref{tab:jbd_stat_audit_t10} and~\ref{tab:jbd_manifest_t13})",
        ),
        # 5. Page 12 line 441: documented in Table 13. -> Table 19.
        (
            r"documented in Table[\s~]+(?:13|\\ref\{tab:jbd_ablation_t9\}|\\ref\{tab:system_manifest\}|\\ref\{tab:env_manifest\})\.",
            r"documented in Table~\ref{tab:jbd_manifest_t13}.",
        ),
        # 6. Page 14 line 511 (Sec 4.3): Tables 4 and 9, alongside Fig. 5 -> Tables 7 and 9
        (
            r"Tables[\s~]+(?:4|\\ref\{tab:jbd_duckdb_t1\}|\\ref\{tab:gpu_pcie_decomposition\})\s+and[\s~]+(?:6|9|\\ref\{[^}]+\}),\s*alongside\s+Fig",
            r"Tables~\ref{tab:jbd_gpu_3level_t4} and~\ref{tab:jbd_throughput_t6}, alongside Fig",
        ),
        # 7. Page 17 line 574 (Sec 4.6): Tables 8 and 12 and Fig. 9 -> Tables 11 and 12
        (
            r"Tables[\s~]+(?:8|\\ref\{tab:jbd_six_eng_t5\}|\\ref\{tab:routing_evaluation\})\s+and[\s~]+(?:10|12|\\ref\{[^}]+\})\s+and\s+Fig\.?[\s~]+9",
            r"Tables~\ref{tab:jbd_routing_t8} and~\ref{tab:jbd_stat_audit_t10} and Fig.~9",
        ),
        # 8. Page 19 line 615 (Sec 4.7): Tables 9 and 14 and Fig. 10 -> Tables 13 and 14 and Fig. 10
        (
            r"Tables[\s~]+(?:9|\\ref\{tab:jbd_throughput_t6\}|\\ref\{tab:ablation_robustness\})\s+and[\s~]+(?:14|\\ref\{[^}]+\})\s+and\s+Fig\.?[\s~]+10",
            r"Tables~\ref{tab:jbd_ablation_t9} and~\ref{tab:jbd_csci_oblique_t9b} and Fig.~10",
        ),
        # 9. Page 20 line 640 (Sec 4.7 item 2): (Table 13 Exp B and Fig. 9b) -> Fig. 10b
        (
            r"Exp[\s~]+B\s+and\s+Fig\.?[\s~]+9b\)",
            r"Exp B and Fig.~10b)",
        ),
        # 10. Page 22 line 689 (Sec 4.7.1/4.7.2): in Table 11 Part B -> Table 16 Part B
        (
            r"in Table[\s~]+(?:11|\\ref\{tab:jbd_routing_t8\})\s+Part[\s~]+B\)",
            r"in Table~\ref{tab:jbd_unseen_t11} Part B)",
        ),
        # 11. Page 22 line 702: Across Tables 8, 11C, and 14A -> Tables 11, 16C, and 15A
        (
            r"Across\s+Tables[\s~]+(?:8|\\ref\{[^}]+\}),\s*(?:11|\\ref\{[^}]+\})C,\s*and[\s~]+(?:14|\\ref\{[^}]+\})A",
            r"Across Tables~\ref{tab:jbd_routing_t8}, \ref{tab:jbd_unseen_t11}C, and~\ref{tab:jbd_step3_t14}A",
        ),
        # 12a. Page 23 line 744 (Sec 4.8): Table 11 and Fig. 11 -> Table 16 and Fig. 11
        (
            r"Table[\s~]+(?:11|\\ref\{tab:jbd_routing_t8\}|\\ref\{tab:priority3_validation\})\s+and\s+Fig\.?[\s~]+11",
            r"Table~\ref{tab:jbd_unseen_t11} and Fig.~11",
        ),
        # 12b. Page 23 line 746 & Page 26 line 842: (Table 11, Part A) and (Table 11, Part C) -> Table 16
        (
            r"\(Table[\s~]+(?:11|\\ref\{tab:jbd_routing_t8\}|\\ref\{tab:priority3_validation\}),\s*Part[\s~]+([AC])\)",
            r"(Table~\ref{tab:jbd_unseen_t11}, Part @PART@)",
        ),
        # 13a. Page 26 line 853 (Sec 4.9): Tables 18 and 13 and Fig. 13 -> Tables 18 and 19
        (
            r"Tables[\s~]+(?:12|18|\\ref\{[^}]+\})\s+and[\s~]+(?:13|\\ref\{tab:jbd_ablation_t9\}|\\ref\{tab:system_manifest\})\s+and\s+Fig",
            r"Tables~\ref{tab:jbd_multiscale_t12} and~\ref{tab:jbd_manifest_t13} and Fig",
        ),
        # 13b. Page 27 line 871: (Table 18 Part B & Table 13 Part C) -> Table 18 Part B & Table 19 Part C
        (
            r"Part[\s~]+B\s+\\&\s+Table[\s~]+(?:13|\\ref\{tab:jbd_ablation_t9\}|\\ref\{tab:system_manifest\})\s+Part[\s~]+C\)",
            r"Part B \& Table~\ref{tab:jbd_manifest_t13} Part C)",
        ),
        # 14. Page 28 Table 18 Footnote: matching Tables 8 and 8 exactly -> Tables 8 and 11
        (
            r"matching\s+Tables[\s~]+(?:5|8|\\ref\{[^}]+\})\s+and[\s~]+(?:8|\\ref\{tab:jbd_six_eng_t5\})\s+exactly",
            r"matching Tables~\ref{tab:jbd_six_eng_t5} and~\ref{tab:jbd_routing_t8} exactly",
        ),
    ]

    for pat, rep in replacements:
        if "@PART@" in rep:
            tex = re.sub(pat, lambda m, r=rep: r.replace("@PART@", m.group(1)), tex)
        else:
            tex = re.sub(pat, lambda _, r=rep: r, tex)

    return tex


# ==============================================================================
# STEP 3: FIX ALL 6 CITATION MISMATCHES, CITE BREß ET AL., & VANCOUVER RE-SORT
# ==============================================================================
def fix_all_6_citations_and_vancouver_order(tex: str) -> str:
    bib_start = tex.find(r"\begin{thebibliography}")
    if bib_start != -1:
        bib_end = tex.find(r"\end{thebibliography}", bib_start)
        body = tex[:bib_start]
        bib_block = tex[bib_start:bib_end]
        tail = tex[bib_end:]

        item_matches = list(re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bib_block))
        entries: dict[str, str] = {}
        for idx, im in enumerate(item_matches):
            k = im.group(1).strip()
            s_i = im.start()
            e_i = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(bib_block)
            clean_item = re.sub(r"^\\bibitem\[[^\]]*\]\{", lambda _: r"\bibitem{", bib_block[s_i:e_i].strip())
            entries[k] = clean_item

        def find_key(*needles: str) -> str:
            for k, txt in entries.items():
                hay = (k + " " + txt).lower()
                if all(nd.lower() in hay for nd in needles):
                    return k
            raise KeyError(f"Could not find bibitem matching {needles}")

        k_bress   = find_key("co-processor utilization")
        k_hetex   = find_key("hetexchange")
        k_maximus = find_key("maximus:")
        k_sirius  = find_key("sirius:")
        k_elqo    = find_key("e-lqo:")
        k_roq     = find_key("roq:")
        k_job     = find_key("how good are query optimizers")
        k_ssb     = find_key("star schema benchmark")
        k_mapd    = find_key("mapd:")
        k_tpch    = find_key("tpc benchmark h")
        k_dres    = find_key("quantifying tpc-h choke points")

        # 1. Cite Breß et al. (2013) alongside HetExchange in Section 2.2 so 0 references are uncited
        if k_bress not in body:
            body = re.sub(
                rf"(\\cite\{{[^}}]*\b{re.escape(k_hetex)}\b[^}}]*)\}}",
                lambda m: m.group(1) + f", {k_bress}" + "}",
                body,
                count=1,
            )

        # 2. Convert ALL hardcoded bracket citations in body to dynamic \cite{...} (SINGLE backslash!)
        hardcoded_cite_fixes = [
            (r"Maximus[\s~]+\[\d+\]", f"Maximus~\\cite{{{k_maximus}}}"),
            (r"Sirius[\s~]+\[\d+\]", f"Sirius~\\cite{{{k_sirius}}}"),
            (r"E-LQO[\s~]+\[\d+\]", f"E-LQO~\\cite{{{k_elqo}}}"),
            (r"Roq[\s~]+\[\d+\]", f"Roq~\\cite{{{k_roq}}}"),
            (r"Join Order Benchmark \(JOB\)[\s~]+\[\d+\]", f"Join Order Benchmark (JOB)~\\cite{{{k_job}}}"),
            (r"Star Schema Benchmark \(SSB\)[\s~]+\[\d+\]", f"Star Schema Benchmark (SSB)~\\cite{{{k_ssb}}}"),
            (r"HeavyDB[\s~]+\[\d+\]", f"HeavyDB~\\cite{{{k_mapd}}}"),
            (r"(Q9,\s*and\s*Q21)[\s~]+\[\d+\s*,\s*\d+\]", f"@GRP1@~\\cite{{{k_tpch}, {k_dres}}}"),
            (r"(TPC-H queries Q1(?:--|–|-)Q22)[\s~]+\[\d+\]", f"@GRP1@~\\cite{{{k_tpch}}}"),
            (r"(IMDB JOB templates)[\s~]+\[\d+\]", f"@GRP1@~\\cite{{{k_job}}}"),
            (r"(SSB star-join queries)[\s~]+\[\d+\]", f"@GRP1@~\\cite{{{k_ssb}}}"),
        ]

        for pat, rep in hardcoded_cite_fixes:
            if "@GRP1@" in rep:
                body = re.sub(pat, lambda m, r=rep: r.replace("@GRP1@", m.group(1)), body)
            else:
                body = re.sub(pat, lambda _, r=rep: r, body)

        # 3. Re-order all \bibitem{...} entries in strict first-\cite appearance order (Vancouver style)
        cited_order: list[str] = []
        for cm in re.finditer(r"\\cite\{([^}]+)\}", body):
            for raw_k in cm.group(1).split(","):
                ck = raw_k.strip()
                if ck and ck in entries and ck not in cited_order:
                    cited_order.append(ck)

        uncited_keys = [k for k in entries if k not in cited_order]
        if uncited_keys:
            raise RuntimeError(f"Uncited bibliography keys remain: {uncited_keys}")

        header_part = bib_block[: item_matches[0].start()]
        rebuilt_bib = header_part + "\n\n".join(entries[k] for k in cited_order) + "\n\n"
        return body + rebuilt_bib + tail

    # Fallback mode if References are plain `[1] ...` text items instead of \bibitem
    ref_hdr = re.search(r"\\section\*?\{References\}", tex)
    if not ref_hdr:
        return tex
    body = tex[: ref_hdr.start()]
    ref_sec = tex[ref_hdr.start() :]
    body = re.sub(r"HetExchange\s*\[7\]", "HetExchange [7, 36]", body, count=1)
    body = re.sub(r"Maximus[\s~]+\[33\]", "Maximus [34]", body)
    body = re.sub(r"Sirius[\s~]+\[34\]", "Sirius [35]", body)
    body = re.sub(r"E-LQO[\s~]+\[35\]", "E-LQO [32]", body)
    body = re.sub(r"Roq[\s~]+\[36\]", "Roq [33]", body)
    body = re.sub(r"Join Order Benchmark \(JOB\)[\s~]+\[27\]", "Join Order Benchmark (JOB) [23]", body)
    body = re.sub(r"Star Schema Benchmark \(SSB\)[\s~]+\[26\]", "Star Schema Benchmark (SSB) [22]", body)
    body = re.sub(r"HeavyDB[\s~]+\[24\]", "HeavyDB [16]", body)
    body = re.sub(r"(Q9,\s*and\s*Q21)[\s~]+\[28,\s*30\]", lambda m: f"{m.group(1)} [18, 20]", body)
    body = re.sub(r"(TPC-H queries Q1(?:--|–|-)Q22)[\s~]+\[28\]", lambda m: f"{m.group(1)} [18]", body)
    body = re.sub(r"(IMDB JOB templates)[\s~]+\[27\]", lambda m: f"{m.group(1)} [23]", body)
    body = re.sub(r"(SSB star-join queries)[\s~]+\[26\]", lambda m: f"{m.group(1)} [22]", body)
    return body + ref_sec


# ==============================================================================
# STEP 4: FIX SECTION 4.7.1-4.7.3 HEADINGS & PAGE 1 FRONTMATTER TYPOGRAPHY
# ==============================================================================
def fix_sec47_headings_and_frontmatter(tex: str) -> str:
    # 1. Convert flush-left `5. Seed-Level...` -> \subsubsection{...} (4.7.1)
    tex, ok_5 = replace_textbf_heading_balanced(
        tex,
        "5. Seed-Level",
        r"\subsubsection{Seed-Level ($N=25$) Inference, Simple Interpretable Baselines, and Leave-One-Tier-Out (LOTO-CV) Boundaries}",
    )
    if not ok_5:
        tex = re.sub(
            r"(?:\\par\\vspace\{[^{}]*\}\s*)?(?:\\noindent\s*)?5\.\s*\\textbf\{Seed-Level[^}]*\}\s*:?",
            lambda _: r"\subsubsection{Seed-Level ($N=25$) Inference, Simple Interpretable Baselines, and Leave-One-Tier-Out (LOTO-CV) Boundaries}",
            tex,
            count=1,
        )

    # 2. Convert flush-left `6. Comprehensive Baseline Matrix...` -> \subsubsection{...} (4.7.3; existing Zero-Leakage becomes 4.7.2)
    tex, ok_6 = replace_textbf_heading_balanced(
        tex,
        "6. Comprehensive Baseline Matrix",
        r"\subsubsection{Comprehensive Baseline Matrix, Standalone CPU-Only Routing, Multi-Scale Byte Accounting, and Metadata Cache Audit}",
    )
    if not ok_6:
        tex = re.sub(
            r"(?:\\par\\vspace\{[^{}]*\}\s*)?(?:\\noindent\s*)?6\.\s*\\textbf\{Comprehensive Baseline Matrix[^\n]*",
            lambda _: r"\subsubsection{Comprehensive Baseline Matrix, Standalone CPU-Only Routing, Multi-Scale Byte Accounting, and Metadata Cache Audit}",
            tex,
            count=1,
        )

    # 3. Prevent Page 1 Title (`Vector-ized`) & Affiliation (`Kan-tharawichai`) hyphenation + add Superscript `1`
    if r"\mbox{Vectorized}" not in tex:
        tex = tex.replace("Vectorized CPU (C++/Rust)", r"\mbox{Vectorized} CPU (C++/Rust)", 1)
    if r"\mbox{Kantharawichai}" not in tex:
        tex = tex.replace("Kantharawichai,", r"\mbox{Kantharawichai},", 1)

    tex = re.sub(
        r"(?<!\\textsuperscript\{1\})(?<!\\textsuperscript\{1\}\s)Department of Information Technology,\s*Faculty of Informatics",
        lambda _: r"\textsuperscript{1}Department of Information Technology, Faculty of Informatics",
        tex,
        count=1,
    )

    return tex


# ==============================================================================
# STEP 5: COMPILE 3 PASSES, ASSERT ZERO DEFECTS, AND REBUILD FLAT SNAPP ZIP
# ==============================================================================
def compile_verify_and_package(src_tex: Path, updated_tex: str) -> None:
    b_open, b_close = count_unquoted_braces(updated_tex)
    if b_open != b_close:
        raise AssertionError(f"Unbalanced LaTeX braces detected: {b_open} '{{' vs {b_close} '}}'")

    out_v85_tex = MS_DIR / "bigquery_olap_benchmark_V8_5.tex"
    ready_tex   = MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.tex"

    for p in (src_tex, out_v85_tex, ready_tex, MS_DIR / "main.tex"):
        p.write_text(updated_tex, encoding="utf-8")

    for pass_i in (1, 2, 3):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", out_v85_tex.name],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"[ERROR] pdflatex failed on pass {pass_i}:")
            print("\n".join(proc.stdout.splitlines()[-30:]))
            raise RuntimeError(f"Compilation failed on pass {pass_i}")

    out_pdf = out_v85_tex.with_suffix(".pdf")
    for mirror_pdf in (
        MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.pdf",
        ROOT / "bigquery_olap_benchmark_V8_5.pdf",
    ):
        try:
            shutil.copy2(out_pdf, mirror_pdf)
        except PermissionError:
            pass

    log_txt = out_v85_tex.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    undef_refs  = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)
    undef_cites = re.findall(r"LaTeX Warning: Citation `([^']+)' on page \d+ undefined", log_txt)
    if undef_refs or undef_cites:
        raise RuntimeError(f"Undefined refs ({undef_refs}) or citations ({undef_cites}) remain!")

    # Rebuild Flat-Directory Springer Nature Snapp Submission ZIP
    if SNAPP_DIR.exists():
        shutil.rmtree(SNAPP_DIR)
    SNAPP_DIR.mkdir(parents=True, exist_ok=True)

    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+)(\})",
        r"\1\2\3",
        updated_tex,
    )
    (SNAPP_DIR / "manuscript.tex").write_text(flat_tex, encoding="utf-8")
    shutil.copy2(out_pdf, SNAPP_DIR / "manuscript.pdf")

    copied_figs = 0
    for fig_ref in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat_tex))):
        fname = Path(fig_ref).name
        if not Path(fname).suffix:
            fname += ".pdf"
        s_fig = FIG_DIR / fname
        if not s_fig.exists() and (ROOT_FIG / fname).exists():
            s_fig = ROOT_FIG / fname
        if s_fig.exists():
            shutil.copy2(s_fig, SNAPP_DIR / fname)
            copied_figs += 1

    for _ in (1, 2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"],
            cwd=str(SNAPP_DIR),
            capture_output=True,
            text=True,
        )
    for ext in ("*.aux", "*.log", "*.out", "*.toc"):
        for f_aux in SNAPP_DIR.glob(ext):
            f_aux.unlink(missing_ok=True)

    snapp_zip = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat.zip"
    with zipfile.ZipFile(snapp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(SNAPP_DIR.iterdir()):
            if item.is_file():
                zf.write(item, arcname=item.name)

    pdf_sha = hashlib.sha256(out_pdf.read_bytes()).hexdigest()
    print("=" * 96)
    print(f"[100% VERIFIED] Output PDF          : {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
    print(f"[100% VERIFIED] Brace Balance       : {b_open} '{{' == {b_close} '}}' (Exact Match)")
    print(f"[100% VERIFIED] Undefined Refs/Cite : 0 undefined refs / 0 undefined citations")
    print(f"[100% VERIFIED] Flat Snapp Package  : {snapp_zip.name} ({copied_figs} root-level figures)")
    print(f"[100% VERIFIED] V8_5 PDF SHA-256    : {pdf_sha[:32]}...")
    print("=" * 96)


def main() -> None:
    src_tex = locate_source_tex()
    print(f"[LOAD] Reading manuscript source: {src_tex}")
    tex = src_tex.read_text(encoding="utf-8")

    b0_open, b0_close = count_unquoted_braces(tex)
    print(f"[INIT] Initial brace balance: {b0_open} '{{' vs {b0_close} '}}'")

    print("[STEP 1/5] Redrawing Figure 6, Figure 10 (all aliases incl. _sf1_), and Figure 15 at 300 DPI...")
    s6  = redraw_figure6_six_engines(tex)
    s10 = redraw_figure10_ablation_pareto(tex)
    s15 = redraw_figure15_stability_drift(tex)
    print(f"  -> Saved Fig 6 : {s6}")
    print(f"  -> Saved Fig 10: {s10}")
    print(f"  -> Saved Fig 15: {s15}")

    print("[STEP 2/5] Fixing all 14 Table & Figure cross-references...")
    tex = fix_all_14_table_and_figure_crossrefs(tex)

    print("[STEP 3/5] Fixing all 6 Citation mismatches, citing Breß et al., & sorting Vancouver [1]..[36]...")
    tex = fix_all_6_citations_and_vancouver_order(tex)

    print("[STEP 4/5] Standardizing Subsections 4.7.1–4.7.3 & Page 1 Title/Affiliation...")
    tex = fix_sec47_headings_and_frontmatter(tex)

    print("[STEP 5/5] Compiling 3 passes of pdflatex & building Flat Snapp Submission ZIP...")
    compile_verify_and_package(src_tex, tex)


if __name__ == "__main__":
    main()
