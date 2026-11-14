#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/123_final_5point_polish.py
Fixes the final 5 visual inspection items in bigquery_olap_benchmark_V8_5:
  1. Remaining 3 table cross-refs before Fig.~\ref{...} (Sec 4.6, 4.7, 4.8).
  2. Moves Table 15 \caption{Comprehensive multi-baseline...} above the table body.
  3. Converts item "6. Comprehensive Baseline Matrix..." into Subsubsection 4.7.3.
  4. Lifts the red dashed line in Figure 15(a) clear of the "12.00%" bar label.
  5. Ensures affiliation superscript "1" appears on Page 1 + adds \usepackage{xurl}.
"""

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


def extract_balanced_cmd(text: str, cmd_idx: int) -> tuple[int, int]:
    open_b = text.find("{", cmd_idx)
    depth = 0
    for i in range(open_b, len(text)):
        if text[i] == "\\" and i + 1 < len(text):
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return cmd_idx, i + 1
    raise ValueError("Unbalanced braces")


def polish_figure15_marker_clearance(tex: str) -> None:
    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 4.45), dpi=300)

    x_lbl = ["0%\nClean", "±5%\nNoise", "±10%\nNoise", "±20%\nNoise"]
    xs = np.arange(len(x_lbl))
    w = 0.34
    flip_unguard = [0.00, 2.99, 7.35, 14.05]
    flip_conf    = [0.00, 3.02, 5.78, 12.00]
    viol_noguard = [10.67, 10.89, 10.85, 11.05]
    viol_conf    = [9.33,  9.33,  9.12,  8.86]

    b1 = ax1.bar(xs - w / 2, flip_unguard, w, color="#e67e22", edgecolor="black", lw=0.85, label="Unguarded ArgMin Flip Rate (%)")
    b2 = ax1.bar(xs + w / 2, flip_conf,    w, color="#27ae60", edgecolor="black", lw=0.85, label="Conformal AST-Route Flip Rate (%)")
    for b in list(b1)[1:] + list(b2)[1:]:
        v = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.30, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_lbl, fontsize=8.2)
    ax1.set_ylim(0, 25.0)
    ax1.set_ylabel("Routing Flip Rate (%)", fontsize=9.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.38)

    ax1_r = ax1.twinx()
    ax1_r.plot(xs, viol_noguard, "s--", color="#c0392b", lw=1.7, markersize=5.5, label="Without 12% PCIe Guard: >100% Viol. (%)")
    ax1_r.plot(xs, viol_conf,    "o-",  color="#1f4e78", lw=1.8, markersize=5.5, label="Conformal AST-Route: >100% Viol. (%)")
    # Setting ylim(4.5, 14.3) places 11.05% cleanly above the 12.00% bar text while staying below the legend!
    ax1_r.set_ylim(4.5, 14.3)
    ax1_r.set_ylabel("Severe Regret (>100%) Rate (%)", fontsize=9.0, color="#1f4e78")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    leg = ax1_r.legend(
        h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.015, 0.985),
        fontsize=6.85, frameon=True, facecolor="white", edgecolor="#bbbbbb",
        framealpha=0.97, borderpad=0.50, labelspacing=0.36,
    )
    leg.set_zorder(20)
    ax1.set_title("(a) Router Decision Stability & 12% PCIe Guard (1,000-MC)", fontsize=9.6, fontweight="bold", pad=8)

    d_lbl = ["D0\nBase", "D1\nMix", "D2a\nStale", "D2b\nRefr.", "D3\nSkew 0.8", "D4a\nSkew 1.4", "D4b\nRecal."]
    xd = np.arange(len(d_lbl))
    pol_ms = [82.47, 92.89, 87.60, 87.60, 86.86, 91.26, 91.26]
    ast_ms = [60.90, 62.34, 68.85, 64.49, 67.01, 78.47, 76.13]
    pi_cov = ["(89%)", "(89%)", "(89%)", "(89%)", "(85%)", "(75%)", "(89%)"]
    ast_cols = ["#2b579a", "#2b579a", "#2b579a", "#2b579a", "#2b579a", "#c0392b", "#27ae60"]

    ax2.bar(xd - w / 2, pol_ms, w, color="#7f8c8d", edgecolor="black", lw=0.85, label="Static Polars Mean Latency (ms)")
    bars_a = ax2.bar(xd + w / 2, ast_ms, w, color=ast_cols, edgecolor="black", lw=0.85, label="AST-Route Net Latency (ms)")
    for b, v, pc in zip(bars_a, ast_ms, pi_cov):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}\n{pc}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")

    ax2.set_xticks(xd)
    ax2.set_xticklabels(d_lbl, fontsize=7.9)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Mean Execution Latency (ms)", fontsize=9.0)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), fontsize=7.3, frameon=True,
               facecolor="white", edgecolor="#bbbbbb", framealpha=0.97, borderpad=0.50)
    ax2.grid(axis="y", linestyle="--", alpha=0.38)
    ax2.set_title("(b) Dynamic Drift & 5-Canary Recalibration Policy", fontsize=9.6, fontweight="bold", pad=8)

    fig.tight_layout()
    stems = ["fig_v7_stability_and_workload_drift_300dpi", "fig_v8_stability_and_workload_drift_300dpi"]
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        fn = Path(m.group(1)).stem
        if "stability" in fn.lower() or "drift" in fn.lower():
            stems.append(fn)
    for st in set(stems):
        for ext in (".pdf", ".png"):
            fig.savefig(FIG_DIR / f"{st}{ext}", dpi=300, bbox_inches="tight")
            if ROOT_FIG.exists():
                fig.savefig(ROOT_FIG / f"{st}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_5point_tex_fixes(tex: str) -> str:
    # 1. Fix the 3 remaining Table cross-references before Fig.~\ref{...}
    # (a) Sec 4.6: Tables 8 and 12 and Fig. 9 -> Tables 11 and 12
    tex = re.sub(
        r"(\\subsection\{Head-to-Head Evaluation[^}]*\}\s*)Tables[\s~]+(?:8|\\ref\{[^}]+\})\s+and[\s~]+(?:10|12|\\ref\{[^}]+\})",
        lambda m: m.group(1) + r"Tables~\ref{tab:jbd_routing_t8} and~\ref{tab:jbd_stat_audit_t10}",
        tex,
        count=1,
    )
    # (b) Sec 4.7: Tables 9 and 14 and Fig. 10 -> Tables 13 and 14
    tex = re.sub(
        r"(validate the internal mechanisms of AST-Route,\s*)Tables[\s~]+(?:9|\\ref\{[^}]+\})\s+and[\s~]+(?:14|\\ref\{[^}]+\})",
        lambda m: m.group(1) + r"Tables~\ref{tab:jbd_ablation_t9} and~\ref{tab:jbd_csci_oblique_t9b}",
        tex,
        count=1,
    )
    # (c) Sec 4.8: Table 11 and Fig. 11 present -> Table 16 and Fig. 11 present
    tex = re.sub(
        r"(\\subsection\{Cost Prediction Q-Error[^}]*\}\s*)Table[\s~]+(?:11|\\ref\{[^}]+\})(\s+and\s+Fig)",
        lambda m: m.group(1) + r"Table~\ref{tab:jbd_unseen_t11}" + m.group(2),
        tex,
        count=1,
    )

    # 2. Move Table 15 \caption{Comprehensive multi-baseline...} above the table body
    sig15 = "Comprehensive multi-baseline and operational robustness evaluation:"
    p15 = tex.find(sig15)
    if p15 != -1:
        t_start = tex.rfind(r"\begin{table", 0, p15)
        t_end_m = re.search(r"\\end\{table\*?\}", tex[p15:])
        if t_start != -1 and t_end_m:
            t_end = p15 + t_end_m.end()
            blk = tex[t_start:t_end]
            c_pos = blk.rfind(r"\caption", 0, blk.find(sig15) + 10)
            body_starts = [blk.find(tok) for tok in (r"\resizebox", r"\begin{tabular", r"\begin{fitasttable", r"\scriptsize", r"\footnotesize") if blk.find(tok) != -1]
            first_body = min(body_starts) if body_starts else -1
            if c_pos != -1 and first_body != -1 and c_pos > first_body:
                cs, ce = extract_balanced_cmd(blk, c_pos)
                # Also grab any trailing \label{...} right after \caption{...}
                after_c = blk[ce:]
                for _ in range(2):
                    lm = re.match(r"\s*\\label\{[^}]+\}", after_c)
                    if lm:
                        ce += lm.end()
                        after_c = blk[ce:]
                cap_chunk = blk[cs:ce].strip()
                blk_no_cap = blk[:cs].rstrip() + "\n" + blk[ce:].lstrip()
                # Insert cap_chunk right after \begin{table...}[...] (and \centering if before body)
                hdr_m = re.match(r"^(\\begin\{table\*?\}(?:\[[^\]]*\])?\s*(?:\\centering\s*)?)", blk_no_cap)
                if hdr_m:
                    ins = hdr_m.end()
                    blk_fixed = blk_no_cap[:ins] + cap_chunk + "\n" + blk_no_cap[ins:]
                    tex = tex[:t_start] + blk_fixed + tex[t_end:]

    # 3. Convert "6. Comprehensive Baseline Matrix..." (any LaTeX formatting) into \subsubsection{...} (4.7.3)
    m6 = re.search(r"6\.\s*(?:\\textbf\{)?Comprehensive Baseline Matrix.*?Cold-Metadata Net Latency Audit\.?\}?", tex, flags=re.DOTALL)
    if m6:
        line_start = tex.rfind("\n", 0, m6.start()) + 1
        line_end = m6.end()
        while line_end < len(tex) and tex[line_end] in ("}", ".", ":", " "):
            # Only consume '}' if it belonged to \textbf{6. ...} on that same paragraph header
            if tex[line_end] == "}" and r"\textbf{" not in tex[line_start:m6.end()]:
                break
            line_end += 1
        new_sub = r"\subsubsection{Comprehensive Baseline Matrix, Standalone CPU-Only Routing, Multi-Scale Byte Accounting, and Metadata Cache Audit}"
        tex = tex[:line_start] + new_sub + "\n" + tex[line_end:].lstrip()

    # 4. Page 1 affiliation superscript '1' (handles \small / \textit / braces before Department)
    def add_affil_sup1(m: re.Match) -> str:
        prefix = m.group(1)
        if "textsuperscript" in prefix or "$^1$" in prefix:
            return m.group(0)
        return prefix + r"\textsuperscript{1}" + "Department of Information Technology"

    tex = re.sub(
        r"([^\n]{0,40}?)Department of Information Technology(?=,?\s*Faculty of Informatics)",
        add_affil_sup1,
        tex,
        count=1,
    )

    # 5. Clean URL line-breaking in References via xurl if available
    if r"\usepackage{xurl}" not in tex:
        tex = tex.replace(r"\usepackage{hyperref}", "\\usepackage{xurl}\n\\usepackage{hyperref}", 1)

    return tex


def main() -> None:
    v85_tex = MS_DIR / "bigquery_olap_benchmark_V8_5.tex"
    src_tex = v85_tex if v85_tex.exists() else (MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.tex")
    tex = src_tex.read_text(encoding="utf-8")

    polish_figure15_marker_clearance(tex)
    tex = apply_5point_tex_fixes(tex)

    for p in (
        MS_DIR / "bigquery_olap_benchmark_V8_5.tex",
        MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.tex",
        MS_DIR / "main.tex",
    ):
        p.write_text(tex, encoding="utf-8")

    for pass_i in (1, 2, 3):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V8_5.tex"],
            cwd=str(MS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            # If xurl package is not installed in local MiKTeX/TeXLive, fallback cleanly without xurl
            if pass_i == 1 and "xurl.sty" in proc.stdout:
                tex = tex.replace("\\usepackage{xurl}\n", "")
                for p in (MS_DIR / "bigquery_olap_benchmark_V8_5.tex", MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.tex", MS_DIR / "main.tex"):
                    p.write_text(tex, encoding="utf-8")
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V8_5.tex"],
                    cwd=str(MS_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
            if proc.returncode != 0:
                print("\n".join(proc.stdout.splitlines()[-25:]))
                raise RuntimeError(f"pdflatex failed on pass {pass_i}")

    out_pdf = MS_DIR / "bigquery_olap_benchmark_V8_5.pdf"
    for mirror in (MS_DIR / "bigquery_olap_benchmark_JBD_Submission_Ready.pdf", ROOT / "bigquery_olap_benchmark_V8_5.pdf"):
        try:
            shutil.copy2(out_pdf, mirror)
        except PermissionError:
            pass

    # Rebuild Flat-Directory Snapp ZIP
    if SNAPP_DIR.exists():
        shutil.rmtree(SNAPP_DIR)
    SNAPP_DIR.mkdir(parents=True, exist_ok=True)

    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+)(\})",
        r"\1\2\3",
        tex,
    )
    (SNAPP_DIR / "manuscript.tex").write_text(flat_tex, encoding="utf-8")
    shutil.copy2(out_pdf, SNAPP_DIR / "manuscript.pdf")

    for fig_ref in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat_tex))):
        fname = Path(fig_ref).name
        if not Path(fname).suffix:
            fname += ".pdf"
        s_fig = FIG_DIR / fname
        if not s_fig.exists() and (ROOT_FIG / fname).exists():
            s_fig = ROOT_FIG / fname
        if s_fig.exists():
            shutil.copy2(s_fig, SNAPP_DIR / fname)

    snapp_zip = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat.zip"
    with zipfile.ZipFile(snapp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(SNAPP_DIR.iterdir()):
            if item.is_file():
                zf.write(item, arcname=item.name)

    h = hashlib.sha256(out_pdf.read_bytes()).hexdigest()
    print("=" * 96)
    print(f"[100% CAMERA-READY] Final PDF  : {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
    print(f"[100% CAMERA-READY] Snapp ZIP  : {snapp_zip}")
    print(f"[100% CAMERA-READY] SHA-256    : {h[:32]}...")
    print("=" * 96)


if __name__ == "__main__":
    main()