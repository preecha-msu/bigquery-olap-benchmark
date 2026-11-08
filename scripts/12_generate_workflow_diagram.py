# ไฟล์: scripts/12_generate_workflow_diagram.py
# วาดแผนภาพสถาปัตยกรรมและการดำเนินงานวิจัย 6 ขั้นตอน (300 DPI Vector PDF/PNG) สำหรับวารสาร IEEE Access
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def draw_box(ax, x, y, w, h, title, bullets, facecolor="#F8FAFC", edgecolor="#1E3A8A", title_bg="#1E3A8A", fontsize=8.3):
    # กล่องหลัก
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3, zorder=2
    )
    ax.add_patch(box)

    # แถบหัวข้อด้านบนของกล่อง
    header_h = min(0.042, h * 0.28)
    header = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor=title_bg, edgecolor=edgecolor, linewidth=1.0, zorder=3
    )
    ax.add_patch(header)

    ax.text(
        x + w / 2, y + h - header_h / 2, title,
        ha="center", va="center", color="white", fontweight="bold", fontsize=fontsize + 0.4, zorder=4
    )

    # เนื้อหาย่อยด้านใน
    body_text = "\n".join(bullets)
    ax.text(
        x + 0.014, y + (h - header_h) / 2, body_text,
        ha="left", va="center", color="#111827", fontsize=fontsize, linespacing=1.32, zorder=4
    )

def draw_arrow(ax, x1, y1, x2, y2, color="#374151"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=1.5, color=color, zorder=5
    )
    ax.add_patch(arr)

def build_workflow_diagram(root: Path):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"]})

    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # หัวเรื่องบนสุดของแผนภาพ
    ax.text(
        0.5, 0.975,
        "End-to-End Experimental Workflow: AST-Driven Multi-Paradigm OLAP Benchmarking (75 Queries, 6 Engines)",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0F172A"
    )

    # -------------------------------------------------------------------------
    # Row 1: Phase 1 (Left) & Phase 2 (Right)
    # -------------------------------------------------------------------------
    draw_box(
        ax, 0.02, 0.77, 0.45, 0.17,
        "PHASE 1: TPC-H Data Generation & Multi-Layout Storage (SF = 1 GB)",
        [
            "• Base Dataset: 8 Relational Tables (8.66M rows; lineitem = 6.00M rows)",
            "• Local Columnar: ZSTD Parquet + DuckDB Native + Apache Arrow Buffers",
            "• Local Row-Store Baseline: SQLite 3 In-Memory B-Tree with PK/FK Indexes",
            "• Cloud Serverless: Google Cloud Storage -> BigQuery Unoptimized vs. Clustered"
        ],
        facecolor="#EFF6FF", edgecolor="#1D4ED8", title_bg="#1D4ED8"
    )

    draw_box(
        ax, 0.53, 0.77, 0.45, 0.17,
        "PHASE 2: 75-Query Factorial Workload Design (25 Seeds x 3 Variations)",
        [
            "• 5 Structural Tiers (5 Canonical Seeds/Tier = 25 Independent Seeds):",
            "  L1 (Scan/Filter), L2 (4-8 Way Joins), L3 (Staged CTEs), L4 (Window), L5 (UNNEST)",
            "• 3 Selectivity Perturbations per Seed (Total N = 75 Queries):",
            "  V1: High Selectivity (1-5%) | V2: Medium (15-30%) | V3: Full-Horizon (65-95%)"
        ],
        facecolor="#ECFDF5", edgecolor="#047857", title_bg="#047857"
    )

    draw_arrow(ax, 0.47, 0.855, 0.53, 0.855, color="#1E40AF")

    # -------------------------------------------------------------------------
    # Row 2: Phase 3 (Static AST Profiling)
    # -------------------------------------------------------------------------
    draw_box(
        ax, 0.02, 0.58, 0.96, 0.14,
        "PHASE 3: Static Abstract Syntax Tree (AST) Profiling & Complexity Quantification (SQLGlot Parser)",
        [
            "• Parses all 75 SQL queries into canonical AST representations to extract structural operator vectors:",
            "  Tree Depth D(q), Total Operator Nodes N(q), Join Count J(q), Nested CTE/Subquery Depth S(q), Window Specs W(q), and Array UNNEST U(q)",
            "• Composite Structural Complexity Index:  CSCI(q) = 1.5 D(q) + 2.5 J(q) + 3.0 S(q) + 3.5 W(q) + 4.0 U(q) + 1.5 C(q)   [Range: 21.0 (L1) to 36.4 (L5)]"
        ],
        facecolor="#FFFBEB", edgecolor="#B45309", title_bg="#B45309"
    )

    draw_arrow(ax, 0.245, 0.77, 0.245, 0.72)
    draw_arrow(ax, 0.755, 0.77, 0.755, 0.72)

    # -------------------------------------------------------------------------
    # Row 3: Phase 4 (Four Computing Paradigms / Six Software Engines)
    # -------------------------------------------------------------------------
    p4_y, p4_h = 0.31, 0.22
    draw_box(
        ax, 0.02, p4_y, 0.225, p4_h,
        "Paradigm 1: Row-Store",
        [
            "• Engine: SQLite 3 (ANSI C)",
            "• Storage: In-Memory B-Tree",
            "• Execution: Tuple-at-a-Time",
            "  Volcano Iterator Model",
            "• Role: Traditional Row-Store",
            "  Baseline (Mean: 5071.1 ms)"
        ],
        facecolor="#F3F4F6", edgecolor="#4B5563", title_bg="#4B5563", fontsize=7.9
    )

    draw_box(
        ax, 0.265, p4_y, 0.225, p4_h,
        "Paradigm 2: Serverless Cloud",
        [
            "• Engine: Google BigQuery",
            "• Storage: Capacitor Columnar",
            "  (Unoptimized vs. Clustered)",
            "• Controls: Cache Disabled,",
            "  Zero-Billing Sandbox Quota",
            "• Telemetry: Wall & Slot Time,",
            "  Bytes Scanned (745.4 ms)"
        ],
        facecolor="#EFF6FF", edgecolor="#1E40AF", title_bg="#1E40AF", fontsize=7.9
    )

    draw_box(
        ax, 0.51, p4_y, 0.225, p4_h,
        "Paradigm 3: Vectorized CPU",
        [
            "• 1) DuckDB (C++ Columnar)",
            "  Parquet vs. Native (183.7ms)",
            "• 2) Polars SQL (Rust Arrow)",
            "  Lazy Streaming (66.5 ms)",
            "• 3) Apache DataFusion (Rust)",
            "  Arrow Planner (88.2 ms)"
        ],
        facecolor="#ECFDF5", edgecolor="#065F46", title_bg="#065F46", fontsize=7.9
    )

    draw_box(
        ax, 0.755, p4_y, 0.225, p4_h,
        "Paradigm 4: SIMT CUDA GPU",
        [
            "• Engine: PyTorch CUDA OLAP",
            "• GPU: NVIDIA RTX 2050 (4GB)",
            "• Hardware cudaEvent Timers:",
            "  T_Cold = T_PCIe_H2D + T_Warm",
            "• Mean PCIe H2D: 34.71 ms",
            "• Mean Warm Kernel: 22.25 ms"
        ],
        facecolor="#FEF2F2", edgecolor="#991B1B", title_bg="#991B1B", fontsize=7.9
    )

    for cx in [0.132, 0.377, 0.622, 0.867]:
        draw_arrow(ax, cx, 0.58, cx, 0.53)
        draw_arrow(ax, cx, 0.31, cx, 0.26)

    # -------------------------------------------------------------------------
    # Row 4: Phase 5 (Left) & Phase 6 (Right)
    # -------------------------------------------------------------------------
    draw_box(
        ax, 0.02, 0.04, 0.46, 0.22,
        "PHASE 5: Statistical Verification & Efficiency Metrics",
        [
            "• Result-Set Multiset Equivalence Check: EX(q) in {0, 1}",
            "• Valid Efficiency Score (VES) & Storage Pruning Ratio (78.4% in V1)",
            "• 10,000-Iteration Seed-Cluster Bootstrap 95% CIs (N_seeds = 25)",
            "• Paired Two-Sided Wilcoxon Signed-Rank Tests (Holm-Bonferroni)",
            "• Architectural Crossover Analysis (L1 PCIe H2D vs. L4/L5 Speedup)"
        ],
        facecolor="#F5F3FF", edgecolor="#5B21B6", title_bg="#5B21B6"
    )

    draw_box(
        ax, 0.52, 0.04, 0.46, 0.22,
        "PHASE 6: Cryptographic Audit & Manuscript Artifacts",
        [
            "• Automated LaTeX Booktabs Tables (Tables I-V) & 300 DPI Figures",
            "• Zero-Discrepancy Verification Suite (08_verify_paper_statistics.py)",
            "• SHA-256 Telemetry Audit Certificate (REPRODUCTION_CERTIFICATE.md)",
            "• Complete Open-Source Benchmark Repository & IEEE Access Manuscript:",
            "  https://github.com/preecha-msu/bigquery-olap-benchmark"
        ],
        facecolor="#F0FDF4", edgecolor="#166534", title_bg="#166534"
    )

    draw_arrow(ax, 0.48, 0.15, 0.52, 0.15, color="#5B21B6")

    fig_dir = root / "results" / "figures_300dpi"
    ms_fig_dir = root / "manuscript" / "figures"
    for d in [fig_dir, ms_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pdf_name = "fig_experimental_workflow_300dpi.pdf"
    png_name = "fig_experimental_workflow_300dpi.png"

    plt.tight_layout()
    fig.savefig(fig_dir / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / png_name, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(ms_fig_dir / pdf_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 105)
    print(f" [OK] Generated 300 DPI Experimental Workflow PDF : {ms_fig_dir / pdf_name}")
    print(f" [OK] Generated 300 DPI Experimental Workflow PNG : {fig_dir / png_name}")
    print("=" * 105)

def inject_workflow_figure_into_main_tex(root: Path):
    tex_path = root / "manuscript" / "main.tex"
    if not tex_path.exists():
        return
    tex = tex_path.read_text(encoding="utf-8")
    if "fig:experimental_workflow" in tex:
        print(" [OK] Workflow figure (fig:experimental_workflow) is already in manuscript/main.tex")
        return

    wf_block = r"""
\begin{figure*}[t!]
\centering
\IfFileExists{figures/fig_experimental_workflow_300dpi.pdf}{%
    \includegraphics[width=0.96\textwidth]{figures/fig_experimental_workflow_300dpi.pdf}%
}{%
    \fbox{\parbox{0.92\textwidth}{\centering\vspace{1.2em}\textbf{Workflow Diagram Placeholder:} Upload \texttt{figures/fig\_experimental\_workflow\_300dpi.pdf} to Overleaf.\vspace{1.2em}}}%
}
\caption{End-to-end six-phase experimental workflow of the AST-driven multi-paradigm OLAP benchmarking framework, illustrating TPC-H multi-layout data provisioning ($\text{SF}=1\text{ GB}$), 75-query factorial synthesis (25 canonical seeds $\times$ 3 selectivity horizons), \texttt{SQLGlot} static AST profiling, six-engine execution across four computing paradigms, non-parametric seed-cluster statistical inference, and cryptographic SHA-256 verification.}
\label{fig:experimental_workflow}
\end{figure*}
"""
    target = r"\section{METHODOLOGY AND SIX-SOFTWARE BENCHMARK DESIGN}"
    if target in tex:
        tex = tex.replace(target, target + "\n" + wf_block)
        tex_path.write_text(tex, encoding="utf-8")
        print(" [OK] Injected Workflow Figure (fig:experimental_workflow) into Section III of manuscript/main.tex")

if __name__ == "__main__":
    build_workflow_diagram(PROJECT_ROOT)
    inject_workflow_figure_into_main_tex(PROJECT_ROOT)
