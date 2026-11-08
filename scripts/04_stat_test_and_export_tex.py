# à¹„à¸Ÿà¸¥à¹Œ: scripts/04_stat_test_and_export_tex.py
import hashlib
import yaml
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# à¹ƒà¸Šà¹‰ Backend à¹à¸šà¸šà¹„à¸¡à¹ˆà¹€à¸›à¸´à¸”à¸«à¸™à¹‰à¸²à¸•à¹ˆà¸²à¸‡ GUI à¹€à¸žà¸·à¹ˆà¸­à¸„à¸§à¸²à¸¡à¸£à¸§à¸”à¹€à¸£à¹‡à¸§à¹ƒà¸™à¸à¸²à¸£à¸£à¸±à¸™à¸œà¹ˆà¸²à¸™ PowerShell
matplotlib.use("Agg")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIER_LABELS = {
    "L1_scan_filter": "L1: Scan & Filter",
    "L2_multi_joins": "L2: Multi-Way Joins",
    "L3_nested_cte": "L3: Nested CTEs",
    "L4_window_analytical": "L4: Window Func.",
    "L5_array_unnest": "L5: Array / UNNEST"
}

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def evaluate_statistics_and_export(root: Path, sf_override: int = None):
    with open(root / "config" / "benchmark_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sf = sf_override if sf_override is not None else cfg["experiment_controls"]["scale_factor"]
    alpha = cfg["experiment_controls"].get("alpha_significance", 0.05)

    raw_csv = root / "results" / "raw_logs" / f"raw_benchmark_sf{sf}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"à¹„à¸¡à¹ˆà¸žà¸šà¹„à¸Ÿà¸¥à¹Œà¸œà¸¥à¸à¸²à¸£à¸—à¸”à¸¥à¸­à¸‡à¸—à¸µà¹ˆ: {raw_csv} à¸à¸£à¸¸à¸“à¸²à¸£à¸±à¸™à¸ªà¸„à¸£à¸´à¸›à¸•à¹Œ 03 à¸à¹ˆà¸­à¸™à¸„à¸£à¸±à¸š")

    df = pd.read_csv(raw_csv)
    print(f"\n[STEP 1] Loaded Raw Telemetry ({len(df)} trial records) from: {raw_csv}")
    print(f"  -> Raw CSV SHA-256: {compute_sha256(raw_csv)}")

    # à¸„à¸³à¸™à¸§à¸“ Valid Efficiency Score (VES) à¸£à¸²à¸¢à¸£à¸­à¸š
    # à¹ƒà¸Šà¹‰à¸„à¹ˆà¸²à¸¡à¸±à¸˜à¸¢à¸à¸²à¸™à¸‚à¸­à¸‡ External Parquet à¹ƒà¸™à¹à¸•à¹ˆà¸¥à¸°à¸„à¸´à¸§à¸£à¸µà¹€à¸›à¹‡à¸™ Baseline (T_baseline)
    q_baselines = df.groupby("query_file")["external_parquet_ms"].transform("median")
    df["ves_external"] = df["ex_accuracy"] * np.sqrt(q_baselines / np.maximum(df["external_parquet_ms"], 1e-6)) * 100.0
    df["ves_native"] = df["ex_accuracy"] * np.sqrt(q_baselines / np.maximum(df["native_storage_ms"], 1e-6)) * 100.0

    # à¸ªà¸£à¸¸à¸›à¸œà¸¥à¸•à¸²à¸¡à¸£à¸°à¸”à¸±à¸šà¸„à¸§à¸²à¸¡à¸‹à¸±à¸šà¸‹à¹‰à¸­à¸™ 5 à¸£à¸°à¸”à¸±à¸š (L1 - L5)
    summary_rows = []
    for tier_key, tier_name in TIER_LABELS.items():
        sub = df[df["complexity_tier"] == tier_key]
        if sub.empty:
            continue

        ext_ms = sub["external_parquet_ms"].values
        nat_ms = sub["native_storage_ms"].values

        # à¸—à¸”à¸ªà¸­à¸šà¸™à¸±à¸¢à¸ªà¸³à¸„à¸±à¸à¸—à¸²à¸‡à¸ªà¸–à¸´à¸•à¸´ Wilcoxon signed-rank test (Paired non-parametric)
        try:
            w_stat, p_val = stats.wilcoxon(ext_ms, nat_ms, alternative="two-sided")
        except ValueError:
            w_stat, p_val = 0.0, 1.0

        summary_rows.append({
            "Complexity_Tier": tier_name,
            "AST_Depth_Mean": round(sub["ast_depth"].mean(), 2),
            "AST_Nodes_Mean": round(sub["ast_nodes"].mean(), 1),
            "Joins_Mean": round(sub["joins"].mean(), 2),
            "Ext_Mean_ms": round(np.mean(ext_ms), 2),
            "Ext_SD_ms": round(np.std(ext_ms, ddof=1), 2),
            "Ext_Median_ms": round(np.median(ext_ms), 2),
            "Ext_P95_ms": round(np.percentile(ext_ms, 95), 2),
            "Nat_Mean_ms": round(np.mean(nat_ms), 2),
            "Nat_SD_ms": round(np.std(nat_ms, ddof=1), 2),
            "Nat_Median_ms": round(np.median(nat_ms), 2),
            "Nat_P95_ms": round(np.percentile(nat_ms, 95), 2),
            "Speedup_x": round(np.mean(ext_ms) / max(np.mean(nat_ms), 1e-6), 2),
            "EX_Pct": round(sub["ex_accuracy"].mean() * 100.0, 1),
            "VES_Ext": round(sub["ves_external"].mean(), 2),
            "VES_Nat": round(sub["ves_native"].mean(), 2),
            "Wilcoxon_W": round(float(w_stat), 2),
            "p_value": float(p_val),
            "Significant": bool(p_val < alpha)
        })

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = root / "results" / "summary_tables" / f"statistical_summary_sf{sf}.csv"
    df_sum.to_csv(sum_csv, index=False)

    print("\n" + "=" * 115)
    print(df_sum[[
        "Complexity_Tier", "AST_Depth_Mean", "Ext_Mean_ms", "Nat_Mean_ms",
        "Speedup_x", "VES_Ext", "VES_Nat", "p_value", "Significant"
    ]].to_string(index=False))
    print("=" * 115)
    print(f"[OK] Saved Statistical Summary CSV to: {sum_csv}")

    # à¸ªà¸£à¹‰à¸²à¸‡à¹„à¸Ÿà¸¥à¹Œà¸•à¸²à¸£à¸²à¸‡ LaTeX (IEEEtran booktabs)
    export_latex_table(root, df_sum, sf)

    # à¸ªà¸£à¹‰à¸²à¸‡à¸à¸£à¸²à¸Ÿà¸„à¸§à¸²à¸¡à¸¥à¸°à¹€à¸­à¸µà¸¢à¸”à¸ªà¸¹à¸‡ 300 DPI (PDF à¹à¸¥à¸° PNG)
    plot_publication_figures(root, df_sum, sf)


def export_latex_table(root: Path, df_sum: pd.DataFrame, sf: int):
    tex_lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\begin{threeparttable}",
        rf"\caption{{Empirical Execution Latency (ms), Valid Efficiency Score (VES), and Wilcoxon Signed-Rank Test Across Five SQL Complexity Levels (Scale Factor = {sf} GB)}}",
        r"\label{tab:empirical_complexity_results}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}lcccccccccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Complexity Category}} & \multicolumn{2}{c}{\textbf{AST Metrics}} & \multicolumn{2}{c}{\textbf{External Parquet (ms)}} & \multicolumn{2}{c}{\textbf{Native Storage (ms)}} & \multirow{2}{*}{\textbf{Speedup}} & \multicolumn{2}{c}{\textbf{Efficiency (VES)}} & \multirow{2}{*}{\textbf{Wilcoxon $p$}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){9-10}",
        r" & \textbf{Depth} & \textbf{Joins} & \textbf{Mean $\pm$ SD} & \textbf{P95} & \textbf{Mean $\pm$ SD} & \textbf{P95} & & \textbf{Ext.} & \textbf{Native} & \\",
        r"\midrule"
    ]

    for _, r in df_sum.iterrows():
        tier_tex = r["Complexity_Tier"].replace("&", r"\&")
        sig_star = r"$^{*}$" if r["Significant"] else ""
        p_str = "< 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"

        line = (
            f"\\textbf{{{tier_tex}}} & ${r['AST_Depth_Mean']:.1f}$ & ${r['Joins_Mean']:.1f}$ & "
            f"${r['Ext_Mean_ms']:.2f} \\pm {r['Ext_SD_ms']:.2f}$ & ${r['Ext_P95_ms']:.2f}$ & "
            f"$\\mathbf{{{r['Nat_Mean_ms']:.2f} \\pm {r['Nat_SD_ms']:.2f}}}$ & $\\mathbf{{{r['Nat_P95_ms']:.2f}}}$ & "
            f"${r['Speedup_x']:.2f}\\times$ & ${r['VES_Ext']:.1f}$ & $\\mathbf{{{r['VES_Nat']:.1f}}}$ & "
            f"${p_str}${sig_star} \\\\"
        )
        tex_lines.append(line)

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item[*] Statistically significant difference between External Parquet and Native Storage modes using paired Wilcoxon signed-rank test ($\alpha = 0.05$).",
        r"\item \textit{Acknowledgment:} This research project was financially supported by Mahasarakham University.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}"
    ])

    tex_out = root / "results" / "summary_tables" / f"table_empirical_results_sf{sf}.tex"
    tex_out.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"[OK] Exported IEEE LaTeX Table to    : {tex_out}")


def plot_publication_figures(root: Path, df_sum: pd.DataFrame, sf: int):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    x = np.arange(len(df_sum))
    width = 0.36

    short_labels = [t.split(":")[0] + "\n" + t.split(":")[1].strip() for t in df_sum["Complexity_Tier"]]

    # Subplot 1: Execution Latency (Mean +/- SD)
    rects1 = ax1.bar(
        x - width / 2, df_sum["Ext_Mean_ms"], width,
        yerr=df_sum["Ext_SD_ms"], capsize=3.5,
        label="External Parquet Mode", color="#4C72B0", edgecolor="black", linewidth=0.7
    )
    rects2 = ax1.bar(
        x + width / 2, df_sum["Nat_Mean_ms"], width,
        yerr=df_sum["Nat_SD_ms"], capsize=3.5,
        label="Native Table Mode", color="#55A868", edgecolor="black", linewidth=0.7
    )

    ax1.set_ylabel("Execution Latency (ms)")
    ax1.set_title(f"(a) Query Execution Latency by SQL Complexity (SF={sf}GB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # Subplot 2: Valid Efficiency Score (VES) & AST Depth Overlay
    bars_ves1 = ax2.bar(
        x - width / 2, df_sum["VES_Ext"], width,
        label="VES (External Parquet)", color="#C44E52", edgecolor="black", linewidth=0.7
    )
    bars_ves2 = ax2.bar(
        x + width / 2, df_sum["VES_Nat"], width,
        label="VES (Native Storage)", color="#8172B2", edgecolor="black", linewidth=0.7
    )

    ax2.set_ylabel("Valid Efficiency Score (VES)")
    ax2.set_title("(b) Valid Efficiency Score (VES) Across Complexity Tiers")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels)
    ax2.axhline(100.0, color="black", linestyle=":", linewidth=1.0, label="Baseline Reference (100)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper left")

    plt.tight_layout()

    fig_dir = root / "results" / "figures_300dpi"
    manuscript_fig_dir = root / "manuscript" / "figures"

    pdf_out = fig_dir / f"fig_complexity_benchmark_sf{sf}_300dpi.pdf"
    png_out = fig_dir / f"fig_complexity_benchmark_sf{sf}_300dpi.png"
    ms_pdf_out = manuscript_fig_dir / f"fig_complexity_benchmark_sf{sf}_300dpi.pdf"

    fig.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(ms_pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved 300 DPI PDF Figure to     : {pdf_out}")
    print(f"[OK] Saved 300 DPI PNG Figure to     : {png_out}")
    print(f"[OK] Synced Manuscript Figure to     : {ms_pdf_out}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Analysis, LaTeX Table & 300 DPI Figure Generator")
    parser.add_argument("--sf", type=int, default=None, help="Scale factor to analyze (e.g., 1 or 10)")
    args = parser.parse_args()
    evaluate_statistics_and_export(PROJECT_ROOT, args.sf)
