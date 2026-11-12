#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, shutil, subprocess, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR, FIG_DIR, ROOT_FIG = ROOT/"manuscript", ROOT/"manuscript"/"figures", ROOT/"figures"
V8_TEX, V8_PDF = MS_DIR/"bigquery_olap_benchmark_V8.tex", MS_DIR/"bigquery_olap_benchmark_V8.pdf"

for bak in ROOT.glob("manuscript_backup_*"):
    if bak.is_dir():
        shutil.rmtree(bak, ignore_errors=True)

def build_cache_busted_fig15():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.4, 4.35), dpi=300)

    x = np.arange(4)
    flip_naive  = [0.00, 2.99, 7.35, 14.05]
    flip_conf   = [0.00, 3.02, 5.78, 12.00]
    viol_nopcie = [10.67, 10.89, 10.87, 11.05]
    viol_conf   = [9.33, 9.33, 9.12, 8.86]

    b1 = axa.bar(x - 0.17, flip_naive, 0.33, color="#e67e22", edgecolor="black", label="Unguarded ArgMin Flip Rate (%)")
    b2 = axa.bar(x + 0.17, flip_conf,  0.33, color="#1e8449", edgecolor="black", label="Conformal AST-Route Flip Rate (%)")
    ax2 = axa.twinx()
    ax2.plot(x, viol_nopcie, "s--", color="#9e2a2b", lw=1.8, markersize=5.5, label="Without 12% PCIe Guard: >100% Viol. (%)")
    ax2.plot(x, viol_conf,   "o-",  color="#1f4e78", lw=2.0, markersize=5.5, label="Conformal AST-Route: >100% Viol. (%)")

    for b in list(b1) + list(b2):
        if b.get_height() > 0:
            axa.text(b.get_x() + b.get_width()/2, b.get_height() + 0.38, f"{b.get_height():.2f}%", ha="center", fontsize=6.9, fontweight="bold")

    axa.set_xticks(x)
    axa.set_xticklabels(["0%\nClean", "+-5%\nNoise", "+-10%\nNoise", "+-20%\nNoise"], fontsize=8.0)
    axa.set_ylabel("Routing Flip Rate (%)", fontsize=8.8)
    ax2.set_ylabel("Severe Regret (>100%) Rate (%)", fontsize=8.8, color="#1f4e78")
    axa.set_ylim(0, 20.5); ax2.set_ylim(5.0, 14.0)
    axa.set_title("(a) Router Decision Stability & 12% PCIe Guard (1,000-MC)", fontsize=9.2, fontweight="bold")
    h1, l1 = axa.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axa.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.7)
    axa.grid(axis="y", ls="--", alpha=0.4)

    xd = np.arange(7)
    pol_ms = [82.47, 92.89, 87.60, 87.60, 86.86, 91.26, 91.26]
    ast_ms = [60.90, 62.34, 68.85, 64.49, 67.01, 78.47, 76.13]
    pi_cov = [89.0, 88.7, 88.7, 88.7, 85.3, 75.0, 89.0]

    axb.bar(xd - 0.17, pol_ms, 0.33, color="#7f8c8d", edgecolor="black", label="Static Polars Mean Latency (ms)")
    ba = axb.bar(xd + 0.17, ast_ms, 0.33, color="#2e5984", edgecolor="black", label="AST-Route Net Latency (ms)")
    ba[5].set_facecolor("#c44e52"); ba[6].set_facecolor("#2ca02c")
    for i, (mv, cv) in enumerate(zip(ast_ms, pi_cov)):
        axb.text(xd[i] + 0.17, mv + 1.8, f"{mv:.1f}\n({cv:.0f}%)", ha="center", fontsize=6.4, fontweight="bold")

    axb.set_xticks(xd)
    axb.set_xticklabels(["D0\nBase", "D1\nMix", "D2a\nStale", "D2b\nRefr", "D3\nSkew.8", "D4a\nSkew1.4", "D4b\nRecal"], fontsize=7.5)
    axb.set_ylabel("Mean Execution Latency (ms)", fontsize=8.8)
    axb.set_ylim(0, 118.0)
    axb.set_title("(b) Dynamic Drift & 5-Canary Recalibration Policy", fontsize=9.2, fontweight="bold")
    axb.legend(loc="upper left", fontsize=7.1)
    axb.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    for fn in ("fig_v8_stability_and_workload_drift_300dpi.pdf", "fig_v7_stability_and_workload_drift_300dpi.pdf"):
        fig.savefig(FIG_DIR / fn, dpi=300, bbox_inches="tight")
        if ROOT_FIG.exists():
            fig.savefig(ROOT_FIG / fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

def unwrap_resize_and_fitbox(tbl_block: str) -> str:
    s = re.sub(r"\\begin\{fitasttable\}\s*", "", tbl_block)
    s = re.sub(r"\s*\\end\{fitasttable\}", "", s)
    pat = re.compile(r"\\resizebox\{[^{}]*\}\{[^{}]*\}\s*\{")
    while True:
        m = pat.search(s)
        if not m:
            break
        ob = m.end() - 1
        depth, cb = 0, -1
        for i in range(ob, len(s)):
            if s[i] == "{" and (i == 0 or s[i-1] != "\\"):
                depth += 1
            elif s[i] == "}" and (i == 0 or s[i-1] != "\\"):
                depth -= 1
                if depth == 0:
                    cb = i; break
        if cb == -1:
            break
        inner = s[ob + 1 : cb].strip()
        if inner.startswith("%"):
            inner = inner[1:].lstrip()
        s = s[:m.start()] + inner + s[cb + 1:]
    return s

def move_caption_to_top(tbl_block: str) -> str:
    fit_pos = tbl_block.find(r"\begin{fitasttable}")
    cap_pos = tbl_block.find(r"\caption", fit_pos if fit_pos != -1 else 0)
    if fit_pos == -1 or cap_pos == -1 or cap_pos < fit_pos:
        return tbl_block
    i = cap_pos + len(r"\caption")
    while i < len(tbl_block) and tbl_block[i].isspace():
        i += 1
    if i < len(tbl_block) and tbl_block[i] == "[":
        brk = 1; i += 1
        while i < len(tbl_block) and brk > 0:
            if tbl_block[i] == "[": brk += 1
            elif tbl_block[i] == "]": brk -= 1
            i += 1
    while i < len(tbl_block) and tbl_block[i].isspace():
        i += 1
    if i >= len(tbl_block) or tbl_block[i] != "{":
        return tbl_block
    depth = 0
    for j in range(i, len(tbl_block)):
        if tbl_block[j] == "{" and (j == 0 or tbl_block[j-1] != "\\"):
            depth += 1
        elif tbl_block[j] == "}" and (j == 0 or tbl_block[j-1] != "\\"):
            depth -= 1
            if depth == 0:
                end_cap = j + 1; break
    else:
        return tbl_block
    m_lbl = re.match(r"\s*(\\label\{[^{}]+\})", tbl_block[end_cap:])
    if m_lbl:
        end_cap += m_lbl.end()
    cap_str = tbl_block[cap_pos:end_cap].strip()
    rem = tbl_block[:cap_pos] + tbl_block[end_cap:]
    nf = rem.find(r"\begin{fitasttable}")
    return rem[:nf] + cap_str + "\n" + rem[nf:]

def normalize_one_table(tbl_block: str) -> str:
    s = unwrap_resize_and_fitbox(tbl_block)
    s = re.sub(r"\\parbox\{[^{}]*\}", r"\\parbox{0.96\\textwidth}", s)
    m_b = re.search(r"\\begin\{(tabular\*?|tabularx)\}", s)
    if m_b:
        env = m_b.group(1)
        m_ends = list(re.finditer(rf"\\end\{{{re.escape(env)}\}}", s))
        if m_ends:
            b_i, e_i = m_b.start(), m_ends[-1].end()
            s = s[:b_i] + "\\begin{fitasttable}\n" + s[b_i:e_i] + "\n\\end{fitasttable}" + s[e_i:]
    return move_caption_to_top(s)

def fix_texttt_and_urls(tex: str) -> str:
    tex = re.sub(
        r"\\(?:no)?link?url\{scripts/run\\?_reproducibility\\?_audit\.py\}|\\url\{scripts/run\\?_reproducibility\\?_audit\.py\}",
        r"\\texttt{scripts/run\\_reproducibility\\_audit.py}",
        tex,
    )
    tex = tex.replace(r"scripts/run\\_reproducibility\\_audit.py", r"scripts/run\_reproducibility\_audit.py")
    def repl_tt(m: re.Match) -> str:
        inner = m.group(1).replace(r"\\_", r"\_")
        inner = re.sub(r"\\_\\allowbreak(?:\{\})?\s*", r"\\_", inner)
        inner = re.sub(r"/\\allowbreak(?:\{\})?\s*", "/", inner)
        if r"\_" in inner or "/" in inner:
            inner = inner.replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")
        return f"\\texttt{{{inner}}}"
    return re.sub(r"\\texttt\{([^{}\n]+)\}", repl_tt, tex)

def main():
    build_cache_busted_fig15()
    tex = V8_TEX.read_text(encoding="utf-8")
    tex = tex.replace("\\RequirePackage[T1]{fontenc}\n", "")

    fit_macro = r"""\newsavebox{\asttblbox}
\newsavebox{\asttblboxB}
\newenvironment{fitasttable}{%
  \begin{lrbox}{\asttblbox}%
}{%
  \end{lrbox}%
  \sbox{\asttblboxB}{%
    \ifdim\wd\asttblbox>0.96\textwidth
      \resizebox{0.96\textwidth}{!}{\usebox{\asttblbox}}%
    \else
      \usebox{\asttblbox}%
    \fi
  }%
  \ifdim\ht\asttblboxB>0.72\textheight
    \resizebox{!}{0.72\textheight}{\usebox{\asttblboxB}}%
  \else
    \usebox{\asttblboxB}%
  \fi
}
\sloppy
\emergencystretch=3em"""

    if r"\newenvironment{fitasttable}" not in tex:
        tex = tex.replace(r"\begin{document}", fit_macro + "\n\\begin{document}", 1)

    tex = tex.replace(
        "figures/fig_v7_stability_and_workload_drift_300dpi.pdf",
        "figures/fig_v8_stability_and_workload_drift_300dpi.pdf",
    )

    t_cnt = 0
    def repl_tbl(m: re.Match) -> str:
        nonlocal t_cnt; t_cnt += 1
        return normalize_one_table(m.group(0))

    tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", repl_tbl, tex, flags=re.DOTALL)

    for cap_anchor in (
        r"\caption{Internal validation of the AST-Route framework under 25-seed LOSO-CV",
        r"\caption{Big Data Multi-Scale",
        r"\caption{Consolidated Step-3 robustness audit",
    ):
        pos = tex.find(cap_anchor)
        if pos != -1:
            tb_pos = tex.rfind(r"\begin{table", 0, pos)
            if tb_pos != -1 and r"\clearpage" not in tex[max(0, tb_pos - 25) : tb_pos]:
                tex = tex[:tb_pos] + "\\clearpage\n" + tex[tb_pos:]

    tex = fix_texttt_and_urls(tex).replace("dispatching..", "dispatching.")

    no_cmt = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    cl_b = re.sub(r"\\[\{\}]", "", no_cmt)
    assert cl_b.count("{") == cl_b.count("}"), f"Brace mismatch: {cl_b.count('{')} vs {cl_b.count('}')}"

    V8_TEX.write_text(tex, encoding="utf-8")
    (MS_DIR / "main.tex").write_text(tex, encoding="utf-8")

    for p_i in (1, 2, 3):
        pr = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", V8_TEX.name],
            cwd=str(MS_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if pr.returncode != 0:
            print("\n".join(pr.stdout.splitlines()[-25:]))
            raise RuntimeError(f"pdflatex failed on pass {p_i}")

    log_txt = (MS_DIR / "bigquery_olap_benchmark_V8.log").read_text(encoding="utf-8", errors="replace")
    severe_hbox = [
        ln for ln in log_txt.splitlines()
        if "Overfull \\hbox" in ln and float(re.search(r"\(([0-9.]+)pt too wide\)", ln).group(1)) > 3.0
    ]
    print(f"[CLAMPED {t_cnt} TABLES] {V8_PDF.name} ({V8_PDF.stat().st_size:,} bytes) | Severe Overfull (>3pt) = {len(severe_hbox)}")

    zip_p = ROOT / "Overleaf_V8_Full_Submission_Ready.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(V8_TEX, arcname="main.tex")
        zf.write(V8_TEX, arcname="bigquery_olap_benchmark_V8.tex")
        if (MS_DIR / "references.bib").exists():
            zf.write(MS_DIR / "references.bib", arcname="references.bib")
        for fg in sorted(FIG_DIR.glob("*.pdf")):
            zf.write(fg, arcname=f"figures/{fg.name}")
    print(f"[OVERLEAF ZIP READY] {zip_p.name} ({zip_p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()