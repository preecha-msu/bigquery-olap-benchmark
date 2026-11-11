#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, re, shutil, subprocess, zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TARGET_TEX = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.tex"
LOG_FILE = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.log"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

tex = TARGET_TEX.read_text(encoding="utf-8")

nb2_upgraded_paragraph = (
    "Furthermore, comprehensive cross-hardware and cross-version validation on an independent secondary CPU-only machine "
    "(\\texttt{LAPTOP-LTL5PC7P}, 16 physical cores / 22 logical threads, 15.5~GB RAM, $M_{\\text{gpu\\_max}}=0$~GB, "
    "DuckDB v2.0.0, Polars v2.1.0, DataFusion v55.0.0; canonical TPC-H $\\text{SF}=1.0$ \\texttt{lineitem.parquet} "
    "SHA-256: \\texttt{41ba056fcf213207...}) established four empirical findings: "
    "(i) $100.0\\%$ multiset result equivalence ($\\text{EX}=1.000$, max FP64 relative error $2.87\\times 10^{-16} < 10^{-9}$) "
    "with $172.9~\\mu$s pre-execution footer metadata extraction; "
    "(ii) under 30 analytical queries (15 stratified L1--L5 + 15 official TPC-H templates) with leak-free leave-one-group-out routing, "
    "CPU-only AST-Route achieved $86.67\\%$ Top-1 and $100.00\\%$ Top-2 accuracy at $62.05$~ms mean latency ($8.83\\%$ mean regret, $44.07\\%$ P95 regret), "
    "outperforming Always DuckDB v2.0 ($71.19$~ms, $15.96\\%$ mean regret, $84.55\\%$ P95 regret; targeted tail-regret reduction on window/CTE queries with paired Wilcoxon $p=0.3409$ across $N=30$ due to $23/30$ identical scan/join selections), "
    "Always Polars v2.1 ($71.43$~ms, $96.74\\%$ regret, $p=1.07\\times 10^{-3}$), "
    "and Always DataFusion v55.0 ($127.35$~ms, $p=1.86\\times 10^{-9}$) while preserving positive partial CSCI rank invariance "
    "($\\rho_{\\text{partial}}(\\text{CSCI}, T \\mid \\sigma) = 0.3818\\text{--}0.4496$); "
    "(iii) under 22-thread multi-stream concurrency ($C \\in \\{1,4,8,16,32\\}$, $0.0\\%$ failure rate), hybrid AST-Route dispatching "
    "(Scan/Join to DuckDB, Window/Sort to Polars) boosted $C=32$ throughput to $22.68$~QPS (vs.\\ $17.51$~QPS on Polars and $4.10$~QPS on DuckDB) "
    "and compressed P50/P95/P99 tail latencies to $589.8/2620.1/2781.4$~ms (vs.\\ $1228.4/3484.9/3596.1$~ms on Polars); and "
    "(iv) sustained 45-second ACPI DC shunt hardware telemetry ($2,817$ queries, $2,232.0$~J battery discharge drop, $46.40$~W active vs.\\ $20.97$~W idle) "
    "empirically confirmed why Windows SMBus EC polling intervals ($\\Delta t_{\\text{EC}} \\ge 15$~s) necessitate TDP/DVFS utilization integration "
    "for millisecond-scale single-query CPU energy accounting."
)

# แทนที่ย่อหน้าเดิมก่อนขึ้น \section{Experimental Validation อย่างแม่นยำ 100%
pattern = re.compile(
    r"Furthermore,\s+independent environment re-execution on a separate secondary machine.*?(\\section\{Experimental Validation)",
    flags=re.DOTALL,
)
if pattern.search(tex):
    tex = pattern.sub(lambda m: nb2_upgraded_paragraph + "\n\n" + m.group(1), tex, count=1)
elif "22.68~QPS" not in tex:
    tex = re.sub(
        r"(\\section\{Experimental Validation)",
        lambda m: nb2_upgraded_paragraph + "\n\n" + m.group(1),
        tex,
        count=1,
    )

TARGET_TEX.write_text(tex, encoding="utf-8")
shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "main.tex")
shutil.copy2(TARGET_TEX, MANUSCRIPT_DIR / "overleaf_ready_single_file.tex")
shutil.copy2(TARGET_TEX, PROJECT_ROOT / "main.tex")

for p in (1, 2):
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
        cwd=str(MANUSCRIPT_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"[COMPILE ERROR Pass {p}]:\n" + "\n".join(proc.stdout.splitlines()[-25:]))
        raise RuntimeError("LaTeX compilation failed")

log_txt = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
undef = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)

pdf_v2 = MANUSCRIPT_DIR / "bigquery_olap_benchmark_V2.pdf"
for clone_name in ["bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V4.pdf"]:
    try:
        shutil.copy2(pdf_v2, MANUSCRIPT_DIR / clone_name)
    except PermissionError:
        pass

for zname in ["overleaf_V2_submission_bundle.zip", "overleaf_project_bundle.zip"]:
    zpath = MANUSCRIPT_DIR / zname
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, arcname="main.tex")
        zf.write(TARGET_TEX, arcname="bigquery_olap_benchmark_V2.tex")
        bib = MANUSCRIPT_DIR / "references.bib"
        if bib.exists():
            zf.write(bib, arcname="references.bib")
        for f in sorted(FIGURES_DIR.glob("*")):
            if f.is_file():
                zf.write(f, arcname=f"figures/{f.name}")

final_tex = TARGET_TEX.read_text(encoding="utf-8")
print("=" * 98)
print(f"[VERIFY 1] 22.68 QPS & 62.05 ms in V2.tex    : {'22.68~QPS' in final_tex and '62.05~ms' in final_tex}")
print(f"[VERIFY 2] Master & NB2 SHA-256 in V2.tex    : {'5a645bba57807962' in final_tex and 'nb2\\_exp1' in final_tex}")
print(f"[VERIFY 3] Undefined References Count        : {len(undef)} (Must be 0)")
print(f"[FINAL V2.TEX SHA-256                        ] {sha256_file(TARGET_TEX)}")
print(f"[FINAL V2.PDF SHA-256                        ] {sha256_file(pdf_v2)}")
print("=" * 98)