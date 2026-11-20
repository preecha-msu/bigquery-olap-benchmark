#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
TEX_V13 = MS_DIR / "bigquery_olap_benchmark_V13.tex"
FLAT13_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat13.zip"
REPO_URL = "[https://github.com/preecha-msu/bigquery-olap-benchmark](https://github.com/preecha-msu/bigquery-olap-benchmark)"
V13_TITLE = (
    "AST-Route: Tail-Safe CPU-GPU-Cloud OLAP Dispatch via PCIe Margin Guarding "
    "and Zero-Compile SQL Metadata"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tracked_files() -> list[str]:
    res = subprocess.run(
        ["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", check=True
    )
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def get_protected_names() -> set[str]:
    keep = {
        "bigquery_olap_benchmark_V13.tex",
        "bigquery_olap_benchmark_V13.pdf",
        "v13_verified_macros.tex",
        "references.bib",
        "sn-jnl.cls",
        "Journal_of_Big_Data_Snapp_Submission_Flat13.zip",
        "frozen_ast_route_v11_gbdt_model.json",
        "verify_and_regenerate_main_tables_R1_R6.py",
        "curate_v13_public_repository.py",
        "verify_zero_contradiction_V13.py",
        "v12_r29_r33_real_execution_overhaul.py",
        "v11_ast_route_live_sql_wrapper.py",
        "v13_b1_cloud_gpu_8to16vcpu_r5.py",
        "v13_groupB_step2_prospective_holdout_jcc_h.py",
        "run_independent_replication.py",
    }
    if TEX_V13.exists():
        raw = TEX_V13.read_text(encoding="utf-8", errors="replace")
        cleaned = raw.replace(r"\_", "_").replace(r"\allowbreak{}", "").replace(r"\allowbreak", "")
        for m in re.findall(r"[A-Za-z0-9_./-]+\.(?:py|csv|json|pdf|png|jpg|tex|bib|bst|cls|sql|txt|cff)", cleaned):
            keep.add(Path(m).name)
        for inc in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", raw):
            stem = Path(inc).name
            keep.add(stem)
            for ext in (".pdf", ".png", ".jpg", ".jpeg"):
                keep.add(stem + ext)
    return keep


def should_untrack(rel_posix: str, protected: set[str]) -> bool:
    p = Path(rel_posix)
    name = p.name
    parts = p.parts
    root_allowed = {
        "README.md", "LICENSE", "CITATION.cff", ".zenodo.json", "SHA256SUMS.txt",
        ".gitignore", "requirements.txt", "requirements-lock.txt",
        "Journal_of_Big_Data_Snapp_Submission_Flat13.zip",
    }
    if len(parts) == 1:
        return name not in root_allowed

    top = parts[0]
    if top.startswith(("_", "manuscript_backup", "clean_archival_repo")):
        return True

    if top == "manuscript":
        if p.suffix.lower() in {".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".synctex", ".fls", ".fdb_latexmk"}:
            return True
        if len(parts) == 2:
            if name in protected or p.suffix.lower() in {".bib", ".bst", ".cls"}:
                return False
            if re.search(r"Additional_File|Supplement", name, flags=re.I):
                return False
            return True
        if len(parts) >= 3 and parts[1] == "figures":
            return name not in protected
        return False

    if top == "scripts":
        if name in protected:
            return False
        patch_pats = (
            r"^(?:2[6-9]|[3-9]\d|1\d\d|2\d\d)_", r"^fix_", r"^patch_", r"^step\d",
            r"^v\d+_step", r"^restore_", r"^finalize_", r"^publish_clean",
            r"overhaul", r"reconcil", r"gatekeeper", r"regenerate_fig",
        )
        return any(re.search(pat, name, flags=re.I) for pat in patch_pats)

    if top == "results" and (name.startswith("_temp") or name.endswith(".tmp")):
        return True
    return False


def untrack_clutter(protected: set[str]) -> None:
    tracked = git_tracked_files()
    to_rm = [f for f in tracked if should_untrack(f, protected)]
    for i in range(0, len(to_rm), 50):
        batch = to_rm[i : i + 50]
        subprocess.run(
            ["git", "rm", "-f", "--cached", "--ignore-unmatch", "--"] + batch,
            cwd=str(ROOT), check=True, capture_output=True
        )
    print(f"[CLEANUP] Untracked {len(to_rm)} legacy/clutter files from GitHub index (0 local files deleted).")


def write_all_v13_metadata() -> None:
    gitignore = "\n".join([
        "*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk", "*.synctex.gz",
        "__pycache__/", "*.py[cod]", ".venv/", "venv/", ".ipynb_checkpoints/",
        "_cleanroom*/", "_tmp*/", "manuscript_backup*/", "clean_archival_repo*/",
        "work_step.txt", "ZENODO_DOI.txt", "Overleaf_V*.zip", "FINAL_SUBMISSION_GATEKEEPER_CERTIFICATE.csv",
        "bigquery_olap_benchmark_V[0-9]*.tex", "bigquery_olap_benchmark_V[0-9]*.pdf",
        "manuscript/main.tex", "manuscript/bigquery_olap_benchmark_JBD_Submission_Ready.*",
        "manuscript/bigquery_olap_benchmark_V[0-9].*", "manuscript/bigquery_olap_benchmark_V1[0-2].*",
        "manuscript/bigquery_olap_benchmark_V8*.*", ""
    ])
    (ROOT / ".gitignore").write_text(gitignore, encoding="utf-8")

    cff = "\n".join([
        "cff-version: 1.2.0",
        'message: "If you use this benchmark, routing framework, or telemetry dataset, please cite our work as below."',
        "type: software",
        f'title: "{V13_TITLE}"',
        'version: "13.0.0"',
        'date-released: "2026-11-20"',
        f'url: "{REPO_URL}"',
        f'repository-code: "{REPO_URL}"',
        "license: MIT",
        "authors:",
        '  - family-names: "Noiumkar"',
        '    given-names: "Preecha"',
        '    email: "preecha.n@msu.ac.th"',
        '    affiliation: "Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand"',
        ""
    ])
    (ROOT / "CITATION.cff").write_text(cff, encoding="utf-8")

    zenodo = {
        "title": V13_TITLE,
        "version": "13.0.0",
        "upload_type": "software",
        "publication_date": "2026-11-20",
        "creators": [{"name": "Noiumkar, Preecha", "affiliation": "Mahasarakham University, Thailand"}],
        "description": (
            "Official open-source benchmark suite, frozen pre-execution GBDT routing model, "
            "495-query row-level SQL registry (400 primary across Categories I-IV + 25 prospective OOD + "
            "70 enterprise stress), 4-node multi-scale telemetry logs (SF = 1, 10, 30, 100 GB), and "
            "Docker replication kit accompanying the V13 manuscript submitted to Journal of Big Data."
        ),
        "access_right": "open",
        "license": "MIT",
    }
    (ROOT / ".zenodo.json").write_text(json.dumps(zenodo, indent=2) + "\n", encoding="utf-8")

    targets = [
        MS_DIR / "bigquery_olap_benchmark_V13.tex",
        MS_DIR / "bigquery_olap_benchmark_V13.pdf",
        FLAT13_ZIP,
        ROOT / "reproducibility" / "frozen_ast_route_v11_gbdt_model.json",
        ROOT / "results" / "v13_master_query_registry_row_level.csv",
        ROOT / "results" / "v12_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv",
        ROOT / "results" / "v13_groupB_prospective_holdout_jcc_h_results.csv",
        ROOT / "results" / "v13_r1_r6_audit_manifest_sha256.json",
    ] + sorted((ROOT / "results").glob("v13_*.csv"))

    seen, sha_lines = set(), []
    for p in targets:
        if p.exists() and p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            if rel not in seen:
                sha_lines.append(f"{sha256_file(p)}  {rel}")
                seen.add(rel)
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    CB = chr(96) * 3
    readme_lines = [
        f"# {V13_TITLE}",
        "",
        "[![Manuscript](https://img.shields.io/badge/Manuscript-V13%20(Journal%20of%20Big%20Data)-005A9C)](manuscript/bigquery_olap_benchmark_V13.pdf)",
        "[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)",
        "[![Replication](https://img.shields.io/badge/Replication-Docker%20%2B%20SHA256%20Verified-success)](docker_replication/)",
        "",
        "**Author:** Preecha Noiumkar (`preecha.n@msu.ac.th`)  ",
        "**Affiliation:** Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand  ",
        "**Manuscript Version:** **V13** (*Journal of Big Data*, SpringerOpen)",
        "",
        "---",
        "",
        "## 1. Overview & Key Empirical Findings",
        "",
        "**AST-Route** is an engine-agnostic, pre-execution analytical SQL dispatch framework combining zero-compile `SQLGlot` AST parsing and Parquet row-group footer sketching (`172 us` cached; `4.15 ms` cold parse+sketch; `1.42 MB` RAM) with a calibrated PCIe transfer-margin abstention guard across **DuckDB**, **Polars**, **Apache DataFusion**, **SQLite 3**, **CUDA GPUs** (PyTorch CUDA, CuPy, NVIDIA RAPIDS `cuDF` / `cudf-polars`), and **Google BigQuery**.",
        "",
        "1. **PCIe Margin Guarding Drives Tail Safety (81.3%):** Stepwise ablation across the 75-query TPC-H development suite ($K=25$ seed clusters, 25-fold LOSO-CV at $\\text{SF}=1\\text{ GB}$) shows that the PCIe margin guard accounts for **81.3%** of P95 tail-regret reduction—cutting P95 regret from `257.31%` (Always GPU) and `358.40%` (Always Polars) to **`118.40%`** (`8%` Nested-CV Protocol P1, `55.37 ms` net) and **`114.11%`** (`12%` Fixed Protocol P2, `54.92 ms` net).",
        "2. **Minimal Two-Engine Pool Sufficiency (98.7%):** A compact **`{Polars, GPU}`** portfolio (`55.28 ms`) retains **98.7%** of full six-engine utility at $\\text{SF}=1\\text{ GB}$, while **DuckDB** handles correlated subqueries (Q16–Q22 / TPC-DS) and **BigQuery** offloads $\\text{SF}=100\\text{ GB}$ NVMe spill regimes.",
        "3. **Cross-Dialect Zero-Compile Role:** Static `SQLGlot` AST + footer sketching avoids target DBMS planner/catalog lock-in across SQL dialects (note: single-dialect native DuckDB `EXPLAIN-DAG` is faster at `1.41 ms` on DuckDB-only SQL). Under $K=25$ cluster bootstrap, aggregate P95 differences vs. `Ridge+Guard` ($p=0.38$) and `EXPLAIN-DAG` ($p=0.74$) are non-significant (`TOST` at $\\pm 25\\text{ pp}$ inconclusive), while GBR+CSCI lowers L3–L5 subgroup mean regret (`18.98%` -> `8.49%`).",
        "4. **External Consortium Holdouts ($K=38$ Clusters):** Across **155 untouched consortium queries** (99 Official TPC-DS + 43 ClickBench [20 native SQL + 23 dialect-transpiled] + 13 SSB), tail-regret reduction (`97.05%` vs. `644.62%` CPU and `588.99%` GPU P95; `0/38` LOCO sign flips) is **directional but not statistically significant** ($p_{\\text{perm}} = 0.125$ vs. CPU, $p_{\\text{perm}} = 0.198$ vs. GPU; random-intercept LME retired).",
        "5. **Prospective OOD & Harness-Shift Boundary (`3/25` Top-1):** On 25 locked single-shot prospective queries (`Seed_S7` $1/15$, `DSB_Subset` $2/5$, `JCC_H_Skew` $0/5$) over compact 1–2 table in-memory Arrow buffers, zero-shot Top-1 drops to **`12.0%`** (`83.98%` mean regret, `346.00%` P95 regret; `19/25` routed to Tier-L1 DataFusion basin; `0/25` static Mahalanobis OOD triggers), establishing that **schema/harness-local calibration ($m=5$ canaries) is required** before deployment on unseen skewed workloads.",
        "",
        "---",
        "",
        "## 2. Clean Repository Layout",
        "",
        "| Path | Contents |",
        "| :--- | :--- |",
        "| `manuscript/bigquery_olap_benchmark_V13.pdf` | Compiled V13 Research Article PDF (26 pages, Springer *Journal of Big Data*) |",
        "| `manuscript/bigquery_olap_benchmark_V13.tex` | Canonical V13 LaTeX source (28/28 consistency gates verified) |",
        "| `Journal_of_Big_Data_Snapp_Submission_Flat13.zip` | Clean-Room verified flat submission package |",
        "| `reproducibility/frozen_ast_route_v11_gbdt_model.json` | Frozen Cat-I GBDT routing ensemble (`SHA-256: f0e5f17c832472e3...`) |",
        "| `results/v13_master_query_registry_row_level.csv` | 495-row SQL registry (`400` Primary Cat. I–IV + `25` Prospective OOD + `70` Enterprise Stress) |",
        "| `docker_replication/` | Single-command containerized replication kit (`EX = 1.000`) |",
        "| `workloads/` | SQL benchmark queries across all 4 epistemic categories and auxiliary suites |",
        "| `results/` | Verified per-query & multi-scale ($\\text{SF}=1\\text{--}100\\text{ GB}$) hardware/energy CSV logs |",
        "| `scripts/` | Benchmark harnesses, router evaluation, and verification scripts |",
        "| `SHA256SUMS.txt` | Cryptographic SHA-256 checksums of primary artifacts and logs |",
        "",
        "---",
        "",
        "## 3. Reproducibility & Verification Commands",
        "",
        "**1. Containerized Clean-Room Replication (Docker):**",
        f"{CB}bash",
        "cd docker_replication",
        "docker compose up --build",
        f"{CB}",
        "",
        "**2. Verify Manuscript Tables & Zero-Contradiction Gates (Python):**",
        f"{CB}bash",
        "pip install -r reproducibility/requirements-lock.txt",
        "python reproducibility/verify_and_regenerate_main_tables_R1_R6.py",
        "python scripts/verify_zero_contradiction_V13.py",
        f"{CB}",
        "",
        "---",
        "",
        "## 4. Protocol Transparency",
        "",
        "This study was **not prospectively preregistered on OSF**. Repository `.zenodo.json` files and Git tags denote code-freeze checkpoints; Git commit chronology and Nested-CV guard harmonization (`12%` V1–V9 Fixed Protocol P2 -> `8%` V10–V13 Inner-Fold Nested-CV Protocol P1) are documented in **Tables 8, 10, and 18** of the V13 manuscript.",
        ""
    ]
    (ROOT / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"[METADATA PASS] Updated README.md, CITATION.cff, .zenodo.json, .gitignore, and SHA256SUMS.txt ({len(sha_lines)} hashes).")


if __name__ == "__main__":
    untrack_clutter(get_protected_names())
    write_all_v13_metadata()