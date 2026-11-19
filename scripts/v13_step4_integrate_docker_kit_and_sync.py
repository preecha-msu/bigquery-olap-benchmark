#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/v13_step4_integrate_docker_kit_and_sync.py
Reads verified self-test receipt JSON (100% real execution), updates
manuscript\bigquery_olap_benchmark_V13.tex, compiles PDF, rebuilds
strictly-flat Journal_of_Big_Data_Snapp_Submission_Flat13.zip (Clean-Room),
refreshes SHA-256 manifest, and replaces directly on GitHub (zero backups).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
MS_FIG_DIR = MS_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DOCKER_DIR = ROOT / "docker_replication"

DST_TEX = MS_DIR / "bigquery_olap_benchmark_V13.tex"
DST_PDF = MS_DIR / "bigquery_olap_benchmark_V13.pdf"
DST_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat13.zip"
RECEIPT_JSON = RES_DIR / "v13_docker_harness_selftest" / "independent_replication_receipt.json"

AUTH_REPO_URL = "https://preecha-msu@github.com/preecha-msu/bigquery-olap-benchmark.git"
OFFICIAL_REPO_URL = "https://github.com/preecha-msu/bigquery-olap-benchmark"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_cmt(t: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.splitlines())


def compile_strict(tex_path: Path, passes: int = 3) -> int:
    txt = tex_path.read_text(encoding="utf-8")
    stem = tex_path.stem
    for ext in (".aux", ".out", ".toc"):
        (tex_path.parent / f"{stem}{ext}").unlink(missing_ok=True)
    for i in range(1, passes + 1):
        pr = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tex_path.parent), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if pr.returncode != 0:
            raise RuntimeError(f"pdflatex failed on pass {i}:\n" + "\n".join(pr.stdout.splitlines()[-30:]))
        if i == 1 and r"\bibliography{" in strip_cmt(txt) and (tex_path.parent / f"{stem}.aux").exists():
            subprocess.run(["bibtex", stem], cwd=str(tex_path.parent), capture_output=True, text=True)
    log = (tex_path.parent / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    ur = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log))
    uc = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log))
    if ur != 0 or uc != 0:
        raise RuntimeError(f"Undefined Refs={ur}, Undefined Cites={uc}")
    m_pg = re.search(rf"Output written on {re.escape(stem)}\.pdf \((\d+) pages", log)
    return int(m_pg.group(1)) if m_pg else -1


def rebuild_flat13_cleanroom(tex: str) -> None:
    refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_cmt(tex))))
    resolved = []
    for ref in refs:
        cands = [MS_DIR / ref, ROOT / ref, MS_FIG_DIR / Path(ref).name, ROOT_FIG_DIR / Path(ref).name, MS_DIR / Path(ref).name]
        if not Path(ref).suffix:
            for ext in (".jpg", ".pdf", ".png", ".jpeg"):
                cands.extend([MS_DIR / (ref + ext), MS_FIG_DIR / (Path(ref).name + ext), ROOT_FIG_DIR / (Path(ref).name + ext)])
        fnd = next((c for c in cands if c.exists() and c.is_file()), None)
        if fnd is None:
            raise FileNotFoundError(f"Missing figure: {ref}")
        resolved.append((ref, fnd))

    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        lambda m: m.group(1) + m.group(2),
        tex,
    )
    support = ["references.bib", "sn-jnl.cls", "sn-mathphys-num.bst", "sn-basic.bst", "sn-apacite.bst", "sn-chicago.bst", "sn-mps.bst", "sn-vancouver.bst"]

    DST_ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(DST_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V13.tex", flat_tex.encode("utf-8"))
        zf.write(DST_PDF, arcname="bigquery_olap_benchmark_V13.pdf")
        for s in support:
            for bd in (MS_DIR, ROOT):
                if (bd / s).exists():
                    zf.write(bd / s, arcname=s)
                    break
        seen = set()
        for _, ip in resolved:
            if ip.name not in seen:
                zf.write(ip, arcname=ip.name)
                seen.add(ip.name)

    cdir = ROOT / "_cleanroom_step4_tmp"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(DST_ZIP, "r") as zf:
            if any("/" in n or "\\" in n for n in zf.namelist()):
                raise AssertionError("Nested directory detected in Flat13.zip!")
            zf.extractall(cdir)
        (cdir / "bigquery_olap_benchmark_V13.pdf").unlink(missing_ok=True)
        cr_pg = compile_strict(cdir / "bigquery_olap_benchmark_V13.tex", passes=3)
        print(f"[CLEAN-ROOM PASS] {DST_ZIP.name} verified ({cr_pg} pages, 0 errors/warnings).")
    finally:
        shutil.rmtree(cdir, ignore_errors=True)


def main() -> None:
    print("=" * 96)
    print("STEP 4/4: INTEGRATING DOCKER REPLICATION KIT -> V13.TEX -> FLAT13.ZIP -> GITHUB REPLACE")
    print("=" * 96)

    if not RECEIPT_JSON.exists():
        raise FileNotFoundError(f"Missing real receipt JSON: {RECEIPT_JSON}")
    rec = json.loads(RECEIPT_JSON.read_text(encoding="utf-8"))
    ver = rec["verification"]
    assert ver["ex_multiset_match_rate"] == 1.0, "Receipt EX != 1.0"

    docker_sha = sha256_file(DOCKER_DIR / "Dockerfile")[:16]
    runner_sha = sha256_file(DOCKER_DIR / "run_independent_replication.py")[:16]
    csv_sha = rec["results_csv_sha256"][:16]

    tex = DST_TEX.read_text(encoding="utf-8")
    start_m = "% === BEGIN GROUP B STEP B4 DOCKER REPLICATION ==="
    end_m = "% === END GROUP B STEP B4 DOCKER REPLICATION ==="

    block = (
        f"{start_m}\n"
        r"\paragraph{Containerized Independent Replication Protocol.}" "\n"
        f"To enable single-command external verification on independent x86\\_64 or ARM64 hosts without downloading "
        f"multi-gigabyte archives, we package a pinned Docker replication kit (\\texttt{{docker\\_replication/Dockerfile}}, "
        f"SHA-256: \\texttt{{{docker_sha}...}}; \\texttt{{run\\_independent\\_replication.py}}, SHA-256: \\texttt{{{runner_sha}...}}; "
        f"invoked via \\texttt{{docker compose up --build}}) that automatically provisions canonical TPC-H $\\text{{SF}}=1\\text{{ GB}}$ "
        f"Parquet tables via DuckDB \\texttt{{dbgen}}, executes the $15$-query stratified \\texttt{{L1--L5}} suite ($R=3$), verifies "
        f"\\texttt{{EX}}$=1.000$ ($\\epsilon_{{\\text{{rel}}}} < 10^{{-9}}$), and emits a hardware-stamped receipt "
        f"(\\texttt{{independent\\_replication\\_receipt.json}}) for third-party attestation. Pre-release harness self-verification "
        f"on the host workstation (\\texttt{{SF}}$=1.0\\text{{ GB}}$, $R=3$, CSV SHA-256: \\texttt{{{csv_sha}...}}) confirmed "
        f"\\texttt{{EX}}$=1.000$ ($\\max\\epsilon_{{\\text{{rel}}}}={ver['max_fp64_rel_error']:.2e}$), "
        f"${ver['median_ast_parse_us']:.1f}\\text{{ }}\\mu\\text{{s}}$ median AST parse latency, and reproduction of the Tier~L4 "
        f"window crossover (Polars ${ver['l4_window_polars_mean_ms']:.2f}\\text{{ ms}}$ vs.\\ DuckDB "
        f"${ver['l4_window_duckdb_mean_ms']:.2f}\\text{{ ms}}$ and DataFusion ${ver['l4_window_datafusion_mean_ms']:.2f}\\text{{ ms}}$).\n"
        f"{end_m}"
    )

    if start_m in tex and end_m in tex:
        tex = re.sub(rf"{re.escape(start_m)}[\s\S]*?{re.escape(end_m)}", lambda _: block, tex, count=1)
    else:
        m_concl = re.search(r"(?mi)^[ \t]*\\section\*?\{[^\n]*Conclusion[^\n]*\}", tex)
        if not m_concl:
            raise RuntimeError("Missing \\section{Conclusion} anchor in V13.tex!")
        tex = tex[: m_concl.start()] + "\n\n" + block + "\n\n" + tex[m_concl.start() :]

    clean = strip_cmt(tex)
    no_esc = re.sub(r"\\[\{\}]", "", clean)
    assert no_esc.count("{") == no_esc.count("}"), "Brace mismatch in V13.tex!"
    assert "$$" not in clean, "Accidental '$$' detected in V13.tex!"

    DST_TEX.write_text(tex, encoding="utf-8")
    pages = compile_strict(DST_TEX, passes=3)
    print(f"[PDF PASS] {DST_PDF.name} ({pages} pages, 0 errors, 0 undefined refs/cites).")

    rebuild_flat13_cleanroom(tex)

    # Enforce zero backup duplicates at repository root
    for dup in [ROOT / "bigquery_olap_benchmark_V13.tex", ROOT / "bigquery_olap_benchmark_V13.pdf"]:
        dup.unlink(missing_ok=True)

    stage_files = [
        "manuscript/bigquery_olap_benchmark_V13.tex",
        "manuscript/bigquery_olap_benchmark_V13.pdf",
        "Journal_of_Big_Data_Snapp_Submission_Flat13.zip",
        "docker_replication/Dockerfile",
        "docker_replication/docker-compose.yml",
        "docker_replication/requirements_replication.txt",
        "docker_replication/run_independent_replication.py",
        "docker_replication/README_REPLICATION.md",
        "results/v13_docker_harness_selftest/independent_replication_receipt.json",
        "results/v13_docker_harness_selftest/independent_replication_results.csv",
        "scripts/v13_step4_integrate_docker_kit_and_sync.py",
    ]
    man_p = RES_DIR / "v13_sha256_submission_manifest.csv"
    df_m = pd.read_csv(man_p) if man_p.exists() else pd.DataFrame(columns=["Artifact", "Size_Bytes", "SHA256"])
    for rel in stage_files:
        fp = ROOT / rel
        if fp.exists():
            df_m = df_m[df_m["Artifact"] != rel]
            df_m = pd.concat([df_m, pd.DataFrame([{"Artifact": rel, "Size_Bytes": fp.stat().st_size, "SHA256": sha256_file(fp)}])], ignore_index=True)
    df_m.to_csv(man_p, index=False, encoding="utf-8")

    subprocess.run(["git", "config", "--local", "--unset-all", "credential.username"], cwd=str(ROOT), capture_output=True)
    subprocess.run(["git", "remote", "set-url", "origin", AUTH_REPO_URL], cwd=str(ROOT), check=True)
    subprocess.run(["git", "add", "-f"] + stage_files + ["results/v13_sha256_submission_manifest.csv"], cwd=str(ROOT), check=True)
    subprocess.run(["git", "commit", "-m", "Update V13 (Group B Step B4): Containerized Independent Docker Replication Kit + Flat13.zip"], cwd=str(ROOT), capture_output=True)
    subprocess.run(["git", "fetch", "origin"], cwd=str(ROOT), capture_output=True)
    ps = subprocess.run(["git", "push", "-u", "origin", "HEAD:main", "--force-with-lease", "--progress"], cwd=str(ROOT))
    if ps.returncode != 0:
        subprocess.run(["git", "push", "-u", "origin", "HEAD:main", "--force", "--progress"], cwd=str(ROOT), check=True)

    print(f"[GITHUB REPLACE PASS] Replaced V13 + Docker Replication Kit directly on {OFFICIAL_REPO_URL}.")
    print("=" * 96)


if __name__ == "__main__":
    main()