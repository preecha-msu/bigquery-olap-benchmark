#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/init_V12_from_V11_baseline.py
# Clones and initializes the complete V12 manuscript, figures, CSV baseline,
# companion Reviewer Response / Cover Letter templates, 18-gate V12 verifier,
# and Clean-Room verified Flat12.zip & Overleaf_V12_Submission_Ready.zip.

import hashlib
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
SCRIPTS_DIR = ROOT / "scripts"

V11_TEX = MS_DIR / "bigquery_olap_benchmark_V11.tex"
V12_TEX = MS_DIR / "bigquery_olap_benchmark_V12.tex"
ROOT_V12_TEX = ROOT / "bigquery_olap_benchmark_V12.tex"
V12_PDF = MS_DIR / "bigquery_olap_benchmark_V12.pdf"
ROOT_V12_PDF = ROOT / "bigquery_olap_benchmark_V12.pdf"

FLAT12_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat12.zip"
OVERLEAF12_ZIP = ROOT / "Overleaf_V12_Submission_Ready.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_tex_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())


def clone_v12_data_and_figures() -> tuple[Path, str]:
    v11_csv = RES_DIR / "v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
    v12_csv = RES_DIR / "v12_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
    if not v11_csv.exists():
        raise FileNotFoundError(f"Missing V11 master CSV: {v11_csv}")
    shutil.copy2(v11_csv, v12_csv)

    old_stem = "fig_v11_4node_multiscale_cudf_sf1_10_30_100"
    new_stem = "fig_v12_4node_multiscale_cudf_sf1_10_30_100"
    for d in (MS_DIR, MS_FIG_DIR, ROOT_FIG_DIR):
        for ext in (".jpg", ".pdf"):
            src = d / f"{old_stem}{ext}"
            if src.exists():
                shutil.copy2(src, d / f"{new_stem}{ext}")

    return v12_csv, sha256_file(v12_csv)


def build_v12_tex_from_v11(v12_csv_sha: str) -> str:
    if not V11_TEX.exists():
        raise FileNotFoundError(f"Missing V11 baseline .tex: {V11_TEX}")

    tex_v11 = V11_TEX.read_text(encoding="utf-8")
    brace_before = tex_v11.count("{") - tex_v11.count("}")

    # Update internal V11 markers, figure/table labels, and CSV artifact references to V12
    tex_v12 = (
        tex_v11
        .replace(
            "% === BEGIN V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
            "% === BEGIN V12 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
        )
        .replace(
            "% === END V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
            "% === END V12 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===",
        )
        .replace("V11-B4-ABS", "V12-B4-ABS")
        .replace("V11-B4-INTRO", "V12-B4-INTRO")
        .replace("V11-B2-B3-SYNC", "V12-B2-B3-SYNC")
        .replace("V11-B1-LIM", "V12-B1-LIM")
        .replace("V11-B4-CONC", "V12-B4-CONC")
        .replace("V11-B5-AVAIL", "V12-B5-AVAIL")
        .replace(
            "fig_v11_4node_multiscale_cudf_sf1_10_30_100",
            "fig_v12_4node_multiscale_cudf_sf1_10_30_100",
        )
        .replace("tab:v11_fournode_multiscale_cudf", "tab:v12_fournode_multiscale_cudf")
        .replace("fig:v11_fournode_multiscale_cudf", "fig:v12_fournode_multiscale_cudf")
        .replace(
            r"v11\_\allowbreak{}4node\_\allowbreak{}6engine\_\allowbreak{}cudf\_\allowbreak{}multiscale\_\allowbreak{}sf1\_\allowbreak{}10\_\allowbreak{}30\_\allowbreak{}100\_\allowbreak{}master.csv",
            r"v12\_\allowbreak{}4node\_\allowbreak{}6engine\_\allowbreak{}cudf\_\allowbreak{}multiscale\_\allowbreak{}sf1\_\allowbreak{}10\_\allowbreak{}30\_\allowbreak{}100\_\allowbreak{}master.csv",
        )
        .replace(
            r"v11\_\allowbreak{}sha256\_\allowbreak{}submission\_\allowbreak{}manifest.csv",
            r"v12\_\allowbreak{}sha256\_\allowbreak{}submission\_\allowbreak{}manifest.csv",
        )
        .replace(
            r"verify\_\allowbreak{}zero\_\allowbreak{}contradiction\_\allowbreak{}V11.py",
            r"verify\_\allowbreak{}zero\_\allowbreak{}contradiction\_\allowbreak{}V12.py",
        )
        .replace("V11 master SHA-256:", "V12 master SHA-256:")
        .replace("V11 fixed four-node", "V12 fixed four-node")
        .replace("The V11 four-node", "The V12 four-node")
        .replace("our V11 external multi-scale audit", "our V12 external multi-scale audit")
    )

    brace_after = tex_v12.count("{") - tex_v12.count("}")
    if brace_after != brace_before:
        raise AssertionError(f"Brace balance changed during V12 clone: {brace_before} vs {brace_after}")

    header_banner = (
        "% ==========================================================================\n"
        "% MANUSCRIPT VERSION: V12 (Initialized from Audited V11 Baseline)\n"
        "% PURPOSE: Active Working Manuscript for Next-Round Peer Reviewer Revisions\n"
        f"% BASELINE V12 MASTER CSV SHA-256: {v12_csv_sha}\n"
        "% ==========================================================================\n"
    )
    if not tex_v12.startswith("% ==========================================================================\n% MANUSCRIPT VERSION: V12"):
        tex_v12 = header_banner + tex_v12

    V12_TEX.write_text(tex_v12, encoding="utf-8")
    shutil.copy2(V12_TEX, ROOT_V12_TEX)
    return tex_v12


def compile_latex_file(tex_path: Path, passes: int = 3) -> int:
    txt = tex_path.read_text(encoding="utf-8")
    stem = tex_path.stem
    for i in range(1, passes + 1):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tex_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            tail = "\n".join(res.stdout.splitlines()[-35:])
            raise RuntimeError(f"pdflatex failed on {tex_path.name} (pass {i}):\n{tail}")
        aux_p = tex_path.parent / f"{stem}.aux"
        if i == 1 and r"\bibliography{" in txt and aux_p.exists():
            subprocess.run(["bibtex", stem], cwd=str(tex_path.parent), capture_output=True, text=True)

    log_p = tex_path.parent / f"{stem}.log"
    log_txt = log_p.read_text(encoding="utf-8", errors="replace") if log_p.exists() else ""
    ur = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    uc = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    if ur != 0 or uc != 0:
        raise RuntimeError(f"{tex_path.name} has Undefined Refs={ur}, Citations={uc}")
    m_pg = re.search(rf"Output written on {re.escape(stem)}\.pdf \((\d+) pages", log_txt)
    return int(m_pg.group(1)) if m_pg else -1


def resolve_all_figures(tex_txt: str) -> list[tuple[str, Path]]:
    clean_tex = strip_tex_comments(tex_txt)
    refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex)))
    resolved = []
    for ref in refs:
        cands = [
            MS_DIR / ref, ROOT / ref,
            MS_FIG_DIR / Path(ref).name,
            ROOT_FIG_DIR / Path(ref).name,
            MS_DIR / Path(ref).name,
        ]
        if not Path(ref).suffix:
            for ext in (".jpg", ".pdf", ".png", ".jpeg"):
                cands.extend([
                    MS_DIR / (ref + ext),
                    MS_FIG_DIR / (Path(ref).name + ext),
                    ROOT_FIG_DIR / (Path(ref).name + ext),
                ])
        found = next((c for c in cands if c.exists() and c.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Missing figure in V12.tex: {ref}")
        resolved.append((ref, found))
    return resolved


def build_and_cleanroom_verify_v12_zips(tex_txt: str) -> None:
    resolved_imgs = resolve_all_figures(tex_txt)
    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        lambda m: m.group(1) + m.group(2),
        tex_txt,
    )
    support_files = [
        "references.bib", "sn-jnl.cls", "sn-mathphys-num.bst",
        "sn-basic.bst", "sn-apacite.bst", "sn-chicago.bst", "sn-mps.bst", "sn-vancouver.bst",
    ]

    with zipfile.ZipFile(FLAT12_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V12.tex", flat_tex.encode("utf-8"))
        if V12_PDF.exists():
            zf.write(V12_PDF, arcname="bigquery_olap_benchmark_V12.pdf")
        for sname in support_files:
            for bd in (MS_DIR, ROOT):
                sp = bd / sname
                if sp.exists() and sp.is_file():
                    zf.write(sp, arcname=sname)
                    break
        added = set()
        for _, img_p in resolved_imgs:
            if img_p.name not in added:
                zf.write(img_p, arcname=img_p.name)
                added.add(img_p.name)

    clean_dir = ROOT / "_cleanroom_v12_init"
    shutil.rmtree(clean_dir, ignore_errors=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(FLAT12_ZIP, "r") as zf:
            if any("/" in n or "\\" in n for n in zf.namelist()):
                raise AssertionError("Nested folder detected inside Flat12.zip")
            zf.extractall(clean_dir)
        (clean_dir / "bigquery_olap_benchmark_V12.pdf").unlink(missing_ok=True)
        compile_latex_file(clean_dir / "bigquery_olap_benchmark_V12.tex", passes=3)
    finally:
        shutil.rmtree(clean_dir, ignore_errors=True)

    with zipfile.ZipFile(OVERLEAF12_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(V12_TEX, arcname="main.tex")
        zf.write(V12_TEX, arcname="bigquery_olap_benchmark_V12.tex")
        if V12_PDF.exists():
            zf.write(V12_PDF, arcname="bigquery_olap_benchmark_V12.pdf")
        for sname in support_files:
            for bd in (MS_DIR, ROOT):
                sp = bd / sname
                if sp.exists() and sp.is_file():
                    zf.write(sp, arcname=sname)
                    break
        added_ov = set()
        for ref, img_p in resolved_imgs:
            arc_p = ref if Path(ref).suffix else f"{ref}{img_p.suffix}"
            if arc_p not in added_ov:
                zf.write(img_p, arcname=arc_p)
                added_ov.add(arc_p)
            sub_p = f"figures/{img_p.name}"
            if sub_p not in added_ov:
                zf.write(img_p, arcname=sub_p)
                added_ov.add(sub_p)


def clone_v12_companion_docs(v12_tex_sha: str, v12_pdf_sha: str) -> None:
    for base_name in ("Cover_Letter", "Response_to_Reviewers"):
        src = MS_DIR / f"{base_name}_V11.tex"
        dst = MS_DIR / f"{base_name}_V12.tex"
        if src.exists():
            txt = src.read_text(encoding="utf-8")
            txt = (
                txt.replace("Version V11", "Version V12")
                .replace("(V11)", "(V12)")
                .replace("in V11:", "in V12:")
                .replace("V11 multi-scale", "V12 multi-scale")
                .replace("bigquery\\_olap\\_benchmark\\_V11", "bigquery\\_olap\\_benchmark\\_V12")
            )
            txt = re.sub(
                r"(bigquery\\_olap\\_benchmark\\_V12\.tex\} \(SHA-256: \\texttt\{)[0-9a-f]+(\.\.\.\}\))",
                rf"\g<1>{v12_tex_sha[:20]}\g<2>",
                txt,
            )
            txt = re.sub(
                r"(bigquery\\_olap\\_benchmark\\_V12\.pdf\} \(SHA-256: \\texttt\{)[0-9a-f]+(\.\.\.\}\))",
                rf"\g<1>{v12_pdf_sha[:20]}\g<2>",
                txt,
            )
            dst.write_text(txt, encoding="utf-8")
            compile_latex_file(dst, passes=2)


def write_v12_verifier() -> Path:
    src_v = SCRIPTS_DIR / "verify_zero_contradiction_V11.py"
    dst_v = SCRIPTS_DIR / "verify_zero_contradiction_V12.py"
    txt = src_v.read_text(encoding="utf-8")
    txt = (
        txt.replace("V11", "V12")
        .replace("v11_", "v12_")
        .replace("Flat11", "Flat12")
        .replace("FLAT11", "FLAT12")
        .replace(
            "fig_v11_4node_multiscale_cudf_sf1_10_30_100",
            "fig_v12_4node_multiscale_cudf_sf1_10_30_100",
        )
    )
    dst_v.write_text(txt, encoding="utf-8")
    return dst_v


def write_v12_manifest() -> Path:
    tracked = [
        "manuscript/bigquery_olap_benchmark_V12.tex",
        "manuscript/bigquery_olap_benchmark_V12.pdf",
        "manuscript/Cover_Letter_V12.pdf",
        "manuscript/Response_to_Reviewers_V12.pdf",
        "manuscript/figures/fig_v12_4node_multiscale_cudf_sf1_10_30_100.jpg",
        "Journal_of_Big_Data_Snapp_Submission_Flat12.zip",
        "Overleaf_V12_Submission_Ready.zip",
        "results/v12_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv",
        "scripts/verify_zero_contradiction_V12.py",
    ]
    rows = []
    for rel in tracked:
        p = ROOT / rel
        if p.exists():
            rows.append({
                "Artifact_Path": rel,
                "Size_Bytes": p.stat().st_size,
                "SHA256": sha256_file(p),
            })
    out_m = RES_DIR / "v12_sha256_submission_manifest.csv"
    pd.DataFrame(rows).to_csv(out_m, index=False, encoding="utf-8")
    return out_m


def main() -> None:
    print("=" * 96)
    print("[INITIALIZING V12 WORKSPACE FROM AUDITED V11 BASELINE]")
    print("=" * 96)

    v12_csv, v12_csv_sha = clone_v12_data_and_figures()
    print(f"  [1/6] Cloned V12 Master CSV & 300-DPI Figures ({v12_csv.name})")

    tex_v12 = build_v12_tex_from_v11(v12_csv_sha)
    pages = compile_latex_file(V12_TEX, passes=3)
    try:
        shutil.copy2(V12_PDF, ROOT_V12_PDF)
    except PermissionError:
        pass
    print(f"  [2/6] Created & Compiled bigquery_olap_benchmark_V12.pdf ({pages} pages, 0 Undefined Refs/Cites)")

    build_and_cleanroom_verify_v12_zips(tex_v12)
    print("  [3/6] Built & Clean-Room Verified Flat12.zip + Overleaf_V12_Submission_Ready.zip")

    clone_v12_companion_docs(sha256_file(V12_TEX), sha256_file(V12_PDF))
    print("  [4/6] Initialized & Compiled Cover_Letter_V12.pdf and Response_to_Reviewers_V12.pdf")

    verifier_v12 = write_v12_verifier()
    subprocess.run(["python", str(verifier_v12)], cwd=str(ROOT), check=True)
    print("  [5/6] Executed verify_zero_contradiction_V12.py (18/18 Gates Passed)")

    manifest_v12 = write_v12_manifest()
    print(f"  [6/6] Saved cryptographic V12 baseline manifest: {manifest_v12.relative_to(ROOT)}")
    print("=" * 96)
    print(pd.read_csv(manifest_v12).to_string(index=False))
    print("=" * 96)
    print("[V12 BASELINE READY 100% FOR REVIEWER REVISIONS]")


if __name__ == "__main__":
    main()