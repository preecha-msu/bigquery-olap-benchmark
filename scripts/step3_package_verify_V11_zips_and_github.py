#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step3_package_verify_V11_zips_and_github.py
ขั้นตอนที่ 3 (V11 Final Packaging, Clean-Room Snapp Compile Verification & SHA-256 Manifest):
1. ตรวจสอบไฟล์รูปภาพทุกไฟล์ที่อ้างอิงใน manuscript/bigquery_olap_benchmark_V11.tex
2. สร้าง Journal_of_Big_Data_Snapp_Submission_Flat11.zip (Flat structure สำหรับระบบ Springer Snapp)
3. แตกไฟล์ Flat11.zip ลงในโฟลเดอร์ชั่วคราวแบบ Clean-Room แล้วสั่งคอมไพล์ pdflatex จริง (ตรวจ Undefined Refs = 0)
4. สร้าง Overleaf_V11_Submission_Ready.zip และบันทึกใบรับรอง results/v11_sha256_submission_manifest.csv
"""

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_tex_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())


def resolve_all_v11_figures(tex_txt: str) -> list[tuple[str, Path]]:
    clean_tex = strip_tex_comments(tex_txt)
    refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex)))
    resolved = []
    for ref in refs:
        cands = [
            MS_DIR / ref,
            ROOT / ref,
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
            raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพที่อ้างอิงใน V11.tex: {ref}")
        resolved.append((ref, found))
    return resolved


def build_and_cleanroom_test_flat11(v11_ms: Path, resolved_imgs: list[tuple[str, Path]]) -> Path:
    tex_txt = v11_ms.read_text(encoding="utf-8")

    # ปรับพาธรูปภาพทั้งหมดให้เป็น Flat (ตัด figures/ หรือ manuscript/figures/ ออกสำหรับ Snapp)
    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        lambda m: m.group(1) + m.group(2),
        tex_txt,
    )

    flat11_zip = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat11.zip"
    v11_pdf = MS_DIR / "bigquery_olap_benchmark_V11.pdf"

    aux_support_names = [
        "references.bib", "sn-jnl.cls", "sn-mathphys-num.bst",
        "sn-basic.bst", "sn-apacite.bst", "sn-chicago.bst", "sn-mps.bst", "sn-vancouver.bst"
    ]

    with zipfile.ZipFile(flat11_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V11.tex", flat_tex.encode("utf-8"))
        if v11_pdf.exists():
            zf.write(v11_pdf, arcname="bigquery_olap_benchmark_V11.pdf")

        for sname in aux_support_names:
            for bd in (MS_DIR, ROOT):
                sp = bd / sname
                if sp.exists() and sp.is_file():
                    zf.write(sp, arcname=sname)
                    break

        added_names = set()
        for _, img_p in resolved_imgs:
            if img_p.name not in added_names:
                zf.write(img_p, arcname=img_p.name)
                added_names.add(img_p.name)

    # Clean-Room Extraction & Live Compilation Verification
    clean_dir = ROOT / "_cleanroom_flat11_verify"
    shutil.rmtree(clean_dir, ignore_errors=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(flat11_zip, "r") as zf:
            names = zf.namelist()
            # ยืนยันว่าไม่มีโฟลเดอร์ย่อยค้างใน Flat11.zip
            nested = [n for n in names if "/" in n or "\\" in n]
            if nested:
                raise AssertionError(f"พบพาธซ้อนใน Flat11.zip: {nested}")
            zf.extractall(clean_dir)

        # ลบ PDF เดิมชั่วคราวในห้องทดลอง เพื่อพิสูจน์ว่าคอมไพล์จาก .tex + assets ใน ZIP ได้ PDF ใหม่จริง
        test_pdf = clean_dir / "bigquery_olap_benchmark_V11.pdf"
        test_pdf.unlink(missing_ok=True)

        for p_idx in (1, 2, 3):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "bigquery_olap_benchmark_V11.tex"],
                cwd=str(clean_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if p_idx == 1 and (clean_dir / "references.bib").exists() and r"\bibliography{" in flat_tex:
                subprocess.run(["bibtex", "bigquery_olap_benchmark_V11"], cwd=str(clean_dir), capture_output=True, text=True)

        if not test_pdf.exists():
            tail = "\n".join(r.stdout.splitlines()[-30:])
            raise RuntimeError(f"Clean-room compile ของ Flat11.zip ล้มเหลว:\n{tail}")

        log_txt = (clean_dir / "bigquery_olap_benchmark_V11.log").read_text(encoding="utf-8", errors="replace")
        u_ref = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
        u_cit = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
        if u_ref != 0 or u_cit != 0:
            raise RuntimeError(f"Clean-room Flat11.zip พบ Undefined Refs={u_ref}, Citations={u_cit}")
        print(f"  [CLEAN-ROOM PASS] Flat11.zip แตกไฟล์และคอมไพล์ผ่าน 100% (Undefined Refs={u_ref}, Citations={u_cit})")
    finally:
        shutil.rmtree(clean_dir, ignore_errors=True)

    return flat11_zip


def build_overleaf_v11_zip(v11_ms: Path, resolved_imgs: list[tuple[str, Path]]) -> Path:
    ov_zip = ROOT / "Overleaf_V11_Submission_Ready.zip"
    v11_pdf = MS_DIR / "bigquery_olap_benchmark_V11.pdf"

    with zipfile.ZipFile(ov_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(v11_ms, arcname="main.tex")
        zf.write(v11_ms, arcname="bigquery_olap_benchmark_V11.tex")
        if v11_pdf.exists():
            zf.write(v11_pdf, arcname="bigquery_olap_benchmark_V11.pdf")

        for sname in ["references.bib", "sn-jnl.cls", "sn-mathphys-num.bst", "sn-basic.bst"]:
            for bd in (MS_DIR, ROOT):
                sp = bd / sname
                if sp.exists() and sp.is_file():
                    zf.write(sp, arcname=sname)
                    break

        added_arc = set()
        for ref, img_p in resolved_imgs:
            arc_path = ref if Path(ref).suffix else f"{ref}{img_p.suffix}"
            if arc_path not in added_arc:
                zf.write(img_p, arcname=arc_path)
                added_arc.add(arc_path)
            fig_sub = f"figures/{img_p.name}"
            if fig_sub not in added_arc:
                zf.write(img_p, arcname=fig_sub)
                added_arc.add(fig_sub)

    return ov_zip


def write_v11_manifest() -> Path:
    tracked = [
        "manuscript/bigquery_olap_benchmark_V11.tex",
        "manuscript/bigquery_olap_benchmark_V11.pdf",
        "manuscript/figures/fig_v11_4node_multiscale_cudf_sf1_10_30_100.jpg",
        "Journal_of_Big_Data_Snapp_Submission_Flat11.zip",
        "Overleaf_V11_Submission_Ready.zip",
        "results/v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv",
        "results/multiscale_gpu_vram_and_arrow_streaming_sf1_10_30_100.csv",
        "results/node2_laptop16c_multiscale_sf1_10_30_100.csv",
        "results/node3_macbook_m1_multiscale_sf1_10_30_100.csv",
        "results/node4_tesla_t4_6engines_cudf_sf1_10_30_100.csv",
        "results/node4_tesla_t4_cudf_and_cudf_polars_sf1_10_30_100.csv",
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
    df_m = pd.DataFrame(rows)
    out_m = RES_DIR / "v11_sha256_submission_manifest.csv"
    df_m.to_csv(out_m, index=False, encoding="utf-8")
    return out_m


def main() -> None:
    print("=" * 98)
    print("[STEP 3 - V11] PACKAGE FLAT11.ZIP & OVERLEAF_V11.ZIP + CLEAN-ROOM COMPILE AUDIT")
    print("=" * 98)

    v11_ms = MS_DIR / "bigquery_olap_benchmark_V11.tex"
    tex_txt = v11_ms.read_text(encoding="utf-8")
    resolved_imgs = resolve_all_v11_figures(tex_txt)
    jpg_cnt = sum(1 for _, p in resolved_imgs if p.suffix.lower() in (".jpg", ".jpeg"))
    print(f"[1/4] ตรวจสอบรูปภาพใน V11.tex ครบ {len(resolved_imgs)} รายการ (ไฟล์ .jpg = {jpg_cnt} รายการ) ไม่พบไฟล์สูญหาย")

    flat11_zip = build_and_cleanroom_test_flat11(v11_ms, resolved_imgs)
    with zipfile.ZipFile(flat11_zip, "r") as zf:
        f11_files = zf.namelist()
    print(f"[2/4] สร้าง {flat11_zip.name} สำเร็จ ({flat11_zip.stat().st_size:,} bytes | {len(f11_files)} flat files)")
    print(f"      Flat11.zip SHA-256: {sha256_file(flat11_zip)}")

    ov11_zip = build_overleaf_v11_zip(v11_ms, resolved_imgs)
    print(f"[3/4] สร้าง {ov11_zip.name} สำเร็จ ({ov11_zip.stat().st_size:,} bytes)")
    print(f"      Overleaf_V11.zip SHA-256: {sha256_file(ov11_zip)}")

    manifest_csv = write_v11_manifest()
    print(f"[4/4] บันทึกใบรับรอง SHA-256 ครบทุกไฟล์รุ่น V11 ที่: {manifest_csv.relative_to(ROOT)}")
    print("      Manifest SHA-256: " + sha256_file(manifest_csv))
    print("=" * 98)
    print(pd.read_csv(manifest_csv).to_string(index=False))
    print("=" * 98)
    print("[ALL V11 STEPS (1-3) COMPLETED 100%]")


if __name__ == "__main__":
    main()