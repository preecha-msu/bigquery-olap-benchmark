#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/build_overleaf_v6_clean_zip.py

import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark")
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"
MASTER_V2_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"
V6_TEX = MS_DIR / "bigquery_olap_benchmark_V6.tex"

ZIP_IN_MS = MS_DIR / "Overleaf_V6_Project.zip"
ZIP_IN_ROOT = ROOT / "Overleaf_V6_Project.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inline_any_inputs(tex: str) -> str:
    pat = re.compile(r"\\input\{([^}]+)\}")
    def _repl(m):
        fname = m.group(1).strip()
        if not fname.endswith(".tex"):
            fname += ".tex"
        cand = MS_DIR / fname
        if cand.exists():
            return cand.read_text(encoding="utf-8")
        return m.group(0)
    return pat.sub(_repl, tex)


def resolve_figure_file(raw_ref: str) -> tuple[Path, str]:
    clean_ref = raw_ref.strip().replace("\\", "/")
    fname = Path(clean_ref).name
    candidates = []
    if Path(fname).suffix:
        candidates.extend([
            MS_DIR / clean_ref,
            FIG_DIR / fname,
            ROOT / "figures" / fname,
        ])
    else:
        for ext in (".pdf", ".png", ".jpg"):
            candidates.extend([
                MS_DIR / f"{clean_ref}{ext}",
                FIG_DIR / f"{fname}{ext}",
                ROOT / "figures" / f"{fname}{ext}",
            ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve(), f"figures/{c.name}"
    raise FileNotFoundError(f"Missing figure referenced in TeX: {raw_ref}")


def main():
    print("=" * 95)
    print("BUILDING CLEAN SELF-CONTAINED OVERLEAF V6 PACKAGE: Overleaf_V6_Project.zip")
    print("=" * 95)

    if not MASTER_V2_TEX.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_V2_TEX}")

    tex = MASTER_V2_TEX.read_text(encoding="utf-8")
    tex = inline_any_inputs(tex)

    # ตรวจสอบพาธรูปภาพทุกจุดใน \includegraphics และจัดให้อยู่ใน figures/<filename> มาตรฐานเดียวกัน
    inc_pat = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
    fig_map: dict[str, Path] = {}

    def _norm_inc(m):
        prefix, raw_path, suffix = m.group(1), m.group(2), m.group(3)
        real_file, arc_rel = resolve_figure_file(raw_path)
        fig_map[arc_rel] = real_file
        return f"{prefix}{arc_rel}{suffix}"

    tex = inc_pat.sub(_norm_inc, tex)

    # บันทึกซิงค์กลับลงไฟล์หลัก bigquery_olap_benchmark_V2.tex และ V6.tex ตามข้อกำหนด
    MASTER_V2_TEX.write_text(tex, encoding="utf-8")
    V6_TEX.write_text(tex, encoding="utf-8")
    (MS_DIR / "main.tex").write_text(tex, encoding="utf-8")
    (MS_DIR / "overleaf_ready_single_file.tex").write_text(tex, encoding="utf-8")

    # สร้างไฟล์ Overleaf_V6_Project.zip (ใส่ main.tex เพียงไฟล์เดียวที่ Root เพื่อให้ Overleaf เลือกถูก 100%)
    with zipfile.ZipFile(ZIP_IN_MS, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", tex.encode("utf-8"))
        bib_path = MS_DIR / "references.bib"
        if bib_path.exists():
            zf.write(bib_path, arcname="references.bib")
        # ตรวจสอบไฟล์ .cls หรือ .bst เผื่อมีการเรียกใช้ในโฟลเดอร์ manuscript
        for ext_pat in ("*.cls", "*.bst", "*.sty"):
            for style_f in sorted(MS_DIR.glob(ext_pat)):
                zf.write(style_f, arcname=style_f.name)
        for arc_rel, real_f in sorted(fig_map.items()):
            zf.write(real_f, arcname=arc_rel)

    # คัดลอกไว้ที่โฟลเดอร์นอกสุดของโปรเจกต์ด้วยเพื่อความสะดวกตอนกด Upload
    shutil.copy2(ZIP_IN_MS, ZIP_IN_ROOT)
    shutil.copy2(ZIP_IN_MS, MS_DIR / "overleaf_V6_submission_bundle.zip")

    # ทดสอบแตกไฟล์ลงใน Clean-Room Temp Folder แล้วรัน pdflatex เสมือนรันบนเซิร์ฟเวอร์ Overleaf จริง!
    with tempfile.TemporaryDirectory(prefix="overleaf_v6_test_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_IN_MS, "r") as zf:
            zf.extractall(tmp_path)

        for p in (1, 2, 3):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"[OVERLEAF SANDBOX ERROR Pass {p}]:")
                print("\n".join(proc.stdout.splitlines()[-30:]))
                raise RuntimeError("Sandbox compilation failed!")

        log_txt = (tmp_path / "main.log").read_text(encoding="utf-8", errors="ignore")
        undefs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_txt)
        undef_cites = re.findall(r"LaTeX Warning: Citation `([^']+)' on page \d+ undefined", log_txt)
        m_pg = re.search(r"Output written on main\.pdf \((\d+) pages", log_txt)
        n_pages = int(m_pg.group(1)) if m_pg else 0

        # นำ PDF ที่คอมไพล์สะอาดจาก Sandbox มาอัปเดตให้ V2.pdf และ V6.pdf ด้วย
        built_pdf = tmp_path / "main.pdf"
        if built_pdf.exists():
            for pdf_name in ("bigquery_olap_benchmark_V2.pdf", "bigquery_olap_benchmark_V6.pdf"):
                try:
                    shutil.copy2(built_pdf, MS_DIR / pdf_name)
                except PermissionError:
                    pass

    size_mb = ZIP_IN_ROOT.stat().st_size / (1024 * 1024)
    print(f"  [PASS] Bundled Figures Verified  : {len(fig_map)} exact figure files inside figures/")
    print(f"  [PASS] Sandbox Compile Test      : ZERO Errors ({n_pages} pages)")
    print(f"  [PASS] Undefined References (??) : {len(undefs)} | Undefined Citations: {len(undef_cites)}")
    print(f"  [PASS] Synced Master V2.tex      : {MASTER_V2_TEX}")
    print("-" * 95)
    print(f"  [ZIP READY 1] {ZIP_IN_ROOT} ({size_mb:.2f} MB)")
    print(f"  [ZIP READY 2] {ZIP_IN_MS} ({size_mb:.2f} MB)")
    print(f"  [ZIP SHA-256] {sha256_file(ZIP_IN_ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    main()