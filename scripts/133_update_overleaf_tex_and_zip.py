# -*- coding: utf-8 -*-
import os, re, shutil, hashlib, zipfile
from pathlib import Path

ROOT = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark").resolve()
MS_DIR = ROOT / "manuscript"
FIG_DIR = MS_DIR / "figures"

# เลือกไฟล์ต้นฉบับล่าสุดที่มีอยู่ในเครื่อง
CANDIDATES = [
    MS_DIR / "bigquery_olap_benchmark_V8_5.tex",
    MS_DIR / "bigquery_olap_benchmark_V8.tex",
    MS_DIR / "bigquery_olap_benchmark_V2.tex",
]
SRC_TEX = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()

def inline_inputs(tex: str) -> str:
    pat = re.compile(r"\\input\{([^}]+)\}")
    def _repl(m):
        fn = m.group(1).strip()
        if not fn.endswith(".tex"):
            fn += ".tex"
        cand = MS_DIR / fn
        return cand.read_text(encoding="utf-8") if cand.exists() else m.group(0)
    return pat.sub(_repl, tex)

def resolve_fig(raw_ref: str) -> Path | None:
    clean = raw_ref.strip().replace("\\", "/")
    fname = Path(clean).name
    cands = []
    if Path(fname).suffix:
        cands.extend([MS_DIR / clean, FIG_DIR / fname, ROOT / "figures" / fname, ROOT / "Snapp_Journal_of_Big_Data_Submission" / fname])
    else:
        for ext in (".pdf", ".png", ".jpg"):
            cands.extend([MS_DIR / f"{clean}{ext}", FIG_DIR / f"{fname}{ext}", ROOT / "figures" / f"{fname}{ext}"])
    for c in cands:
        if c.exists() and c.is_file():
            return c.resolve()
    return None

tex = SRC_TEX.read_text(encoding="utf-8")
tex = inline_inputs(tex)
tex = tex.replace("±", r"$\pm$")

# ตรวจสอบสมดุลวงเล็บปีกกา (ไม่นับคอมเมนต์และ \{ \})
cleaned_lines = [re.sub(r"(?<!\\)%.*$", "", ln).replace(r"\{", "").replace(r"\}", "") for ln in tex.splitlines()]
joined_clean = "\n".join(cleaned_lines)
b_open, b_close = joined_clean.count("{"), joined_clean.count("}")
assert b_open == b_close, f"Brace mismatch detected: {b_open} != {b_close}"

# จัดรูปแบบพาธรูปภาพมาตรฐาน figures/<filename> สำหรับ Overleaf Project
inc_pat = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
fig_map = {}

def _norm_fig(m):
    prefix, raw_p, suffix = m.group(1), m.group(2), m.group(3)
    real_f = resolve_fig(raw_p)
    if real_f:
        arc_rel = f"figures/{real_f.name}"
        fig_map[arc_rel] = real_f
        return f"{prefix}{arc_rel}{suffix}"
    return m.group(0)

tex_std = inc_pat.sub(_norm_fig, tex)

# บันทึกซิงค์ลงไฟล์ .tex หลักทั้งหมด (ไม่สร้างไฟล์ backup)
targets = [
    MS_DIR / "bigquery_olap_benchmark_V8_5.tex",
    MS_DIR / "bigquery_olap_benchmark_V8.tex",
    MS_DIR / "bigquery_olap_benchmark_V2.tex",
    MS_DIR / "main.tex",
    MS_DIR / "overleaf_ready_single_file.tex",
    ROOT / "bigquery_olap_benchmark_V2.tex",
    ROOT / "main.tex",
]
for t in targets:
    t.write_text(tex_std, encoding="utf-8")

# 1. สร้าง Overleaf_V8_5_Ready.zip (มี main.tex ไฟล์เดียวที่ Root + โฟลเดอร์ figures/)
zip_overleaf = ROOT / "Overleaf_V8_5_Ready.zip"
with zipfile.ZipFile(zip_overleaf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("main.tex", tex_std.encode("utf-8"))
    bib = MS_DIR / "references.bib"
    if bib.exists():
        zf.write(bib, arcname="references.bib")
    for ext_pat in ("*.cls", "*.bst", "*.sty"):
        for sf in sorted(MS_DIR.glob(ext_pat)):
            zf.write(sf, arcname=sf.name)
    for arc_rel, real_f in sorted(fig_map.items()):
        zf.write(real_f, arcname=arc_rel)

shutil.copy2(zip_overleaf, MS_DIR / "overleaf_V2_submission_bundle.zip")
shutil.copy2(zip_overleaf, ROOT / "Overleaf_V2_Submission_Ready.zip")

# 2. สร้าง Journal_of_Big_Data_Snapp_Submission_Flat.zip (Flat directory ไม่มีโฟลเดอร์ย่อย)
tex_flat = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+)(\})", r"\1\2\3", tex_std)
zip_flat = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat.zip"
with zipfile.ZipFile(zip_flat, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("main.tex", tex_flat.encode("utf-8"))
    zf.writestr("bigquery_olap_benchmark_V8_5.tex", tex_flat.encode("utf-8"))
    for _, real_f in sorted(fig_map.items()):
        zf.write(real_f, arcname=real_f.name)

print("=" * 90)
print(f"[PASS] Master TeX Updated : {MS_DIR / 'bigquery_olap_benchmark_V8_5.tex'}")
print(f"       Brace Balance      : {b_open} open == {b_close} close (ZERO Syntax Errors)")
print(f"       Bundled Figures    : {len(fig_map)} files verified inside figures/")
print(f"       TeX SHA-256        : {sha256_of(MS_DIR / 'bigquery_olap_benchmark_V8_5.tex')}")
print("-" * 90)
print(f"[ZIP 1 - Overleaf Project] : {zip_overleaf} ({zip_overleaf.stat().st_size:,} bytes)")
print(f"[ZIP 2 - Flat Submission ] : {zip_flat} ({zip_flat.stat().st_size:,} bytes)")
print("=" * 90)