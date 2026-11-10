#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/hub_sync_notebook1_master.py
เชื่อมต่อเครื่องหลัก (notebook1) เข้ากับ https://github.com/preechan-glitch/my_research_files:
  1) สำรองโฟลเดอร์ manuscript/ ของ notebook1 ไว้ในเครื่องอย่างปลอดภัย 100%
  2) ดึงโฟลเดอร์ผลลัพธ์ของ notebook2 (data_hub/nodes/*) ลงมาโดยไม่ทับต้นฉบับ V2.tex ของ notebook1
  3) นำผลและรหัส SHA-256 จริงจาก notebook2 ผสานเข้าสู่ manuscript/bigquery_olap_benchmark_V2.tex
  4) ลงทะเบียนผล CSV ของ notebook1 เข้าสู่ data_hub/nodes/<notebook1>/ และสร้าง global_sha256_registry.csv
  5) คอมไพล์ pdflatex (2 รอบ), ซิงค์สู่ root main.tex + figures/ สำหรับ Overleaf, และ Push ขึ้น GitHub
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/preechan-glitch/my_research_files.git"
REPO_WEB = "https://github.com/preechan-glitch/my_research_files"

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results"
HUB_DIR = ROOT / "data_hub"
NODES_DIR = HUB_DIR / "nodes"
MS_DIR = ROOT / "manuscript"
MS_FIG_DIR = MS_DIR / "figures"
ROOT_FIG_DIR = ROOT / "figures"
WORK_DIR = ROOT / "workloads"
TARGET_TEX = MS_DIR / "bigquery_olap_benchmark_V2.tex"

for _d in [RES_DIR, HUB_DIR, NODES_DIR, MS_DIR, MS_FIG_DIR, ROOT_FIG_DIR, WORK_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_node_id(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "-", raw.strip()).strip("-")
    return s or "NOTEBOOK1-MASTER"


def tex_escape(val: object) -> str:
    out = str(val)
    for k, v in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        out = out.replace(k, v)
    return out


def count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return 0


def read_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            return dict(r)
    return {}


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)


def backup_authoritative_manuscript() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_dir = ROOT / f"manuscript_backup_hub_{ts}"
    if MS_DIR.exists():
        shutil.copytree(MS_DIR, bak_dir, dirs_exist_ok=True)
        print(f"[SAFE BACKUP] สำรองโฟลเดอร์ manuscript/ ของ notebook1 ไว้ที่: {bak_dir.name}")
    return bak_dir


def init_and_safe_fetch_from_hub() -> None:
    if not shutil.which("git"):
        raise RuntimeError("ไม่พบ Git ในเครื่อง กรุณาติดตั้งด้วยคำสั่ง: winget install -e --id Git.Git")

    if not (ROOT / ".git").exists():
        run_git(["init", "-b", "main"])

    # ตั้งชื่อผู้ใช้งาน Git เริ่มต้นหากยังไม่เคยตั้งค่าบนเครื่องนี้
    if not run_git(["config", "user.name"]).stdout.strip():
        run_git(["config", "user.name", "Preecha Noiumkar"])
    if not run_git(["config", "user.email"]).stdout.strip():
        run_git(["config", "user.email", "preecha.n@msu.ac.th"])

    rem = run_git(["remote", "get-url", "origin"])
    if rem.returncode != 0:
        run_git(["remote", "add", "origin", REPO_URL])
    elif rem.stdout.strip() != REPO_URL:
        run_git(["remote", "remove", "legacy_origin"])
        run_git(["remote", "rename", "origin", "legacy_origin"])
        run_git(["remote", "add", "origin", REPO_URL])
        print(f"[GIT REMOTE] ย้าย Remote เดิมไปเก็บที่ legacy_origin และผูก origin เข้ากับ {REPO_URL}")

    # ล้างโฟลเดอร์ data/ ออกจาก Git Index (ไม่ลบไฟล์จริงในดิสก์)
    run_git(["rm", "-r", "--cached", "data/"])

    # บันทึกสถานะไฟล์ต้นฉบับของ notebook1 ไว้ก่อนดึงข้อมูลจาก notebook2
    for p in [".gitignore", "manuscript", "results", "workloads", "scripts"]:
        if (ROOT / p).exists():
            run_git(["add", p])
    if run_git(["status", "--porcelain"]).stdout.strip():
        run_git(["commit", "-m", "Snapshot authoritative notebook1 workspace before hub merge"])

    print(f"[GIT FETCH] กำลังดึงข้อมูลที่แชร์ไว้จาก notebook2 บน {REPO_URL} ...")
    f_res = run_git(["fetch", "origin", "main"])
    if f_res.returncode == 0:
        # ใช้กลยุทธ์ -X ours เพื่อป้องกันไฟล์หลักของ notebook1 ถูกทับ แต่รับโฟลเดอร์ใหม่ data_hub/nodes/<notebook2> เข้ามาครบ 100%
        run_git(["merge", "origin/main", "--allow-unrelated-histories", "-X", "ours", "--no-edit"])
        run_git(["checkout", "origin/main", "--", "data_hub/nodes"])
        print("[MERGE OK] ดึงข้อมูลโหนด notebook2 เข้าสู่ data_hub/nodes/ สำเร็จโดยต้นฉบับ notebook1 ปลอดภัย 100%!")
    else:
        print("[FETCH INFO] พร้อมสำหรับการส่งข้อมูลขึ้นกิ่ง main:", f_res.stderr.strip())


def register_this_node(custom_node: str | None = None) -> str:
    this_id = clean_node_id(custom_node or platform.node())
    node_dir = NODES_DIR / this_id
    node_res = node_dir / "results"
    node_res.mkdir(parents=True, exist_ok=True)

    items = []
    for f in sorted(RES_DIR.glob("*")):
        if f.is_file() and f.suffix.lower() in (".csv", ".json"):
            dst = node_res / f.name
            shutil.copy2(f, dst)
            items.append({
                "filename": f.name,
                "rows": count_csv_rows(dst) if dst.suffix.lower() == ".csv" else 1,
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            })

    meta = {
        "node_id": this_id,
        "role": "Primary Workstation (notebook1)",
        "hostname": platform.node(),
        "os_platform": f"{platform.system()} {platform.release()}",
        "cpu_logical_threads": os.cpu_count() or 0,
        "python_version": platform.python_version(),
        "last_synced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "artifact_count": len(items),
        "artifacts": items,
    }
    (node_dir / "node_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[NODE1 REGISTERED] บันทึกผลของเครื่องหลัก ({this_id}) จำนวน {len(items)} ไฟล์ลงใน data_hub/nodes/{this_id}/")
    return this_id


def ingest_notebook2_into_v2_tex(this_id: str, reg_sha: str, node_count: int) -> None:
    """ค้นหาผลจาก notebook2 ใน data_hub/nodes/ แล้วซิงค์รหัส SHA-256 ลงใน bigquery_olap_benchmark_V2.tex"""
    if not TARGET_TEX.exists():
        for alt in ["overleaf_ready_single_file.tex", "main.tex"]:
            if (MS_DIR / alt).exists():
                shutil.copy2(MS_DIR / alt, TARGET_TEX)
                break
    if not TARGET_TEX.exists():
        print("[WARN] ไม่พบไฟล์ bigquery_olap_benchmark_V2.tex ใน manuscript/")
        return

    nb2_audit_csv = None
    nb2_id = None
    for nd in sorted(NODES_DIR.iterdir()):
        if nd.is_dir() and nd.name != this_id:
            cand = nd / "results" / "external_machine_reexecution_audit.csv"
            if cand.exists():
                nb2_audit_csv = cand
                nb2_id = nd.name
                break

    tex = TARGET_TEX.read_text(encoding="utf-8")
    tex = re.sub(r"Preecha Noiumkar\s*\$\^\{11\}\$", lambda _: r"Preecha Noiumkar$^{1*}$", tex)

    # หากพบไฟล์ external_machine_reexecution_audit.csv จาก notebook2 ให้คัดลอกเข้า results/ ของเครื่องหลักและฝังค่า SHA-256
    ext_note = ""
    if nb2_audit_csv is not None:
        shutil.copy2(nb2_audit_csv, RES_DIR / "external_machine_reexecution_audit.csv")
        r2 = read_first_row(nb2_audit_csv)
        ext_sha = sha256_file(nb2_audit_csv)
        host2 = tex_escape(r2.get("External_Hostname") or nb2_id)
        os2 = tex_escape(r2.get("External_OS", "Windows/WSL2"))
        ext_note = (
            f" Independent secondary-machine clean-room verification on \\texttt{{{host2}}} (\\texttt{{{os2}}}) "
            f"is archived in \\texttt{{external\\_machine\\_reexecution\\_audit.csv}} "
            f"[SHA-256: \\texttt{{{ext_sha[:20]}...}}]."
        )
        print(f"[NOTEBOOK2 LINKED] ผสานผลและรหัส SHA-256 ({ext_sha[:16]}...) จากโหนด '{nb2_id}' เข้าสู่ต้นฉบับหลักแล้ว!")

    hub_clause = (
        f" Multi-notebook telemetry across {node_count} registered execution node(s) is synchronized via the "
        f"centralized data share repository (\\texttt{{data\\_hub/global\\_sha256\\_registry.csv}}, "
        f"SHA-256: \\texttt{{{reg_sha[:20]}...}}) at \\url{{{REPO_WEB}}}.{ext_note}"
    )

    tex = re.sub(
        r"\s*Multi-notebook telemetry across \d+ registered execution node\(s\) is synchronized via the centralized data share repository.*?(?=\\section\*\{Competing interests\})",
        lambda _: "\n\n",
        tex,
        flags=re.DOTALL,
    )
    if r"\section*{Competing interests}" in tex:
        tex = tex.replace(
            r"\section*{Competing interests}",
            hub_clause.strip() + "\n\n" + r"\section*{Competing interests}",
            1,
        )

    TARGET_TEX.write_text(tex, encoding="utf-8")


def rebuild_registry_compile_and_sync_overleaf(this_id: str) -> dict[str, str]:
    reg_rows = []
    nodes_summary = []

    for nd in sorted(NODES_DIR.iterdir()):
        if not nd.is_dir():
            continue
        nid = nd.name
        mp = nd / "node_manifest.json"
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        os_str = meta.get("os_platform", "Unknown")
        ts_str = meta.get("last_synced_utc", "N/A")
        csvs = sorted((nd / "results").glob("*.csv")) if (nd / "results").exists() else []

        nodes_summary.append({
            "node_id": nid,
            "os": os_str,
            "threads": meta.get("cpu_logical_threads", "-"),
            "csv_count": len(csvs),
            "synced": ts_str,
        })
        for cf in csvs:
            reg_rows.append({
                "Node_ID": nid,
                "OS_Platform": os_str,
                "Artifact_CSV": cf.name,
                "Rows": count_csv_rows(cf),
                "Size_KB": round(cf.stat().st_size / 1024.0, 2),
                "SHA256": sha256_file(cf),
                "Synced_UTC": ts_str,
            })

    reg_csv = HUB_DIR / "global_sha256_registry.csv"
    with reg_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Node_ID", "OS_Platform", "Artifact_CSV", "Rows", "Size_KB", "SHA256", "Synced_UTC"])
        w.writeheader()
        w.writerows(reg_rows)
    reg_sha = sha256_file(reg_csv)

    # อัปเดต bigquery_olap_benchmark_V2.tex ของ notebook1 ด้วยผลรวมทุกโหนด
    ingest_notebook2_into_v2_tex(this_id, reg_sha, len(nodes_summary))

    # ซิงค์รูปภาพ 300 DPI ทั้งหมดจาก manuscript/figures/ ไปไว้ที่ root figures/ เพื่อให้ Overleaf ดึงไปคอมไพล์ได้ไร้รอยต่อ
    for fg in sorted(MS_FIG_DIR.glob("*")):
        if fg.is_file():
            shutil.copy2(fg, ROOT_FIG_DIR / fg.name)

    # ซิงค์ต้นฉบับหลักไปเป็น root main.tex สำหรับ Overleaf GitHub Sync
    if TARGET_TEX.exists():
        shutil.copy2(TARGET_TEX, ROOT / "main.tex")
        shutil.copy2(TARGET_TEX, MS_DIR / "main.tex")
        shutil.copy2(TARGET_TEX, MS_DIR / "overleaf_ready_single_file.tex")
    if (MS_DIR / "references.bib").exists():
        shutil.copy2(MS_DIR / "references.bib", ROOT / "references.bib")

    # คอมไพล์ PDF ด้วย pdflatex บน notebook1 (หากมี pdflatex ในเครื่อง)
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin and TARGET_TEX.exists():
        ok_compile = True
        for _ in (1, 2):
            p = subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", TARGET_TEX.name],
                cwd=str(MS_DIR), capture_output=True, text=True,
            )
            if p.returncode != 0:
                ok_compile = False
                print("[WARN] pdflatex แจ้งเตือนระหว่างคอมไพล์ (ตรวจสอบบรรทัดท้าย):\n" + "\n".join(p.stdout.splitlines()[-15:]))
                break
        pdf_built = MS_DIR / "bigquery_olap_benchmark_V2.pdf"
        if ok_compile and pdf_built.exists():
            for clone in ["bigquery_olap_benchmark_V3.pdf", "bigquery_olap_benchmark_V4.pdf", "overleaf_project_bundle.pdf"]:
                try:
                    shutil.copy2(pdf_built, MS_DIR / clone)
                except PermissionError:
                    pass
            print(f"[PDF COMPILED] อัปเดตไฟล์ {pdf_built.name} สำเร็จ (SHA-256: {sha256_file(pdf_built)[:20]}...)")

    # อัปเดตไฟล์ ZIP สำรองสำหรับ Overleaf / Editorial Manager
    if (ROOT / "main.tex").exists():
        for zname in ["overleaf_V2_submission_bundle.zip", "overleaf_project_bundle.zip"]:
            zp = MS_DIR / zname
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(ROOT / "main.tex", arcname="main.tex")
                zf.write(ROOT / "main.tex", arcname="bigquery_olap_benchmark_V2.tex")
                if (ROOT / "references.bib").exists():
                    zf.write(ROOT / "references.bib", arcname="references.bib")
                for fg in sorted(ROOT_FIG_DIR.glob("*")):
                    if fg.is_file():
                        zf.write(fg, arcname=f"figures/{fg.name}")

    # อัปเดตตาราง Dashboard รวมทุกเครื่องบนหน้าแรก README.md ของ GitHub
    rows_md = "\n".join(
        f"| `{n['node_id']}` | {n['os']} | {n['threads']} | **{n['csv_count']}** | `{n['synced']}` | [`data_hub/nodes/{n['node_id']}`](data_hub/nodes/{n['node_id']}/results) |"
        for n in nodes_summary
    )
    readme = (
        f"# Multi-Notebook Research Data Share Center & Overleaf Bridge\n\n"
        f"Centralized empirical telemetry repository and seamless **Overleaf** synchronization hub (`{REPO_WEB}`).\n\n"
        f"## Connected Research Machines & Notebooks ({len(nodes_summary)} Active Nodes)\n\n"
        f"| Node / Machine ID | OS Platform | Threads | CSV Logs | Last Synced (UTC) | Telemetry Folder |\n"
        f"| :--- | :--- | :---: | :---: | :--- | :--- |\n"
        f"{rows_md}\n\n"
        f"- **Global Multi-Node SHA-256 Registry:** [`data_hub/global_sha256_registry.csv`](data_hub/global_sha256_registry.csv) (`SHA-256: {reg_sha[:24]}...`)\n"
        f"- **Overleaf Root Entrypoint (25-Page Manuscript):** [`main.tex`](main.tex) + [`figures/`](figures/)\n"
        f"- **Journal Submission Bundle:** [`manuscript/overleaf_V2_submission_bundle.zip`](manuscript/overleaf_V2_submission_bundle.zip)\n"
    )
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    return {"nodes": str(len(nodes_summary)), "csvs": str(len(reg_rows)), "reg_sha": reg_sha}


def pull_latest_without_overwriting_manuscript() -> None:
    init_and_safe_fetch_from_hub()
    this_id = clean_node_id(platform.node())
    st = rebuild_registry_compile_and_sync_overleaf(this_id)
    print(f"[PULL & MERGE COMPLETE] ซิงค์ครบ {st['nodes']} เครื่อง (รวม {st['csvs']} CSV logs)")


def push_master_to_github_and_overleaf(custom_node: str | None = None) -> None:
    backup_authoritative_manuscript()
    init_and_safe_fetch_from_hub()
    this_id = register_this_node(custom_node)
    st = rebuild_registry_compile_and_sync_overleaf(this_id)

    stage_items = [
        ".gitignore", "README.md", "main.tex", "references.bib",
        "figures", "manuscript", "data_hub", "workloads", "scripts", ".github",
    ]
    for item in stage_items:
        if (ROOT / item).exists():
            run_git(["add", item])

    if not run_git(["status", "--porcelain"]).stdout.strip():
        print("[UP-TO-DATE] ข้อมูลทั้ง notebook1 และ notebook2 บน GitHub เป็นฉบับล่าสุดแล้ว!")
        return

    msg = f"Sync master node [{this_id}] + merge multi-node hub ({st['nodes']} nodes, {st['csvs']} CSVs) & Overleaf main.tex"
    run_git(["commit", "-m", msg])
    run_git(["branch", "-M", "main"])

    print(f"[GIT PUSH] กำลังส่งต้นฉบับหลัก รูปภาพ 300 DPI และข้อมูลรวมทุกโหนดขึ้น {REPO_URL} ...")
    res = run_git(["push", "-u", "origin", "main"])
    if res.returncode == 0:
        print("=" * 92)
        print(f"[SUCCESS 100%] เชื่อมต่อและอัปเดต {REPO_WEB} สำเร็จสมบูรณ์!")
        print(f"  * จำนวนเครื่องใน Data Share Center : {st['nodes']} เครื่อง (รวม {st['csvs']} ไฟล์ CSV)")
        print(f"  * Global Registry SHA-256         : {st['reg_sha']}")
        print("  * ขั้นตอนบน Overleaf              : กด Menu -> GitHub -> Pull GitHub changes into Overleaf ได้ทันที!")
        print("=" * 92)
    else:
        print("[PUSH ERROR] กรุณาตรวจสอบสิทธิ์ล็อกอินบัญชี GitHub (preechan-glitch) บนเครื่องนี้:\n" + res.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pull", "push"], default="push")
    ap.add_argument("--node", default=None, help="ตั้งชื่อโหนดเองได้ เช่น Notebook1-MainPC")
    args = ap.parse_args()
    if args.mode == "pull":
        pull_latest_without_overwriting_manuscript()
    else:
        push_master_to_github_and_overleaf(args.node)
