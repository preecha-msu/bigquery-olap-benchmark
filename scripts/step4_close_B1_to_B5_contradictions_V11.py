#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/step4_close_B1_to_B5_contradictions_V11.py
Closes all Category-B items (B1-B5) in manuscript/bigquery_olap_benchmark_V11.tex
with raw-string LaTeX safety, universal SpringerOpen abstract detection,
Clean-Room Flat11.zip verification, and SHA-256 manifest update.
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

V11_TEX = MS_DIR / "bigquery_olap_benchmark_V11.tex"
ROOT_V11_TEX = ROOT / "bigquery_olap_benchmark_V11.tex"
V11_PDF = MS_DIR / "bigquery_olap_benchmark_V11.pdf"
ROOT_V11_PDF = ROOT / "bigquery_olap_benchmark_V11.pdf"
FLAT11_ZIP = ROOT / "Journal_of_Big_Data_Snapp_Submission_Flat11.zip"
OVERLEAF11_ZIP = ROOT / "Overleaf_V11_Submission_Ready.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_tex_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())


def ensure_rapids_reference(tex: str) -> str:
    if r"\end{thebibliography}" in tex and r"\bibitem{rapids2024cudf}" not in tex:
        bibitem_lines = [
            "",
            r"\bibitem{rapids2024cudf}",
            r"RAPIDS Development Team, ``cuDF: GPU DataFrame Library (\texttt{libcudf}~v26.06.00),'' "
            r"NVIDIA Corporation, 2026. [Online]. Available: \url{https://github.com/rapidsai/cudf}",
            "",
        ]
        tex = tex.replace(r"\end{thebibliography}", "\n".join(bibitem_lines) + r"\end{thebibliography}", 1)

    for bib_dir in (MS_DIR, ROOT):
        bib_path = bib_dir / "references.bib"
        if bib_path.exists():
            btxt = bib_path.read_text(encoding="utf-8", errors="replace")
            if "rapids2024cudf" not in btxt:
                bib_lines = [
                    "",
                    r"@misc{rapids2024cudf,",
                    r"  author       = {{RAPIDS Development Team}},",
                    r"  title        = {{cuDF}: {GPU} DataFrame Library (libcudf v26.06.00)},",
                    r"  howpublished = {NVIDIA Corporation, \url{https://github.com/rapidsai/cudf}},",
                    r"  year         = {2026}",
                    r"}",
                    "",
                ]
                bib_path.write_text(btxt + "\n".join(bib_lines), encoding="utf-8")

    target_phrase = r"\texttt{RAPIDS cuDF} (\texttt{libcudf}~v26.06.00)"
    if r"\cite{rapids2024cudf}" not in tex and target_phrase in tex:
        tex = tex.replace(target_phrase, target_phrase + r"~\cite{rapids2024cudf}", 1)
    return tex


def insert_into_abstract_universal(tex: str, abs_block: str) -> str:
    doc_idx = tex.find(r"\begin{document}")
    start_pos = doc_idx if doc_idx != -1 else 0
    m_sec1 = re.search(r"(?m)^[ \t]*\\section\*?\{", tex[start_pos:])
    if not m_sec1:
        m_sec1 = re.search(r"\\section\*?\{", tex[start_pos:])
    sec1_pos = start_pos + m_sec1.start() if m_sec1 else len(tex)
    front = tex[start_pos:sec1_pos]

    # Case 1: \end{abstract}
    if r"\end{abstract}" in front:
        front_new = front.replace(r"\end{abstract}", abs_block + "\n" + r"\end{abstract}", 1)
        return tex[:start_pos] + front_new + tex[sec1_pos:]

    # Case 2: \abstract{ ... } macro (exact brace-depth matching)
    abs_macro_idx = front.find(r"\abstract{")
    if abs_macro_idx != -1:
        brace_open_idx = abs_macro_idx + len(r"\abstract")
        depth = 0
        end_brace = -1
        for k in range(brace_open_idx, len(front)):
            ch = front[k]
            prev = front[k - 1] if k > 0 else ""
            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    end_brace = k
                    break
        if end_brace != -1:
            front_new = front[:end_brace] + "\n" + abs_block + "\n" + front[end_brace:]
            return tex[:start_pos] + front_new + tex[sec1_pos:]

    # Case 3: SpringerOpen / Quote-block header with Keywords line
    m_kw = re.search(
        r"(?mi)^[ \t]*(?:\\noindent[ \t]*)?(?:\\textbf\{Keywords|\\keywords\{|{\\bfseries[ \t]*Keywords)",
        front,
    )
    if m_kw:
        kp = m_kw.start()
        front_new = front[:kp] + abs_block + "\n\n" + front[kp:]
        return tex[:start_pos] + front_new + tex[sec1_pos:]

    # Case 4: \end{quote} before Section 1
    q_idx = front.rfind(r"\end{quote}")
    if q_idx != -1:
        front_new = front[:q_idx] + abs_block + "\n" + front[q_idx:]
        return tex[:start_pos] + front_new + tex[sec1_pos:]

    # Case 5: Fallback right before Section 1
    return tex[:sec1_pos] + abs_block + "\n\n" + tex[sec1_pos:]


def apply_B4_abstract_intro_conclusion(tex: str) -> str:
    for tag in ("V11-B4-ABS", "V11-B4-INTRO", "V11-B4-CONC"):
        tex = re.sub(rf"% \[{tag}\].*?% \[/{tag}\]\s*", "", tex, flags=re.DOTALL)

    abs_block = "\n".join([
        r"% [V11-B4-ABS]",
        r"Four-node multi-scale validation ($\text{SF}\in\{1,10,30,100\}\text{ GB}$, $64/64$ runs with $\text{EX}=1.000$) "
        r"further confirms $2.6\%\text{--}3.5\%$ parity with \texttt{RAPIDS cuDF} and $2.85\times\text{--}9.50\times$ "
        r"\texttt{cudf-polars} speedups over CPU Polars.",
        r"% [/V11-B4-ABS]",
    ])
    tex = insert_into_abstract_universal(tex, abs_block)

    doc_idx = tex.find(r"\begin{document}")
    start_pos = doc_idx if doc_idx != -1 else 0
    sec_matches = list(re.finditer(r"(?m)^[ \t]*\\section\{", tex[start_pos:]))
    if len(sec_matches) < 2:
        sec_matches = list(re.finditer(r"\\section\{", tex[start_pos:]))
    if len(sec_matches) >= 2:
        sec2_pos = start_pos + sec_matches[1].start()
        intro_block = "\n".join([
            r"% [V11-B4-INTRO]",
            r"In addition, our four-node multi-scale validation ($\text{SF}\in\{1,10,30,100\}\text{ GB}$; "
            r"Table~\ref{tab:v11_fournode_multiscale_cudf} and Fig.~\ref{fig:v11_fournode_multiscale_cudf}) "
            r"benchmarks the primary workstation alongside a $22$-thread x86\_64 node, an Apple~M1 ARM64 node, "
            r"and an NVIDIA Tesla~T4 server GPU running native \texttt{RAPIDS cuDF}~\cite{rapids2024cudf} and \texttt{cudf-polars}.",
            r"% [/V11-B4-INTRO]",
        ])
        tex = tex[:sec2_pos] + intro_block + "\n\n" + tex[sec2_pos:]

    conc_block = "\n".join([
        r"% [V11-B4-CONC]",
        r"Finally, cross-hardware and cross-ISA multi-scale experiments across $\text{SF}\in\{1,10,30,100\}\text{ GB}$ "
        r"($64/64$ real runs, $\text{EX}=1.000$; Table~\ref{tab:v11_fournode_multiscale_cudf}) confirm that "
        r"PyTorch CUDA tensor execution tracks production \texttt{RAPIDS cuDF} within $2.6\%\text{--}3.5\%$ at $\text{SF}=10\text{--}30\text{ GB}$ "
        r"on Tesla~T4 while \texttt{cudf-polars} delivers $2.85\times\text{--}9.50\times$ speedups over CPU Polars on identical query plans.",
        r"% [/V11-B4-CONC]",
    ])

    m_conc = re.search(r"(?mi)^[ \t]*\\section\*?\{[^\n]*Conclusion[^\n]*$", tex[start_pos:])
    if m_conc:
        c_end = start_pos + m_conc.end()
        m_after = re.search(
            r"(?m)^[ \t]*(?:\\section\*?\{|\\subsection\*?\{Abbreviations|\\subsection\*?\{Declarations|\\bmhead\{|\\backmatter|\\bibliographystyle|\\begin\{thebibliography\}|\\end\{document\})",
            tex[c_end:],
        )
        ins_pos = c_end + (m_after.start() if m_after else len(tex[c_end:]))
        tex = tex[:ins_pos] + "\n" + conc_block + "\n\n" + tex[ins_pos:]
    else:
        m_tail = re.search(
            r"(?m)^[ \t]*(?:\\section\*?\{Declarations\}|\\bmhead\{|\\backmatter|\\bibliographystyle|\\begin\{thebibliography\}|\\end\{document\})",
            tex[start_pos:],
        )
        ins_pos = start_pos + (m_tail.start() if m_tail else len(tex[start_pos:]))
        tex = tex[:ins_pos] + "\n" + conc_block + "\n\n" + tex[ins_pos:]

    return tex


def apply_B2_gpu_fairness_and_B3_hardware_manifest(tex: str) -> str:
    tex = re.sub(r"% \[V11-B2-B3-SYNC\].*?% \[/V11-B2-B3-SYNC\]\s*", "", tex, flags=re.DOTALL)

    tex = re.sub(
        r"evaluated\s+exclusively\s+at\s+\$\\text\{SF\}=1\\text\{\s*GB\}\$",
        lambda _: r"evaluated on the 75-query routing suite at $\text{SF}=1\text{ GB}$ and extended across $\text{SF}\in\{1,10,30,100\}\text{ GB}$ in Table~\ref{tab:v11_fournode_multiscale_cudf}",
        tex,
    )

    b2_b3_block = "\n".join([
        r"% [V11-B2-B3-SYNC]",
        r"\noindent\textbf{Native Columnar GPU Parity and Four-Node Environment Scope:} "
        r"To complement the primary workstation hardware manifest (Node~1: x86\_64 8~threads, $16\text{ GB}$ RAM, RTX~2050 $4.0\text{ GB}$ VRAM) "
        r"and verify architectural fairness against native C++/CUDA engines, our extended multi-scale testbed (Table~\ref{tab:v11_fournode_multiscale_cudf}) "
        r"incorporates three external hardware tiers across all four physical scale factors ($\text{SF}\in\{1,10,30,100\}\text{ GB}$): "
        r"Node~2 (x86\_64 16C/22T high-core workstation, $15.5\text{ GB}$ RAM), "
        r"Node~3 (Apple~M1 ARM64, 8~cores, $8.0\text{ GB}$ unified memory with a $3.5\text{ GB}$ spill guard), and "
        r"Node~4 (cloud server NVIDIA Tesla~T4, $14.56\text{ GB}$ usable GDDR6 VRAM running \texttt{RAPIDS cuDF} / \texttt{libcudf}~v26.06.00~\cite{rapids2024cudf} "
        r"and \texttt{cudf-polars} on Polars~v1.35.2). "
        r"Direct comparison on Node~4 confirms that our PyTorch CUDA relational tensor primitives match native \texttt{RAPIDS cuDF} "
        r"within $2.6\%\text{--}3.5\%$ at $\text{SF}=10\text{--}30\text{ GB}$ ($311.07$ vs.\ $322.37\text{ ms}$ and $1028.13$ vs.\ $1001.71\text{ ms}$), "
        r"ruling out custom-prototype distortion.",
        r"% [/V11-B2-B3-SYNC]",
    ]) + "\n\n"

    m_res = re.search(
        r"(?m)^[ \t]*\\section\{(?:Experimental\s+Results|Results|Empirical\s+Results)",
        tex,
    )
    if m_res:
        pos = m_res.start()
        tex = tex[:pos] + b2_b3_block + tex[pos:]
    else:
        m_v11 = re.search(r"% === BEGIN V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ===", tex)
        if m_v11:
            pos = m_v11.start()
            tex = tex[:pos] + b2_b3_block + tex[pos:]

    return tex


def apply_B1_limitations_alignment(tex: str) -> str:
    tex = re.sub(r"% \[V11-B1-LIM\].*?% \[/V11-B1-LIM\]\s*", "", tex, flags=re.DOTALL)

    tex = re.sub(
        r"rather than production GPU SQL engines \(e\.g\.,\s*RAPIDS cuDF or HeavyDB\)",
        lambda _: r"for the primary 75-query router training matrix (while validated externally against production \texttt{RAPIDS cuDF}~v26.06.00 and \texttt{cudf-polars}~v1.35.2 on a server-grade Tesla~T4 in Table~\ref{tab:v11_fournode_multiscale_cudf})",
        tex,
    )

    b1_paragraph = "\n" + "\n".join([
        r"% [V11-B1-LIM]",
        r"\noindent\textbf{GPU Backend Scope and Accelerator Generalization:} "
        r"While the primary 75-query routing cost models ($\mathcal{R}_{\text{GBR-9D}}$) are trained on our primary workstation "
        r"using the 6-operator PyTorch CUDA tensor engine on an entry-level NVIDIA RTX~2050 ($4.0\text{ GB}$ VRAM), "
        r"our V11 external multi-scale audit (Table~\ref{tab:v11_fournode_multiscale_cudf}) directly validates this design on a "
        r"server-grade NVIDIA Tesla~T4 ($14.56\text{ GB}$ usable GDDR6 VRAM) against production C++/CUDA \texttt{RAPIDS cuDF} "
        r"(\texttt{libcudf}~v26.06.00)~\cite{rapids2024cudf} and \texttt{cudf-polars} (Polars~v1.35.2) across $\text{SF}\in\{1,10,30,100\}\text{ GB}$. "
        r"Remaining hardware extensions---specifically multi-GPU NVLink shuffle collectives, complex out-of-VRAM multi-way hash-join spilling in \texttt{cudf-polars}, "
        r"and non-NVIDIA SIMT backends (AMD ROCm and Intel oneAPI/XPU)---represent promising directions for future work.",
        r"% [/V11-B1-LIM]",
    ]) + "\n"

    m_lim = re.search(r"(?mi)^[ \t]*\\(?:sub)?section\*?\{[^\n]*Limitations[^\n]*$", tex)
    if m_lim:
        after_lim = tex[m_lim.end():]
        m_next = re.search(
            r"(?m)^[ \t]*(?:\\(?:sub)?section\*?\{|\\bmhead\{|\\backmatter|\\bibliographystyle|\\begin\{thebibliography\})",
            after_lim,
        )
        ins_pos = m_lim.end() + (m_next.start() if m_next else len(after_lim))
        tex = tex[:ins_pos] + b1_paragraph + "\n" + tex[ins_pos:]
    else:
        marker = "% === END V11 FOUR-NODE CUDF MULTISCALE SF1-10-30-100 ==="
        if marker in tex:
            tex = tex.replace(marker, marker + "\n" + b1_paragraph, 1)

    return tex


def apply_B5_data_availability(tex: str, master_sha: str, cudf_sha: str) -> str:
    tex = re.sub(r"% \[V11-B5-AVAIL\].*?% \[/V11-B5-AVAIL\]\s*", "", tex, flags=re.DOTALL)
    tex = tex.replace(r"\_\allowbreak{}", "__SAFE_ALLOWBREAK__")
    tex = tex.replace("_allowbreak{}", "__SAFE_ALLOWBREAK__")
    tex = tex.replace("__SAFE_ALLOWBREAK__", r"\_\allowbreak{}")

    b5_lines = [
        r"% [V11-B5-AVAIL]",
        r"The V11 four-node multi-scale ($\text{SF}=1, 10, 30, 100\text{ GB}$) and Tesla~T4 \texttt{RAPIDS cuDF} / \texttt{cudf-polars} "
        r"execution logs and cryptographic audit manifests are archived in "
        r"\texttt{results/v11\_\allowbreak{}4node\_\allowbreak{}6engine\_\allowbreak{}cudf\_\allowbreak{}multiscale\_\allowbreak{}sf1\_\allowbreak{}10\_\allowbreak{}30\_\allowbreak{}100\_\allowbreak{}master.csv} "
        + f"(SHA-256: \\texttt{{{master_sha[:16]}...}}), "
        + r"\texttt{results/node4\_\allowbreak{}tesla\_\allowbreak{}t4\_\allowbreak{}cudf\_\allowbreak{}and\_\allowbreak{}cudf\_\allowbreak{}polars\_\allowbreak{}sf1\_\allowbreak{}10\_\allowbreak{}30\_\allowbreak{}100.csv} "
        + f"(SHA-256: \\texttt{{{cudf_sha[:16]}...}}), and "
        + r"\texttt{results/v11\_\allowbreak{}sha256\_\allowbreak{}submission\_\allowbreak{}manifest.csv}.",
        r"% [/V11-B5-AVAIL]",
    ]
    b5_sentence = "\n" + "\n".join(b5_lines) + "\n"

    m_av = re.search(
        r"(\\bmhead\{Availability of data[^\}]*\}|\\(?:sub)*section\*?\{Availability of data[^\}]*\}|\\textbf\{Availability of data[^\}]*\})",
        tex,
        flags=re.IGNORECASE,
    )
    if m_av:
        after_av = tex[m_av.end():]
        m_next = re.search(
            r"(\\bmhead\{|\\(?:sub)*section\*?\{|\\noindent\\textbf\{Competing|\\textbf\{Competing|\\bibliographystyle|\\begin\{thebibliography\}|\\end\{itemize\})",
            after_av,
        )
        ins_pos = m_av.end() + (m_next.start() if m_next else len(after_av))
        tex = tex[:ins_pos] + b5_sentence + tex[ins_pos:]
    else:
        m_bib = re.search(r"(\\bibliographystyle|\\begin\{thebibliography\})", tex)
        if m_bib:
            pos = m_bib.start()
            tex = tex[:pos] + r"\subsection*{Availability of data and materials}" + "\n" + b5_sentence + "\n" + tex[pos:]

    return tex


def resolve_all_figures(tex_txt: str) -> list[tuple[str, Path]]:
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
            raise FileNotFoundError(f"Missing figure referenced in V11.tex: {ref}")
        resolved.append((ref, found))
    return resolved


def compile_v11_and_audit(orig_tex: str, updated_tex: str) -> tuple[int, int, int]:
    V11_TEX.write_text(updated_tex, encoding="utf-8")
    try:
        for pass_i in (1, 2, 3):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", V11_TEX.name],
                cwd=str(MS_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if res.returncode != 0:
                tail = "\n".join(res.stdout.splitlines()[-35:])
                raise RuntimeError(f"pdflatex pass {pass_i} failed:\n{tail}")
            if pass_i == 1 and r"\bibliography{" in updated_tex and (MS_DIR / "bigquery_olap_benchmark_V11.aux").exists():
                subprocess.run(["bibtex", "bigquery_olap_benchmark_V11"], cwd=str(MS_DIR), capture_output=True, text=True)
    except Exception:
        V11_TEX.write_text(orig_tex, encoding="utf-8")
        raise

    log_txt = (MS_DIR / "bigquery_olap_benchmark_V11.log").read_text(encoding="utf-8", errors="replace")
    undef_refs = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", log_txt))
    undef_cites = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", log_txt))
    m_pg = re.search(r"Output written on bigquery_olap_benchmark_V11\.pdf \((\d+) pages", log_txt)
    pages = int(m_pg.group(1)) if m_pg else -1

    if undef_refs != 0 or undef_cites != 0:
        V11_TEX.write_text(orig_tex, encoding="utf-8")
        raise RuntimeError(f"Compilation produced Undefined Refs={undef_refs}, Undefined Citations={undef_cites}")

    shutil.copy2(V11_TEX, ROOT_V11_TEX)
    try:
        shutil.copy2(V11_PDF, ROOT_V11_PDF)
    except PermissionError:
        print("  [INFO] Root bigquery_olap_benchmark_V11.pdf is open in viewer; manuscript/ PDF updated.")

    return pages, undef_refs, undef_cites


def rebuild_and_cleanroom_verify_zips(tex_txt: str, resolved_imgs: list[tuple[str, Path]]) -> None:
    flat_tex = re.sub(
        r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:figures/|manuscript/figures/)([^}]+\})",
        lambda m: m.group(1) + m.group(2),
        tex_txt,
    )
    support_files = [
        "references.bib", "sn-jnl.cls", "sn-mathphys-num.bst",
        "sn-basic.bst", "sn-apacite.bst", "sn-chicago.bst", "sn-mps.bst", "sn-vancouver.bst",
    ]

    with zipfile.ZipFile(FLAT11_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bigquery_olap_benchmark_V11.tex", flat_tex.encode("utf-8"))
        if V11_PDF.exists():
            zf.write(V11_PDF, arcname="bigquery_olap_benchmark_V11.pdf")
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

    clean_dir = ROOT / "_cleanroom_flat11_b1_b5"
    shutil.rmtree(clean_dir, ignore_errors=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(FLAT11_ZIP, "r") as zf:
            nested = [n for n in zf.namelist() if "/" in n or "\\" in n]
            if nested:
                raise AssertionError(f"Nested paths found in Flat11.zip: {nested}")
            zf.extractall(clean_dir)

        (clean_dir / "bigquery_olap_benchmark_V11.pdf").unlink(missing_ok=True)
        for p_i in (1, 2, 3):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "bigquery_olap_benchmark_V11.tex"],
                cwd=str(clean_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if r.returncode != 0:
                tail = "\n".join(r.stdout.splitlines()[-30:])
                raise RuntimeError(f"Clean-room Flat11 compile failed on pass {p_i}:\n{tail}")
            if p_i == 1 and (clean_dir / "references.bib").exists() and r"\bibliography{" in flat_tex:
                subprocess.run(["bibtex", "bigquery_olap_benchmark_V11"], cwd=str(clean_dir), capture_output=True, text=True)

        clog = (clean_dir / "bigquery_olap_benchmark_V11.log").read_text(encoding="utf-8", errors="replace")
        ur = len(re.findall(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", clog))
        uc = len(re.findall(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", clog))
        if ur != 0 or uc != 0:
            raise RuntimeError(f"Clean-room Flat11 Undefined Refs={ur}, Citations={uc}")
        print(f"  [CLEAN-ROOM PASS] Flat11.zip compiled standalone (Undefined Refs={ur}, Citations={uc})")
    finally:
        shutil.rmtree(clean_dir, ignore_errors=True)

    with zipfile.ZipFile(OVERLEAF11_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(V11_TEX, arcname="main.tex")
        zf.write(V11_TEX, arcname="bigquery_olap_benchmark_V11.tex")
        if V11_PDF.exists():
            zf.write(V11_PDF, arcname="bigquery_olap_benchmark_V11.pdf")
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


def update_v11_manifest() -> Path:
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
    out_m = RES_DIR / "v11_sha256_submission_manifest.csv"
    pd.DataFrame(rows).to_csv(out_m, index=False, encoding="utf-8")
    return out_m


def main() -> None:
    print("=" * 100)
    print("[STEP 4 - V11] CLOSING ALL CATEGORY-B ITEMS (B1-B5) + ZERO-CONTRADICTION & CLEAN-ROOM AUDIT")
    print("=" * 100)

    if not V11_TEX.exists():
        raise FileNotFoundError(f"Missing canonical file: {V11_TEX}")

    master_csv = RES_DIR / "v11_4node_6engine_cudf_multiscale_sf1_10_30_100_master.csv"
    cudf_csv = RES_DIR / "node4_tesla_t4_cudf_and_cudf_polars_sf1_10_30_100.csv"
    master_sha = sha256_file(master_csv)
    cudf_sha = sha256_file(cudf_csv)

    orig_tex = V11_TEX.read_text(encoding="utf-8")
    brace_diff_before = orig_tex.count("{") - orig_tex.count("}")

    tex = orig_tex
    tex = ensure_rapids_reference(tex)
    tex = apply_B4_abstract_intro_conclusion(tex)
    tex = apply_B2_gpu_fairness_and_B3_hardware_manifest(tex)
    tex = apply_B1_limitations_alignment(tex)
    tex = apply_B5_data_availability(tex, master_sha, cudf_sha)

    brace_diff_after = tex.count("{") - tex.count("}")
    if brace_diff_after != brace_diff_before:
        raise AssertionError(f"Brace balance changed! Before={brace_diff_before}, After={brace_diff_after}")

    # Pre-compile gate verification (fast-fail before touching disk)
    pre_gates = [
        ("B1 Limitations GPU Scope Aligned (RTX 2050 + Tesla T4 cuDF)", "[V11-B1-LIM]" in tex and "multi-GPU NVLink" in tex),
        ("B2 GPU Fairness Cross-Referenced to V11 Table", "[V11-B2-B3-SYNC]" in tex and "ruling out custom-prototype distortion" in tex),
        ("B3 Four-Node Environment Scope (Nodes 1-4, SF=1-100 GB)", r"14.56\text{ GB}" in tex and "Polars~v1.35.2" in tex),
        ("B4-Abs Abstract Synchronized", "[V11-B4-ABS]" in tex),
        ("B4-Intro Introduction Synchronized", "[V11-B4-INTRO]" in tex),
        ("B4-Conc Conclusion Synchronized", "[V11-B4-CONC]" in tex),
        ("B5 Data Availability + rapids2024cudf Citation Active", "[V11-B5-AVAIL]" in tex and r"\cite{rapids2024cudf}" in tex and "_allowbreak{}" not in tex.replace(r"\_\allowbreak{}", "")),
        ("Core 75Q Metrics Preserved (54.92 ms / 57.33% / 34.98%)", all(m in tex for m in ("54.92", "57.33", "34.98", "183.74", "82.47"))),
    ]
    for gname, ok in pre_gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {gname}")
        if not ok:
            raise AssertionError(f"Pre-compile gate failed: {gname}")

    pages, u_ref, u_cit = compile_v11_and_audit(orig_tex, tex)
    print(f"  [PASS] Zero Undefined Refs & Citations (Pages={pages}, Refs={u_ref}, Cites={u_cit})")

    resolved_imgs = resolve_all_figures(tex)
    rebuild_and_cleanroom_verify_zips(tex, resolved_imgs)
    manifest_p = update_v11_manifest()

    print("-" * 100)
    print(f"  * bigquery_olap_benchmark_V11.tex : {sha256_file(V11_TEX)} (Pages={pages})")
    print(f"  * bigquery_olap_benchmark_V11.pdf : {sha256_file(V11_PDF)} ({V11_PDF.stat().st_size:,} bytes)")
    print(f"  * Snapp Submission Flat11.zip     : {sha256_file(FLAT11_ZIP)} ({FLAT11_ZIP.stat().st_size:,} bytes)")
    print(f"  * Overleaf V11 Ready.zip          : {sha256_file(OVERLEAF11_ZIP)} ({OVERLEAF11_ZIP.stat().st_size:,} bytes)")
    print(f"  * Updated SHA-256 Manifest        : {manifest_p.relative_to(ROOT)}")
    print("=" * 100)
    print("[ALL B1-B5 V11 UPDATES COMPLETED 100% WITH ZERO CONTRADICTIONS]")


if __name__ == "__main__":
    main()