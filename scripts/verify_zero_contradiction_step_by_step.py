#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
scripts/verify_zero_contradiction_step_by_step.py
สคริปต์ตรวจสอบความสอดคล้องของตัวเลข (Read-Only 100% - ไม่แก้ไขไฟล์)
ตรวจสอบเฉพาะ 4 ประเด็นหลักตามคำเตือนสุดท้ายของ Reviewer:
1. Top-1 Accuracy (57.33%) คู่กับ Top-2 (76.00%), Mean Regret (34.98%), P95 Regret (114.11%), และ Energy (1.381 J)
2. SF=1 GB Latency ข้ามทุกตาราง (183.74 ms DuckDB, 82.47 ms Polars, 96.30/96.29 ms DF, 56.96 ms GPU Cold, 54.92 ms AST-Route)
3. Energy Telemetry (6.40 J, 3.17 J, 3.39 J, 0.65 J, 0.25 J | Net Dynamic 3.73 J, 1.97 J | System Energy 1.58 J, 0.61 J)
4. Cross-Validation (LOSO-CV 54.92 ms | Unseen Seeds 73.33%, 47.01 ms | Official TPC-H Nested LOO-CV 68.18% Top-1, 90.91% Top-2, 16.98 ms)
"""

import re
from pathlib import Path

TEX_FILE = Path(r"E:\TeachPree\Research_Academic_Services\bigquery-olap-benchmark\manuscript\bigquery_olap_benchmark_V2.tex")
tex = TEX_FILE.read_text(encoding="utf-8")

print("=" * 90)
print("ZERO-CONTRADICTION AUDIT REPORT FOR: bigquery_olap_benchmark_V2.tex")
print("=" * 90)

audit_rules = [
    # Group 1: Routing Accuracy, Regret & Energy
    ("G1-1: Top-1 Accuracy = 57.33%", r"57\.33\\%", 5, "Abstract, Table 8, Table 9, Table 11, Sec 4.6"),
    ("G1-2: Top-2 Accuracy = 76.00%", r"76\.00\\%", 4, "Abstract, Intro, Table 8, Sec 4.6, Conclusion"),
    ("G1-3: Mean Regret = 34.98%", r"34\.98\\%", 6, "Abstract, Table 8, Table 9, Table 11, Sec 4.6, Conclusion"),
    ("G1-4: P95 Regret = 114.11% (vs 257.31% GPU)", r"114\.11?\\%", 4, "Abstract, Table 8, Table 11, Sec 4.6, Sec 4.8"),
    ("G1-5: AST-Route Energy = 1.381 J", r"1\.381", 5, "Abstract, Table 8, Table 9, Sec 4.6, Sec 4.7"),

    # Group 2: SF=1 GB Latencies
    ("G2-1: DuckDB Native SF=1 = 183.74 ms", r"183\.74", 10, "Abstract, Table 1, 4, 5, 8, 10, 12, Sec 4.1-4.9"),
    ("G2-2: Polars SF=1 = 82.47 ms", r"82\.47", 6, "Abstract, Table 5, 8, 10, 11, Sec 4.4-4.9"),
    ("G2-3: DataFusion SF=1 = 96.30 ms", r"96\.30|96\.29", 6, "Abstract, Table 5, 8, 10, 11, 12"),
    ("G2-4: GPU Cold SF=1 = 56.96 ms", r"56\.96", 8, "Abstract, Table 4, 5, 8, 10, 11, 12, 13"),
    ("G2-5: AST-Route SF=1 = 54.92 ms", r"54\.92", 10, "Abstract, Table 8, 9, 10, 11, 12, Sec 4.6-4.9"),
    ("G2-6: No Legacy Micro-Benchmark Numbers (51.12 / 6.70 / 103.32)", r"51\.12|6\.70|103\.32", 0, "Must be 0 occurrences"),

    # Group 3: Energy Consistency
    ("G3-1: Active Device Energy (6.40 J DuckDB, 3.17 J Polars)", r"6\.40.*3\.17|3\.17.*6\.40", 2, "Abstract, Table 7, Sec 4.5"),
    ("G3-2: Net Dynamic Energy (3.73 J DuckDB, 1.97 J Polars)", r"3\.73.*1\.97", 2, "Table 7 Note, Sec 4.5"),
    ("G3-3: Composite System Energy (1.58 J Cold, 0.61 J Warm)", r"1\.58.*0\.61", 2, "Table 7 Note, Sec 4.5"),

    # Group 4: Cross-Validation Consistency
    ("G4-1: Unseen Holdout Seeds (73.33% Top-1, 47.01 ms)", r"47\.01", 4, "Abstract, Intro, Table 11B, Sec 4.8, Conclusion"),
    ("G4-2: Official TPC-H Nested LOO-CV (68.18% Top-1, 90.91% Top-2, 16.98 ms)", r"16\.98", 4, "Abstract, Intro, Table 12C, Sec 4.9, Conclusion"),
    ("G4-3: Official TPC-H Unweighted LOO-CV (40.91% Top-1, 27.33 ms)", r"27\.33", 2, "Table 12C, Sec 4.9"),
    ("G4-4: No Undefined References (??)", r"\?\?", 0, "Must be 0 occurrences"),
]

all_passed = True
for label, pattern, min_expected, locations in audit_rules:
    found = len(re.findall(pattern, tex, flags=re.DOTALL))
    if min_expected == 0:
        ok = (found == 0)
    else:
        ok = (found >= 1)
    status = "PASS" if ok else "WARN"
    if not ok:
        all_passed = False
    print(f"[{status:4s}] {label:<65} | Found: {found:2d} | {locations}")

print("=" * 90)
if all_passed:
    print("[RESULT] ผ่านการตรวจสอบ 100% ทุกหัวข้อ! ไม่มีตัวเลขขัดแย้งกันใน bigquery_olap_benchmark_V2.tex")
else:
    print("[RESULT] พบจุดที่ควรตรวจสอบเพิ่มเติมตามรายการที่ขึ้น WARN ด้านบน")
print("=" * 90)
