#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roadmap Modular Script: scripts/fairness/generate_boundary_report.py (V86-01)
Target Output Artifact: results/fairness/boundary_report.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import importlib
mod = importlib.import_module("213_v9_roadmap_v86_00_to_v86_09_master_closure")

if __name__ == "__main__":
    summary, _ = getattr(mod, "execute_v86_01_fairness_audit")()
    out_p = ROOT / "results/fairness/boundary_report.csv"
    print(f"[V86-01] scripts/fairness/generate_boundary_report.py -> {out_p.name} (SHA-256: {mod.sha256_of_file(out_p)[:16]}...)")
