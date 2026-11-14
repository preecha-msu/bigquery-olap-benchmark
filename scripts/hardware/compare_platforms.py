#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roadmap Modular Script: scripts/hardware/compare_platforms.py (V86-07)
Target Output Artifact: results/hardware/portability_summary.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import importlib
mod = importlib.import_module("213_v9_roadmap_v86_00_to_v86_09_master_closure")

if __name__ == "__main__":
    summary, _ = getattr(mod, "execute_v86_07_hardware_portability")()
    out_p = ROOT / "results/hardware/portability_summary.csv"
    print(f"[V86-07] scripts/hardware/compare_platforms.py -> {out_p.name} (SHA-256: {mod.sha256_of_file(out_p)[:16]}...)")
