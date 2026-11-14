#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roadmap Modular Script: scripts/faults/collect_recovery_events.py (V86-08)
Target Output Artifact: results/faults/recovery_events.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import importlib
mod = importlib.import_module("213_v9_roadmap_v86_00_to_v86_09_master_closure")

if __name__ == "__main__":
    summary, _ = getattr(mod, "execute_v86_08_fault_injection")()
    out_p = ROOT / "results/faults/recovery_events.csv"
    print(f"[V86-08] scripts/faults/collect_recovery_events.py -> {out_p.name} (SHA-256: {mod.sha256_of_file(out_p)[:16]}...)")
