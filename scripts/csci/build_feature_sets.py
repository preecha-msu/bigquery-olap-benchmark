#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roadmap Modular Script: scripts/csci/build_feature_sets.py (V86-03)
Target Output Artifact: results/csci/features_M1_M4.parquet
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import importlib
mod = importlib.import_module("213_v9_roadmap_v86_00_to_v86_09_master_closure")

if __name__ == "__main__":
    summary, _ = getattr(mod, "execute_v86_03_csci_evaluation")()
    out_p = ROOT / "results/csci/features_M1_M4.parquet"
    print(f"[V86-03] scripts/csci/build_feature_sets.py -> {out_p.name} (SHA-256: {mod.sha256_of_file(out_p)[:16]}...)")
