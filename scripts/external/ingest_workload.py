#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roadmap Modular Script: scripts/external/ingest_workload.py (V86-06)
Target Output Artifact: results/external/workload_manifest.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import importlib
mod = importlib.import_module("213_v9_roadmap_v86_00_to_v86_09_master_closure")

if __name__ == "__main__":
    summary, _ = getattr(mod, "execute_v86_06_external_extension")()
    out_p = ROOT / "results/external/workload_manifest.json"
    print(f"[V86-06] scripts/external/ingest_workload.py -> {out_p.name} (SHA-256: {mod.sha256_of_file(out_p)[:16]}...)")
