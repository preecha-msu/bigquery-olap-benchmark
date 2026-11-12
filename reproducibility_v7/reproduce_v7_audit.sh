#!/usr/bin/env bash
set -euo pipefail
python3 reproducibility_v7/verify_manifest.py
python3 scripts/verify_zero_contradiction_step_by_step.py
echo "[PASS] V7 Clean-Room Reproduction Verified!"
