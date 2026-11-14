# AST-Route V7 External Verification Guide
1. Run `docker build -f reproducibility_v7/Dockerfile -t ast-route-v7 . && docker run --rm ast-route-v7`
2. Confirm `verify_manifest.py` and `18/18` Zero-Contradiction checks pass.
