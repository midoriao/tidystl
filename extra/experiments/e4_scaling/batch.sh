#!/usr/bin/env bash
set -euo pipefail

# Sweep every tidystl backend the in-process runner can time. stlcgpp_torch is
# omitted: it requires a TorchSignal input (run_backend.py feeds a plain Signal).
# breach-real (run_breach.py) is omitted: MATLAB-only; run it separately when a
# Breach/MATLAB install is available.
for b in native breach rtamt rtamt_dense pymtl stlcgpp taliro tidystl_simd; do
  uv run python run_backend.py --backend "$b"
done
uv run --only-group experiments python run_rtamt.py
uv run --only-group experiments python run_pymtl.py
uv run --only-group experiments python run_stlcgpp.py
