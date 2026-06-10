#!/usr/bin/env bash
set -euo pipefail

for b in native breach rtamt tidystl_simd; do
  uv run python run_backend.py --backend "$b"
done
uv run --only-group experiments python run_rtamt.py
uv run --only-group experiments python run_stlcgpp.py
