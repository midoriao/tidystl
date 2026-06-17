#!/usr/bin/env bash
set -euo pipefail

for v in breach rtamt rtamt_dense pymtl taliro stlcgpp stlcgpp_torch tidystl_simd full_suite; do
  uv run python run.py --variant "$v"
done
