#!/usr/bin/env bash
set -euo pipefail

for v in breach rtamt stlcgpp stlcgpp_torch tidystl_simd full_suite; do
  uv run python run.py --variant "$v"
done
