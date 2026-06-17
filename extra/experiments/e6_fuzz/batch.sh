#!/usr/bin/env bash
set -euo pipefail

# Three seeds pooled by aggregate.py for a larger sample.
for seed in 0 1 2; do
  uv run python run.py --n-pairs 500 --seed "$seed"
done
