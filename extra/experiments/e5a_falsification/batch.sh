#!/usr/bin/env bash

set -euo pipefail

BACKENDS=(breach rtamt rtamt_dense pymtl taliro stlcgpp)
OPTIMIZERS=(cma anneal)
BENCHMARKS=(m2_mass_spring m1_speed m3_coupled)
# difficulty ladder for the hero (m2); must match specs.BENCHMARKS["m2_mass_spring"].thresholds
HERO=m2_mass_spring
HERO_THRESHOLDS=(0.40 0.45 0.50 0.54)

# Main matrix: every optimizer x backend x benchmark at the hero threshold.
for opt in "${OPTIMIZERS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    for b in "${BACKENDS[@]}"; do
      uv run --no-sync python run.py --backend "$b" --benchmark "$bench" --optimizer "$opt"
    done
  done
done

# Difficulty sweep on the hero benchmark.
for opt in "${OPTIMIZERS[@]}"; do
  for thr in "${HERO_THRESHOLDS[@]}"; do
    for b in "${BACKENDS[@]}"; do
      uv run --no-sync python run.py --backend "$b" --benchmark "$HERO" --optimizer "$opt" --threshold "$thr"
    done
  done
done
