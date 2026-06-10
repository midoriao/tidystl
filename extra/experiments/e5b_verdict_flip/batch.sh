#!/usr/bin/env bash

set -euo pipefail
RUN=(uv run python run.py)
# Sweep resolution. Verdicts are deterministic, so N only sets how
# precisely the flip/inversion boundary is located: the reported rates
# are (points in the flip region) / N. N=41 (step 0.25 over [-5,5]) gives
# ~1-decimal rates at ~160 runs; raise it for finer boundaries.
N="${1:-41}"
SWEEP=$(uv run python -c "import numpy; print(' '.join(repr(float(v)) for v in numpy.linspace(-5, 5, $N)))")

# boundary-rule family: F[2,4](x >= 0), clamping (native) vs pessimistic (rtamt)
for b in native rtamt; do
  for s in $SWEEP; do
    "${RUN[@]}" --backend "$b" --signal.s "$s"
  done
done

# terminal-step family: (x >= 0) and (y >= 0), as-is (native) vs extend-penultimate (breach)
for b in native breach; do
  for s in $SWEEP; do
    "${RUN[@]}" --backend "$b" --spec "(x >= 0) and (y >= 0)" \
      --signal.names x y --signal.base-values 5.0 5.0 \
      --signal.s "$s" --verdict-t-index 2
  done
  # ranking reference signal (interior violation, visible under both rules)
  "${RUN[@]}" --backend "$b" --spec "(x >= 0) and (y >= 0)" \
    --signal.names x y --signal.base-values 5.0 -1.0 \
    --signal.s 5.0 --verdict-t-index 2
done
