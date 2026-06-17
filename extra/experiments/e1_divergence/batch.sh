#!/usr/bin/env bash

set -euo pipefail

REG=../registry
# Honor the Makefile's OUTPUTS_ROOT (../../outputs, i.e. extra/outputs) so that
# `make run` writes columns where `make report`/aggregate.py reads them; the
# default keeps direct `./batch.sh` invocation (from this dir) working too.
OUT="${OUTPUTS_ROOT:-../../outputs}/e1_divergence/columns"

echo $UV_PROJECT_ENVIRONMENT

for b in native breach rtamt stlcgpp; do
  for blk in baseline div; do
    uv run --only-group experiments \
      python ../tools/eval_tidystl.py --backend "$b" \
      --specs "$REG/specs_$blk.json" --signals "$REG/signals_$blk.json" \
      --output "$OUT/backend_${b}_${blk}.json"
  done
done

for t in rtamt stlcgpp; do
  for blk in baseline div; do
    uv run --only-group experiments \
      python ../tools/eval_external_tools.py --tool "$t" \
      --specs "$REG/specs_$blk.json" --signals "$REG/signals_$blk.json" \
      --output "$OUT/tool_${t}_${blk}.json"
  done
done

# rtamt_dense + py-mtl share rtamt's antlr4 pin, which needs Python 3.12
# (typing.io was removed in 3.13; see other_tools/rtamt/generate_ground_truth.py).
for t in rtamt_dense pymtl; do
  for blk in baseline div; do
    uv run --python 3.12 --only-group experiments \
      python ../tools/eval_external_tools.py --tool "$t" \
      --specs "$REG/specs_$blk.json" --signals "$REG/signals_$blk.json" \
      --output "$OUT/tool_${t}_${blk}.json"
  done
done
