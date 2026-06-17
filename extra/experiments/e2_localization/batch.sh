#!/usr/bin/env bash

set -euo pipefail

RUN=(uv run python run.py)

# All five divergence rows (Table 1), so the localizer is demonstrated on each.
# native vs breach: terminal-step (and), equality (Eq kernel), and the two
# window-kernel cases (interp_sparse, nonuniform) where rtamt/stlcgpp lack
# coverage. native vs rtamt: the boundary-window case.
for b in native breach; do
  "${RUN[@]}" --backend "$b" --spec div_terminal_and --signal div_terminal_and
  "${RUN[@]}" --backend "$b" --spec div_equality --signal div_equality
  "${RUN[@]}" --backend "$b" --spec div_interp_sparse --signal div_interp_sparse
  "${RUN[@]}" --backend "$b" --spec div_nonuniform_always --signal div_nonuniform_always
done
for b in native rtamt; do
  "${RUN[@]}" --backend "$b" --spec div_boundary_F24 --signal div_boundary_F24
done
# Tool-vs-tool divergence on a bounded until (RTAMT is the lone outlier).
for b in breach rtamt taliro stlcgpp; do
  "${RUN[@]}" --backend "$b" --spec div_until_boundary --signal div_until_boundary
done
# Tool-vs-tool divergence on a between-samples window. Only the dense-time tools
# run here: RTAMT and STLCG++ both reject the non-uniform grid (coverage finding).
for b in breach taliro; do
  "${RUN[@]}" --backend "$b" --spec div_window_sampling --signal div_window_sampling
done
