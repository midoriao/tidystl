#!/usr/bin/env bash

set -euo pipefail

RUN=(uv run python run.py)

for b in native breach; do
  "${RUN[@]}" --backend "$b" --spec div_terminal_and --signal div_terminal_and
done
for b in native rtamt; do
  "${RUN[@]}" --backend "$b" --spec div_boundary_F24 --signal div_boundary_F24
done
