# E1: Cross-Tool Divergence Matrix

## Purpose

Show where real STL tools agree and diverge on documented decision points, and verify that each tidystl backend reproduces its corresponding real tool.

## Batch and Aggregate

No runners: batch.sh produces one column file per backend/tool via the shared eval tools (`tools/eval_tidystl.py`, `tools/eval_external_tools.py`).

Aggregate normalizes the diagonal (spec name == signal name) of each column, computes the pairwise agreement matrix and the backend-reproduction check, and emits the paper table (`--emit-tex PATH` to write it to a file).
