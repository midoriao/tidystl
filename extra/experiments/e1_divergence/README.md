# E1: Cross-Tool Divergence Matrix

## Purpose

Show where real STL tools agree and diverge on documented decision points, and verify that each tidystl backend reproduces its corresponding real tool.

## Batch and Aggregate

No runners: batch.sh produces one column file per backend/tool via the shared eval tools (`../tools/eval_tidystl.py`, `../tools/eval_external_tools.py`).

Aggregate normalizes the diagonal (spec name == signal name) of each column, computes the pairwise agreement matrix and the backend-reproduction check, and emits the supplementary `tab:divergence-matrix` (`--emit-tex PATH` to write it to a file). This matrix is the statistical companion to the paper's profile table.

The paper's `tab:divergence-profile` is emitted by `profile.py`. It reads the Breach row from the committed cache and the rtamt/stlcgpp/pymtl rows from `extra/outputs/e1_divergence/columns/`, so run it after `make run` (which `batch.sh` populates). From the repo root, in the experiments environment:

```bash
UV_PROJECT_ENVIRONMENT=extra/experiments/.venv-exp UV_PYTHON=3.12 UV_NO_SYNC=true \
  uv run --only-group experiments python extra/experiments/e1_divergence/profile.py
```
