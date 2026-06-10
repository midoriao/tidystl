# Breach handoff: E1 real-Breach column (RV 2026 evaluation)

One MATLAB run on a Breach-equipped machine produces the entire
real-Breach column of the E1 divergence matrix (eval-execution.md
Sec. 2). Everything else in the campaign runs without it; the matrix
marks the missing cells `pending` until the CSVs land.

## What to run (on the MATLAB machine)

```sh
git clone --branch feat/eval-campaign https://github.com/midoriao/tidystl
git clone https://github.com/decyphir/breach
cd tidystl
matlab -batch "addpath(genpath('../breach')); InitBreach; addpath('extra/other_tools/breach'); generate_ground_truth"
```

Notes:

- `generate_ground_truth` regenerates the 37 regression CSVs under
  `tests/breach_ground_truth/` AND calls `generate_e1_column`, which
  writes the E1 column (21 CSVs + `breach_meta.txt`) to
  `extra/experiments/e1_divergence/cache/breach_column/`.
- If short on time, the E1 column alone is enough:
  `matlab -batch "addpath(genpath('../breach')); InitBreach; addpath('extra/other_tools/breach'); generate_e1_column"`.
- `generate_e1_column.m` is AUTO-GENERATED from
  the spec/signal registries (`extra/experiments/registry/`) by
  `extra/experiments/e1_divergence/breach_handoff.py`; do not edit it by hand.

## Expected output

- 21 `[OK]` lines (16 baseline + 5 divergence cases) under
  `extra/experiments/e1_divergence/cache/breach_column/`.
  (`div_equality` was originally an expected-FAIL probe; the live run
  on Breach 1.11.4 / R2022b showed Breach parses `x[t] == 3` and
  returns BigM sign semantics, so it is now a regular case.)
- `git status` on `tests/breach_ground_truth/` should stay CLEAN
  (byte-identical regeneration). A diff there means a different
  Breach/MATLAB version than the recorded baseline -- do not discard
  it; commit nothing under `tests/` and report the diff instead.

## Handing the results back

```sh
git add extra/experiments/e1_divergence/cache/breach_column
git commit -m "exp(rv2026-eval): real-Breach E1 column from MATLAB run"
git push origin feat/eval-campaign
```

## Integration (back on the campaign machine)

```sh
git pull
uv run python extra/experiments/e1_divergence/aggregate.py
```

`aggregate.py` imports the Breach column directly from `breach_column/`,
preferring those CSVs over the recorded regression CSVs (provenance
recorded per cell), so it picks up the live Breach column, fills the
`pending` matrix cells, and completes the V(b) reproduction check for the
Breach column. The hero cells (`div_boundary_F24`, `div_terminal_and`)
are genuine Breach output by construction.
