# E3: Variation Cost Table

## Purpose

Measure the code cost and locality of each semantic/computational variant.

## Runner

Count one variant's specific LOC and run its tests.

| Kind           | Items                                  |
| -------------- | -------------------------------------- |
| Conditions     | `variant`                              |
| Parameters     | `loc_rule`, ...                        |
| Recorded facts | `python_loc`, `rust_loc`, tests passed |

## Batch and Aggregate

Batch sweeps the variants (including the full-suite check entry).

Aggregate reads the records under `--result-dir`, builds the variation table, writes `variation.json` there, and emits the paper table (`--emit-tex PATH` to write it to a file; `make report` writes `variation.tex` under `$OUTPUTS_ROOT/e3_variation`).
