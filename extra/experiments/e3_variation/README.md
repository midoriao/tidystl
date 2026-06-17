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

Batch sweeps every variant (all compatibility backends -- Breach, RTAMT, RTAMT dense-time, py-MTL, TaLiRo, STLCG++ and its Torch executor, the Rust SIMD executor -- plus the full-suite check entry).

Aggregate reads the records under `--result-dir`, builds the variation table over all gathered variants, writes `variation.json` there, and emits the paper table (`--emit-tex PATH` to write it to a file; `make report` writes `variation.tex` under `$OUTPUTS_ROOT/e3_variation`).

Which variants land in the emitted tex table is configurable and independent of what is gathered: `--tex-variants VARIANT...` selects the rows in table order (default: `breach taliro rtamt`). Data is always gathered for the full catalog regardless of this selection.
