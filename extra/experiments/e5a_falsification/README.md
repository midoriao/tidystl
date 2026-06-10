# E5a: Falsification-Performance Benchmark

## Purpose

Measure whether different robustness semantics affect falsification performance.

## Runner

For a specified spec, run seeded falsification repetitions under one backend.

| Kind       | Items                                                    |
| ---------- | -------------------------------------------------------- |
| Conditions | `backend`                                                |
| Parameters | `spec`, `seed_base`, `n_repetitions`, `eval_budget`, ... |
| Metrics    | `falsifying_rate`, `evals_to_falsification`              |

## Batch and Aggregate

Batch sweeps the backends.

Aggregate groups by backend and reports the metrics (ECDF figure); refuses reduced runs.
