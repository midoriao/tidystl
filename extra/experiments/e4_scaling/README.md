# E4: Scaling Benchmark

## Purpose

Measure how evaluation time scales with batch size and signal length, across tidystl backends and the real tools on the identical workload.

## Runners

Two runners, one per column kind; both time evaluation over the same batch x timestep grid.

### Backend runner

Time one tidystl backend (in-process). `--backend tidystl_simd` selects the
no-trace Rust executor; it ships in the `experiments` dependency group (uv
builds it from `extra/tidystl_simd`), so run it under that group.

| Kind       | Items                                                 |
| ---------- | ----------------------------------------------------- |
| Conditions | `backend` (native, breach, rtamt, tidystl_simd)       |
| Parameters | `spec`, `batch_grid`, `timestep_grid`, `repeats`, ... |
| Metrics    | `mean_ms` / `std_ms`                                  |

### Real-tool runner

Time one real tool, executed in the shared `experiments` venv (`breach` shells out to `matlab`).

| Kind       | Items                                                   |
| ---------- | ------------------------------------------------------- |
| Conditions | `tool`                                                  |
| Parameters | `spec`, `batch_grid`, `timestep_grid`, `repeats`, ...   |
| Metrics    | `mean_ms` / `std_ms`, `repeats`, cell status (e.g. OOM) |

## Batch and Aggregate

Batch sweeps the backends (including `tidystl_simd`) and the tools.

Aggregate reads the records under `--result-dir` and merges the latest record per column into `scaling_merged.json` there, deriving `per_trace_ms` and `throughput_msts` from the recorded means; `--emit-tex` regenerates the paper table (`make report` writes `scaling.tex` under `$OUTPUTS_ROOT/e4_scaling`).
