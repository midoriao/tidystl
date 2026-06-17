# E4: Scaling Benchmark

## Purpose

Measure how evaluation time scales with batch size and signal length, across tidystl backends and the real tools on the identical workload.

## Runners

Two runners, one per column kind; both time evaluation over the same batch x timestep grid.

### Backend runner

Time one tidystl backend (in-process). `--backend tidystl_simd` selects the
no-trace Rust executor; it ships in the `experiments` dependency group (uv
builds it from `packages/tidystl-simd`), so run it under that group.

| Kind       | Items                                                 |
| ---------- | ----------------------------------------------------- |
| Conditions | `backend` (native + compat backends + tidystl_simd)   |
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

Batch sweeps every tidystl backend the in-process runner can time (`native`, `breach`, `rtamt`, `rtamt_dense`, `pymtl`, `stlcgpp`, `taliro`, `tidystl_simd`) and the real tools (`rtamt-real`, `pymtl-real`, `stlcgpp-real`). Two backends are out of the default sweep: `stlcgpp_torch` needs a `TorchSignal` input the runner does not build, and `breach-real` (`run_breach.py`) is MATLAB-only; run those separately when their toolchain is available.

Aggregate reads the records under `--result-dir` and merges the latest record per column into `scaling_merged.json` there (all collected backends, deriving `per_trace_ms` and `throughput_msts` from the recorded means); `--emit-tex` regenerates the paper table (`make report` writes `scaling.tex` under `$OUTPUTS_ROOT/e4_scaling`).

Which columns land in the emitted tex table is configurable and independent of what is merged: `--tex-columns COLUMN...` selects them in table order (default: `breach taliro rtamt rtamt-real pymtl pymtl-real tidystl_simd`). Choices are the keys of `COLUMN_LABELS` in `aggregate.py` (every backend plus `stlcgpp_torch` and `breach-real`); a selected column with no merged data prints `--`. The real single-trace tools (`rtamt-real`, `pymtl-real`, `breach-real`) are timed at N in {1,8}; larger-N cells are extrapolated linearly and marked with a dagger. py-MTL does not scale to large `T`, so `run_pymtl.py` caps each cell at a 10 s wall-clock budget (`--timeout-s`, `<=0` disables); a capped cell is recorded as `timeout` and renders as `t/o`.
