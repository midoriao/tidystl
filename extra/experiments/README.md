# Experiment Suite

Evaluation-campaign experiments, one directory per experiment; each is
independently runnable and documented by its own README:

- [`e1_divergence/`](e1_divergence/README.md) -- cross-tool divergence matrix + backend reproduction (V(b))
- [`e2_localization/`](e2_localization/README.md) -- mechanical divergence localization to the responsible operator
- [`e3_variation/`](e3_variation/README.md) -- code cost and locality of each semantic/computational variant
- [`e4_scaling/`](e4_scaling/README.md) -- batch x signal-length scaling, tidystl backends vs real tools
- [`e5a_falsification/`](e5a_falsification/README.md) -- falsification performance under different objective semantics
- [`e5b_verdict_flip/`](e5b_verdict_flip/README.md) -- verdict-flip rate between two named semantic rules

## Conventions

Each experiment has a `Makefile` (`make run` = regenerate the records
via `batch.sh`; `make report` = aggregate + paper emitters; `make all`
= both; `make smoke` = one cheap harness check into `/tmp/smoke/`,
never touching the real records). The suite-level `Makefile` here fans
all four targets out across the experiments. Each experiment directory holds
`run.py` (run one config; `tyro` CLI flags override the frozen defaults), an
optional `batch.sh` (sweeps = repeated `run.py` calls), and
`aggregate.py` (consume saved records; never reruns). A run writes a
committed record `extra/outputs/<experiment>/run_NNNN/` with
`params.json` / `result.json` / `metadata.json`; paper tables and
figures regenerate from those records, never by hand. Experiments that
span two column kinds (e4) have `run_backend.py` and per-tool real-tool
runners (`run_rtamt.py` / `run_stlcgpp.py` / `run_breach.py`, sharing
`_real.py`) instead of a single `run.py`. Each runner declares its config
(`RunParams`, the tyro CLI surface) and its facts (`RunResult`) as frozen
dataclasses, with a `runner(...) -> RunResult` entry point. Aggregates write
their product files
(e.g. `matrix.json`, `scaling_merged.json`, `variation.json`) next to
the records under `extra/outputs/<experiment>/`.

Shared run-tracking helpers (env capture, run-dir allocation, record
writing) live in `_lib/infra.py` (stdlib-only). The registries under
`registry/` are the source of truth for named specs and signals
(hand-editable JSON; see below).

## Exploration tools

`registry/` holds named specs (per-tool syntax under each name) and
named signals as plain hand-editable JSON. `tools/eval_tidystl.py` and
`tools/eval_external_tools.py` evaluate the spec x signal cross product
under one backend / one real tool into one flat JSON
(`--filter-specs` / `--filter-signals` for partial selection) -- no run
records, for quick exploration.

## Environments

Everything runs in one uv environment, `extra/experiments/.venv-exp`,
provisioned from the `experiments` dependency group (the real tools
`rtamt` and `stlcgpp`, plus `tyro` and `tidystl-simd`, all live there).
Each experiment's `Makefile` pins it via `UV_PROJECT_ENVIRONMENT` /
`UV_PYTHON`, and the runners invoke it with
`uv run --only-group experiments python ...`. The `breach` column is the
exception: it has no Python package and shells out to the `matlab` command.
