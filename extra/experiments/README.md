# Experiment Suite

Evaluation-campaign experiments, one directory per experiment; each is
independently runnable and documented by its own README:

- [`e1_divergence/`](e1_divergence/) -- cross-tool divergence matrix 
- [`e2_localization/`](e2_localization/) -- mechanical divergence localization to the responsible operator
- [`e3_variation/`](e3_variation/) -- code cost and locality of each semantic/computational variant
- [`e4_scaling/`](e4_scaling/) -- batch x signal-length scaling, tidystl backends vs real tools
- [`e5b_verdict_flip/`](e5b_verdict_flip/) -- verdict-flip rate between two named semantic rules


## Reproduction

Run these `make` commands from this directory (`extra/experiments/`)
There are three environment layers; provision only what the
target tier needs.

1. **Core layer** (library + core tests): no extra toolchain.

   ```bash
   make check   # core test suite + Core API probe (numpy+lark only, seconds)
   ```

2. **Experiments layer** (the full campaign, including the real tools):

   ```bash
   make bootstrap-exp     # builds .venv-exp
   make smoke   # one cheap run per configured experiment
   make reproduce-full    # regenerate the paper tables and numbers, then list the comparisons
   make compare # show results of the comparisons in the paper
   ```

  Instead of `reproduce-full`, you can run a single experiment directly in its own directory, e.g. `make -C e5b_verdict_flip smoke` or `make -C e5b_verdict_flip all`.

3. **MATLAB layer** (optional): MATLAB + Breach, needed only to *regenerate* the
   committed Breach/TaLiRo reference cache, never to
   *replay* them. A default reproduction does not need MATLAB.

### Per-Experiment Notes

- **e1**: the paper's `tab:divergence-profile` comes from the `profile` target,
  which reads the columns produced by `run`: `make -C e1_divergence run profile`.
  (`report`/`all` emit the supplementary `tab:divergence-matrix`.)
- **e2**: `report` writes the table under `extra/outputs/`; `make -C e2_localization
  emit-ref` refreshes the committed `figs/` reference the paper mirrors.



## Conventions

* Each experiment is driven by its local `Makefile`.

  * `make run` regenerates records.
  * `make report` aggregates existing records and emits paper artifacts.
  * `make all` runs both.
  * `make smoke` performs a cheap check without touching real records.

* A typical experiment contains:

  * `run.py`: run one configuration
  * `batch.sh`: optional sweep script
  * `aggregate.py`: aggregate saved records only

* Runs write records under:

  ```text
  extra/outputs/<experiment>/run_NNNN/
  ```

  Each record contains `params.json`, `result.json`, and `metadata.json`.

* Runner inputs and outputs are frozen dataclasses:

  * `RunParams`
  * `RunResult`


## Exploration tools

- `registry/` holds named fixture specs (per-tool syntax under each name) and named signals as plain hand-editable JSON. 

- `tools/eval_tidystl.py` and `tools/eval_external_tools.py` evaluate the spec x signal cross product under one backend / one real tool into one flat JSON (`--filter-specs` / `--filter-signals` for partial selection)
