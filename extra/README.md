# extra/

This directory holds material that is not part of the installed `tidystl` package but is
useful for development, research, and cross-tool validation.

**`other_tools/`** — Scripts that generate ground-truth CSV fixtures by running external
reference tools (Breach/MATLAB, RTAMT, STLCG++). Run these once in an isolated environment
to regenerate `tests/{breach,rtamt,stlcgpp}_ground_truth/` when the test cases change.
Each script's docstring contains setup instructions.

**`tidystl_simd/`** — Optional Rust/SIMD backend (`tidystl-simd` package). A self-contained
maturin sub-project; build it with `maturin develop` from this directory. Once installed,
`import tidystl_simd` registers `SIMDBackend` with the tidystl backend registry.

**`experiments/`** — Evaluation-campaign experiments (divergence matrix, localization,
variation cost, scaling, falsification studies), one directory per experiment with
`config.toml` / `run.py` / `aggregate.py`. See `experiments/README.md` for the layout.

**`outputs/`** — Run records: `outputs/<experiment>/run_NNNN/` with the resolved config,
result, and provenance metadata of one run. Gitignored scratch: every reported number
regenerates by re-running the seeded per-experiment runners. The one committed artifact
is E1's Breach column cache (`e1_divergence/cache/breach_column/`), kept in git because
regenerating it needs the dockerized MATLAB / Breach toolchain.
