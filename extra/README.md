# extra/

This directory holds material that is not part of the installed `tidystl` package but is
useful for development, research, and cross-tool validation.

**`other_tools/`:** Scripts that generate ground-truth CSV fixtures by running external
reference tools (Breach, RTAMT, etc.). Run these once in an isolated environment
to regenerate the committed fixtures `packages/tidystl-compat/tests/*_ground_truth.jsonl`
when the test cases change. Each script's docstring contains setup instructions.

**`experiments/`:** Evaluation-campaign experiments (divergence matrix, localization,
variation cost, scaling, falsification studies). See `experiments/README.md` for the layout.

**`outputs/`:** Run records: `outputs/<experiment>/run_NNNN/` with the resolved config,
result, and provenance metadata of one run. Gitignored scratch: every reported number
regenerates by re-running the seeded per-experiment runners.
