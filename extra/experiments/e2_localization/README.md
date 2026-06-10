# E2: Divergence Localization

## Purpose

Show that a divergence between two semantics can be localized mechanically to the responsible operator.

## Runner

Evaluate one (spec, signal) from the registries under one backend; record per-node traces.

| Kind           | Items                       |
| -------------- | --------------------------- |
| Conditions     | `backend`, `spec`, `signal` |
| Recorded facts | per-node robustness traces  |

## Aggregate

Read the records under `--result-dir`, compare two recorded trace sets over the shared AST, and report the minimal divergent nodes (console output; no product file).
