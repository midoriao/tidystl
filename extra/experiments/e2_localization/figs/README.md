# E2 localization figures

TikZ/`forest` and `booktabs` artifacts for the paper. The hero cases compare two
real-tool-compatible backends against each other (no `native` reference); the
divergence cause lives in the temporal semantics, not the syntax.

| File | Registry case | Tools | Shows |
| --- | --- | --- | --- |
| `localization_window_sampling.tex` | `div_window_sampling` | Breach vs TaLiRo | `forest` only (depth-3 AST; one origin `G[1,2]`; value-only gap); wrap in your own `figure` float / caption |
| `localization_summary_table.tex` | `until_boundary` + `window_sampling` | Breach/RTAMT/TaLiRo/STLCG++ | a single `tabular` (one row per case: tools, `|phi|`/depth, both roots, and a short divergence cell e.g. "verdict flip on $U_{[1,2]}$"); the formulas, the signals (times + per-variable values), the kernel each tool runs at the origin, the 4-tool verdict, and the first-divergence step are emitted as `%` comments. **Auto-generated** by `aggregate.py --emit-tex`; wrap in your own `table` float / caption |

Legend (figure): shaded node = trace differs between the two tools; double-bordered
node = a minimal divergent node (the origin -- its trace differs while all
descendants agree); white node = trace agrees.

The `until_boundary` case is a 3-node tree (`U[1,2]` over two predicates); it is
presented via the summary table rather than its own AST figure.

## Preamble

```latex
\usepackage[edges]{forest}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta}
\usepackage{booktabs}        % for the table
```

Each file contains only its inner environment (`forest` / `tabular`), no float
or caption; `\input` it inside your own `figure` / `table` and supply the caption.

`localization_summary_table.tex` is regenerated from the records by
`make report` (which runs `aggregate.py --emit-tex figs/localization_summary_table.tex`);
its numbers (`|phi|`/depth, roots, origins, the outlier highlight) are computed,
not hand-edited. The AST figure is hand-authored (the tree layout is not
data-driven).

## Reproducing the numbers

All values come from the committed records; regenerate with `make -C .. run report`.
The localizer outputs (verified):

```text
until_boundary  breach vs rtamt : U[1,2]  BoundedUntilKernel[1,2] vs DiscreteBoundedUntil[1,2]   (-5.0 vs +2.0, flip)
until_boundary  rtamt  vs taliro: U[1,2]  DiscreteBoundedUntil[1,2] vs SampleBoundedUntil[1,2]
window_sampling breach vs taliro: G[1,2]  WindowMin[1,2] vs SampleMin[1,2]                       (+2.0 vs +5.0, value gap)
```

The bounded-until verdict across all four tools at `t=0`: Breach `-5.0`,
RTAMT `+2.0` (outlier), TaLiRo `-5.0`, STLCG++ `-5.0`.
