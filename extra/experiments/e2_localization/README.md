# E2: Divergence Localization

## Purpose

Show that when two real STL tools disagree on the same formula, the
disagreement can be localized mechanically to the minimal divergent formula node
and the first divergent timestep, reporting each backend's genuine primitive op
there -- with no reference semantics and no syntactic giveaway. The cause lives
in the temporal semantics (how each tool treats a window or a bounded until),
which a human cannot read off the formula. When both backends share the same op
at that node, the divergence is a root-output or executor convention rather than
an op mismatch; the tool flags this but does not name the implicit choice (that
attribution is made in prose from the controlled single-choice cases, Table 1).

The hero cases compare pairs of tool-compatible backends directly
(Breach / RTAMT / TaLiRo / STLCG++), never `native`. STLCG++ reproduces standard
hard robustness and tends to agree with the dense-time reading, so it rarely
diverges -- itself a finding.

## Hero 1: a verdict flip between Breach and RTAMT (`div_until_boundary`)

The bounded-until cell from the E1 matrix, `(x >= 0) U[1,2] (y >= 0)`, on the
2-sample trace `x = [5, -5]`, `y = [-1, 2]`. The end-to-end verdict disagrees:

- **Breach**: root robustness `-5.0` -> **VIOLATED**
- **RTAMT**: root robustness `+2.0` -> **SATISFIED**

Across all four tools, RTAMT is the lone outlier (Breach, TaLiRo, STLCG++ all
give `-5.0`); the cause is RTAMT's discrete-time bounded-until composition at the
trace boundary, not anything visible in the syntax. The localizer pins every
pairwise disagreement to the single `U[1,2]` operator:

```text
$ localize --spec div_until_boundary --signal div_until_boundary --backends breach,rtamt
  3 nodes, temporal depth 1  |  root robustness: breach=-5, rtamt=+2
  first divergent op at `U[1,2]`: BoundedUntilKernel[1,2] vs DiscreteBoundedUntil[1,2] (t-index 0)
```

## Hero 2: a value-only gap between Breach and TaLiRo (`div_window_sampling`)

A divergence need not flip the verdict. `div_window_sampling` buries the cause at
temporal nesting depth 3 (`G[0,3] > F[0,1] > G[1,2]`) on a **non-uniform** grid
with a wide segment `[0,3]`:

```text
G[0,3]( (mode >= 0) and F[0,1]( (enable >= 0) and G[1,2](level >= 0) ) ) and (safety_margin >= 0)
```

The buried `G[1,2](level >= 0)` reduces over a window lying entirely **between
samples** (samples only at `t=0` and `t=3`):

- **Breach** uses the interpolated predicate value at the window start -> the
  branch contributes `+2.0`.
- **TaLiRo** finds no sample inside the closed window and returns the
  always-identity (vacuously satisfied) -> the branch stays high.

Both **SATISFY** the spec; the root robustness differs (`+2.0` vs `+5.0`) -- a
subtle value-only gap with no syntactic giveaway. The two agree everywhere else,
so the localizer reports the single origin:

```text
$ localize --spec div_window_sampling --signal div_window_sampling --backends breach,taliro
  10 nodes, temporal depth 3  |  root robustness: breach=+2, taliro=+5
  first divergent op at `G[1,2]`: WindowMin[1,2] vs SampleMin[1,2] (t-index 0)
```

(RTAMT and STLCG++ cannot run this case at all -- both reject the non-uniform
grid -- so only the dense-time tools apply, which is itself a coverage finding.)

All cases are pure-numpy via `tidystl_compat` -- no MATLAB. `div_terminal_and`
(native vs breach) and `div_boundary_F24` (native vs rtamt) remain as minimal,
single-operator illustrations.

## Runner

Evaluate one (spec, signal) from the registries under one backend; record
per-node robustness traces.

| Kind           | Items                       |
| -------------- | --------------------------- |
| Conditions     | `backend`, `spec`, `signal` |
| Recorded facts | per-node robustness traces  |

## Aggregate

Read the records under `--result-dir`, compare two recorded trace sets over the
shared AST, and report, per (spec, signal, tool-pair): a summary line (AST node
count, temporal nesting depth, and each tool's root robustness at `t=0`) followed
by the minimal divergent nodes -- those whose traces disagree while every
descendant agrees (console output; no product file). With no
`--spec`/`--signal`/`--backends` it compares every recorded tool pair.
