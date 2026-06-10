# tidystl: Design Reference

## Design and Implementation

### Formula Representation

`parse()` produces a tree of `Node` objects from an inline STL string.
Temporal operators, boolean connectives, and arithmetic predicates are each
a distinct node kind; the tree is the sole formula representation passed
between all layers.

### Signal Representation

Signals are batch-first numpy arrays with shape `(N, S, T)` — N batch
instances, S signal dimensions, T timesteps — held in a `Signal` object
together with a time vector and a label-to-dimension mapping.
The batch axis makes it possible to evaluate robustness over thousands of
candidate traces in a single call, which is the common pattern in
falsification and parameter synthesis loops.

### Backend Dispatch

Evaluation is dispatched through a `BackendRegistry`.
Each backend implements a single `evaluate(formula, signal)` method that
returns an `EvaluationResult` exposing `robustness` (an ndarray) and,
optionally, per-node traces via `trace_for(node)`.
`robustness()` and `evaluate()` in the public API are thin wrappers over
the registry; the backend is resolved by name or passed directly.
Third-party backends (such as `extra/tidystl_simd`) register themselves at
import time by calling into the same registry, requiring no changes to the
core package.

### Semantics

`NativeBackend` implements exact piecewise-linear dense-time semantics:
signals are linearly interpolated between samples, and temporal operators
compute exact min/max over the continuous interval.
This is the principled reference semantics.
Other backends (`BreachBackend`, `RtamtBackend`, `StlcgppTorchBackend`)
replicate the runtime behavior of their respective tools, including
tool-specific deviations, so that tidystl can serve as a drop-in
replacement or validation layer for existing workflows.

## Cross-Tool Validation

Each backend is regression-tested against ground-truth data from its
reference tool. The test suites cover nested temporal operators, arithmetic
predicates, interpolation-sensitive inputs, and boundary cases.

Known semantics differences between backends are documented rather than
hidden. `NativeBackend` (the default) implements principled piecewise-linear
dense-time semantics:

- Signal model: Piecewise-Linear (PL) continuous time with linear interpolation.
- `G[a,b]` / `F[a,b]`: exact PL window min/max, including interpolated boundary values.
- Until: Maler-Nickovic sliding window.

`BreachBackend` replicates Breach-specific runtime behaviors on top of the same
PL infrastructure. For example, it overwrites the final timestep of top-level
AND/OR results with the penultimate value. It also replicates Breach's
behavior for `G[a,b]` with `a > 0` when no samples fall inside `[t+a, t+b]`:
using `rho(t+a)` at the first sample and `rho(t)` at subsequent samples rather
than the PL window minimum.

Ground truth fixtures and compatibility tests live in `tests/test_breach_compat.py`.
They always run against `BreachBackend` explicitly.

Floating point tolerance for Breach comparison: `atol=1e-6`.

If you find an undocumented discrepancy, please open an issue.

## See Also

- `docs/usage.md` — user manual: concepts, backend selection, and worked examples.
- `docs/reference.md` — API reference: grammar tables, call signatures, contracts.
- `docs/tool_specifications.md` — implementation status and compatibility scope against external tools.
