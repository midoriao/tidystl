# tidystl: Design Reference

## Design and Implementation

### Formula Representation

`parse()` produces a tree of `Node` objects from an inline STL string.
Temporal operators, boolean connectives, and arithmetic predicates are each
a distinct node kind; the tree is the sole formula representation passed
between all layers.

### Signal Representation

Signals are batch-first numpy arrays with shape `(N, S, T)` (N batch
instances, S signal dimensions, T timesteps) held in a `Signal` object
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
Third-party backends (such as `tidystl-simd`) expose a plugin module that
`tidystl.use(...)` registers with the same registry, requiring no changes to
the core package.

### Extension Surface

The supported plugin seam is `tidystl.use(plugin)`. A plugin module should
expose `register(registry)`; it may also expose `backends()` for callers that
want to inspect or register backend instances manually.

The modules `tidystl.backends.algorithms`, `tidystl.backends.helper`, and
`tidystl.backends._pl_dag` form a stable internal SDK that out-of-tree backend
packages (e.g. `tidystl_compat`, `tidystl_simd`) may import. Changes to those
modules are treated as API changes and will be reflected in the package version.

### Semantics

`NativeBackend` implements exact piecewise-linear dense-time semantics:
signals are linearly interpolated between samples, and temporal operators
compute exact min/max over the continuous interval.
This is the principled reference semantics.

## See Also

- `usage.md`: user manual: concepts, backend selection, and worked examples.
- `language.md`: specification language: grammar tables and robustness rules.
- `api.md`: API reference: call signatures and contracts.
